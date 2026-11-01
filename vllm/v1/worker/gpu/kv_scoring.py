# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TriAttention frequency scores from packed paged K_rot and a shared plan."""

import hashlib
import io
from pathlib import Path

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _future_query_phase(logical_length, frequencies, coefficients, F: tl.constexpr):
    frequency = tl.arange(0, triton.next_power_of_2(F))
    omega = tl.load(frequencies + frequency, frequency < F, 0)
    logical_next = tl.load(logical_length).to(tl.float32)
    cosine = tl.full((triton.next_power_of_2(F),), 0, tl.float32)
    sine = tl.full((triton.next_power_of_2(F),), 0, tl.float32)
    for offset in tl.static_range(17):
        phase = (logical_next + (1 << offset)) * omega
        cosine += tl.cos(phase) / 17
        sine += tl.sin(phase) / 17
    tl.store(coefficients + frequency, cosine, frequency < F)
    tl.store(coefficients + F + frequency, sine, frequency < F)


@triton.jit
def _paged_frequency_scores(
    cache,
    row,
    stats,
    coefficients,
    scores,
    E,
    OUTPUT_STRIDE: tl.constexpr,
    SB: tl.constexpr,
    SH: tl.constexpr,
    ST: tl.constexpr,
    SC: tl.constexpr,
    B: tl.constexpr,
    F: tl.constexpr,
    GROUP: tl.constexpr,
    TILE: tl.constexpr,
):
    head = tl.program_id(0)
    member = tl.program_id(1) * TILE + tl.arange(0, TILE)
    frequency = tl.arange(0, triton.next_power_of_2(F))
    page = tl.load(row + member // B, member < E, 0).to(tl.int64)
    address = page[:, None] * SB + head * SH + (member[:, None] % B) * ST
    mask = (member[:, None] < E) & (frequency[None, :] < F)
    real = tl.load(cache + address + frequency[None, :] * SC, mask, 0).to(tl.float32)
    imag = tl.load(cache + address + (frequency[None, :] + F) * SC, mask, 0).to(
        tl.float32
    )
    magnitude = tl.sqrt(real * real + imag * imag + 1.0e-8)
    cosine = tl.load(coefficients + frequency, frequency < F, 0)
    sine = tl.load(coefficients + F + frequency, frequency < F, 0)
    for query in tl.static_range(GROUP):
        query_head = head * GROUP + query
        qr = tl.load(stats + (query_head * 3) * F + frequency, frequency < F, 0)
        qi = tl.load(stats + (query_head * 3 + 1) * F + frequency, frequency < F, 0)
        qa = tl.load(stats + (query_head * 3 + 2) * F + frequency, frequency < F, 0)
        mu_abs = tl.sqrt(qr * qr + qi * qi + 1.0e-8)
        prod_real = qr[None, :] * real + qi[None, :] * imag
        prod_imag = qi[None, :] * real - qr[None, :] * imag
        score = tl.sum(
            prod_real * cosine[None, :]
            - prod_imag * sine[None, :]
            + (qa - mu_abs)[None, :] * magnitude,
            1,
        )
        tl.store(scores + query_head * OUTPUT_STRIDE + member, score, member < E)


@triton.jit
def _normalize_reduce_scores(
    scores,
    variance,
    mean,
    reduced,
    E,
    STRIDE: tl.constexpr,
    HEADS: tl.constexpr,
    GROUP: tl.constexpr,
    TILE: tl.constexpr,
    WEIGHT: tl.constexpr = 1.0,
    ACCUMULATE: tl.constexpr = False,
):
    member = tl.program_id(0) * TILE + tl.arange(0, TILE)
    head = tl.arange(0, triton.next_power_of_2(HEADS))
    query = tl.arange(0, triton.next_power_of_2(GROUP))
    statistics_head = head[:, None] * GROUP + query[None, :]
    valid_head = (head[:, None] < HEADS) & (query[None, :] < GROUP)
    var = tl.load(variance + statistics_head, valid_head, 0)
    mu = tl.load(mean + statistics_head, valid_head, 0)
    std = tl.sqrt(var)
    denominator = tl.where(std > 0, std, 1)
    value = tl.load(
        scores + statistics_head[:, :, None] * STRIDE + member[None, None, :],
        valid_head[:, :, None] & (member[None, None, :] < E),
        0,
    )
    normalized = tl.div_rn(value - mu[:, :, None], denominator[:, :, None])
    grouped = tl.max(tl.where(valid_head[:, :, None], normalized, -float("inf")), 1)
    result = tl.sum(tl.where(head[:, None] < HEADS, grouped, 0), 0) / HEADS
    if ACCUMULATE:
        previous = tl.load(reduced + member, member < E, 0)
        result = tl.fma(result, tl.full((), WEIGHT, tl.float32), previous)
    tl.store(reduced + member, result, member < E)


def protected_topk(
    scores: torch.Tensor, keep: int, prefix: int, recent: int
) -> torch.Tensor:
    """Stable score ties use member order; the resulting shared plan is sorted."""
    source = scores.numel()
    budget = keep - prefix - (source - recent)
    selected = torch.argsort(scores[prefix:recent], descending=True, stable=True)[
        :budget
    ]
    return (
        torch.cat(
            (
                torch.arange(prefix, device=scores.device),
                selected + prefix,
                torch.arange(recent, source, device=scores.device),
            )
        )
        .sort()
        .values.to(torch.int32)
    )


class TriAttentionScorer:
    def __init__(
        self,
        stats: torch.Tensor,
        kv_heads: int,
        theta: float,
        max_len: int,
        tp_size: int = 1,
        tp_rank: int = 0,
        device: torch.device | None = None,
    ):
        # [layer, statistics_head, real/imag/abs, frequency].
        self.global_kv_heads = kv_heads
        self.calibration_sha256: str | None = None
        self.group = stats.shape[1] // kv_heads
        self.kv_heads = max(1, kv_heads // tp_size)
        replicas = max(1, tp_size // kv_heads)
        self.scoring_owner = tp_rank % replicas == 0
        first = tp_rank * self.kv_heads if kv_heads >= tp_size else tp_rank // replicas
        heads = self.kv_heads if self.scoring_owner else 0
        self.stats = (
            stats[:, first * self.group : (first + heads) * self.group]
            .to(device=device, dtype=torch.float32)
            .contiguous()
        )
        stats = self.stats
        self.theta = theta
        self.frequencies = 1 / (
            theta
            ** (
                torch.arange(stats.shape[3], dtype=torch.float32, device=stats.device)
                / stats.shape[3]
            )
        )
        self.coefficients = torch.empty(
            (2, stats.shape[3]), dtype=torch.float32, device=stats.device
        )
        self.scores = torch.empty(
            (stats.shape[1], max_len), dtype=torch.float32, device=stats.device
        )
        self.total = torch.empty(max_len, dtype=torch.float32, device=stats.device)

    @classmethod
    def from_file(cls, path, model_config, max_len, device, tp_size=1, tp_rank=0):
        artifact = Path(path).read_bytes()
        payload = torch.load(
            io.BytesIO(artifact), map_location="cpu", weights_only=True
        )
        metadata, stats = payload["metadata"], payload["stats"]
        hf = model_config.hf_text_config
        rope = hf.rope_parameters
        expected = {
            "model_type": hf.model_type,
            "num_layers": hf.num_hidden_layers,
            "num_attention_heads": hf.num_attention_heads,
            "num_kv_heads": hf.num_key_value_heads,
            "head_dim": hf.head_dim,
            "rope_parameters": rope,
            "rope_style": "half",
            "q_normalized": True,
            "model_config_sha256": hashlib.sha256(
                (Path(model_config.model) / "config.json").read_bytes()
            ).hexdigest(),
        }
        if (
            metadata.get("calibration") is None
            or any(metadata.get(k) != value for k, value in expected.items())
            or rope.get("rope_type", "default") != "default"
        ):
            raise ValueError("Frequency calibration does not match model/RoPE contract")
        shape = (hf.num_hidden_layers, hf.num_attention_heads, 3, hf.head_dim // 2)
        if stats.shape != shape or not torch.isfinite(stats).all():
            raise ValueError("Frequency stats require all model layers/query heads")
        scorer = cls(
            stats,
            hf.num_key_value_heads,
            rope["rope_theta"],
            max_len,
            tp_size,
            tp_rank,
            device,
        )
        scorer.calibration_sha256 = hashlib.sha256(artifact).hexdigest()
        return scorer

    def score_layer(self, cache, row, logical_length, source_len, layer):
        self.prepare_phase(logical_length)
        return self._score_layer(cache, row, source_len, layer)

    def prepare_phase(self, logical_length):
        _future_query_phase[(1,)](
            logical_length,
            self.frequencies,
            self.coefficients,
            self.stats.shape[3],
            enable_fp_fusion=False,
        )

    def _score_layer(self, cache, row, source_len, layer):
        _paged_frequency_scores[(self.kv_heads, triton.cdiv(source_len, 16))](
            cache,
            row,
            self.stats[layer],
            self.coefficients,
            self.scores,
            source_len,
            self.scores.stride(0),
            *cache.stride(),
            cache.shape[2],
            self.stats.shape[3],
            self.group,
            16,
            enable_fp_fusion=False,
        )
        return self.scores[:, :source_len]

    def shared_scores(self, layers, row, logical_length, source_len, prefix, recent):
        total = self.total[:source_len]
        total.zero_()
        if prefix == recent or not self.scoring_owner:
            return total
        self.prepare_phase(logical_length)
        for layer, cache in enumerate(layers):
            scores = self._score_layer(cache, row, source_len, layer)
            variance, mean = torch.var_mean(
                scores[:, prefix:recent], dim=1, correction=0
            )
            _normalize_reduce_scores[(triton.cdiv(source_len, 128),)](
                scores,
                variance,
                mean,
                total,
                source_len,
                scores.stride(0),
                self.kv_heads,
                self.group,
                128,
                self.kv_heads / (self.global_kv_heads * len(layers)),
                True,
                enable_fp_fusion=False,
            )
        return total

    def plan(self, layers, row, logical_length, source_len, keep, prefix, recent):
        scores = self.shared_scores(
            layers, row, logical_length, source_len, prefix, recent
        )
        return protected_topk(scores, keep, prefix, recent)


@triton.jit
def _grouped_raw(
    pointers,
    strides,
    row,
    stats,
    coefficients,
    scores,
    E,
    FIRST: tl.constexpr,
    LAYERS: tl.constexpr,
    STRIDE: tl.constexpr,
    STATS_STRIDE: tl.constexpr,
    F: tl.constexpr,
    GROUP: tl.constexpr,
    HEADS: tl.constexpr,
    B: tl.constexpr,
    TILE: tl.constexpr,
):
    head = tl.program_id(0)
    local_layer = tl.program_id(2)
    layer = FIRST + local_layer
    cache = tl.load(pointers + layer).to(tl.pointer_type(tl.bfloat16))
    sb = tl.load(strides + layer * 4)
    sh = tl.load(strides + layer * 4 + 1)
    st = tl.load(strides + layer * 4 + 2)
    sc = tl.load(strides + layer * 4 + 3)
    member = tl.program_id(1) * TILE + tl.arange(0, TILE)
    frequency = tl.arange(0, triton.next_power_of_2(F))
    page = tl.load(row + member // B, member < E, 0).to(tl.int64)
    address = page[:, None] * sb + head * sh + (member[:, None] % B) * st
    mask = (member[:, None] < E) & (frequency[None, :] < F)
    real = tl.load(cache + address + frequency[None, :] * sc, mask, 0).to(tl.float32)
    imag = tl.load(cache + address + (frequency[None, :] + F) * sc, mask, 0).to(
        tl.float32
    )
    magnitude = tl.sqrt(real * real + imag * imag + 1.0e-8)
    cosine = tl.load(coefficients + frequency, frequency < F, 0)
    sine = tl.load(coefficients + F + frequency, frequency < F, 0)
    for query in tl.static_range(GROUP):
        query_head = head * GROUP + query
        base = stats + layer * STATS_STRIDE + query_head * 3 * F
        qr = tl.load(base + frequency, frequency < F, 0)
        qi = tl.load(base + F + frequency, frequency < F, 0)
        qa = tl.load(base + 2 * F + frequency, frequency < F, 0)
        mu_abs = tl.sqrt(qr * qr + qi * qi + 1.0e-8)
        prod_real = qr[None, :] * real + qi[None, :] * imag
        prod_imag = qi[None, :] * real - qr[None, :] * imag
        score = tl.sum(
            prod_real * cosine[None, :]
            - prod_imag * sine[None, :]
            + (qa - mu_abs)[None, :] * magnitude,
            1,
        )
        tl.store(
            scores + (local_layer * HEADS * GROUP + query_head) * STRIDE + member,
            score,
            member < E,
        )


@triton.jit
def _grouped_normalize(
    scores,
    variance,
    mean,
    total,
    E,
    STRIDE: tl.constexpr,
    HEADS: tl.constexpr,
    GROUP: tl.constexpr,
    LAYERS: tl.constexpr,
    WEIGHT: tl.constexpr,
):
    for layer in tl.static_range(LAYERS):
        _normalize_reduce_scores(
            scores + layer * HEADS * GROUP * STRIDE,
            variance + layer * HEADS * GROUP,
            mean + layer * HEADS * GROUP,
            total,
            E,
            STRIDE,
            HEADS,
            GROUP,
            128,
            WEIGHT,
            True,
        )


class GroupedTriAttentionScorer(TriAttentionScorer):
    def initialize_groups(self, compactor):
        self.compactor = compactor
        self.group_scores = torch.empty(
            (4, self.stats.shape[1], self.total.numel()),
            device=self.total.device,
            dtype=torch.float32,
        )
        self.group_variance = torch.empty(
            (4, self.stats.shape[1]),
            device=self.total.device,
        )
        self.group_mean = torch.empty_like(self.group_variance)

    def shared_scores(self, layers, row, logical_length, source_len, prefix, recent):
        total = self.total[:source_len]
        total.zero_()
        if prefix == recent or not self.scoring_owner:
            return total
        self.prepare_phase(logical_length)
        for first in range(0, len(layers), 4):
            count = min(4, len(layers) - first)
            scores = self.group_scores[:count, :, :source_len]
            _grouped_raw[(self.kv_heads, triton.cdiv(source_len, 16), count)](
                self.compactor.layer_ptrs,
                self.compactor.strides,
                row,
                self.stats,
                self.coefficients,
                scores,
                source_len,
                first,
                count,
                self.total.numel(),
                self.stats.stride(0),
                self.stats.shape[3],
                self.group,
                self.kv_heads,
                layers[0].shape[2],
                16,
                enable_fp_fusion=False,
            )
            torch.ops.aten.var_mean.correction_out(
                scores[:, :, prefix:recent],
                [2],
                correction=0,
                out0=self.group_variance[:count],
                out1=self.group_mean[:count],
            )
            _grouped_normalize[(triton.cdiv(source_len, 128),)](
                scores,
                self.group_variance,
                self.group_mean,
                total,
                source_len,
                self.total.numel(),
                self.kv_heads,
                self.group,
                count,
                self.kv_heads / (self.global_kv_heads * len(layers)),
                enable_fp_fusion=False,
            )
        return total
