# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate retained bytes, not undefined page tails or FullKV model quality."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from tests.v1.worker.kv_compaction_utils import scalar_retained_kv


def test_reclamation_rejects_alibi_before_cache_mutation():
    """Compacted physical distances cannot preserve ALiBi's logical position bias."""
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        KVCacheConfig,
        KVCacheGroupSpec,
    )
    from vllm.v1.kv_reclamation import ReclamationConfig
    from vllm.v1.worker.gpu.kv_reclamation import KVReclamationWorker

    config = KVCacheConfig(
        num_blocks=4,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer"],
                FullAttentionSpec(
                    block_size=16,
                    num_kv_heads=1,
                    head_size=128,
                    dtype=torch.bfloat16,
                ),
            )
        ],
    )
    worker = KVReclamationWorker.__new__(KVReclamationWorker)
    worker.config = ReclamationConfig()
    layer = SimpleNamespace(impl=SimpleNamespace(alibi_slopes=torch.tensor([0.5])))
    with pytest.raises(ValueError, match="ALiBi position bias"):
        worker.initialize_cache(config, {"layer": torch.empty(0)}, {"layer": layer})


@pytest.mark.parametrize("physical", [False, True])
def test_reclamation_lengths_share_native_prepare_and_postprocess(physical):
    """Reordered requests retain logical positions and advance E only after work."""
    from vllm.v1.worker.gpu.input_batch import post_update, prepare_pos_seq_lens

    logical = torch.tensor([20, 7, 98, 1], dtype=torch.int32, device="cuda")
    effective = torch.tensor([9, 2, 33, 0], dtype=torch.int32, device="cuda")
    indices = torch.tensor([3, 1], dtype=torch.int32, device="cuda")
    starts = torch.tensor([0, 1, 4], dtype=torch.int32, device="cuda")
    positions = torch.full((8,), -1, dtype=torch.int64, device="cuda")
    lengths = torch.full((4,), -1, dtype=torch.int32, device="cuda")
    cache_positions = torch.full_like(positions, -1)
    kv_lengths = torch.full_like(lengths, -1)
    prepare_pos_seq_lens(
        indices,
        starts,
        logical,
        positions,
        lengths,
        (effective, cache_positions, kv_lengths) if physical else None,
    )
    assert positions[:4].tolist() == [1, 7, 8, 9]
    assert lengths.tolist() == [2, 10, 0, 0]
    assert effective.tolist() == [9, 2, 33, 0]
    if physical:
        assert cache_positions[:4].tolist() == [0, 2, 3, 4]
        assert kv_lengths.tolist() == [1, 5, 0, 0]
    else:
        assert kv_lengths.tolist() == [-1] * 4

    # The third sampler row is masked; the second is an unfinished prefill.
    post_update(
        torch.tensor([3, 1, -1], dtype=torch.int32, device="cuda"),
        logical,
        torch.zeros_like(logical),
        None,
        torch.tensor([[42], [0], [0]], dtype=torch.int64, device="cuda"),
        torch.tensor([1, 0, 0], dtype=torch.int32, device="cuda"),
        torch.zeros(3, dtype=torch.int32, device="cuda"),
        starts,
        torch.zeros((4, 16), dtype=torch.int32, device="cuda"),
        torch.tensor([0, 3, 0, 4], dtype=torch.int32, device="cuda"),
        effective if physical else None,
    )
    assert logical.tolist() == [20, 10, 98, 2]
    assert effective.tolist() == ([9, 5, 33, 1] if physical else [9, 2, 33, 0])


def test_graph_attention_lengths_keep_logical_inputs_and_refresh_dummy_padding():
    from vllm.v1.worker.gpu.cudagraph_utils import prepare_inputs_to_capture
    from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
    from vllm.v1.worker.gpu.kv_reclamation import KVReclamationWorker

    buffers = InputBuffers(4, 8, torch.device("cpu"))
    physical = torch.full_like(buffers.seq_lens, 999)
    model_state = MagicMock()
    tables = SimpleNamespace(
        cp_size=1,
        get_dummy_block_tables=lambda n: (torch.zeros(n, 2, dtype=torch.int32),),
        get_dummy_slot_mappings=lambda n: torch.full((1, n), -1),
    )
    cache = SimpleNamespace(kv_cache_groups=[SimpleNamespace(layer_names=["layer"])])
    prepare_inputs_to_capture(
        2, 2, model_state, buffers, tables, [], cache, attn_seq_lens=physical
    )
    captured_batch = model_state.prepare_attn.call_args.args[0]
    assert captured_batch.seq_lens.data_ptr() == physical.data_ptr()
    assert captured_batch.seq_lens.data_ptr() != buffers.seq_lens.data_ptr()
    assert physical.tolist() == [1, 1, 0, 0]
    physical[:2] = torch.tensor([13, 31])
    assert captured_batch.seq_lens.tolist() == [13, 31]
    assert buffers.seq_lens.tolist() == [1, 1, 0, 0]

    worker = KVReclamationWorker.__new__(KVReclamationWorker)
    worker.kv_seq_lens = physical
    dummy = InputBatch.make_dummy(1, 3, buffers)
    attn_dummy = worker.prepare_dummy_attn(dummy)
    assert physical.tolist() == [3, 0, 0, 0]
    assert attn_dummy.seq_lens.data_ptr() == captured_batch.seq_lens.data_ptr()
    assert dummy.seq_lens.data_ptr() == buffers.seq_lens.data_ptr()
    assert attn_dummy.positions is dummy.positions


def dense_frequency_reference(kv, stats, logical_next, theta):
    """Independent FP32 complex arithmetic; no production score/address helper."""
    keys = kv[:, :, : kv.shape[2] // 2].float()
    frequency = keys.shape[2] // 2
    k = torch.complex(keys[..., :frequency], keys[..., frequency:]).transpose(0, 1)
    mu = torch.complex(stats[:, 0], stats[:, 1])
    omega = 1 / theta ** (torch.arange(frequency).float() / frequency)
    outputs = []
    group = stats.shape[0] // keys.shape[1]
    for head, mean in enumerate(mu):
        rotated = mean * k[head // group].conj()
        score = torch.stack(
            [
                (
                    rotated
                    * torch.polar(torch.ones_like(omega), (logical_next + 2**o) * omega)
                ).real.sum(dim=-1)
                for o in range(17)
            ]
        ).mean(dim=0)
        norm = (stats[head, 0].square() + stats[head, 1].square() + 1e-8).sqrt()
        magnitude = (k[head // group].abs().square() + 1e-8).sqrt()
        outputs.append(score + ((stats[head, 2] - norm) * magnitude).sum(dim=-1))
    return torch.stack(outputs)


@pytest.mark.parametrize("statistics_heads", [2, 6, 8])
def test_paged_shared_frequency_scores_use_logical_time_and_gqa(statistics_heads):
    from vllm.v1.worker.gpu.kv_scoring import TriAttentionScorer, protected_topk

    generator = torch.Generator().manual_seed(91)
    cpu_layers, layers = [], []
    for _ in range(2):
        cpu = torch.empty_strided((7, 2, 16, 256), (16513, 512, 1024, 2))
        cpu.copy_(torch.randn(cpu.shape, generator=generator))
        gpu = torch.empty_strided(
            cpu.shape, cpu.stride(), device="cuda", dtype=torch.bfloat16
        )
        gpu.copy_(cpu)
        cpu_layers.append(gpu.cpu().float())
        layers.append(gpu)
    row, source, logical_next = [6, 1, 4, 0, 5, 2], 95, 18123
    row_gpu = torch.tensor(row, dtype=torch.int32, device="cuda")
    logical_gpu = torch.tensor([logical_next], dtype=torch.int32, device="cuda")
    stats = torch.randn((2, statistics_heads, 3, 64), generator=generator) * 0.2
    stats[:, :, 2] = (stats[:, :, 0].square() + stats[:, :, 1].square()).sqrt() + 0.3
    scorer = TriAttentionScorer(stats.cuda(), 2, 1e6, 128)
    expected = []
    prefix, recent, keep = 3, 88, 41
    for layer in range(2):
        dense, _ = scalar_retained_kv([cpu_layers[layer]], row, list(range(source)))
        raw = dense_frequency_reference(dense[0], stats[layer], logical_next, 1e6)
        actual = scorer.score_layer(layers[layer], row_gpu, logical_gpu, source, layer)
        torch.testing.assert_close(actual.cpu(), raw, atol=2e-3, rtol=1e-3)
        var, mean = torch.var_mean(raw[:, prefix:recent], dim=1, correction=0)
        normalized = (raw - mean[:, None]) / var.sqrt()[:, None]
        expected.append(
            normalized.view(2, statistics_heads // 2, source).amax(1).mean(0)
        )
    reference = torch.stack(expected).mean(0)
    actual = scorer.shared_scores(layers, row_gpu, logical_gpu, source, prefix, recent)
    torch.testing.assert_close(actual.cpu(), reference, atol=2e-3, rtol=1e-3)
    actual_plan = protected_topk(actual, keep, prefix, recent).cpu()
    ranked = sorted(range(prefix, recent), key=lambda i: (-float(reference[i]), i))
    expected_plan = sorted(
        [
            *range(prefix),
            *ranked[: keep - prefix - (source - recent)],
            *range(recent, source),
        ]
    )
    assert actual_plan.tolist() == expected_plan


def test_shared_frequency_zero_variance_and_stable_protection():
    from vllm.v1.worker.gpu.kv_scoring import TriAttentionScorer, protected_topk

    cache = torch.ones((2, 2, 16, 256), dtype=torch.bfloat16, device="cuda")
    stats = torch.ones((1, 8, 3, 64), device="cuda")
    scorer = TriAttentionScorer(stats, 2, 1e6, 32)
    scores = scorer.shared_scores(
        [cache],
        torch.tensor([1], dtype=torch.int32, device="cuda"),
        torch.tensor([1234], dtype=torch.int32, device="cuda"),
        10,
        2,
        7,
    )
    assert torch.equal(scores, torch.zeros_like(scores))
    assert protected_topk(scores, 7, 2, 7).tolist() == [0, 1, 2, 3, 7, 8, 9]


def test_shared_frequency_reduction_preserves_ties_at_partial_tile_boundary(
    monkeypatch,
):
    """Non-power-of-two head groups retain exact ties and a nearby cutoff."""
    from vllm.v1.worker.gpu.kv_scoring import TriAttentionScorer, protected_topk

    source, prefix, recent, keep = 131, 3, 128, 68
    raw = torch.arange(source, dtype=torch.float32).repeat(9, 1)
    raw[:, 65:68] = 66
    raw[:, 68] = torch.nextafter(torch.tensor(66.0), torch.tensor(67.0))
    raw[0].fill_(1)
    var, mean = torch.var_mean(raw.double()[:, prefix:recent], dim=1, correction=0)
    std = var.sqrt()
    normalized = (raw.double() - mean[:, None]) / torch.where(std > 0, std, 1)[:, None]
    expected = normalized.reshape(3, 3, source).amax(1).mean(0)
    scorer = TriAttentionScorer(torch.zeros(1, 9, 3, 64, device="cuda"), 3, 1e6, 144)
    device_raw = raw.cuda()
    monkeypatch.setattr(scorer, "prepare_phase", lambda _: None)
    monkeypatch.setattr(scorer, "_score_layer", lambda *args: device_raw)
    scores = scorer.shared_scores([None], None, None, source, prefix, recent)
    torch.testing.assert_close(scores.cpu().double(), expected, atol=2e-7, rtol=2e-6)
    expected_plan = protected_topk(expected, keep, prefix, recent)
    assert torch.equal(
        protected_topk(scores, keep, prefix, recent).cpu(), expected_plan
    )


@pytest.mark.parametrize("kv_heads", [1, 4])
def test_tp_frequency_contributions_match_complete_unique_gqa_reference(kv_heads):
    from vllm.v1.worker.gpu.kv_scoring import TriAttentionScorer, protected_topk

    generator = torch.Generator().manual_seed(142)
    source, logical, prefix, recent, keep = 47, 17031, 3, 41, 23
    row = torch.tensor([3, 1, 0], dtype=torch.int32, device="cuda")
    clock = torch.tensor([logical], dtype=torch.int32, device="cuda")
    stats = torch.randn(2, 8, 3, 64, generator=generator) * 0.2
    stats[:, :, 2] = stats[:, :, :2].square().sum(2).sqrt() + 0.3
    layers = [
        torch.randn(4, kv_heads, 16, 256, generator=generator).to(
            device="cuda", dtype=torch.bfloat16
        )
        for _ in range(2)
    ]
    expected = []
    for layer, cache in enumerate(layers):
        dense, _ = scalar_retained_kv([cache.cpu()], row.tolist(), list(range(source)))
        raw = dense_frequency_reference(dense[0], stats[layer], logical, 1e6)
        var, mean = torch.var_mean(raw[:, prefix:recent], dim=1, correction=0)
        normalized = (raw - mean[:, None]) / var.sqrt()[:, None]
        expected.append(
            normalized.view(kv_heads, 8 // kv_heads, source).amax(1).mean(0)
        )
    reference = torch.stack(expected).mean(0)
    total = torch.zeros(source, device="cuda")
    for rank in range(2):
        scorer = TriAttentionScorer(stats, kv_heads, 1e6, 64, 2, rank, row.device)
        local = (
            layers
            if kv_heads == 1
            else [v[:, rank * 2 : (rank + 1) * 2] for v in layers]
        )
        contribution = scorer.shared_scores(local, row, clock, source, prefix, recent)
        if kv_heads == 1 and rank == 1:
            assert scorer.stats.shape[1] == 0 and contribution.count_nonzero() == 0
        else:
            assert scorer.stats.shape[1] == (8 if kv_heads == 1 else 4)
        total += contribution
    torch.testing.assert_close(total.cpu(), reference, atol=2e-3, rtol=1e-3)
    ranked = sorted(range(prefix, recent), key=lambda i: (-float(reference[i]), i))
    expected_plan = sorted(
        [
            *range(prefix),
            *ranked[: keep - prefix - (source - recent)],
            *range(recent, source),
        ]
    )
    assert protected_topk(total, keep, prefix, recent).tolist() == expected_plan


@pytest.mark.parametrize("whole_blocks", [False, True])
def test_scalar_oracle_decodes_nonmonotonic_pages_and_partial_tail(whole_blocks):
    layers = []
    for layer in range(2):
        cache = torch.empty_strided((7, 2, 4, 6), (128, 6, 12, 1))
        for page in range(7):
            for head in range(2):
                for token in range(4):
                    for channel in range(6):
                        cache[page, head, token, channel] = (
                            layer * 10000
                            + page * 1000
                            + token * 100
                            + head * 10
                            + channel
                        )
        layers.append(cache)
    keep = [0, 1, 2, 3, 8, 9]
    values, row = scalar_retained_kv(layers, [6, 1, 4], keep, whole_blocks=whole_blocks)
    assert row == ([6, 4] if whole_blocks else [6, 1])
    for layer in range(2):
        for j, source_page in enumerate([6, 6, 6, 6, 4, 4]):
            assert values[layer][j, 1, 5] == (
                layer * 10000 + source_page * 1000 + j % 4 * 100 + 15
            )


@pytest.mark.parametrize("plan", ["identity", "shift_one", "alternating", "recent"])
@pytest.mark.parametrize("channel_stride", [1, 2])
def test_stable_gpu_compaction_preserves_overlapping_members(plan, channel_stride):
    """Cross-warp/tile overlap, padding and nonmonotonic IDs match scalar bytes."""
    from vllm.v1.worker.gpu.kv_compaction import StableKVCompactor

    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    shape = (7, 2, 16, 256)
    strides = (
        16 * 512 * channel_stride + 128,
        256 * channel_stride,
        512 * channel_stride,
        channel_stride,
    )
    generator = torch.Generator().manual_seed(19)
    cpu_layers, gpu_layers = [], []
    for _ in range(2):
        cpu = torch.empty_strided(shape, strides, dtype=torch.bfloat16)
        cpu.copy_(torch.randn(shape, generator=generator))
        backing = torch.empty_strided(
            (8, *shape[1:]), strides, dtype=torch.bfloat16, device="cuda"
        )
        gpu = backing[1:]
        gpu.copy_(cpu)
        cpu_layers.append(cpu)
        gpu_layers.append(gpu)
    row = [6, 1, 4, 0, 5, 2]
    plans = {
        "identity": list(range(95)),
        "shift_one": [0, *range(2, 95)],
        "alternating": list(range(0, 95, 2)),
        "recent": [*range(16), *range(63, 95)],
    }
    keep = plans[plan]
    expected, new_row = scalar_retained_kv(cpu_layers, row, keep)
    compactor = StableKVCompactor(gpu_layers)
    compiled = compactor.compact(
        torch.tensor(row, dtype=torch.int32, device="cuda"),
        torch.tensor(keep, dtype=torch.int32, device="cuda"),
    )
    for gpu, target in zip(gpu_layers, expected):
        actual, _ = scalar_retained_kv([gpu.cpu()], new_row, list(range(len(keep))))
        assert torch.equal(actual[0], target)
    # Inspect the compiler used by this test, not a remembered Triton version.
    assert "bar.sync" in compiled.asm["ptx"]
    import os
    from pathlib import Path

    if output := os.environ.get("P1_RUN_DIR"):
        directory = Path(output) / "compiler" / f"{plan}-stride{channel_stride}"
        directory.mkdir(parents=True, exist_ok=True)
        for kind in ("ttir", "ttgir", "llir", "ptx"):
            (directory / f"stable.{kind}").write_text(compiled.asm[kind])
