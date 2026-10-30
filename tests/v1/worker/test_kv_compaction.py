# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate retained bytes, not undefined page tails or FullKV model quality."""

import pytest
import torch

from tests.v1.worker.kv_compaction_utils import scalar_retained_kv


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


@pytest.mark.parametrize("statistics_heads", [2, 8])
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
