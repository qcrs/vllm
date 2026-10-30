# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate retained bytes, not undefined page tails or FullKV model quality."""

import pytest
import torch

from tests.v1.worker.kv_compaction_utils import scalar_retained_kv


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
