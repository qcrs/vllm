# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA-event latency and profiler entrypoint for stable paged BF16 compaction."""

import argparse
import json
import os
import statistics
from functools import partial
from pathlib import Path

import torch

from vllm.v1.worker.gpu.kv_compaction import StableKVCompactor


def reset_layers(layers, sources):
    for dest, source in zip(layers, sources):
        dest.copy_(source)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lengths", type=int, nargs="+", default=[4096, 32768])
    parser.add_argument("--plans", nargs="+", default=["recent", "alternating"])
    parser.add_argument("--configs", nargs="+", default=["16:4"])
    parser.add_argument("--iterations", type=int, default=15)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    run = Path(os.environ["P1_RUN_DIR"])
    torch.cuda.set_per_process_memory_fraction(0.5)
    torch.manual_seed(11)
    records = []
    for length in args.lengths:
        pages = (length + 15) // 16
        # NHD storage exposed as the runtime's packed [page,head,token,2D] view.
        sources = [
            torch.randn(
                pages, 16, 8, 256, dtype=torch.bfloat16, device="cuda"
            ).transpose(1, 2)
            for _ in range(36)
        ]
        layers = [torch.empty_like(t) for t in sources]
        compactor = StableKVCompactor(layers)
        row = torch.randperm(pages, device="cuda", dtype=torch.int32)
        for plan in args.plans:
            keep = (
                torch.arange(0, length, 2, dtype=torch.int32, device="cuda")
                if plan == "alternating"
                else torch.cat(
                    (
                        torch.arange(16, dtype=torch.int32, device="cuda"),
                        torch.arange(
                            length // 2 + 16, length, dtype=torch.int32, device="cuda"
                        ),
                    )
                )
            )
            changed = int((keep != torch.arange(keep.numel(), device="cuda")).sum())
            for setting in args.configs:
                tile, warps = map(int, setting.split(":"))

                reset = partial(reset_layers, layers, sources)
                launch = partial(
                    compactor.compact, row, keep, tile=tile, num_warps=warps
                )

                reset()
                compiled = launch()
                torch.accelerator.synchronize()
                # This benchmark's correctness check also covers the measured shape.
                for source, dest in zip(sources, layers):
                    expected = source[row[keep // 16].long(), :, keep % 16, :]
                    actual = dest[row[: (keep.numel() + 15) // 16].long()]
                    actual = actual.transpose(1, 2).reshape(-1, 8, 256)[: keep.numel()]
                    assert torch.equal(actual, expected)
                for _ in range(3):
                    reset()
                    launch()
                torch.accelerator.synchronize()
                samples = []
                if args.profile:
                    reset()
                    torch.accelerator.synchronize()
                    torch.cuda.cudart().cudaProfilerStart()
                    with torch.cuda.nvtx.range("compact"):
                        launch()
                    torch.accelerator.synchronize()
                    torch.cuda.cudart().cudaProfilerStop()
                else:
                    for _ in range(args.iterations):
                        reset()
                        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
                        start.record()
                        launch()
                        end.record()
                        end.synchronize()
                        samples.append(start.elapsed_time(end))
                item = {
                    "E": length,
                    "K": keep.numel(),
                    "plan": plan,
                    "tile": tile,
                    "warps": warps,
                    "cuda_event_ms": samples,
                    "median_ms": statistics.median(samples) if samples else None,
                    "estimated_payload_bytes": 2 * changed * 36 * 8 * 256 * 2,
                    "registers": compiled.n_regs,
                    "spills": compiled.n_spills,
                    "shared_bytes": compiled.metadata.shared,
                }
                records.append(item)
                directory = run / "compiler" / f"E{length}-{plan}-T{tile}-W{warps}"
                directory.mkdir(parents=True, exist_ok=True)
                for kind in ("ttir", "ttgir", "llir", "ptx"):
                    (directory / f"stable.{kind}").write_text(compiled.asm[kind])
                (run / "benchmark.json").write_text(json.dumps(records, indent=2))
                print(json.dumps(item), flush=True)
        del compactor, layers, sources
        torch.accelerator.empty_cache()


if __name__ == "__main__":
    main()
