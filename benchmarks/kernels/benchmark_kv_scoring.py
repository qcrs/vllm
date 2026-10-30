# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure raw paged scores, normalization and stable selection separately."""

import argparse
import json
import os
import statistics
from functools import partial
from pathlib import Path

import torch

from vllm.v1.worker.gpu.kv_scoring import TriAttentionScorer, protected_topk


def raw_scores(scorer, layers, row, logical, length):
    scorer.prepare_phase(logical)
    for layer, cache in enumerate(layers):
        scorer._score_layer(cache, row, length, layer)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats", required=True)
    parser.add_argument("--lengths", nargs="+", type=int, default=[4096, 32768])
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    run = Path(os.environ["P1_RUN_DIR"])
    torch.manual_seed(11)
    stats = torch.load(args.stats, weights_only=True, map_location="cuda")["stats"]
    records = []
    for length in args.lengths:
        layers = [
            torch.randn(
                length // 16, 16, 8, 256, device="cuda", dtype=torch.bfloat16
            ).transpose(1, 2)
            for _ in range(36)
        ]
        row = torch.randperm(length // 16, device="cuda", dtype=torch.int32)
        logical = torch.tensor([39743], device="cuda", dtype=torch.int32)
        scorer = TriAttentionScorer(stats, 8, 1e6, length)
        raw = partial(raw_scores, scorer, layers, row, logical, length)
        shared = partial(
            scorer.shared_scores, layers, row, logical, length, 16, length - 16
        )
        select = partial(protected_topk, scorer.total, length // 2, 16, length - 16)
        plan = partial(
            scorer.plan, layers, row, logical, length, length // 2, 16, length - 16
        )
        if args.profile:
            raw()
            torch.accelerator.synchronize()
            torch.cuda.cudart().cudaProfilerStart()
            raw()
            torch.accelerator.synchronize()
            torch.cuda.cudart().cudaProfilerStop()
            continue
        for name, launch in [
            ("raw_36_layers", raw),
            ("shared_with_normalization", shared),
            ("selection", select),
            ("complete_plan", plan),
        ]:
            for _ in range(3):
                launch()
            torch.accelerator.synchronize()
            torch.accelerator.reset_peak_memory_stats()
            base = torch.accelerator.memory_allocated()
            samples = []
            for _ in range(args.iterations):
                start, end = torch.cuda.Event(True), torch.cuda.Event(True)
                start.record()
                launch()
                end.record()
                end.synchronize()
                samples.append(start.elapsed_time(end))
            item = {
                "E": length,
                "part": name,
                "median_ms": statistics.median(samples),
                "samples_ms": samples,
                "peak_temporary_bytes": torch.accelerator.max_memory_allocated() - base,
                "stats_bytes": stats.numel() * stats.element_size(),
                "score_workspace_bytes": (
                    scorer.scores.numel()
                    + scorer.total.numel()
                    + scorer.coefficients.numel()
                )
                * 4,
            }
            records.append(item)
            print(json.dumps(item), flush=True)
        del layers, scorer, raw, shared, plan, select, launch
        torch.accelerator.empty_cache()
    (run / "scoring.json").write_text(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
