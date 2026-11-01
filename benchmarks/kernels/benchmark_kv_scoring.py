# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure raw paged scores, normalization and stable selection separately."""

import argparse
import hashlib
import json
import os
import shutil
import statistics
import time
from datetime import timedelta
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from vllm.v1.kv_reclamation import ReclaimRequest, ReclamationConfig
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.buffer_utils import StagedWriteTensor
from vllm.v1.worker.gpu.kv_compaction import StableKVCompactor
from vllm.v1.worker.gpu.kv_reclamation import KVReclamationWorker
from vllm.v1.worker.gpu.kv_scoring import TriAttentionScorer, protected_topk


def raw_scores(scorer, layers, row, logical, length):
    scorer.prepare_phase(logical)
    for layer, cache in enumerate(layers):
        scorer._score_layer(cache, row, length, layer)


@torch.inference_mode()
def runtime_worker(rank, args, run):
    os.sched_setaffinity(0, {args.worker_cpus[rank]})
    torch.accelerator.set_device_index(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        "nccl",
        init_method="file://" + str(run / "store"),
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=120),
        device_id=device,
    )
    records = []
    native_all_reduce = dist.all_reduce
    try:
        stats = torch.load(args.stats, weights_only=True, map_location="cuda")["stats"]
        for shape in json.loads(Path(args.runtime_shapes).read_text())["shapes"]:
            records.extend(benchmark_runtime_shape(rank, args, device, stats, shape))
            torch.accelerator.empty_cache()
        (run / f"runtime-rank-{rank}.json").write_text(json.dumps(records, indent=2))
    finally:
        dist.all_reduce = native_all_reduce
        dist.destroy_process_group()


def benchmark_runtime_shape(rank, args, device, stats, shape):
    records = []
    native_all_reduce = dist.all_reduce
    source, keep = shape["E"], shape["keep"]
    prefix, recent = shape["P"], shape["recent"]
    blocks = (source + 15) // 16 + 7
    torch.manual_seed(11 + rank)
    layers = [
        torch.randn(blocks, 16, 4, 256, device=device, dtype=torch.bfloat16).transpose(
            1, 2
        )
        for _ in range(36)
    ]
    assert list(layers[0].stride()) == shape["cache_stride"]
    saved = [v.clone() for v in layers]
    ids = torch.randperm(blocks, device=device, dtype=torch.int32)
    row_ids = ids[: (source + 15) // 16].cpu().tolist()
    tables = BlockTables([16], 1, 1, [576], device, [16])
    logical = torch.tensor([shape["L"]], device=device, dtype=torch.int32)
    scorer = TriAttentionScorer(stats, 8, 1e6, 9216, 2, rank, device)
    worker = object.__new__(KVReclamationWorker)
    worker.config = ReclamationConfig(policy="triattention")
    worker.req_states = SimpleNamespace(
        req_id_to_index={"benchmark": 0},
        device=device,
        num_computed_tokens=SimpleNamespace(gpu=logical),
    )
    worker.layers, worker.scorer = layers, scorer
    worker.compactor = StableKVCompactor(layers)
    if args.grouped_probe:
        from vllm.v1.worker.gpu.kv_scoring import GroupedTriAttentionScorer

        scorer = GroupedTriAttentionScorer(stats, 8, 1e6, 9216, 2, rank, device)
        scorer.initialize_groups(worker.compactor)
        worker.scorer = scorer
    worker.tp_group = SimpleNamespace(
        world_size=2,
        rank_in_group=rank,
        device_group=dist.group.WORLD,
        broadcast=lambda value, src: dist.broadcast(value, src=src),
    )
    worker.local_reclaim_done = torch.cuda.Event()
    worker.completion_token = torch.zeros((), dtype=torch.int32, device=device)
    worker.effective_kv_len_np = np.array([source], dtype=np.int32)
    worker.effective_kv_len = StagedWriteTensor(1, torch.int32, device)
    intent = ReclaimRequest("benchmark", source, row_ids, keep, prefix)

    def restore():
        for value, original in zip(layers, saved):
            value.copy_(original)
        tables.append_block_ids(0, (row_ids,), overwrite=True)
        tables.apply_staged_writes()
        worker.effective_kv_len_np[0] = source

    restore()
    row = tables.block_tables[0].gpu[0]
    if args.launch_audit and source == 2051:
        from collections import Counter

        scorer.shared_scores(layers, row, logical, source, prefix, recent)
        torch.accelerator.synchronize()
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as profile:
            scorer.shared_scores(layers, row, logical, source, prefix, recent)
            torch.accelerator.synchronize()
        records.append(
            dict(
                part="launch_audit",
                rank=rank,
                shape=shape,
                kernels=dict(
                    Counter(
                        e.name
                        for e in profile.events()
                        if e.device_type == torch.autograd.DeviceType.CUDA
                    )
                ),
            )
        )
    torch.accelerator.synchronize()
    dist.barrier()
    cold_start, cold_end = torch.cuda.Event(True), torch.cuda.Event(True)
    cold_before = time.perf_counter_ns()
    cold_start.record()
    worker.reclaim(intent, tables)
    cold_end.record()
    cold_submitted = time.perf_counter_ns()
    cold_end.synchronize()
    records.append(
        dict(
            shape=shape,
            rank=rank,
            part="first_reclaim_in_process",
            device_ms=cold_start.elapsed_time(cold_end),
            host_submit_ms=(cold_submitted - cold_before) / 1e6,
            wall_complete_ms=(time.perf_counter_ns() - cold_before) / 1e6,
        )
    )
    restore()
    raw_layers = torch.empty(
        (36, scorer.stats.shape[1], 9216), device=device, dtype=torch.float32
    )
    scorer.prepare_phase(logical)
    for layer, cache in enumerate(layers):
        raw_layers[layer, :, :source].copy_(
            scorer._score_layer(cache, row, source, layer)
        )
    norm_layers = torch.empty_like(raw_layers)
    original_phase, original_layer = scorer.prepare_phase, scorer._score_layer

    def normalization():
        scorer.prepare_phase = lambda _: None
        scorer._score_layer = lambda cache, row, E, layer: norm_layers[layer, :, :E]
        try:
            return scorer.shared_scores(layers, row, logical, source, prefix, recent)
        finally:
            scorer.prepare_phase, scorer._score_layer = (
                original_phase,
                original_layer,
            )

    score_buffer = torch.ones(source, device=device)
    keep_buffer = torch.empty(keep, device=device, dtype=torch.int32)

    def plan():
        scores = scorer.shared_scores(layers, row, logical, source, prefix, recent)
        dist.all_reduce(scores)
        chosen = (
            protected_topk(scores, keep, prefix, recent) if rank == 0 else keep_buffer
        )
        dist.broadcast(chosen, src=0)
        return chosen

    for collective in ("all_reduce", "reduce_root"):

        def score_collective(value, *args, mode=collective, **kwargs):
            if mode == "reduce_root" and value.dtype == torch.float32:
                return dist.reduce(
                    value,
                    dst=dist.get_global_rank(dist.group.WORLD, 0),
                    group=dist.group.WORLD,
                )
            return native_all_reduce(value, *args, **kwargs)

        dist.all_reduce = score_collective
        parts = [
            ("phase", partial(scorer.prepare_phase, logical)),
            (
                "raw_with_phase",
                partial(raw_scores, scorer, layers, row, logical, source),
            ),
            ("normalization_reduction", normalization),
            ("score_collective", partial(dist.all_reduce, score_buffer)),
            (
                "stable_selection",
                (
                    partial(
                        protected_topk,
                        scorer.total[:source],
                        keep,
                        prefix,
                        recent,
                    )
                    if rank == 0
                    else lambda: None
                ),
            ),
            ("plan_broadcast", partial(dist.broadcast, keep_buffer, src=0)),
            ("complete_plan", plan),
            ("complete_reclaim", partial(worker.reclaim, intent, tables)),
        ]
        for name, launch in parts:
            if args.grouped_probe and name in (
                "raw_with_phase",
                "normalization_reduction",
            ):
                continue
            if collective == "reduce_root" and name not in (
                "score_collective",
                "complete_plan",
                "complete_reclaim",
            ):
                continue
            samples, host_samples = [], []
            for iteration in range(-3, args.iterations):
                restore()
                norm_layers.copy_(raw_layers)
                torch.accelerator.synchronize()
                dist.barrier()
                start, end = torch.cuda.Event(True), torch.cuda.Event(True)
                before = time.perf_counter_ns()
                start.record()
                launch()
                end.record()
                submitted = time.perf_counter_ns()
                end.synchronize()
                if iteration >= 0:
                    samples.append(start.elapsed_time(end))
                    host_samples.append((submitted - before) / 1e6)
            item = dict(
                shape=shape,
                rank=rank,
                collective=collective,
                part=name,
                samples_ms=samples,
                host_submit_ms=host_samples,
                median_ms=statistics.median(samples),
                affinity=sorted(os.sched_getaffinity(0)),
            )
            records.append(item)
            print(
                json.dumps(
                    {
                        k: v
                        for k, v in item.items()
                        if k not in ("samples_ms", "host_submit_ms")
                    }
                ),
                flush=True,
            )
        restore()
        selected = plan()
        torch.accelerator.synchronize()
        records.append(
            dict(
                shape=shape,
                rank=rank,
                collective=collective,
                part="plan_identity",
                sha256=hashlib.sha256(selected.cpu().numpy().tobytes()).hexdigest(),
            )
        )
    dist.all_reduce = native_all_reduce
    return records


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--launch-audit", action="store_true")
    parser.add_argument("--grouped-probe", action="store_true")
    parser.add_argument("--stats", required=True)
    parser.add_argument("--lengths", nargs="+", type=int, default=[4096, 32768])
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--runtime-shapes")
    parser.add_argument("--worker-cpus", nargs=2, type=int, default=[20, 21])
    args = parser.parse_args()
    run = Path(os.environ["P1_RUN_DIR"])
    if args.runtime_shapes:
        shutil.copyfile(args.runtime_shapes, run / "runtime-shapes.json")
        source_root = Path(__file__).resolve().parents[2]
        for name in (
            "kv_scoring",
            "kv_reclamation",
            "kv_compaction",
            "block_table",
            "buffer_utils",
        ):
            source = source_root / f"vllm/v1/worker/gpu/{name}.py"
            shutil.copyfile(source, run / source.name)
        mp.spawn(runtime_worker, args=(args, run), nprocs=2, join=True)
        return
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
