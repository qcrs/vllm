# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded Controller, actual TP2 reclaim batch and unwrapped DP route micro."""

import argparse
import hashlib
import json
import os
import shutil
import statistics
import time
from collections import defaultdict
from dataclasses import asdict
from datetime import timedelta
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager
from vllm.v1.engine.core_client import DPLBAsyncMPClient
from vllm.v1.kv_reclamation import (
    PolicyBinding,
    ReclaimRequest,
    ReclamationConfig,
    protected_reclaim_floor,
    sink_recent_ranges,
)
from vllm.v1.request import RequestStatus
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.buffer_utils import StagedWriteTensor
from vllm.v1.worker.gpu.kv_compaction import StableKVCompactor
from vllm.v1.worker.gpu.kv_reclamation import KVReclamationWorker
from vllm.v1.worker.gpu.kv_retention_policy import POLICY_FACTORIES

MEMBER_BYTES = 36 * 4 * 256 * 2


def controller(shape, mode, count, demand):
    """Run the actual Scheduler scan/selection on explicit canonical CPU state."""
    config = ReclamationConfig(budget_mode=mode)
    binding = PolicyBinding(0, config)
    scheduler = object.__new__(Scheduler)
    scheduler.reclamation_config = config
    scheduler.block_size = 16
    scheduler.max_num_scheduled_tokens = 1024
    scheduler.max_model_len = 9216
    scheduler.num_sampled_tokens_per_step = 1
    scheduler.scheduler_config = SimpleNamespace(long_prefill_token_threshold=0)
    scheduler.running = [
        SimpleNamespace(
            request_id=f"r{i}",
            num_computed_tokens=shape["L"],
            num_prompt_tokens=shape["L"],
            max_tokens=128,
            num_tokens=shape["L"] + 1,
            num_in_flight_tokens=0,
            has_encoder_inputs=False,
            status=RequestStatus.RUNNING,
            effective_kv_len=shape["E"],
            arrival_time=float(i),
            policy_binding=binding,
        )
        for i in range(count)
    ]
    scheduler.requests = {r.request_id: r for r in scheduler.running}
    pages = (shape["E"] + 15) // 16
    rows = {
        f"r{i}": [
            SimpleNamespace(
                block_id=1 + i * pages + j,
                ref_cnt=2 if j < shape["P"] // 16 else 1,
                is_null=False,
            )
            for j in range(pages)
        ]
        for i in range(count)
    }
    manager = object.__new__(FullAttentionManager)
    manager.block_size = 16
    manager.req_to_blocks = rows
    manager.num_cached_block = dict.fromkeys(rows, 0)
    running_tail_pages = count * ((shape["E"] + 16) // 16 - pages)
    scheduler.kv_cache_manager = SimpleNamespace(
        coordinator=SimpleNamespace(single_type_managers=[manager]),
        block_pool=SimpleNamespace(get_num_free_blocks=lambda: running_tail_pages),
        watermark_blocks=0,
        last_allocation_shortfall=demand,
        trace=None,
    )
    scheduler.kv_cache_config = SimpleNamespace(
        kv_cache_groups=[
            SimpleNamespace(
                kv_cache_spec=SimpleNamespace(page_size_bytes=MEMBER_BYTES * 16),
                layer_names=["all-layers"],
            )
        ]
    )
    return scheduler


def controller_calls(shape, mode, count, demand):
    scheduler = controller(shape, mode, count, demand)
    yield "scan_only", partial(scheduler._select_reclaim_plan, {})
    if mode != "protected":
        return
    waiting = SimpleNamespace(
        request_id="waiting",
        num_tokens=demand * 16,
        num_prompt_tokens=demand * 16,
        max_tokens=128,
        num_computed_tokens=0,
        effective_kv_len=0,
        has_encoder_inputs=False,
        status=RequestStatus.WAITING,
    )
    yield "waiting_full_forecast", partial(scheduler._select_reclaim_plan, {}, waiting)
    scheduler = controller(shape, mode, count, demand)
    scheduler.running[0].num_tokens += 1023
    yield (
        "prefill_reservation",
        partial(scheduler._protected_prefill_reserved_blocks, scheduler.running[-1]),
    )


def cases(shapes):
    for shape in shapes:
        floor = protected_reclaim_floor(
            shape["E"], shape["L"], shape["P"], ReclamationConfig()
        )
        capacity = (shape["E"] + 15) // 16 - (floor + 15) // 16
        yield shape, "fixed", "single", 1
        yield shape, "protected", "single", 1
        yield shape, "protected", "pair", max(2, capacity + 1)


def cpu_micro(shapes):
    records = []
    for shape, mode, label, demand in cases(shapes):
        for count in (2, 8):
            for context, call in controller_calls(shape, mode, count, demand):
                for _ in range(3):
                    call()
                samples = []
                for _ in range(30):
                    start = time.perf_counter_ns()
                    result = call()
                    samples.append((time.perf_counter_ns() - start) / 1e3)
                records.append(
                    dict(
                        shape=shape,
                        mode=mode,
                        label=label,
                        context=context,
                        candidates=count,
                        demand=demand,
                        decision_us=samples,
                        median_us=statistics.median(samples),
                        plan=(
                            [asdict(c) for c in result]
                            if context != "prefill_reservation"
                            else result
                        ),
                    )
                )
    routes = []
    for mode in ("requests", "known_work"):
        client = object.__new__(DPLBAsyncMPClient)
        client.core_engines = [b"0", b"1"]
        client.client_count = 1
        client.lb_loads = [SimpleNamespace(unfinished_tokens=n) for n in (512, 8192)]
        client.known_work_batch_size = 512 if mode == "known_work" else None
        request = SimpleNamespace(
            data_parallel_rank=None,
            pooling_params=None,
            request_id="micro",
            prompt_token_ids=[1] * 512,
            prompt_embeds=None,
        )
        samples = []
        for _ in range(1003):
            client.lb_engines = [[0, 2], [0, 2]]
            client.eng_start_index = 0
            client.reqs_in_flight = {}
            client.dp_add_sequences = defaultdict(int)
            client.pending_adds = defaultdict(dict)
            start = time.perf_counter_ns()
            chosen = client.get_core_engine_for_request(request)
            samples.append((time.perf_counter_ns() - start) / 1e3)
        routes.append(
            dict(
                mode=mode,
                route_us=samples[3:],
                chosen=str(chosen),
                median_us=statistics.median(samples[3:]),
            )
        )
    return dict(controller=records, routing=routes)


@torch.inference_mode()
def gpu_worker(rank, args, shapes, run):
    os.sched_setaffinity(0, {args.worker_cpus[rank]})
    torch.accelerator.set_device_index(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        "nccl",
        init_method="file://" + str(run / "micro-store"),
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=120),
        device_id=device,
    )
    records = []
    for shape, mode, label, demand in cases(shapes):
        plan = controller(shape, mode, 2, demand)._select_reclaim_plan({})
        if not plan:
            records.append(dict(shape=shape, mode=mode, label=label, feasible=False))
            continue
        source = shape["E"]
        pages = (source + 15) // 16
        layers = [
            torch.randn(
                1 + 2 * pages, 16, 4, 256, device=device, dtype=torch.bfloat16
            ).transpose(1, 2)
            for _ in range(36)
        ]
        assert list(layers[0].stride()) == [16384, 256, 1024, 1]
        saved = [v.clone() for v in layers]
        worker = object.__new__(KVReclamationWorker)
        binding = plan[0].policy_binding
        worker.config = binding.config
        worker.req_states = SimpleNamespace(
            req_id_to_index={f"r{i}": i for i in range(2)},
            device=device,
            num_computed_tokens=SimpleNamespace(
                gpu=torch.full((2,), shape["L"], dtype=torch.int32, device=device)
            ),
        )
        worker.layers = layers
        worker.compactor = StableKVCompactor(layers)
        worker.tp_group = SimpleNamespace(
            world_size=2,
            rank_in_group=rank,
            device_group=dist.group.WORLD,
            broadcast=lambda value, src: dist.broadcast(value, src=src),
        )
        policy = POLICY_FACTORIES["sink_recent"](
            binding.config, None, device, worker.tp_group, worker.compactor
        )
        worker.policies = {0: (binding, policy)}
        worker.slot_bindings = [binding, binding]
        worker.local_reclaim_done = torch.cuda.Event()
        worker.completion_token = torch.zeros((), dtype=torch.int32, device=device)
        worker.effective_kv_len_np = np.full(2, source, dtype=np.int32)
        worker.effective_kv_len = StagedWriteTensor(2, torch.int32, device)
        tables = BlockTables([16], 2, 2, [pages], device, [16])
        intents = tuple(
            ReclaimRequest(
                c.request_id,
                source,
                list(c.source_block_ids),
                c.keep_tokens,
                shape["P"],
                binding,
            )
            for c in plan
        )
        keeps = []
        for c in plan:
            prefix, recent = sink_recent_ranges(
                source, c.keep_tokens, binding.config.sink_tokens, shape["P"]
            )
            keeps.append(policy.root_plan(None, source, c.keep_tokens, prefix, recent))

        def restore(
            layers=layers,
            saved=saved,
            pages=pages,
            tables=tables,
            worker=worker,
            source=source,
        ):
            for value, original in zip(layers, saved):
                value.copy_(original)
            for i in range(2):
                tables.append_block_ids(
                    i, (list(range(1 + i * pages, 1 + (i + 1) * pages)),), True
                )
            tables.apply_staged_writes()
            worker.effective_kv_len_np[:] = source

        def compact(plan=plan, keeps=keeps, worker=worker, tables=tables):
            for c, keep in zip(plan, keeps):
                if not c.zero_copy:
                    idx = worker.req_states.req_id_to_index[c.request_id]
                    worker.compactor.compact(tables.block_tables[0].gpu[idx], keep)

        for part, call in (
            ("worker_batch", partial(worker.reclaim_batch, intents, tables)),
            ("compaction", compact),
        ):
            samples = []
            for repeat in range(13):
                restore()
                torch.accelerator.synchronize()
                dist.barrier()
                start, end = torch.cuda.Event(True), torch.cuda.Event(True)
                wall = time.perf_counter_ns()
                start.record()
                call()
                end.record()
                submit = time.perf_counter_ns()
                end.synchronize()
                if repeat >= 3:
                    samples.append(
                        dict(
                            device_ms=start.elapsed_time(end),
                            host_submit_ms=(submit - wall) / 1e6,
                            wall_complete_ms=(time.perf_counter_ns() - wall) / 1e6,
                        )
                    )
            records.append(
                dict(
                    shape=shape,
                    mode=mode,
                    label=label,
                    part=part,
                    victims=len(plan),
                    released_pages=sum(c.released_blocks for c in plan),
                    estimated_copy_bytes=sum(c.copy_bytes for c in plan),
                    zero_copy_victims=sum(c.zero_copy for c in plan),
                    samples=samples,
                )
            )
    output = dict(
        rank=rank,
        cpu_affinity=sorted(os.sched_getaffinity(0)),
        gpu_uuid=str(torch.cuda.get_device_properties(device).uuid),
        records=records,
    )
    (run / f"micro-rank-{rank}.json").write_text(json.dumps(output, indent=2))
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-shapes", required=True)
    parser.add_argument("--worker-cpus", nargs=2, type=int, default=[20, 21])
    args = parser.parse_args()
    run = Path(os.environ["P1_RUN_DIR"])
    shapes = [
        dict(E=s["E"], L=s["L"], P=s["P"])
        for s in json.loads(Path(args.runtime_shapes).read_text())["shapes"]
    ]
    shapes.append(dict(E=272, L=272, P=0))
    shutil.copyfile(args.runtime_shapes, run / "runtime-shapes.json")
    manifest = {}
    for path in (
        "vllm/v1/kv_reclamation.py",
        "vllm/v1/core/sched/scheduler.py",
        "vllm/v1/core/single_type_kv_cache_manager.py",
        "vllm/v1/engine/core_client.py",
        "vllm/v1/worker/gpu/kv_reclamation.py",
        "vllm/v1/worker/gpu/kv_compaction.py",
        "vllm/v1/worker/gpu/block_table.py",
        "vllm/v1/worker/gpu/kv_retention_policy.py",
    ):
        target = run / "source" / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        manifest[path] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    (run / "source-hashes.json").write_text(json.dumps(manifest, indent=2))
    (run / "cpu-micro.json").write_text(json.dumps(cpu_micro(shapes), indent=2))
    mp.spawn(gpu_worker, args=(args, shapes, run), nprocs=2, join=True)


if __name__ == "__main__":
    main()
