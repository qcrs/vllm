# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only observations for a separate dense DP diagnostic, never formal runs."""

import json
import os
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine import EngineCoreOutputs
from vllm.v1.engine.coordinator import DPCoordinatorProc
from vllm.v1.engine.core import EngineCoreProc
from vllm.v1.engine.core_client import (
    DPLBAsyncMPClient,
    length_from_prompt_token_ids_or_embeds,
)
from vllm.v1.serial_utils import MsgpackDecoder


def install():
    stream = None

    def record(site, **fields):
        nonlocal stream
        if stream is None:
            stream = (
                Path(os.environ["P1_RUN_DIR"]) / f"dp-load-{os.getpid()}.jsonl"
            ).open("a", buffering=1)
        stream.write(
            json.dumps(dict(site=site, time_ns=time.monotonic_ns(), **fields)) + "\n"
        )

    decode = MsgpackDecoder.decode

    def observe_decode(self, *args, **kwargs):
        value = decode(self, *args, **kwargs)
        if isinstance(value, EngineCoreOutputs) and value.scheduler_stats:
            stats = value.scheduler_stats
            if stats.dp_load is not None:
                record(
                    "coordinator_receive",
                    engine=value.engine_index,
                    snapshot=asdict(stats.dp_load),
                    counts=[stats.num_waiting_reqs, stats.num_running_reqs],
                )
        return value

    counts = DPCoordinatorProc._get_engine_counts

    def observe_counts(self, *args, **kwargs):
        value = counts(self, *args, **kwargs)
        record(
            "coordinator_publish",
            counts=value,
            snapshots=[asdict(e.load) if e.load else None for e in self.engines],
        )
        return value

    apply_stats = DPLBAsyncMPClient._apply_load_stats

    def observe_apply(self, buf):
        apply_stats(self, buf)
        record(
            "frontend_receive",
            counts=self.lb_engines,
            snapshots=[asdict(s) if s else None for s in self.lb_loads],
            published_ns=self.lb_stats_published_ns,
            received_ns=self.lb_stats_received_ns,
        )

    step = EngineCoreProc._process_engine_step

    def observe_step(self):
        result = step(self)
        scheduler = self.scheduler
        now = time.monotonic_ns()
        current_counts = scheduler.get_request_counts()
        previous_ns, previous_counts = getattr(self, "_p1_observed", (0, None))
        if now - previous_ns >= 100_000_000 or current_counts != previous_counts:
            pool = scheduler.kv_cache_manager.block_pool
            record(
                "scheduler_completed",
                rank=self.engine_index,
                counts=list(reversed(current_counts)),
                snapshot=asdict(scheduler.get_dp_load_snapshot(0, now)),
                free_blocks=pool.get_num_free_blocks(),
                usable_blocks=pool.num_gpu_blocks - 1,
            )
            self._p1_observed = now, current_counts
        return result

    allocate = KVCacheManager.allocate_slots
    initialize = Scheduler.__init__

    def observe_initialize(self, *args, **kwargs):
        initialize(self, *args, **kwargs)
        self.kv_cache_manager._p1_rank = self.parallel_config.data_parallel_index

    def observe_allocate(self, request, *args, **kwargs):
        result = allocate(self, request, *args, **kwargs)
        if result is None:
            record(
                "allocation_miss",
                rank=self._p1_rank,
                request=request.request_id,
                required_available=self.last_allocation_capacity,
                free_blocks=self.block_pool.get_num_free_blocks(),
                usage=self.usage,
            )
        return result

    schedule = Scheduler.schedule

    def observe_schedule(self, *args, **kwargs):
        result = schedule(self, *args, **kwargs)
        if result.reclaim_requests or result.preempted_req_ids:
            record(
                "pressure_action",
                rank=self.parallel_config.data_parallel_index,
                step=self.current_step,
                intents=[asdict(i) for i in result.reclaim_requests],
                preempted=sorted(result.preempted_req_ids),
                usage=self.kv_cache_manager.usage,
            )
        return result

    route = DPLBAsyncMPClient.get_core_engine_for_request

    def observe_route(self, request):
        local = Counter(self.reqs_in_flight.values())
        now = time.monotonic_ns()
        current = [list(c) for c in self.lb_engines]
        inflight = [local[engine] for engine in self.core_engines]
        base = [max(4 * w + r, f) for (w, r), f in zip(current, inflight)]
        work = [s.unfinished_tokens if s else 0 for s in self.lb_loads]
        pending = [dict(self.pending_adds[e]) for e in self.core_engines]
        pending_tokens = [sum(n for _, n in p.values()) for p in pending]
        ages = [(now - s.generated_ns) / 1e9 if s else None for s in self.lb_loads]
        batch = self.known_work_batch_size
        scores = [
            c + (u + p) / batch if batch is not None else c
            for c, u, p in zip(base, work, pending_tokens)
        ]
        order = [(self.eng_start_index + i) % len(base) for i in range(len(base))]
        shadow = min(order, key=lambda i: scores[i])
        start = self.eng_start_index
        chosen = route(self, request)
        record(
            "route",
            request=request.request_id,
            external=request.external_req_id,
            counts=current,
            F=inflight,
            C=base,
            U=work,
            pending_adds=pending,
            pending_tokens=pending_tokens,
            prompt_tokens=length_from_prompt_token_ids_or_embeds(
                request.prompt_token_ids, request.prompt_embeds
            ),
            scores=scores,
            tied=[i for i, score in enumerate(scores) if score == min(scores)],
            start=start,
            chosen=self.core_engines.index(chosen),
            shadow=shadow,
            snapshots=[asdict(s) if s else None for s in self.lb_loads],
            state_age_s=ages,
            fresh_250ms_diagnostic=all(age is not None and age <= 0.25 for age in ages),
            published_ns=self.lb_stats_published_ns,
            received_ns=self.lb_stats_received_ns,
        )
        return chosen

    MsgpackDecoder.decode = observe_decode
    DPCoordinatorProc._get_engine_counts = observe_counts
    DPLBAsyncMPClient._apply_load_stats = observe_apply
    EngineCoreProc._process_engine_step = observe_step
    KVCacheManager.allocate_slots = observe_allocate
    Scheduler.__init__ = observe_initialize
    Scheduler.schedule = observe_schedule
    DPLBAsyncMPClient.get_core_engine_for_request = observe_route
