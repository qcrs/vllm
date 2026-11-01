# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only observations for a separate dense DP diagnostic, never formal runs."""

import json
import os
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from vllm.v1.engine import EngineCoreOutputs
from vllm.v1.engine.coordinator import DPCoordinatorProc
from vllm.v1.engine.core import EngineCoreProc
from vllm.v1.engine.core_client import DPLBAsyncMPClient
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
        requests = [
            dict(
                id=r.request_id,
                L=r.num_tokens,
                C=r.num_computed_tokens,
                inflight=r.num_in_flight_tokens,
                E=r.effective_kv_len,
                status=r.status.name,
            )
            for r in scheduler.requests.values()
        ]
        record(
            "scheduler_completed",
            rank=self.engine_index,
            counts=list(reversed(scheduler.get_request_counts())),
            U=sum(max(0, r["L"] - r["C"] + r["inflight"]) for r in requests),
            usage=scheduler.kv_cache_manager.usage,
            requests=requests,
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
        ages = [(now - s.generated_ns) / 1e9 if s else None for s in self.lb_loads]
        batch = self.vllm_config.scheduler_config.max_num_batched_tokens
        scores = [c + u / batch for c, u in zip(base, work)]
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
            scores=scores,
            start=start,
            chosen=self.core_engines.index(chosen),
            shadow=shadow,
            snapshots=[asdict(s) if s else None for s in self.lb_loads],
            state_age_s=ages,
            fresh=all(age is not None and age <= 0.25 for age in ages),
            published_ns=self.lb_stats_published_ns,
            received_ns=self.lb_stats_received_ns,
        )
        return chosen

    MsgpackDecoder.decode = observe_decode
    DPCoordinatorProc._get_engine_counts = observe_counts
    DPLBAsyncMPClient._apply_load_stats = observe_apply
    EngineCoreProc._process_engine_step = observe_step
    DPLBAsyncMPClient.get_core_engine_for_request = observe_route
