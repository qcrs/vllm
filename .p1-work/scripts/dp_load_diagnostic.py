# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only observations for a separate dense DP diagnostic, never formal runs."""

import json
import os
import time
from collections import Counter
from pathlib import Path

from vllm.v1.engine import EngineCoreOutputs
from vllm.v1.engine.coordinator import DPCoordinatorProc
from vllm.v1.engine.core import EngineCoreProc
from vllm.v1.engine.core_client import DPLBAsyncMPClient
from vllm.v1.serial_utils import MsgpackDecoder


def install():
    stream = None
    latest = {}

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
            latest[value.engine_index] = dict(
                generated_ns=int(value.timestamp * 1e9),
                counts=[stats.num_waiting_reqs, stats.num_running_reqs],
            )
        return value

    counts = DPCoordinatorProc._get_engine_counts

    def observe_counts(self, *args, **kwargs):
        value = counts(self, *args, **kwargs)
        record("coordinator_publish", counts=value, latest=latest)
        return value

    setattr_client = DPLBAsyncMPClient.__setattr__

    def observe_setattr(self, name, value):
        setattr_client(self, name, value)
        if name == "lb_engines":
            record("frontend_receive", counts=value)

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
        record(
            "route",
            request=request.request_id,
            external=request.external_req_id,
            counts=self.lb_engines,
            F=[local[engine] for engine in self.core_engines],
            start=self.eng_start_index,
        )
        return route(self, request)

    MsgpackDecoder.decode = observe_decode
    DPCoordinatorProc._get_engine_counts = observe_counts
    DPLBAsyncMPClient.__setattr__ = observe_setattr
    EngineCoreProc._process_engine_step = observe_step
    DPLBAsyncMPClient.get_core_engine_for_request = observe_route
