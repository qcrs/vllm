# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native API server with one routing observation per new request."""

import atexit
import json
import os
import runpy
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from vllm.v1.engine.core_client import DPLBAsyncMPClient

if os.environ.get("P1_DP_LOAD_DIAGNOSTIC") == "1":
    from dp_load_diagnostic import install

    install()


def main():
    run = Path(os.environ["P1_RUN_DIR"])
    stream = (run / "routing.jsonl").open("a", buffering=1)
    original = DPLBAsyncMPClient.get_core_engine_for_request
    costs = []

    def route(self, request):
        start = time.perf_counter_ns()
        counts = [list(c) for c in self.lb_engines]
        local = Counter(self.reqs_in_flight.values())
        inflight = [local[e] for e in self.core_engines]
        snapshots = [asdict(s) if s else None for s in self.lb_loads]
        scan_start = self.eng_start_index
        used_ns = time.monotonic_ns()
        before = time.perf_counter_ns()
        chosen = original(self, request)
        selected = time.perf_counter_ns()
        stream.write(
            json.dumps(
                {
                    "request": request.request_id,
                    "external": request.external_req_id,
                    "replica": self.core_engines.index(chosen),
                    "explicit": request.data_parallel_rank,
                    "counts": counts,
                    "F": inflight,
                    "snapshots": snapshots,
                    "scan_start": scan_start,
                    "batch": self.known_work_batch_size,
                    "used_ns": used_ns,
                    "published_ns": self.lb_stats_published_ns,
                    "received_ns": self.lb_stats_received_ns,
                    "route_cpu_ns": selected - before,
                    "time_ns": time.monotonic_ns(),
                }
            )
            + "\n"
        )
        costs.append(time.perf_counter_ns() - start)
        return chosen

    def finish():
        (run / "routing-costs.json").write_text(
            json.dumps(
                {
                    "calls": len(costs),
                    "wrapper_including_write_ns": costs,
                },
                indent=2,
            )
        )
        stream.close()

    atexit.register(finish)
    DPLBAsyncMPClient.get_core_engine_for_request = route
    runpy.run_module("vllm.entrypoints.openai.api_server", run_name="__main__")


if __name__ == "__main__":
    main()
