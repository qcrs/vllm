# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native API server with one routing observation per new request."""

import atexit
import json
import os
import runpy
import time
from pathlib import Path

from vllm.v1.engine.core_client import DPLBAsyncMPClient


def main():
    run = Path(os.environ["P1_RUN_DIR"])
    stream = (run / "routing.jsonl").open("a", buffering=1)
    original = DPLBAsyncMPClient.get_core_engine_for_request
    costs = []

    def route(self, request):
        start = time.perf_counter_ns()
        counts = [list(c) for c in self.lb_engines]
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
