# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One existing pressure workload, stable HTTP IDs, and request/step profiling."""

import contextvars
import json
import os
import shutil
import subprocess
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path


def main():
    run = Path(os.environ["P1_RUN_DIR"])
    scripts = Path(__file__).parent
    if "--nsys-child" not in sys.argv:
        for name in (
            "profile_tp_ttft.py",
            "tp_ttft_hooks.py",
            "serving_reclamation.py",
            "worker_affinity.py",
        ):
            shutil.copyfile(scripts / name, run / name)
        subprocess.run(
            [
                "/opt/nvidia/nsight-systems/2024.7.1/bin/nsys",
                "profile",
                "--trace=cuda,nvtx,osrt",
                "--cuda-graph-trace=graph",
                "--cuda-flush-interval=0",
                "--sample=none",
                "--cpuctxsw=none",
                "--trace-fork-before-exec=true",
                "--wait=all",
                "--output=" + str(run / "ttft"),
                sys.executable,
                __file__,
                "--nsys-child",
                *sys.argv[1:],
            ],
            check=True,
        )
        return

    sys.argv.remove("--nsys-child")
    import httpx
    import serving_reclamation

    os.environ["PYTHONPATH"] = str(scripts) + os.pathsep + os.environ["PYTHONPATH"]
    request_name = contextvars.ContextVar("request_name")
    events = []
    original_completion = serving_reclamation.completion
    original_stream = httpx.AsyncClient.stream
    original_popen = subprocess.Popen

    async def completion(client, origin, ids, rid, *args):
        token = request_name.set(rid)
        try:
            return await original_completion(client, origin, ids, rid, *args)
        finally:
            request_name.reset(token)

    @asynccontextmanager
    async def stream(client, method, url, *args, **kwargs):
        assert url.endswith("/v1/completions"), url
        rid = request_name.get()
        kwargs["json"]["request_id"] = rid
        events.append(
            dict(event="client_start", request=rid, time_ns=time.monotonic_ns())
        )
        async with original_stream(client, method, url, *args, **kwargs) as reply:
            original_lines = reply.aiter_lines

            async def lines():
                first = True
                async for line in original_lines():
                    if first and line.startswith("data: ") and line != "data: [DONE]":
                        payload = json.loads(line[6:])
                        if any(c.get("token_ids") for c in payload.get("choices", [])):
                            events.append(
                                dict(
                                    event="client_first_token",
                                    request=rid,
                                    response_id=payload["id"],
                                    time_ns=time.monotonic_ns(),
                                )
                            )
                            first = False
                    yield line

            reply.aiter_lines = lines
            yield reply
        events.append(
            dict(event="client_end", request=rid, time_ns=time.monotonic_ns())
        )

    def profiled_server(command, *args, **kwargs):
        if "vllm.entrypoints.openai.api_server" in command:
            index = command.index("--worker-cls")
            command = list(command)
            command[index + 1] = "tp_ttft_hooks.TimelineWorker"
            command += [
                "--scheduler-cls",
                "tp_ttft_hooks.TimelineScheduler",
                "--shutdown-timeout",
                "20",
            ]
            (run / "diagnostic-command.json").write_text(json.dumps(command, indent=2))
        return original_popen(command, *args, **kwargs)

    serving_reclamation.completion = completion
    httpx.AsyncClient.stream = stream
    subprocess.Popen = profiled_server
    serving_reclamation.main()
    (run / "timeline-client.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events)
    )


if __name__ == "__main__":
    main()
