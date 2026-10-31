# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NSYS diagnostic with HTTP phase marks and unchanged server/runtime paths."""

import os
import subprocess
import sys
from pathlib import Path


def main():
    if "--nsys-child" not in sys.argv:
        output = Path(os.environ["P1_RUN_DIR"]) / "graph-serving"
        subprocess.run(
            [
                "/opt/nvidia/nsight-systems/2024.7.1/bin/nsys",
                "profile",
                "--trace=cuda,nvtx,osrt",
                "--cuda-graph-trace=graph",
                "--cuda-flush-interval=500",
                "--sample=none",
                "--cpuctxsw=none",
                "--trace-fork-before-exec=true",
                "--wait=all",
                "--output=" + str(output),
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
    import torch

    original = httpx.AsyncClient.get
    original_popen = subprocess.Popen
    metrics_index = 0

    def profiled_server(command, *args, **kwargs):
        if "vllm.entrypoints.openai.api_server" in command:
            command = [*command, "--shutdown-timeout", "20"]
        return original_popen(command, *args, **kwargs)

    async def marked_get(client, url, *args, **kwargs):
        nonlocal metrics_index
        index = metrics_index
        if url.endswith("/metrics"):
            metrics_index += 1
            torch.cuda.nvtx.mark(f"metrics-{index}-before")
        result = await original(client, url, *args, **kwargs)
        if url.endswith("/metrics"):
            torch.cuda.nvtx.mark(f"metrics-{index}-after")
        return result

    httpx.AsyncClient.get = marked_get
    subprocess.Popen = profiled_server
    serving_reclamation.main()


if __name__ == "__main__":
    main()
