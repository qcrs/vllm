# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Separate TP Serving or slow-rank trajectory NSYS diagnostic."""

import os
import shutil
import subprocess
import sys
from pathlib import Path


def main():
    run = Path(os.environ["P1_RUN_DIR"])
    scripts = Path(__file__).parent
    if "--nsys-child" not in sys.argv:
        for name in (
            "tp_profile_hooks.py",
            "tp_worker_hooks.py",
            "tp_reclamation.py",
            "serving_reclamation.py",
        ):
            shutil.copyfile(scripts / name, run / name)
        subprocess.run(
            [
                "/opt/nvidia/nsight-systems/2024.7.1/bin/nsys",
                "profile",
                "--trace=cuda,nvtx,osrt",
                "--cuda-graph-trace=graph",
                "--cuda-flush-interval="
                + os.environ.get("P1_NSYS_CUDA_FLUSH_MS", "500"),
                "--sample=none",
                "--cpuctxsw=none",
                "--trace-fork-before-exec=true",
                "--wait=all",
                "--output=" + str(run / "tp"),
                sys.executable,
                __file__,
                "--nsys-child",
                *sys.argv[1:],
            ],
            check=True,
        )
        return
    sys.argv.remove("--nsys-child")
    os.environ["PYTHONPATH"] = str(scripts) + os.pathsep + os.environ["PYTHONPATH"]
    if "--trajectory" in sys.argv:
        sys.argv.remove("--trajectory")
        import tp_reclamation

        sys.argv.append("--profile")
        tp_reclamation.main()
        return
    import httpx
    import serving_reclamation
    import torch

    original_get, original_popen = httpx.AsyncClient.get, subprocess.Popen
    metrics_index = 0

    def profiled_server(command, *args, **kwargs):
        if "vllm.entrypoints.openai.api_server" in command:
            command = [
                *command,
                "--shutdown-timeout",
                "20",
                "--worker-extension-cls",
                "tp_profile_hooks.P1TPProfileWorker",
            ]
        return original_popen(command, *args, **kwargs)

    async def marked_get(client, url, *args, **kwargs):
        nonlocal metrics_index
        index = metrics_index
        if url.endswith("/metrics"):
            metrics_index += 1
            torch.cuda.nvtx.mark(f"metrics-{index}-before")
        result = await original_get(client, url, *args, **kwargs)
        if url.endswith("/metrics"):
            torch.cuda.nvtx.mark(f"metrics-{index}-after")
        return result

    httpx.AsyncClient.get, subprocess.Popen = marked_get, profiled_server
    serving_reclamation.main()


if __name__ == "__main__":
    main()
