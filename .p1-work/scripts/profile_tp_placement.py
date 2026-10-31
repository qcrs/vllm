# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Same-server CPU placement intervention; excluded from official throughput."""

import ctypes
import json
import os
import shutil
import subprocess
import time
from functools import wraps
from pathlib import Path


class P1TPPlacementWorker:
    pass


def install():
    from vllm.distributed import get_tp_group
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    from vllm.v1.worker.gpu_worker import Worker

    run = Path(os.environ["P1_RUN_DIR"])
    getcpu = ctypes.CDLL(None).sched_getcpu
    events = []
    original = GPUModelRunner.execute_model
    shutdown = Worker.shutdown

    @wraps(original)
    def execute(runner, *args, **kwargs):
        rank = get_tp_group().rank_in_group
        if not events:
            (run / f"worker-{rank}.json").write_text(
                json.dumps({"pid": os.getpid(), "rank": rank})
            )
        start = time.perf_counter_ns()
        cpu_start = time.thread_time_ns()
        cpu = getcpu()
        try:
            return original(runner, *args, **kwargs)
        finally:
            events.append(
                [
                    start,
                    time.perf_counter_ns(),
                    time.thread_time_ns() - cpu_start,
                    cpu,
                    getcpu(),
                ]
            )

    @wraps(shutdown)
    def save(worker):
        rank = get_tp_group().rank_in_group
        (run / f"placement-rank-{rank}.json").write_text(json.dumps(events))
        return shutdown(worker)

    GPUModelRunner.execute_model = execute
    Worker.shutdown = save


def main():
    import serving_reclamation

    scripts = Path(__file__).parent
    run = Path(os.environ["P1_RUN_DIR"])
    shutil.copyfile(scripts / "serving_reclamation.py", run / "serving_reclamation.py")
    os.environ["PYTHONPATH"] = str(scripts) + os.pathsep + os.environ["PYTHONPATH"]
    original_popen = subprocess.Popen
    original_workload = serving_reclamation.workload

    def server(command, *args, **kwargs):
        if "vllm.entrypoints.openai.api_server" in command:
            command = [
                *command,
                "--worker-extension-cls",
                "profile_tp_placement.P1TPPlacementWorker",
            ]
        return original_popen(command, *args, **kwargs)

    async def workload(origin, prompts, repeats, directory, prefix, warmup, arrivals):
        workers = [
            json.loads((run / f"worker-{rank}.json").read_text()) for rank in range(2)
        ]
        placements = [
            ("unbound", None),
            ("slow-remote-a", [1, 0]),
            ("slow-local", [0, 1]),
            ("fast-local", [20, 21]),
            ("fast-remote", [21, 20]),
            ("slow-remote-b", [1, 0]),
            ("fast-local-b", [20, 21]),
        ]
        for label, cpus in placements:
            if cpus is not None:
                for worker in workers:
                    os.sched_setaffinity(worker["pid"], {cpus[worker["rank"]]})
            phase = run / label
            phase.mkdir()
            (phase / "placement.json").write_text(
                json.dumps(
                    {
                        "workers": workers,
                        "cpus": cpus,
                        "affinity": [
                            sorted(os.sched_getaffinity(w["pid"])) for w in workers
                        ],
                    },
                    indent=2,
                )
            )
            print("PLACEMENT=" + label, flush=True)
            await original_workload(
                origin, prompts, repeats, phase, prefix, warmup, arrivals
            )

    subprocess.Popen = server
    serving_reclamation.workload = workload
    serving_reclamation.main()


if __name__ == "__main__":
    main()
else:
    install()
