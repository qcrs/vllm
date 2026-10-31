# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Separate host-phase diagnostic; no device synchronization or payload hooks."""

import json
import os
import shutil
import subprocess
import time
from functools import wraps
from pathlib import Path


class P1TPCPUWorker:
    pass


def install():
    import cProfile

    from vllm.distributed import get_tp_group
    from vllm.v1.worker.gpu.kv_reclamation import KVReclamationWorker
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    from vllm.v1.worker.gpu_worker import Worker

    events = []
    thread_events = []
    profile = cProfile.Profile()
    forward_profile = os.environ.get("P1_TP_PROFILE_FORWARD") == "1"
    step = 0

    def instrument(cls, name, detailed=False):
        original = getattr(cls, name)

        @wraps(original)
        def measured(*args, **kwargs):
            nonlocal step
            if name == "execute_model":
                step += 1
            start = time.perf_counter_ns()
            thread_start = time.thread_time_ns()
            profiling = detailed and (not forward_profile or step >= 150)
            if profiling:
                profile.enable()
            try:
                return original(*args, **kwargs)
            finally:
                if profiling:
                    profile.disable()
                events.append(
                    (step, cls.__name__ + "." + name, start, time.perf_counter_ns())
                )
                thread_events.append(
                    (
                        step,
                        cls.__name__ + "." + name,
                        time.thread_time_ns() - thread_start,
                    )
                )

        setattr(cls, name, measured)

    for name in (
        "execute_model",
        "sample_tokens",
        "prepare_inputs",
        "prepare_attn",
        "update_requests",
        "add_requests",
        "postprocess_sampled",
    ):
        instrument(
            GPUModelRunner,
            name,
            name == ("execute_model" if forward_profile else "update_requests"),
        )
    for name in ("prepare_attn", "advance"):
        instrument(KVReclamationWorker, name, not forward_profile)
    shutdown = Worker.shutdown

    @wraps(shutdown)
    def save(worker):
        rank = get_tp_group().rank_in_group
        run = Path(os.environ["P1_RUN_DIR"])
        (run / f"cpu-rank-{rank}.json").write_text(json.dumps(events))
        (run / f"cpu-thread-rank-{rank}.json").write_text(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "affinity": sorted(os.sched_getaffinity(0)),
                    "events": thread_events,
                }
            )
        )
        profile.dump_stats(str(run / f"cpu-rank-{rank}.prof"))
        return shutdown(worker)

    Worker.shutdown = save


def main():
    import serving_reclamation

    scripts = Path(__file__).parent
    run = Path(os.environ["P1_RUN_DIR"])
    shutil.copyfile(scripts / "serving_reclamation.py", run / "serving_reclamation.py")
    os.environ["PYTHONPATH"] = str(scripts) + os.pathsep + os.environ["PYTHONPATH"]
    original_popen = subprocess.Popen

    def profiled_server(command, *args, **kwargs):
        if "vllm.entrypoints.openai.api_server" in command:
            command = [
                *command,
                "--shutdown-timeout",
                "20",
                "--worker-extension-cls",
                "profile_tp_cpu.P1TPCPUWorker",
            ]
        return original_popen(command, *args, **kwargs)

    subprocess.Popen = profiled_server
    serving_reclamation.main()


if __name__ == "__main__":
    main()
else:
    install()
