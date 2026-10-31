# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Idle-server intervention to separate worker framework cost from placement."""

import ctypes
import json
import os
import shutil
import subprocess
import time
from functools import wraps
from pathlib import Path


class P1TPToggleWorker:
    def p1_mode(self, mode):
        import torch

        runner = self.model_runner
        if not hasattr(self, "_p1_reclamation"):
            self._p1_reclamation = runner.reclamation
        worker = self._p1_reclamation
        assert worker.config.retention == 1.0
        torch.accelerator.synchronize()
        if mode == "on":
            worker.effective_kv_len_np[:] = runner.req_states.num_computed_tokens_np
            worker.effective_kv_len.gpu.copy_(runner.req_states.num_computed_tokens.gpu)
            runner.reclamation = worker
        else:
            assert mode == "off"
            runner.reclamation = None
        return {"mode": mode, "pid": os.getpid()}


def install():
    from vllm.distributed import get_tp_group
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    from vllm.v1.worker.gpu_worker import Worker

    getcpu = ctypes.CDLL(None).sched_getcpu
    events = []
    execute = GPUModelRunner.execute_model
    shutdown = Worker.shutdown

    @wraps(execute)
    def measured(runner, *args, **kwargs):
        start = time.perf_counter_ns()
        cpu_start = time.thread_time_ns()
        cpu = getcpu()
        try:
            return execute(runner, *args, **kwargs)
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
        path = Path(os.environ["P1_RUN_DIR"]) / f"toggle-rank-{rank}.json"
        path.write_text(json.dumps(events))
        return shutdown(worker)

    GPUModelRunner.execute_model = measured
    Worker.shutdown = save


def main():
    import httpx
    import serving_reclamation

    scripts = Path(__file__).parent
    run = Path(os.environ["P1_RUN_DIR"])
    shutil.copyfile(scripts / "serving_reclamation.py", run / "serving_reclamation.py")
    os.environ["PYTHONPATH"] = str(scripts) + os.pathsep + os.environ["PYTHONPATH"]
    os.environ["VLLM_SERVER_DEV_MODE"] = "1"
    original_popen = subprocess.Popen
    original_workload = serving_reclamation.workload

    def server(command, *args, **kwargs):
        if "vllm.entrypoints.openai.api_server" in command:
            command = [
                *command,
                "--shutdown-timeout",
                "20",
                "--worker-extension-cls",
                "profile_tp_toggle.P1TPToggleWorker",
            ]
        return original_popen(command, *args, **kwargs)

    async def workload(origin, prompts, repeats, directory, prefix, warmup, arrivals):
        async with httpx.AsyncClient(timeout=60, trust_env=False) as client:
            for label, mode, cpus in (
                ("on-a", "on", None),
                ("off-a", "off", None),
                ("off-b", "off", None),
                ("on-b", "on", None),
                ("on-local", "on", [20, 21]),
                ("off-local", "off", [20, 21]),
            ):
                response = await client.post(
                    origin + "/collective_rpc",
                    json={"method": "p1_mode", "args": [mode]},
                )
                response.raise_for_status()
                workers = response.json()["results"]
                if cpus is not None:
                    for rank, worker in enumerate(workers):
                        os.sched_setaffinity(worker["pid"], {cpus[rank]})
                phase = run / label
                phase.mkdir()
                (phase / "mode.json").write_text(
                    json.dumps(
                        {
                            "workers": workers,
                            "cpus": cpus,
                            "boundary": (
                                "Idle transition, E=L, retention=1, eager only; "
                                "scheduler remains enabled. Diagnostic, not native."
                            ),
                        },
                        indent=2,
                    )
                )
                print("MODE=" + label, flush=True)
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
