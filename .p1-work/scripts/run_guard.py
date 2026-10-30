# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Record one experiment and monitor aggregate project NVML memory.

Runs are serialized. Stop the experiment's process group at 44 GiB, retaining
1 GiB below the project's 45 GiB limit. Never signal unrelated GPU processes.
"""

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pynvml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    base = Path("/data_ssd/qcrs/experiments/p1-v8")
    lock = (base / "gpu.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    run = base / args.stage / (time.strftime("%Y%m%d-%H%M%S") + "-" + args.name)
    run.mkdir(parents=True)
    command = args.command
    if command[0] == "--":
        command = command[1:]
    command = [str(root / ".venv/bin/python") if c == "@python" else c for c in command]
    env = dict(os.environ, P1_RUN_DIR=str(run))
    record = {
        "command": command,
        "cwd": str(root),
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "diff": subprocess.check_output(["git", "diff", "HEAD"], text=True),
        "environment": {
            k: v
            for k, v in env.items()
            if k.startswith(("P1_", "VLLM_", "CUDA_", "HF_", "TRITON_"))
        },
        "hard_limit_bytes": 45 * 2**30,
        "stop_threshold_bytes": 44 * 2**30,
        "sampling_interval_s": 0.1,
    }
    record["source_files"] = {}
    changed = (
        subprocess.check_output(
            [
                "git",
                "ls-files",
                "--modified",
                "--others",
                "--exclude-standard",
                "-z",
                "vllm",
                "tests",
                "benchmarks",
            ],
            text=True,
        )
        .strip("\0")
        .split("\0")
    )
    sources = {Path(__file__), *(Path(c) for c in command if c.endswith(".py"))}
    sources.update(Path(p) for p in changed if p)
    for source in sorted(sources):
        source = source.resolve()
        target = run / "source" / source.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        record["source_files"][str(source)] = hashlib.sha256(
            source.read_bytes()
        ).hexdigest()
    (run / "invocation.json").write_text(json.dumps(record, indent=2))
    pynvml.nvmlInit()
    handles = [
        pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(pynvml.nvmlDeviceGetCount())
    ]
    peak = 0
    exceeded = False
    stopped_at = None
    start = time.time()
    print(f"P1_RUN_DIR={run}", flush=True)
    with (
        (run / "stdout.log").open("w") as log,
        (run / "nvml.jsonl").open("w") as memory,
    ):
        proc = subprocess.Popen(
            command,
            cwd=root,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        (run / "pid").write_text(str(proc.pid))
        while True:
            gpu_records = []
            total = 0
            for index, handle in enumerate(handles):
                for gpu_proc in pynvml.nvmlDeviceGetComputeRunningProcesses(handle):
                    try:
                        project = (
                            b"P1_WORKTREE=" + str(root).encode() + b"\0"
                            in Path(f"/proc/{gpu_proc.pid}/environ").read_bytes()
                        )
                        ours = os.getpgid(gpu_proc.pid) == proc.pid
                    except (FileNotFoundError, ProcessLookupError, PermissionError):
                        continue
                    if project or ours:
                        used = gpu_proc.usedGpuMemory
                        total += used
                        gpu_records.append(
                            {"gpu": index, "pid": gpu_proc.pid, "bytes": used}
                        )
            peak = max(peak, total)
            memory.write(
                json.dumps(
                    {"t": time.time() - start, "bytes": total, "processes": gpu_records}
                )
                + "\n"
            )
            memory.flush()
            if (
                total >= record["stop_threshold_bytes"]
                and proc.poll() is None
                and stopped_at is None
            ):
                exceeded = True
                stopped_at = time.monotonic()
                os.killpg(proc.pid, signal.SIGTERM)
            if (
                stopped_at is not None
                and time.monotonic() - stopped_at > 10
                and proc.poll() is None
            ):
                os.killpg(proc.pid, signal.SIGKILL)
            code = proc.poll()
            if code is not None:
                break
            time.sleep(0.1)
        code = proc.wait()
    result = {
        "status": "PASS" if code == 0 and not exceeded else "FAIL",
        "exit_code": code,
        "peak_project_nvml_bytes": peak,
        "limit_triggered": exceeded,
        "elapsed_s": time.time() - start,
    }
    (run / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
    pynvml.nvmlShutdown()
    sys.exit(1 if exceeded else code)


if __name__ == "__main__":
    main()
