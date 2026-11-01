# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fixed DP2 HTTP workload, native routes and per-replica latency evidence."""

import argparse
import asyncio
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import numpy as np
from serving_reclamation import (
    configure_worker_cpus,
    counters,
    quantiles,
    tuning_arguments,
)
from transformers import AutoTokenizer


async def request(
    client, origin, prompt, rid, arrival, epoch, replica=None, maximum=128
):
    await asyncio.sleep(max(0, epoch + arrival - time.perf_counter()))
    started = time.perf_counter()
    times, tokens = [], []
    headers = {} if replica is None else {"X-data-parallel-rank": str(replica)}
    body = {
        "model": "/data/models/Qwen3-8B",
        "prompt": prompt,
        "request_id": rid,
        "temperature": 0,
        "seed": 0,
        "max_tokens": maximum,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
        "return_token_ids": True,
    }
    usage = None
    async with client.stream(
        "POST", origin + "/v1/completions", json=body, headers=headers
    ) as response:
        assert response.status_code == 200, await response.aread()
        async for line in response.aiter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            value = json.loads(line[6:])
            assert "error" not in value, value
            if value.get("usage"):
                usage = value["usage"]
            for choice in value["choices"]:
                chunk = choice.get("token_ids") or []
                times.extend([time.perf_counter()] * len(chunk))
                tokens.extend(chunk)
    assert len(tokens) == maximum and usage is not None
    return {
        "request": rid,
        "prompt_tokens": len(prompt),
        "output_tokens": len(tokens),
        "start_s": started - epoch,
        "arrival_s": arrival,
        "finish_s": time.perf_counter() - epoch,
        "ttft_s": times[0] - started,
        "tpot_s": (times[-1] - times[0]) / (len(tokens) - 1),
        "itl_s": np.diff(times).tolist(),
        "token_ids": tokens,
        "usage": usage,
    }


def aggregate(items, elapsed):
    good = sum(i["ttft_s"] <= 2 and i["tpot_s"] <= 0.08 for i in items)
    return {
        "requests": len(items),
        "slo_good_requests": good,
        "output_tokens_per_s": sum(i["output_tokens"] for i in items) / elapsed,
        "slo_goodput_requests_per_s": good / elapsed,
        "ttft_s": quantiles([i["ttft_s"] for i in items]),
        "tpot_s": quantiles([i["tpot_s"] for i in items]),
        "itl_s": quantiles([dt for i in items for dt in i["itl_s"]]),
        "active_span_s": max(i["finish_s"] for i in items)
        - min(i["start_s"] for i in items),
    }


async def workload(args, run, origin, prompts, arrivals):
    rounds = []
    async with httpx.AsyncClient(timeout=600, trust_env=False) as client:
        await request(
            client,
            origin,
            prompts[0][:32],
            "small-warmup",
            0,
            time.perf_counter(),
            maximum=8,
        )
        for repeat in ["warmup", *range(args.repeats)]:
            before = (await client.get(origin + "/metrics")).text
            (run / f"metrics-before-{repeat}.txt").write_text(before)
            start_ns = time.monotonic_ns()
            epoch = time.perf_counter()
            items = await asyncio.gather(
                *[
                    request(
                        client,
                        origin,
                        prompt,
                        f"r{repeat}-{i}",
                        arrivals[i],
                        epoch,
                        (0 if i < 6 else 1) if args.scenario == "skewed" else None,
                    )
                    for i, prompt in enumerate(prompts)
                ]
            )
            elapsed = time.perf_counter() - epoch
            end_ns = time.monotonic_ns()
            routes = [
                json.loads(line)
                for line in (run / "routing.jsonl").read_text().splitlines()
            ]
            for item in items:
                assigned = [
                    r for r in routes if r["external"] == f"cmpl-{item['request']}-0"
                ]
                assert len(assigned) == 1, (item["request"], assigned, routes)
                item["replica"] = assigned[0]["replica"]
            after = (await client.get(origin + "/metrics")).text
            prior = counters(before)
            row = {
                "repeat": repeat,
                "start_monotonic_ns": start_ns,
                "end_monotonic_ns": end_ns,
                "elapsed_s": elapsed,
                "aggregate": aggregate(items, elapsed),
                "replicas": {
                    str(rank): aggregate(
                        [i for i in items if i["replica"] == rank], elapsed
                    )
                    for rank in range(2)
                },
                "counter_deltas": {k: v - prior[k] for k, v in counters(after).items()},
            }
            (run / f"requests-{repeat}.json").write_text(json.dumps(items, indent=2))
            (run / f"metrics-{repeat}.txt").write_text(after)
            if repeat == "warmup":
                (run / "warmup-summary.json").write_text(json.dumps(row, indent=2))
            else:
                rounds.append(row)
            print(json.dumps(row), flush=True)
    (run / "serving.json").write_text(json.dumps(rounds, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--policy",
        choices=["full", "noop", "sink_recent", "random", "triattention"],
        required=True,
    )
    parser.add_argument("--pool-gib", type=float, required=True)
    parser.add_argument(
        "--scenario",
        choices=["native", "skewed", "uniform", "w1", "w0"],
        default="native",
    )
    parser.add_argument("--stats")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--load-diagnostic", action="store_true")
    parser.add_argument(
        "--routing", choices=["requests", "known_work"], default="requests"
    )
    parser.add_argument("--work-counters", action="store_true")
    tuning_arguments(parser)
    args = parser.parse_args()
    run = Path(os.environ["P1_RUN_DIR"])
    scripts = Path(__file__).parent
    for name in (
        "dp_server.py",
        "serving_reclamation.py",
        "dp_hooks.py",
        "tp_worker_hooks.py",
        "dp_load_diagnostic.py",
    ):
        shutil.copyfile(scripts / name, run / name)
    source_root = scripts.parents[1]
    source_hashes = {}
    for relative in (
        "vllm/v1/engine/core.py",
        "vllm/v1/engine/coordinator.py",
        "vllm/v1/engine/core_client.py",
        "vllm/v1/core/sched/scheduler.py",
        "vllm/v1/core/sched/interface.py",
        "vllm/v1/metrics/stats.py",
    ):
        source = source_root / relative
        target = run / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        source_hashes[relative] = hashlib.sha256(source.read_bytes()).hexdigest()
    (run / "dp-source-hashes.json").write_text(json.dumps(source_hashes, indent=2))
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    origin = f"http://127.0.0.1:{port}"
    env = dict(os.environ, VLLM_ENABLE_V1_MULTIPROCESSING="1")
    if args.load_diagnostic:
        env["P1_DP_LOAD_DIAGNOSTIC"] = "1"
    env["PYTHONPATH"] = str(scripts) + os.pathsep + env["PYTHONPATH"]
    env.pop("VLLM_KV_TRACE_FILE", None)
    command = [
        sys.executable,
        str(scripts / "dp_server.py"),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--model",
        "/data/models/Qwen3-8B",
        "--dtype",
        "bfloat16",
        "--data-parallel-size",
        "2",
        "--data-parallel-size-local",
        "2",
        "--attention-backend",
        "FLASH_ATTN",
        "--block-size",
        "16",
        "--no-enable-prefix-caching",
        "--enable-prompt-tokens-details",
        "--no-async-scheduling",
        "--disable-cascade-attn",
        "--no-scheduler-reserve-full-isl",
        "--max-model-len",
        "9216",
        "--max-num-batched-tokens",
        str(args.max_num_batched_tokens),
        "--max-num-seqs",
        str(args.max_num_seqs),
        "--kv-cache-memory-bytes",
        str(int(args.pool_gib * 2**30)),
        "--gpu-memory-utilization",
        "0.4",
        "--seed",
        "0",
        "--no-enable-log-requests",
        "--shutdown-timeout",
        "20",
        "--compilation-config",
        json.dumps(
            {
                "mode": 0,
                "cudagraph_mode": "FULL_DECODE_ONLY",
                "cudagraph_capture_sizes": [1, 2, 4, 8],
            }
        ),
        "--cudagraph-metrics",
    ]
    configure_worker_cpus(args, command, env, run)
    additional = {"dp_load_balancing": args.routing}
    if args.policy != "full":
        additional["kv_reclamation"] = {
            "policy": "sink_recent" if args.policy == "noop" else args.policy,
            "retention": 1.0 if args.policy == "noop" else 0.5,
            "pressure": True,
            "reserve_cap": 0,
            "stats_path": args.stats,
        }
    command.extend(["--additional-config", json.dumps(additional)])
    if args.trace:
        command.extend(["--scheduler-cls", "dp_hooks.P1DPScheduler"])
    elif args.work_counters:
        command.extend(["--scheduler-cls", "dp_hooks.P1DPWorkScheduler"])
    tokenizer = AutoTokenizer.from_pretrained(
        "/data/models/Qwen3-8B", local_files_only=True
    )
    seed = tokenizer.encode(
        "Mary went to the garden. John carried the bag. Explain their journey. ",
        add_special_tokens=False,
    )
    lengths = [4096, 8192] * 4
    if args.scenario == "skewed":
        lengths[-2:] = [128, 256]
    elif args.scenario == "uniform":
        lengths = [4096] * 8
    elif args.scenario == "w1":
        lengths = [8192, 512] * 4
    elif args.scenario == "w0":
        lengths = [4096] * 8
    prompts = [(seed * (length // len(seed) + 1))[:length] for length in lengths]
    arrivals = [0, 0.3, 0.6, 0.9, 5, 5.3, 5.6, 5.9]
    if args.scenario in ("w1", "w0"):
        arrivals = [i * 0.05 for i in range(8)]
    (run / "prompts.json").write_text(json.dumps(prompts))
    (run / "protocol.json").write_text(
        json.dumps(
            {
                "args": vars(args),
                "command": command,
                "prompt_tokens": lengths,
                "output_tokens": 128,
                "arrivals_s": arrivals,
                "warmup": "32/8, then full workload; both excluded",
                "slo": {"ttft_s": 2, "tpot_s": 0.08},
                "transport_timeout_s": 600,
                "pool": "bytes per replica, independent pools",
                "prompt_sha256": [
                    hashlib.sha256(json.dumps(p).encode()).hexdigest() for p in prompts
                ],
                "routing": (
                    f"{args.routing}, one observation per ADD; "
                    "fixed rank only for skewed case"
                ),
            },
            indent=2,
        )
    )
    with (run / "server.log").open("w") as log:
        server = subprocess.Popen(
            command, env=env, stdout=log, stderr=subprocess.STDOUT
        )
        try:
            deadline = time.monotonic() + 240
            with httpx.Client(timeout=1, trust_env=False) as client:
                while time.monotonic() < deadline:
                    if server.poll() is not None:
                        raise RuntimeError(f"Server exited {server.returncode}")
                    try:
                        if client.get(origin + "/health").status_code == 200:
                            break
                    except httpx.TransportError:
                        pass
                    time.sleep(0.5)
                else:
                    raise TimeoutError("DP server did not become healthy")
            asyncio.run(workload(args, run, origin, prompts, arrivals))
        finally:
            if server.poll() is None:
                server.terminate()
            try:
                server.wait(timeout=45)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()


if __name__ == "__main__":
    main()
