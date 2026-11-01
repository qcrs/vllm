# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fixed HTTP streaming workload; all timings include maintenance and waiting."""

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
from transformers import AutoTokenizer


def tuning_arguments(parser):
    parser.add_argument("--max-num-batched-tokens", type=int, default=2048)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument(
        "--worker-cpus",
        type=int,
        nargs="+",
        help="Optional Worker main-thread CPU per local GPU; host-specific",
    )


def configure_worker_cpus(args, command, env, run):
    if args.worker_cpus is None:
        return
    import shutil

    scripts = Path(__file__).parent
    shutil.copyfile(scripts / "worker_affinity.py", run / "worker_affinity.py")
    env["PYTHONPATH"] = str(scripts) + os.pathsep + env["PYTHONPATH"]
    env["P1_WORKER_CPUS"] = json.dumps(args.worker_cpus)
    command.extend(["--worker-cls", "worker_affinity.AffinityWorker"])


async def completion(client, origin, ids, rid, output_tokens, arrival, epoch):
    await asyncio.sleep(max(0, epoch + arrival - time.perf_counter()))
    started = time.perf_counter()
    times, tokens, chunks = [], [], []
    usage = None
    body = {
        "model": "/data/models/Qwen3-8B",
        "prompt": ids,
        "temperature": 0,
        "seed": 0,
        "max_tokens": output_tokens,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
        "return_token_ids": True,
    }
    async with client.stream("POST", origin + "/v1/completions", json=body) as reply:
        if reply.status_code != 200:
            raise RuntimeError((reply.status_code, (await reply.aread()).decode()))
        async for line in reply.aiter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            payload = json.loads(line[6:])
            if "error" in payload:
                raise RuntimeError(payload["error"])
            if payload.get("usage"):
                usage = payload["usage"]
            for choice in payload["choices"]:
                ids_out = choice.get("token_ids") or []
                if ids_out:
                    now = time.perf_counter()
                    times.extend([now] * len(ids_out))
                    tokens.extend(ids_out)
                    chunks.append(len(ids_out))
    assert len(tokens) == output_tokens, (rid, len(tokens), output_tokens)
    assert usage is not None, rid
    return {
        "request": rid,
        "prompt_tokens": len(ids),
        "output_tokens": len(tokens),
        "arrival_s": arrival,
        "start_s": started - epoch,
        "ttft_s": times[0] - started,
        "tpot_s": (times[-1] - times[0]) / (len(tokens) - 1),
        "itl_s": np.diff(times).tolist(),
        "e2e_s": time.perf_counter() - started,
        "token_ids": tokens,
        "stream_chunk_sizes": chunks,
        "usage": usage,
    }


def quantiles(values):
    return dict(zip(("p50", "p95", "p99"), np.percentile(values, [50, 95, 99])))


def counters(metrics):
    names = (
        "vllm:num_preemptions_total",
        "vllm:prefix_cache_queries_total",
        "vllm:prefix_cache_hits_total",
        "vllm:prompt_tokens_total",
        "vllm:prompt_tokens_by_source_total",
    )
    return {
        key: float(value)
        for line in metrics.splitlines()
        if line.startswith(names)
        for key, value, *_ in [line.split()]
    }


async def workload(
    origin,
    prompts,
    repeats,
    run,
    prefix_caching,
    full_warmup,
    arrivals,
    reclamation_counts=False,
):
    results = []
    limits = httpx.Limits(max_connections=16)
    async with httpx.AsyncClient(timeout=600, limits=limits, trust_env=False) as client:
        await completion(
            client, origin, prompts[0][:32], "warmup", 8, 0, time.perf_counter()
        )
        rounds = (
            ["warmup", *range(repeats)]
            if prefix_caching or full_warmup
            else range(repeats)
        )
        for repeat in rounds:

            async def read_reclamation_counts():
                reply = await client.post(
                    origin + "/collective_rpc",
                    json={"method": "p1_reclamation_counts"},
                )
                reply.raise_for_status()
                return reply.json()["results"]

            counts_before = (
                await read_reclamation_counts() if reclamation_counts else None
            )
            if prefix_caching:
                reply = await client.post(origin + "/reset_prefix_cache")
                reply.raise_for_status()
                assert reply.json() == {"success": True}, reply.text
            before = (await client.get(origin + "/metrics")).text
            (run / f"metrics-before-{repeat}.txt").write_text(before)
            if repeat == 0:
                (run / "metrics-before.txt").write_text(before)
            start_ns = time.monotonic_ns()
            epoch = time.perf_counter()
            items = await asyncio.gather(
                *[
                    completion(
                        client,
                        origin,
                        prompts[i % len(prompts)],
                        f"r{repeat}-{i}",
                        128,
                        arrivals[i],
                        epoch,
                    )
                    for i in range(len(arrivals))
                ]
            )
            elapsed = time.perf_counter() - epoch
            end_ns = time.monotonic_ns()
            good = sum(i["ttft_s"] <= 2 and i["tpot_s"] <= 0.08 for i in items)
            summary = {
                "repeat": repeat,
                "start_monotonic_ns": start_ns,
                "end_monotonic_ns": end_ns,
                "requests": len(items),
                "elapsed_s": elapsed,
                "output_tokens_per_s": sum(i["output_tokens"] for i in items) / elapsed,
                "request_throughput_per_s": len(items) / elapsed,
                "slo_goodput_requests_per_s": good / elapsed,
                "slo_good_requests": good,
                "ttft_s": quantiles([i["ttft_s"] for i in items]),
                "tpot_s": quantiles([i["tpot_s"] for i in items]),
                "itl_s": quantiles([dt for i in items for dt in i["itl_s"]]),
            }
            if prefix_caching:
                cached = [
                    i["usage"]["prompt_tokens_details"]["cached_tokens"] for i in items
                ]
                summary["cached_prompt_tokens_by_request"] = cached
                summary["cached_prompt_tokens"] = sum(cached)
            after = (await client.get(origin + "/metrics")).text
            if reclamation_counts:
                counts_after = await read_reclamation_counts()
                summary["reclamation_counts_before"] = counts_before
                summary["reclamation_counts_after"] = counts_after
            before_counts = counters(before)
            summary["counter_deltas"] = {
                key: value - before_counts[key]
                for key, value in counters(after).items()
            }
            if repeat != "warmup":
                results.append(summary)
            else:
                (run / "warmup-summary.json").write_text(json.dumps(summary, indent=2))
            (run / f"requests-{repeat}.json").write_text(json.dumps(items, indent=2))
            (run / f"metrics-{repeat}.txt").write_text(after)
            print(json.dumps(summary), flush=True)
    (run / "serving.json").write_text(json.dumps(results, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--policy",
        choices=["full", "noop", "sink_recent", "random", "triattention"],
        required=True,
    )
    parser.add_argument("--pool-gib", type=float, required=True)
    parser.add_argument("--stats")
    parser.add_argument("--reserve", type=int, default=0)
    parser.add_argument(
        "--budget-mode",
        choices=["fixed", "demand", "retry", "retry_pass", "prefill_fixed"],
        default="fixed",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--reclamation-counts", action="store_true")
    parser.add_argument("--prefix-caching", action="store_true")
    parser.add_argument(
        "--graph-mode", choices=["eager", "full", "piecewise"], default="eager"
    )
    parser.add_argument("--full-warmup", action="store_true")
    parser.add_argument("--workload", choices=["long", "short"], default="long")
    parser.add_argument("--tp", type=int, choices=[1, 2], default=1)
    tuning_arguments(parser)
    args = parser.parse_args()
    if args.workload == "short" and args.prefix_caching:
        parser.error("short workload does not define shared prefix identities")
    run = Path(os.environ["P1_RUN_DIR"])
    source_root = Path(__file__).resolve().parents[2]
    sources = [
        "vllm/v1/worker/gpu/kv_scoring.py",
        "vllm/v1/worker/gpu/kv_reclamation.py",
        "vllm/v1/worker/gpu/kv_compaction.py",
        "vllm/v1/kv_reclamation.py",
        "vllm/v1/core/kv_cache_manager.py",
        "vllm/v1/core/sched/scheduler.py",
    ]
    hashes = {}
    for name in sources:
        source = source_root / name
        target = run / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        hashes[name] = hashlib.sha256(source.read_bytes()).hexdigest()
    if args.stats:
        hashes[args.stats] = hashlib.sha256(Path(args.stats).read_bytes()).hexdigest()
    (run / "runtime-source-hashes.json").write_text(json.dumps(hashes, indent=2))
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    origin = f"http://127.0.0.1:{port}"
    env = dict(os.environ, VLLM_ENABLE_V1_MULTIPROCESSING="1")
    if args.reclamation_counts:
        scripts = Path(__file__).parent
        shutil.copyfile(
            scripts / "reclamation_counts.py", run / "reclamation_counts.py"
        )
        (run / "sitecustomize.py").write_text(
            "import reclamation_counts\nreclamation_counts.install()\n"
        )
        env["PYTHONPATH"] = str(run) + os.pathsep + env["PYTHONPATH"]
        env["VLLM_SERVER_DEV_MODE"] = "1"
    env.pop("VLLM_KV_TRACE_FILE", None)
    if args.prefix_caching:
        env["VLLM_SERVER_DEV_MODE"] = "1"
    if args.graph_mode == "piecewise":
        env["VLLM_USE_BREAKABLE_CUDAGRAPH"] = "1"
    if args.trace:
        env["VLLM_KV_TRACE_FILE"] = str(run / "kv-trace.jsonl")
        env["VLLM_KV_TRACE_IDS"] = "0"
    command = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--model",
        "/data/models/Qwen3-8B",
        "--dtype",
        "bfloat16",
        "--tensor-parallel-size",
        str(args.tp),
        "--attention-backend",
        "FLASH_ATTN",
        "--block-size",
        "16",
        "--enable-prefix-caching"
        if args.prefix_caching
        else "--no-enable-prefix-caching",
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
    ]
    configure_worker_cpus(args, command, env, run)
    if args.graph_mode == "eager":
        command.append("--enforce-eager")
    else:
        command.extend(
            [
                "--compilation-config",
                json.dumps(
                    {
                        "mode": 0,
                        "cudagraph_mode": "FULL_DECODE_ONLY"
                        if args.graph_mode == "full"
                        else "PIECEWISE",
                        "cudagraph_capture_sizes": [1, 2, 4, 8],
                    }
                ),
                "--cudagraph-metrics",
            ]
        )
    if args.policy != "full":
        config = {
            "kv_reclamation": {
                "policy": "sink_recent" if args.policy == "noop" else args.policy,
                "retention": 1.0 if args.policy == "noop" else 0.5,
                "pressure": True,
                "reserve_cap": args.reserve,
                "budget_mode": args.budget_mode,
                "stats_path": args.stats,
            }
        }
        command.extend(["--additional-config", json.dumps(config)])
    tokenizer = AutoTokenizer.from_pretrained(
        "/data/models/Qwen3-8B", local_files_only=True
    )
    seed = tokenizer.encode(
        "Mary went to the garden. John carried the bag. Explain their journey. ",
        add_special_tokens=False,
    )
    prompts = [(seed * (length // len(seed) + 1))[:length] for length in (4096, 8192)]
    arrivals = [0, 0.3, 0.6, 0.9, 5, 5.3, 5.6, 5.9]
    if args.workload == "short":
        prompts = [(seed * (length // len(seed) + 1))[:length] for length in (128, 256)]
        arrivals = [0, 0.3]
    if args.prefix_caching:
        shared = (seed * (256 // len(seed) + 1))[:256]
        prompts = []
        for i, length in enumerate([4096, 8192] * 4):
            marker = tokenizer.encode(str(i), add_special_tokens=False)
            assert len(marker) == 1
            private = marker + (seed * (length // len(seed) + 1))
            prompts.append(shared + private[: length - len(shared)])
        assert len({prompt[256] for prompt in prompts}) == 8
    (run / "prompts.json").write_text(json.dumps(prompts))
    protocol = {
        "args": vars(args),
        "command": command,
        "origin": origin,
        "prompt_tokens": [len(prompts[i % len(prompts)]) for i in range(len(arrivals))],
        "output_tokens": 128,
        "arrivals_s": arrivals,
        "warmup": "one 32-token prompt / 8 output tokens; excluded",
        "slo": {"ttft_s": 2, "tpot_s": 0.08},
        "timing": "client token_ids stream; same-chunk tokens share receipt time",
        "transport_timeout_s": 600,
        "quality": "synthetic serving load; quality measured separately",
        "prompt_sha256": [
            hashlib.sha256(json.dumps(prompt).encode()).hexdigest()
            for prompt in prompts
        ],
        "trace": args.trace,
    }
    if args.prefix_caching:
        protocol.update(
            shared_prefix_tokens=256,
            private_suffix="eight different first tokens after the shared prefix",
            warmup="32/8 request, then one entire workload round; both excluded",
            cache_state="native reset_prefix_cache before warmup and each timed round",
            cached_tokens=(
                "first streamed output usage; prefix counters exclude preempted queries"
            ),
        )
    elif args.graph_mode != "eager" or args.full_warmup:
        protocol["warmup"] = (
            "32/8 request, then one entire workload round; both excluded"
        )
    (run / "protocol.json").write_text(json.dumps(protocol, indent=2))
    with (run / "server.log").open("w") as log:
        server = subprocess.Popen(
            command, env=env, stdout=log, stderr=subprocess.STDOUT
        )
        (run / "server.pid").write_text(str(server.pid))
        try:
            deadline = time.monotonic() + 240
            with httpx.Client(timeout=1, trust_env=False) as client:
                while time.monotonic() < deadline:
                    if server.poll() is not None:
                        raise RuntimeError(f"Server exited with {server.returncode}")
                    try:
                        if client.get(origin + "/health").status_code == 200:
                            break
                    except httpx.TransportError:
                        pass
                    time.sleep(0.5)
                else:
                    raise TimeoutError("Server did not become healthy in 240 seconds")
            asyncio.run(
                workload(
                    origin,
                    prompts,
                    args.repeats,
                    run,
                    args.prefix_caching,
                    args.full_warmup or args.graph_mode != "eager",
                    arrivals,
                    args.reclamation_counts,
                )
            )
        finally:
            if server.poll() is None:
                server.terminate()
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()
    print(json.dumps({"status": "PASS"}), flush=True)


if __name__ == "__main__":
    main()
