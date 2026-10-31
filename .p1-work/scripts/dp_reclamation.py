# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native internal DP2: idle peer, local pressure/reuse, abort and routing."""

import argparse
import asyncio
import json
import os
import shutil
import time
from pathlib import Path

from transformers import AutoTokenizer

from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.core_client import DPLBAsyncMPClient


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


async def exercise(args, run):
    options = AsyncEngineArgs(
        model="/data/models/Qwen3-8B",
        dtype="bfloat16",
        data_parallel_size=2,
        data_parallel_size_local=2,
        worker_extension_cls="dp_hooks.P1DPWorker",
        scheduler_cls="dp_hooks.P1DPScheduler",
        enforce_eager=True,
        attention_backend="FLASH_ATTN",
        block_size=16,
        enable_prefix_caching=False,
        async_scheduling=False,
        disable_cascade_attn=True,
        scheduler_reserve_full_isl=False,
        max_model_len=512,
        max_num_seqs=4,
        max_num_batched_tokens=256,
        kv_cache_memory_bytes=256 * 2**20,
        num_gpu_blocks_override=33,
        gpu_memory_utilization=0.3,
        seed=0,
        disable_log_stats=True,
        additional_config={}
        if args.policy == "full"
        else {
            "kv_reclamation": {
                "policy": args.policy,
                "pressure": True,
                "min_keep_tokens": 32,
            }
        },
    )
    llm = AsyncLLM.from_engine_args(options)
    assert isinstance(llm.engine_core, DPLBAsyncMPClient)
    routes, outputs, token_counts = [], {}, {}
    original_route = llm.engine_core.get_core_engine_for_request

    def route(request):
        counts = [list(c) for c in llm.engine_core.lb_engines]
        begin = time.perf_counter_ns()
        chosen = original_route(request)
        routes.append(
            {
                "request": request.request_id,
                "replica": llm.engine_core.core_engines.index(chosen),
                "explicit": request.data_parallel_rank,
                "counts": counts,
                "cpu_ns": time.perf_counter_ns() - begin,
                "time_ns": time.monotonic_ns(),
            }
        )
        return chosen

    llm.engine_core.get_core_engine_for_request = route
    tokenizer = AutoTokenizer.from_pretrained(options.model, local_files_only=True)
    text = tokenizer.encode("Mary went to the garden. John carried the bag. " * 100)
    first = {name: asyncio.Event() for name in ("A", "C", "B")}

    async def generate(name, length, maximum, rank):
        result = None
        async for item in llm.generate(
            {"prompt_token_ids": text[:length]},
            SamplingParams(temperature=0, seed=0, ignore_eos=True, max_tokens=maximum),
            name,
            data_parallel_rank=rank,
        ):
            result = item
            token_counts[name] = len(item.outputs[0].token_ids)
            if name in first:
                first[name].set()
        outputs[name] = {
            "tokens": list(result.outputs[0].token_ids),
            "finish_reason": result.outputs[0].finish_reason,
        }

    async def until(predicate):
        deadline = time.monotonic() + 60
        while not predicate():
            for task in tasks:
                if task.done():
                    task.result()
            assert time.monotonic() < deadline, "DP diagnostic condition timeout"
            await asyncio.sleep(0.01)

    tasks = []
    try:
        await llm.collective_rpc("p1_install")
        identities = [
            json.loads((run / f"replica-{i}" / "worker.json").read_text())
            for i in range(2)
        ]
        assert len({i["pid"] for i in identities}) == 2
        assert len({i["gpu_uuid"] for i in identities}) == 2
        assert all(i["worker_dp_size"] == i["tp_size"] == 1 for i in identities)
        scheduler_identities = [
            lines(run / f"replica-{i}" / "scheduler.jsonl")[0] for i in range(2)
        ]
        assert len({i["pid"] for i in scheduler_identities}) == 2
        assert [i["engine"] for i in scheduler_identities] == [0, 1]
        assert all(
            i["total_blocks"] == 33 and i["free"] == 32 for i in scheduler_identities
        )
        tasks.append(asyncio.create_task(generate("A", 256, 256, 0)))
        await until(first["A"].is_set)
        tasks.append(asyncio.create_task(generate("C", 256, 64, 0)))
        trace0 = run / "replica-0" / "kv-trace.jsonl"
        await until(lambda: any(e["event"] == "allocation_miss" for e in lines(trace0)))
        first_miss = next(e for e in lines(trace0) if e["event"] == "allocation_miss")
        shortfall = first_miss["required"] - first_miss["available"]
        assert shortfall > 0
        idle = lines(run / "replica-1" / "kv-trace.jsonl")
        assert not idle, "Idle replica allocated or freed foreign blocks"
        idle_free = lines(run / "replica-1" / "scheduler.jsonl")[-1]["free"]
        assert idle_free == 32
        if args.policy != "full":
            await until(
                lambda: any(e["event"] == "reclaim_commit" for e in lines(trace0))
            )
            await until(first["C"].is_set)
        tasks.append(asyncio.create_task(generate("B", 128, 128, 1)))
        await until(first["B"].is_set)
        await until(lambda: token_counts.get("B", 0) >= 16)
        affinity = dict(llm.engine_core.reqs_in_flight)
        a_internal = next(r for r in affinity if r.startswith("A-"))
        assert affinity[a_internal] == llm.engine_core.core_engines[0]
        abort_time = time.monotonic_ns()
        peer_tokens_at_abort = token_counts["B"]
        assert peer_tokens_at_abort < 128
        await llm.abort("A")
        await asyncio.wait_for(asyncio.gather(*tasks), timeout=90)
        assert outputs["A"]["finish_reason"] == "abort"
        assert len(outputs["B"]["tokens"]) == 128 and len(outputs["C"]["tokens"]) == 64
        await asyncio.wait_for(
            asyncio.gather(
                *[generate(f"balanced-{i}", 128, 16, None) for i in range(4)]
            ),
            timeout=60,
        )
        await until(
            lambda: all(
                lines(run / f"replica-{i}" / "scheduler.jsonl")[-1].get("free") == 32
                for i in range(2)
            )
        )
        balanced = [r for r in routes if r["request"].startswith("balanced-")]
        assert {r["replica"] for r in balanced} == {0, 1}
        payload = [
            json.loads((run / f"replica-{i}" / "payload.json").read_text())
            for i in range(2)
        ]
        if args.policy != "full":
            assert payload[0]["counts"]["reclaim"] and payload[0]["rewritten"]
            assert payload[0]["counts"]["retained_forward"] > 0
            reused_by_c = {
                rid: pages
                for rid, pages in payload[0]["rewritten_by_request"].items()
                if rid.startswith("C-")
            }
            assert reused_by_c and all(reused_by_c.values())
        assert not payload[1]["released"]
        assert payload[1]["counts"]["peer_retained_forward"] > 1
        peer_after_abort = [
            e
            for e in lines(run / "replica-1" / "worker-trace.jsonl")
            if e["event"] == "forward"
            and e["time_ns"] > abort_time
            and any(rid.startswith("B-") for rid in e["peer_verified"])
        ]
        assert peer_after_abort
        (run / "evidence.json").write_text(
            json.dumps(
                {
                    "status": "PASS",
                    "policy": args.policy,
                    "identities": identities,
                    "scheduler_identities": scheduler_identities,
                    "first_local_miss": {**first_miss, "shortfall": shortfall},
                    "peer_free_at_first_miss": idle_free,
                    "routes": routes,
                    "outputs": outputs,
                    "payload": payload,
                    "peer_idle_before_first_miss": True,
                    "free_final": [32, 32],
                    "abort_time_ns": abort_time,
                    "peer_tokens_at_abort": peer_tokens_at_abort,
                    "peer_verified_forwards_after_abort": len(peer_after_abort),
                    "abort_affinity": {
                        k: llm.engine_core.core_engines.index(v)
                        for k, v in affinity.items()
                    },
                },
                indent=2,
            )
        )
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        llm.shutdown()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--policy", choices=["full", "sink_recent"], default="sink_recent"
    )
    args = parser.parse_args()
    run = Path(os.environ["P1_RUN_DIR"])
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "1"
    os.environ["VLLM_KV_TRACE_IDS"] = "1"
    for name in ("dp_hooks.py", "tp_worker_hooks.py"):
        shutil.copyfile(Path(__file__).with_name(name), run / name)
    asyncio.run(exercise(args, run))


if __name__ == "__main__":
    main()
