# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real TP2 workers, delayed non-output rank, release/reuse and combinations."""

import argparse
import json
import os
import shutil
import threading
import time
from pathlib import Path

from transformers import AutoTokenizer

from vllm import EngineArgs, LLMEngine, SamplingParams
from vllm.config import CompilationConfig, CompilationMode
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_reclamation import sink_recent_ranges


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["eager", "full"], default="eager")
    parser.add_argument("--path", choices=["whole", "token"], default="token")
    parser.add_argument("--prefix", action="store_true")
    parser.add_argument(
        "--policy",
        choices=["sink_recent", "random", "triattention"],
        default="sink_recent",
    )
    parser.add_argument("--delay", action="store_true")
    parser.add_argument("--fail", action="store_true")
    parser.add_argument(
        "--lifecycle", choices=["repeat", "q0", "maintenance"], default="repeat"
    )
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--worker-cpus", nargs=2, type=int)
    parser.add_argument("--demand-pages", type=int)
    args = parser.parse_args()
    run = Path(os.environ["P1_RUN_DIR"])
    hook = Path(__file__).with_name("tp_worker_hooks.py")
    shutil.copyfile(hook, run / "tp_worker_hooks.py")
    if args.profile:
        shutil.copyfile(
            hook.with_name("tp_profile_hooks.py"), run / "tp_profile_hooks.py"
        )
    if args.worker_cpus:
        os.environ["P1_WORKER_CPUS"] = json.dumps(args.worker_cpus)
        shutil.copyfile(
            hook.with_name("worker_affinity.py"), run / "worker_affinity.py"
        )
    os.environ["VLLM_KV_TRACE_FILE"] = str(run / "kv-trace.jsonl")
    os.environ["VLLM_KV_TRACE_IDS"] = "1"
    os.environ["VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS"] = "30"
    options = EngineArgs(
        model="/data/models/Qwen3-8B",
        dtype="bfloat16",
        tensor_parallel_size=2,
        distributed_executor_backend="mp",
        worker_extension_cls=(
            "tp_profile_hooks.P1TPProfileWorker"
            if args.profile
            else "tp_worker_hooks.P1TPWorker"
        ),
        enforce_eager=args.mode == "eager",
        attention_backend="FLASH_ATTN",
        block_size=16,
        enable_prefix_caching=args.prefix,
        async_scheduling=False,
        disable_cascade_attn=True,
        scheduler_reserve_full_isl=False,
        max_model_len=512,
        max_num_seqs=4,
        max_num_batched_tokens=256,
        kv_cache_memory_bytes=128 * 2**20,
        num_gpu_blocks_override=33,
        gpu_memory_utilization=0.3,
        seed=0,
        disable_log_stats=True,
        compilation_config=CompilationConfig(
            mode=CompilationMode.NONE,
            cudagraph_mode="FULL_DECODE_ONLY",
            cudagraph_capture_sizes=[2, 4],
        ),
        additional_config={
            "kv_reclamation": {
                "min_keep_tokens": 32,
                "policy": args.policy,
                "stats_path": os.environ.get("P1_STATS"),
            }
        },
    )
    if args.worker_cpus:
        options.worker_cls = "worker_affinity.AffinityWorker"
    engine = LLMEngine.from_engine_args(options)
    core = engine.engine_core.engine_core
    scheduler, executor = core.scheduler, core.model_executor
    manager = scheduler.kv_cache_manager
    pool = manager.block_pool
    assert core.batch_queue is None and engine.vllm_config.max_concurrent_batches == 1
    non_output = 1 - executor.output_rank
    worker_processes = [handle.proc for handle in executor.workers]
    identities = executor.collective_rpc(
        "p1_install",
        args=(
            {
                "delay_rank": non_output if args.delay else -1,
                "delay_s": 3,
                "fail_rank": non_output if args.fail else -1,
            },
        ),
    )
    (run / "identity.json").write_text(
        json.dumps(
            {
                "options": vars(options),
                "output_rank": executor.output_rank,
                "ranks": identities,
            },
            indent=2,
            default=str,
        )
    )
    tokenizer = AutoTokenizer.from_pretrained(options.model, local_files_only=True)
    common = tokenizer.encode("Read the following facts carefully. " * 20)[:64]
    prompts = {
        "A": common + tokenizer.encode("Alice visited the garden. " * 60)[:192],
        "B": common + tokenizer.encode("Bob stayed at the office. " * 40)[:64],
        "C": common + tokenizer.encode("Carol crossed the river. " * 60)[:128],
    }
    ids, outputs, events = {}, {}, []
    original_schedule, original_update = (
        scheduler.schedule,
        scheduler.update_from_output,
    )
    step = 0
    inflight, early_errors = {}, []
    stop = threading.Event()

    def watch():
        marker = run / f"rank-{executor.output_rank}" / "local-done-1"
        delayed_done = run / f"rank-{non_output}" / "local-done-1"
        while not stop.wait(0.001):
            if marker.exists() and not delayed_done.exists() and inflight:
                try:
                    assert pool.get_num_free_blocks() == inflight["free"]
                    assert manager.get_block_ids(ids["A"])[0] == inflight["row"]
                    inflight["observed"] = True
                except BaseException as error:
                    early_errors.append(repr(error))
                    return

    def schedule(*args_, **kwargs):
        q0 = step == 16 and args.lifecycle == "q0"
        if q0:
            scheduler.running.remove(scheduler.requests[ids["A"]])
        output = (
            SchedulerOutput.make_empty()
            if step == 16 and args.lifecycle == "maintenance"
            else original_schedule(*args_, **kwargs)
        )
        if q0:
            scheduler.running.insert(0, scheduler.requests[ids["A"]])
            assert output.total_num_scheduled_tokens > 0
        if step in (16, 32):
            request = scheduler.requests[ids["A"]]
            keep = (
                (160 if step == 16 else 128)
                if args.prefix
                else (128 if step == 16 else 64)
            )
            if args.path == "token":
                keep -= 1 if step == 16 else 2
            if args.demand_pages is not None:
                source = request.effective_kv_len + output.num_scheduled_tokens.get(
                    request.request_id, 0
                )
                keep = min(
                    source - 1,
                    16
                    * (
                        len(manager.get_block_ids(request.request_id)[0])
                        - args.demand_pages
                    ),
                )
            output.reclaim_request = scheduler.make_reclaim_request(
                request, output, keep
            )
            intent = output.reclaim_request
            if (
                args.demand_pages is None
                and args.path == "token"
                and args.policy == "sink_recent"
            ):
                _, recent = sink_recent_ranges(
                    intent.source_effective_kv_len,
                    keep,
                    scheduler.reclamation_config.sink_tokens,
                    intent.protected_prefix_tokens,
                )
                if recent % 16 == 0:
                    output.reclaim_request = scheduler.make_reclaim_request(
                        request, output, keep - 1
                    )
            if step == 16 and args.lifecycle != "repeat":
                assert ids["A"] not in output.num_scheduled_tokens
            assert output.reclaim_request.protected_prefix_tokens == (
                64 if args.prefix else 0
            )
            inflight.clear()
            inflight.update(
                free=pool.get_num_free_blocks(),
                row=manager.get_block_ids(ids["A"])[0].copy(),
                observed=False,
            )
        if step % 2:
            output.num_scheduled_tokens = dict(
                reversed(list(output.num_scheduled_tokens.items()))
            )
        for new in output.scheduled_new_reqs:
            assert new.num_computed_tokens == (
                64 if args.prefix and new.req_id != ids["A"] else 0
            )
        return output

    def update(output, result):
        intent = output.reclaim_request
        if intent:
            markers = [
                run / f"rank-{rank}" / f"local-done-{len(events) + 1}"
                for rank in range(2)
            ]
            # The device ACK gates release; EventProbe only observes local completion.
            deadline = time.monotonic() + 2
            while not all(p.exists() for p in markers) and time.monotonic() < deadline:
                time.sleep(0.001)
            assert all(p.exists() for p in markers)
            if args.delay and not events:
                assert inflight["observed"] and not early_errors
            row = result.reclaim_result.retained_block_ids
            free = pool.get_num_free_blocks()
        answer = original_update(output, result)
        if intent:
            released = sorted(set(intent.source_block_ids) - set(row))
            assert pool.get_num_free_blocks() == free + len(released)
            events.append(
                {
                    "step": step,
                    "q": output.num_scheduled_tokens.get(intent.request_id, 0),
                    "batch_tokens": output.total_num_scheduled_tokens,
                    "intent": vars(intent),
                    "row": row,
                    "released": released,
                    "local_done_ns": [int(p.read_text()) for p in markers],
                    "ack_ns": time.monotonic_ns(),
                    "early_free_observed": inflight["observed"],
                }
            )
        return answer

    scheduler.schedule, scheduler.update_from_output = schedule, update
    observer = threading.Thread(target=watch, daemon=True)
    observer.start()
    last_checks = None
    try:
        for step in range(120):
            external = {0: "A", 2: "B", 17: "C"}.get(step)
            if external:
                params = SamplingParams(
                    temperature=0,
                    seed=0,
                    ignore_eos=True,
                    max_tokens={"A": 64, "B": 40, "C": 20}[external],
                )
                ids[external] = engine.add_request(
                    external, {"prompt_token_ids": prompts[external]}, params
                )
            try:
                results = engine.step()
            except Exception as error:
                if not args.fail:
                    raise
                assert step == 16 and not events
                assert manager.get_block_ids(ids["A"])[0] == inflight["row"]
                assert pool.get_num_free_blocks() == inflight["free"]
                assert (run / f"rank-{non_output}" / "injected-failure").exists()
                deadline = time.monotonic() + 15
                while any(p.is_alive() for p in worker_processes):
                    assert time.monotonic() < deadline, "TP group did not stop"
                    time.sleep(0.01)
                assert executor.is_failed
                (run / "evidence.json").write_text(
                    json.dumps(
                        {
                            "status": "PASS",
                            "expected_fail_stop": repr(error),
                            "step": step,
                            "canonical_commits": 0,
                            "free_unchanged": True,
                            "executor_failed": executor.is_failed,
                            "worker_exit_codes": [p.exitcode for p in worker_processes],
                            "args": vars(args),
                        },
                        indent=2,
                    )
                )
                return
            assert not early_errors, early_errors
            for output in results:
                outputs[output.request_id] = {
                    "tokens": list(output.outputs[0].token_ids),
                    "finished": output.finished,
                }
            last_checks = executor.collective_rpc(
                "p1_check", args=(list(scheduler.requests),)
            )
            if step in (16, 32):
                left, right = [c["reclaim"] for c in last_checks]
                assert (
                    left["keep"] == right["keep"]
                    and left["row"] == right["row"]
                    and left["E"] == right["E"]
                )
                assert left["path"] == right["path"]
                if args.demand_pages is None:
                    assert left["path"] == args.path
                else:
                    assert len(events[-1]["released"]) == args.demand_pages
                events[-1]["rank_results"] = [left, right]
                executor.collective_rpc("p1_poison", args=(events[-1]["released"],))
            if step >= 17 and not engine.has_unfinished_requests():
                break
        assert not args.fail and len(events) == 2
        assert all(item["finished"] for item in outputs.values())
        assert pool.get_num_free_blocks() == 32
        assert all(c["rewritten"] for c in last_checks)
        if args.mode == "full":
            assert all(
                c["counts"]["FULL"] > 0 and c["counts"]["graph_eager"] > 0
                for c in last_checks
            )
        (run / "evidence.json").write_text(
            json.dumps(
                {
                    "status": "PASS",
                    "args": vars(args),
                    "events": events,
                    "ranks": last_checks,
                    "outputs": outputs,
                    "free_final": pool.get_num_free_blocks(),
                },
                indent=2,
            )
        )
        print(json.dumps({"status": "PASS", "ranks": last_checks}), flush=True)
    finally:
        stop.set()
        observer.join(timeout=2)
        executor.shutdown()


if __name__ == "__main__":
    main()
