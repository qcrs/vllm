# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real TP2 workers, delayed non-output rank, release/reuse and combinations."""

import argparse
import json
import os
import shutil
import threading
import time
from dataclasses import asdict
from pathlib import Path

from transformers import AutoTokenizer

from vllm import EngineArgs, LLMEngine, SamplingParams
from vllm.config import CompilationConfig, CompilationMode
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_reclamation import (
    ReclaimAllocation,
    ReclaimCandidate,
    ReclaimRetry,
    protected_reclaim_floor,
    protected_reclaim_plan,
    retry_reclaim_candidates,
    sink_recent_ranges,
)


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
    parser.add_argument("--tp", type=int, choices=[1, 2], default=2)
    parser.add_argument("--policy-updates", action="store_true")
    parser.add_argument("--worker-cpus", nargs="+", type=int)
    parser.add_argument("--demand-pages", type=int)
    parser.add_argument("--retry-tail", action="store_true")
    parser.add_argument("--protected", action="store_true")
    parser.add_argument("--victims", type=int, choices=[1, 2], default=1)
    parser.add_argument("--decode-reclaim-rounds", type=int, choices=[1, 2], default=1)
    args = parser.parse_args()
    two_round = args.decode_reclaim_rounds == 2
    if two_round and (not args.protected or args.prefix or args.policy_updates):
        parser.error("two rounds require protected mode without prefix/policy updates")
    pool_blocks = (28 if args.victims == 1 else 43) if two_round else 33
    if args.retry_tail and args.demand_pages is None:
        parser.error("--retry-tail requires --demand-pages")
    if args.worker_cpus:
        assert len(args.worker_cpus) == args.tp
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
        tensor_parallel_size=args.tp,
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
        max_model_len=448 if two_round else 512,
        max_num_seqs=4,
        max_num_batched_tokens=256,
        kv_cache_memory_bytes=128 * 2**20,
        num_gpu_blocks_override=pool_blocks,
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
                "pressure": args.victims == 2 or two_round,
                "budget_mode": "protected" if args.protected else "fixed",
                "decode_reclaim_rounds": args.decode_reclaim_rounds,
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
    non_output = 1 - executor.output_rank if args.tp == 2 else 0
    worker_processes = [handle.proc for handle in executor.workers]
    identities = executor.collective_rpc(
        "p1_install",
        args=(
            {
                "delay_rank": non_output if args.delay else -1,
                "delay_s": 3,
                "delay_number": 2 if args.victims == 2 else 1,
                "fail_rank": non_output if args.fail else -1,
                "policy_updates": args.policy_updates,
                "prepare_delay_rank": non_output,
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
        "D": common + tokenizer.encode("David stayed at home. " * 40)[:64],
    }
    if args.victims == 2:
        assert args.protected and not args.prefix and not args.policy_updates
        prompts["C"] = prompts["C"][:160]
    if two_round:
        prompts["C"] = prompts["A"].copy()
        prompts["D"] = prompts["D"][:16]
        if args.victims == 2:
            prompts["B"] = prompts["A"].copy()
    ids, outputs, events = {}, {}, []
    original_schedule, original_update = (
        scheduler.schedule,
        scheduler.update_from_output,
    )
    step = 0
    inflight, early_errors = {}, []
    stop = threading.Event()
    policy_events, cold_steps, schedule_records = [], [], []
    updating = {}
    reclaim_steps = (
        {16: "A", 24: "B", 32: "C"} if args.policy_updates else {16: "A", 32: "A"}
    )

    if args.victims == 2 or two_round:
        reclaim_steps = {}

    def watch():
        number = 2 if args.victims == 2 else 1
        marker = run / f"rank-{executor.output_rank}" / f"local-done-{number}"
        delayed_done = run / f"rank-{non_output}" / f"local-done-{number}"
        while not stop.wait(0.001):
            if updating:
                version = updating["version"]
                started = run / f"rank-{non_output}" / f"prepare-start-{version}"
                done = run / f"rank-{non_output}" / f"prepare-end-{version}"
                if started.exists() and not done.exists():
                    if core.default_reclamation_version != updating["old"]:
                        early_errors.append("default changed before all-rank prepare")
                        return
                    updating["observed"] = True
            if marker.exists() and not delayed_done.exists() and inflight:
                try:
                    assert pool.get_num_free_blocks() == inflight["free"]
                    assert all(
                        manager.get_block_ids(rid)[0] == row
                        for rid, row in inflight["rows"].items()
                    )
                    inflight["observed"] = True
                except BaseException as error:
                    early_errors.append(repr(error))
                    return

    def schedule(*args_, **kwargs):
        q0 = step == 16 and args.lifecycle == "q0" and args.victims == 1
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
        if step in reclaim_steps:
            victim = reclaim_steps[step]
            request = scheduler.requests[ids[victim]]
            keep = (
                (160 if step == 16 else 128)
                if args.prefix
                else (128 if step == 16 else 64)
            )
            if args.path == "token":
                keep -= 1 if step == 16 else 2
            if args.policy_updates:
                keep = {"A": 127, "B": 63, "C": 95}[victim]
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
                if args.retry_tail:
                    row = tuple(manager.get_block_ids(request.request_id)[0])
                    candidate = ReclaimCandidate(
                        request.request_id,
                        32,
                        len(row) - 2,
                        0,
                        False,
                        0,
                        source_effective_kv_len=source,
                        source_block_ids=row,
                        policy_binding=request.policy_binding,
                    )
                    retry = ReclaimRetry(
                        0,
                        (
                            ReclaimAllocation(request.request_id, 1, 0),
                            ReclaimAllocation(
                                "requester", 16 * args.demand_pages, args.demand_pages
                            ),
                        ),
                    )
                    keep = retry_reclaim_candidates(
                        [candidate],
                        retry,
                        0,
                        scheduler.reclamation_config,
                        16,
                        1,
                    )[0].keep_tokens
            if args.protected:
                assert args.demand_pages is not None
                source = request.effective_kv_len + output.num_scheduled_tokens.get(
                    request.request_id, 0
                )
                logical = request.num_computed_tokens
                config = request.policy_binding.config
                floor = protected_reclaim_floor(source, logical, 0, config)
                row = tuple(manager.get_block_ids(request.request_id)[0])
                candidate = ReclaimCandidate(
                    request.request_id,
                    floor,
                    len(row) - (floor + 15) // 16,
                    0,
                    False,
                    request.arrival_time,
                    0,
                    source,
                    row,
                    logical,
                    policy_binding=request.policy_binding,
                )
                plan = protected_reclaim_plan(
                    [candidate], args.demand_pages, config, 16, 1
                )
                assert len(plan) == 1
                keep = plan[0].keep_tokens
            output.reclaim_requests = (
                scheduler.make_reclaim_request(request, output, keep),
            )
            intent = output.reclaim_requests[0] if output.reclaim_requests else None
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
                    output.reclaim_requests = (
                        scheduler.make_reclaim_request(request, output, keep - 1),
                    )
            if step == 16 and args.lifecycle != "repeat":
                assert ids["A"] not in output.num_scheduled_tokens
            assert output.reclaim_requests[0].protected_prefix_tokens == (
                64 if args.prefix else 0
            )
        if output.reclaim_requests:
            if two_round:
                for intent in output.reclaim_requests:
                    request = scheduler.requests[intent.request_id]
                    assert (
                        5 * (intent.source_effective_kv_len - intent.target_keep_tokens)
                        <= intent.source_effective_kv_len
                    )
                    assert 2 * intent.target_keep_tokens >= request.num_computed_tokens
                window = scheduler._decode_reclaim_window
                if window is not None and window.rounds == 2:
                    first_window = next(e for e in events if e["window"])
                    assert all(
                        scheduler.requests[i.request_id].num_computed_tokens
                        - output.num_scheduled_tokens.get(i.request_id, 0)
                        > first_window["logical_tokens"][i.request_id]
                        for i in output.reclaim_requests
                    )
            inflight.clear()
            inflight.update(
                free=pool.get_num_free_blocks(),
                rows={
                    i.request_id: manager.get_block_ids(i.request_id)[0].copy()
                    for i in output.reclaim_requests
                },
                observed=False,
            )
        schedule_records.append(
            dict(
                step=step,
                scheduled=dict(output.num_scheduled_tokens),
                window=asdict(scheduler._decode_reclaim_window)
                if scheduler._decode_reclaim_window
                else None,
                preempted=list(output.preempted_req_ids or ()),
            )
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
        intents = output.reclaim_requests
        if intents:
            first = sum(len(e["intents"]) for e in events) + 1
            markers = [
                run / f"rank-{rank}" / f"local-done-{number}"
                for rank in range(args.tp)
                for number in range(first, first + len(intents))
            ]
            deadline = time.monotonic() + 2
            while not all(p.exists() for p in markers) and time.monotonic() < deadline:
                time.sleep(0.001)
            assert all(p.exists() for p in markers)
            if args.delay and not events:
                assert inflight["observed"] and not early_errors
            rows = [r.retained_block_ids for r in result.reclaim_results]
            free = pool.get_num_free_blocks()
        answer = original_update(output, result)
        if intents:
            released = sorted(
                {
                    page
                    for intent, row in zip(intents, rows)
                    for page in set(intent.source_block_ids) - set(row)
                }
            )
            assert pool.get_num_free_blocks() == free + len(released)
            events.append(
                {
                    "step": step,
                    "q": [
                        output.num_scheduled_tokens.get(i.request_id, 0)
                        for i in intents
                    ],
                    "batch_tokens": output.total_num_scheduled_tokens,
                    "intents": [asdict(i) for i in intents],
                    "logical_tokens": {
                        i.request_id: scheduler.requests[
                            i.request_id
                        ].num_computed_tokens
                        for i in intents
                    },
                    "rows": rows,
                    "released": released,
                    "local_done_ns": [int(p.read_text()) for p in markers],
                    "ack_ns": time.monotonic_ns(),
                    "canonical_unchanged_while_waiting": inflight["observed"],
                    "window": asdict(scheduler._decode_reclaim_window)
                    if scheduler._decode_reclaim_window
                    else None,
                }
            )
        return answer

    scheduler.schedule, scheduler.update_from_output = schedule, update
    observer = threading.Thread(target=watch, daemon=True)
    observer.start()
    last_checks = None
    try:
        for step in range(120):
            external = (
                {0: "A", 2: "B", 17: "C", 34: "D"}
                if args.policy_updates
                else {0: "A", 2: "B", (16 if args.victims == 2 else 17): "C"}
            ).get(step)
            if two_round:
                external = (
                    {0: "A", 1: "C", 2: "D"}
                    if args.victims == 1
                    else {0: "A", 1: "B", 3: "C", 4: "D"}
                ).get(step)
            if args.policy_updates and external in ("B", "C", "D"):
                policy = {"B": "random", "C": "triattention", "D": "full"}[external]
                updating.update(
                    version=core.next_reclamation_version,
                    old=core.default_reclamation_version,
                    observed=False,
                )
                changed = core.update_reclamation_policy(
                    {
                        "policy": policy,
                        "stats_path": os.environ.get("P1_STATS"),
                    }
                )
                published_ns = time.monotonic_ns()
                version = changed["binding"]["version"]
                assert all(
                    int((run / f"rank-{rank}" / f"prepare-end-{version}").read_text())
                    <= published_ns
                    for rank in range(args.tp)
                )
                if version == 1:
                    assert updating["observed"]
                changed.update(
                    published_ns=published_ns,
                    step=step,
                    observed_old=updating["observed"],
                )
                policy_events.append(changed)
                updating.clear()
            if external:
                params = SamplingParams(
                    temperature=0,
                    seed=0,
                    ignore_eos=True,
                    max_tokens={"A": 64, "B": 40, "C": 20, "D": 12}[external],
                )
                ids[external] = engine.add_request(
                    external, {"prompt_token_ids": prompts[external]}, params
                )
                if args.policy_updates:
                    binding = scheduler.requests[ids[external]].policy_binding
                    assert binding.version == {"A": 0, "B": 1, "C": 2, "D": 3}[external]
            try:
                step_started = time.perf_counter()
                results = engine.step()
                if external:
                    cold_steps.append(
                        dict(
                            request=external,
                            step=step,
                            seconds=time.perf_counter() - step_started,
                        )
                    )
            except Exception as error:
                if not args.fail:
                    raise
                assert step == 16 and not events
                assert all(
                    manager.get_block_ids(rid)[0] == row
                    for rid, row in inflight["rows"].items()
                )
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
            if args.policy_updates:
                assert all(
                    c["policy_versions"] == sorted(core.reclamation_policies)
                    for c in last_checks
                )
                for request in scheduler.requests.values():
                    assert request.policy_binding.version in core.reclamation_policies
            if events and events[-1]["step"] == step:
                left = last_checks[0]["reclaims"]
                right = last_checks[-1]["reclaims"]
                assert len(left) == len(right) == len(events[-1]["intents"])
                for left_rank, right_rank in zip(left, right):
                    assert all(
                        left_rank[k] == right_rank[k]
                        for k in ("keep", "row", "E", "path")
                    )
                    if (
                        args.demand_pages is None
                        and args.victims == 1
                        and not two_round
                    ):
                        assert left_rank["path"] == args.path
                if args.demand_pages is not None:
                    assert len(events[-1]["released"]) == args.demand_pages
                events[-1]["rank_results"] = [left, right]
                executor.collective_rpc("p1_poison", args=(events[-1]["released"],))
            if step >= 17 and not engine.has_unfinished_requests():
                break
        assert not args.fail
        if two_round:
            window_events = [e for e in events if e["window"]]
            assert len(window_events) == 2, events
            assert [e["window"]["rounds"] for e in window_events] == [1, 2]
            assert all(len(e["intents"]) <= 2 for e in events)
            assert scheduler._decode_reclaim_window is None
            assert not any(e["preempted"] for e in schedule_records)
            c_step = next(
                e["step"] for e in schedule_records if ids["C"] in e["scheduled"]
            )
            assert c_step < 63 and window_events[1]["step"] < c_step
            assert not any(
                ids["D"] in e["scheduled"]
                for e in schedule_records
                if e["step"] < c_step
            )
        if args.victims == 2:
            assert any(len(e["intents"]) == 2 for e in events)
        elif not two_round:
            assert len(events) == (3 if args.policy_updates else 2)
        assert all(item["finished"] for item in outputs.values())
        assert pool.get_num_free_blocks() == pool_blocks - 1
        assert all(c["rewritten"] for c in last_checks)
        if args.policy_updates:
            assert sorted(core.reclamation_policies) == [3]
            assert all(c["policy_versions"] == [3] for c in last_checks)
            for rank in range(args.tp):
                for version in range(3):
                    retired_ns = int(
                        (run / f"rank-{rank}" / f"retire-{version}").read_text()
                    )
                    completed_ns = int(
                        (run / f"rank-{rank}" / f"local-done-{version + 1}").read_text()
                    )
                    assert retired_ns > completed_ns
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
                    "policy_events": policy_events,
                    "cold_steps": cold_steps,
                    "schedule": schedule_records,
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
