# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Join one TP TTFT diagnostic by HTTP ID, step, process, and CUDA correlation."""

import argparse
import bisect
import json
import sqlite3
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import regex as re


def read_events(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def candidate_summary(run, output):
    """Audit host-only decisions without requiring a profiler database."""
    events = read_events(run / "kv-trace.jsonl")
    decisions = []
    for event in events:
        if event["event"] != "reclaim_candidates":
            continue
        mode = event["budget"].get("budget_mode", "fixed")
        active = event[
            {
                "fixed": "candidates",
                "demand": "demand_candidates",
                "retry": "retry_candidates",
                "retry_pass": "retry_candidates",
            }[mode]
        ]
        selected = next(
            (c for c in active if c["request_id"] == event["selected"]),
            None,
        )
        row = dict(event)
        row["hard_covered"] = any(
            c["released_blocks"] >= event["shortfall"] for c in event["candidates"]
        )
        if selected is not None:
            forecast = event.get("retry_forecast")
            if forecast is not None:
                for name, allocations in (
                    ("retry", forecast["allocations"]),
                    (
                        "pass",
                        forecast["allocations"]
                        + forecast.get("following_allocations", []),
                    ),
                ):
                    victim = next(
                        (
                            a
                            for a in allocations
                            if a["request_id"] == selected["request_id"]
                        ),
                        None,
                    )
                    other = sum(a["required_blocks"] for a in allocations) - (
                        victim["required_blocks"] if victim else 0
                    )
                    keep = selected["keep_tokens"]
                    next_tokens = victim["num_tokens"] if victim else 0
                    growth = (keep + next_tokens + 15) // 16 - (keep + 15) // 16
                    row[f"predicted_{name}_slack"] = (
                        forecast["free_blocks"]
                        + selected["released_blocks"]
                        - other
                        - growth
                    )
            commit = next(
                (
                    e
                    for e in events
                    if e["event"] == "reclaim_commit"
                    and e["request_id"] == event["selected"]
                    and e["time_ns"] > event["time_ns"]
                ),
                None,
            )
            assert commit is not None, event
            assert commit["released_blocks"] == selected["released_blocks"]
            attempts = [
                e
                for e in events
                if e.get("request_id") == event["requester"]
                and e["time_ns"] > commit["time_ns"]
                and e["event"]
                in ("allocation_done", "allocation_miss", "native_preempt")
            ]
            row.update(commit=commit, first_retry=attempts[0] if attempts else None)
            row["first_retry_success"] = bool(
                attempts and attempts[0]["event"] == "allocation_done"
            )
            row["subsequent_admission"] = next(
                (e for e in attempts if e["event"] == "allocation_done"), None
            )
            row["preemptions_before_admission"] = [
                e
                for e in events
                if e["event"] == "native_preempt"
                and commit["time_ns"]
                < e["time_ns"]
                < (row["subsequent_admission"] or {"time_ns": float("inf")})["time_ns"]
            ]
        decisions.append(row)
    result = dict(
        decisions=decisions,
        count=len(decisions),
        insufficient=sum(not d["hard_covered"] for d in decisions),
        first_retry_success=sum(d.get("first_retry_success", False) for d in decisions),
        predicted_retry_covered=sum(
            d.get("predicted_retry_slack", -1) >= 0 for d in decisions
        ),
        predicted_pass_covered=sum(
            d.get("predicted_pass_slack", -1) >= 0 for d in decisions
        ),
        timing="host monotonic; no GPU incremental cost claim",
    )
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "decisions"}))


def union_ns(intervals):
    end, total = 0, 0
    for start, finish in sorted(intervals):
        total += max(0, finish - max(start, end))
        end = max(end, finish)
    return total


def overlap_ns(start, end, intervals):
    return union_ns(
        (max(start, a), min(end, b)) for a, b in intervals if a < end and b > start
    )


def controller_summary(run, analysis, output):
    schedules = [
        e
        for e in read_events(run / "timeline-scheduler.jsonl")
        if e["event"] == "schedule"
    ]
    by_step = {e["step"]: e for e in schedules}
    events = read_events(run / "kv-trace.jsonl")
    rows = []
    for reclaim in analysis["reclaims"]:
        scheduled = by_step[reclaim["step"]]
        victim = reclaim["intent"]["request_id"]
        selections = [
            e
            for e in events
            if e["event"] == "reclaim_select"
            and scheduled["start_ns"] <= e["time_ns"] <= scheduled["end_ns"]
        ]
        assert len(selections) == 1 and selections[0]["request_id"] == victim
        selected = selections[0]
        misses = [
            e
            for e in events
            if e["event"] == "allocation_miss"
            and scheduled["start_ns"] <= e["time_ns"] < selected["time_ns"]
        ]
        miss = misses[-1]
        assert miss["required"] - miss["available"] == selected["shortfall"]
        commit = reclaim["commit_ns"]
        requester = miss["request_id"]
        attempts = [
            e
            for e in events
            if e.get("request_id") == requester
            and e["time_ns"] > commit
            and e["event"] in ("allocation_done", "allocation_miss", "native_preempt")
        ]
        first = attempts[0]
        done = next(e for e in attempts if e["event"] == "allocation_done")
        outcome_events = [
            e
            for e in events
            if commit < e["time_ns"] <= done["time_ns"]
            and (
                e["event"] in ("native_preempt", "reclaim_select", "reclaim_commit")
                or (
                    e.get("request_id") == requester
                    and e["event"] in ("allocation_miss", "allocation_done")
                )
            )
        ]
        native_preemptions = [
            e for e in outcome_events if e["event"] == "native_preempt"
        ]
        retry_step = next(
            s for s in schedules if s["start_ns"] <= first["time_ns"] <= s["end_ns"]
        )
        victim_next = next(
            s for s in schedules if s["start_ns"] > commit and victim in s["requests"]
        )
        gpu_end = max(r["last_gpu_end_ns"] for r in reclaim["ranks"].values())
        gpu_begin = min(r["first_gpu_start_ns"] for r in reclaim["ranks"].values())
        before_done = [e for e in attempts if e["time_ns"] <= done["time_ns"]]
        rows.append(
            dict(
                step=reclaim["step"],
                requester=requester,
                victim=victim,
                miss_status=miss["request_status"],
                miss=miss,
                selection=selected,
                source_e=reclaim["intent"]["source_e"],
                keep=reclaim["intent"]["target_e"],
                reserve_blocks=selected["goal_blocks"] - selected["shortfall"],
                released_blocks=reclaim["released_blocks"],
                released_covers_goal=reclaim["released_blocks"]
                >= selected["goal_blocks"],
                useful_released_blocks=min(
                    reclaim["released_blocks"], selected["goal_blocks"]
                ),
                retry_step=retry_step["step"],
                first_retry=first,
                first_retry_success=first["event"] == "allocation_done",
                allocation_success=done,
                outcome_events=outcome_events,
                retry_misses_before_success=sum(
                    e["event"] == "allocation_miss" for e in before_done
                ),
                preemptions_before_success=len(native_preemptions),
                victim_next_step=victim_next["step"],
                timing_ms=dict(
                    select_to_commit_inclusive=(commit - selected["time_ns"]) / 1e6,
                    select_to_first_gpu_inclusive=(gpu_begin - selected["time_ns"])
                    / 1e6,
                    reclaim_gpu_span=(gpu_end - gpu_begin) / 1e6,
                    last_gpu_to_commit=(commit - gpu_end) / 1e6,
                    commit_to_first_retry=(first["time_ns"] - commit) / 1e6,
                    commit_to_success=(done["time_ns"] - commit) / 1e6,
                    commit_to_victim_schedule=(victim_next["start_ns"] - commit) / 1e6,
                ),
                ranks=reclaim["ranks"],
            )
        )

    def timing_ranges(selected_rows):
        return {
            name: dict(
                min=min(values), max=max(values), median=statistics.median(values)
            )
            for name in selected_rows[0]["timing_ms"]
            for values in [[r["timing_ms"][name] for r in selected_rows]]
        }

    cost_groups = {}
    for kind, selected_rows in (
        ("zero_copy", [r for r in rows if r["selection"]["zero_copy"]]),
        ("token_copy", [r for r in rows if not r["selection"]["zero_copy"]]),
    ):
        if not selected_rows:
            continue
        cost_groups[kind] = dict(
            intents=len(selected_rows),
            source_e_range=[
                min(r["source_e"] for r in selected_rows),
                max(r["source_e"] for r in selected_rows),
            ],
            keep_range=[
                min(r["keep"] for r in selected_rows),
                max(r["keep"] for r in selected_rows),
            ],
            first_retry_success=sum(r["first_retry_success"] for r in selected_rows),
            timing_ranges_ms=timing_ranges(selected_rows),
        )
    result = dict(
        source=str(run),
        reclaims=rows,
        counts=dict(
            intents=len(rows),
            first_retry_success=sum(r["first_retry_success"] for r in rows),
            allocation_success_after_preemption=sum(
                r["preemptions_before_success"] > 0 for r in rows
            ),
            released_covers_goal=sum(r["released_covers_goal"] for r in rows),
            miss_status=dict(Counter(r["miss_status"] for r in rows)),
            zero_copy=sum(r["selection"]["zero_copy"] for r in rows),
            requester_is_victim=sum(r["requester"] == r["victim"] for r in rows),
            released_blocks=sum(r["released_blocks"] for r in rows),
            candidates=dict(Counter(r["selection"]["candidates"] for r in rows)),
        ),
        timing_ranges_ms=timing_ranges(rows),
        observed_cost_groups=cost_groups,
        clock_tolerance_ns=analysis["clock_tolerance_ns"],
        limits=[
            "SinkRecent diagnostic with profiler perturbation and six preemptions",
            "select-to-commit includes normal forward; not incremental reclaim cost",
            "GPU span contains launch gaps and TP waits, not just kernel busy time",
            "unselected candidate costs and matched preemption counterfactual "
            "are absent",
            "allocation success after native preemption is a new recompute "
            "admission, not reclaim retry success",
            "C1/C2 NOT_RUN; no supported time-cost ordering or action gate",
        ],
    )
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ("reclaims",)}, indent=2
        )
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--controller-output", type=Path)
    parser.add_argument("--candidate-output", type=Path)
    parser.add_argument("--analysis", type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    if args.candidate_output:
        candidate_summary(run, args.candidate_output)
        return
    if args.controller_output:
        if args.analysis is None:
            parser.error("--controller-output requires --analysis")
        controller_summary(
            run, json.loads(args.analysis.read_text()), args.controller_output
        )
        return
    scheduler = read_events(run / "timeline-scheduler.jsonl")
    clients = read_events(run / "timeline-client.jsonl")
    kv = read_events(run / "kv-trace.jsonl")
    window = json.loads((run / "serving.json").read_text())[0]
    begin, end = window["start_monotonic_ns"], window["end_monotonic_ns"]
    schedules = {e["step"]: e for e in scheduler if e["event"] == "schedule"}
    completes = {e["step"]: e for e in scheduler if e["event"] == "complete"}
    assert len(schedules) == len(completes)
    assert schedules.keys() == completes.keys()
    timed = {k: s for k, s in schedules.items() if begin <= s["start_ns"] < end}
    worker = {rank: read_events(run / f"timeline-rank-{rank}.jsonl") for rank in (0, 1)}
    host = {}
    for rank, events in worker.items():
        for event in events:
            key = (rank, event["step"], event["event"])
            assert key not in host, key
            host[key] = event
        assert {e["step"] for e in events} == schedules.keys()

    db = sqlite3.connect(f"file:{run / 'trace.sqlite'}?mode=ro", uri=True)
    clocks = defaultdict(list)
    for stamp, tid, label in db.execute(
        "SELECT start,globalTid,text FROM NVTX_EVENTS "
        "WHERE text LIKE 'p1-ttft-clock|%' ORDER BY start"
    ):
        _, pid, mono = label.split("|")
        clocks[tid & -16777216].append((int(pid), stamp - int(mono)))
    offsets, clock_check = {}, []
    for pid, marks in clocks.items():
        assert len(marks) == 2 and marks[0][0] == marks[1][0]
        # Use the final offset and retain the first/last difference as uncertainty.
        offsets[pid] = marks[-1][1]
        clock_check.append(
            dict(
                pid=marks[0][0],
                global_pid=pid,
                final_offset_ns=marks[-1][1],
                first_to_final_offset_delta_ns=marks[-1][1] - marks[0][1],
            )
        )
    tolerance = max(abs(c["first_to_final_offset_delta_ns"]) for c in clock_check)

    phases = defaultdict(list)
    worker_pids = {}
    for start, finish, tid, label in db.execute(
        "SELECT n.start,n.end,n.globalTid,coalesce(n.text,s.value) "
        "FROM NVTX_EVENTS n LEFT JOIN StringIds s ON n.textId=s.id "
        "WHERE coalesce(n.text,s.value) LIKE 'p1-ttft|%'"
    ):
        _, rank, step, name = label.split("|")
        rank, step = int(rank), int(step)
        pid = tid & -16777216
        worker_pids[pid] = rank
        if step not in timed:
            continue
        assert (rank, step, name) in host
        phases[tid].append((start, finish, rank, step, name))
    for ranges in phases.values():
        ranges.sort()
    starts = {tid: [r[0] for r in ranges] for tid, ranges in phases.items()}

    launches = {}
    graph_launches = set()
    for a, b, tid, correlation, name in db.execute(
        "SELECT r.start,r.end,r.globalTid,r.correlationId,s.value "
        "FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON r.nameId=s.id"
    ):
        if tid not in phases:
            continue
        index = bisect.bisect_right(starts[tid], a) - 1
        for start, finish, rank, step, phase in reversed(phases[tid][: index + 1]):
            if start <= a and b <= finish:
                key = (tid & -16777216, correlation)
                assert key not in launches
                launches[key] = (rank, step, phase, name)
                if name.startswith("cudaGraphLaunch"):
                    graph_launches.add(key)
                break
            # Do not scan previous steps once outside the enclosing execute range.
            if finish < a and phase == "GPUModelRunner.execute_model":
                break

    activities = defaultdict(list)
    seen_graphs, unjoined = set(), Counter()
    columns = {
        "KERNEL": "s.value",
        "GRAPH_TRACE": "'opaque_cuda_graph'",
        "MEMCPY": "'memcpy'",
        "MEMSET": "'memset'",
    }
    for kind, name in columns.items():
        query = (
            f"SELECT k.start,k.end,k.globalPid,k.correlationId,{name} "
            f"FROM CUPTI_ACTIVITY_KIND_{kind} k "
        )
        if kind == "KERNEL":
            query += "JOIN StringIds s ON k.shortName=s.id"
        for a, b, pid, correlation, label in db.execute(query):
            key = (pid, correlation)
            if key not in launches:
                if pid in worker_pids and begin <= a - offsets[pid] < end:
                    unjoined[kind] += 1
                continue
            rank, step, phase, _ = launches[key]
            if kind == "GRAPH_TRACE":
                category = "opaque_graph"
                seen_graphs.add(key)
            elif label == "_compact_stable":
                category = "compactor"
            elif "nccl" in label.lower() or "cross_device_reduce" in label:
                category = (
                    "reclaim_completion_collective"
                    if phase == "KVReclamationWorker.reclaim"
                    else "model_collective"
                )
            elif kind in ("MEMCPY", "MEMSET"):
                category = kind.lower()
            else:
                category = "other_kernel"
            activities[rank, step].append(
                dict(
                    start_ns=a - offsets[pid],
                    end_ns=b - offsets[pid],
                    phase=phase,
                    category=category,
                    name=label,
                )
            )
    db.close()
    assert not graph_launches - seen_graphs, "CUDA graph spans missing"
    assert not unjoined, dict(unjoined)

    steps = []
    for step, scheduled in timed.items():
        item = dict(
            step=step,
            start_ns=scheduled["start_ns"],
            end_ns=completes[step]["end_ns"],
            q=sum(r["q"] for r in scheduled["requests"].values()),
            prefill=any(
                r["logical_start"] < r["prompt_tokens"]
                for r in scheduled["requests"].values()
            ),
            requests=scheduled["requests"],
            reclaim=scheduled["reclaim"],
            ranks={},
        )
        for rank in (0, 1):
            activity = activities[rank, step]
            groups = defaultdict(list)
            for event in activity:
                groups[event["category"]].append((event["start_ns"], event["end_ns"]))
            item["ranks"][rank] = dict(
                gpu_busy_ms=union_ns((e["start_ns"], e["end_ns"]) for e in activity)
                / 1e6,
                category_ms={k: union_ns(v) / 1e6 for k, v in groups.items()},
                last_gpu_end_ns=max((e["end_ns"] for e in activity), default=None),
                gpu_tail_after_core_start_ms=max(
                    0,
                    max(
                        (e["end_ns"] - completes[step]["start_ns"] for e in activity),
                        default=0,
                    ),
                )
                / 1e6,
                schedule_to_worker_ms=(
                    host[rank, step, "GPUModelRunner.execute_model"]["start_ns"]
                    - scheduled["end_ns"]
                )
                / 1e6,
                host={
                    phase: dict(
                        wall_ms=(e["end_ns"] - e["start_ns"]) / 1e6,
                        thread_ms=e["thread_ns"] / 1e6,
                    )
                    for (r, s, phase), e in host.items()
                    if r == rank and s == step
                },
            )
        steps.append(item)
    step_map = {s["step"]: s for s in steps}

    def occupying_work(a, b):
        groups = defaultdict(list)
        for step in steps:
            kind = (
                "maintenance"
                if step["q"] == 0 and step["reclaim"]
                else "prefill_step"
                if step["prefill"]
                else "decode_or_empty_step"
            )
            groups[kind].append((step["start_ns"], step["end_ns"]))
        return {k: overlap_ns(a, b, v) / 1e6 for k, v in groups.items()}

    requests = []
    raw_requests = json.loads((run / "requests-0.json").read_text())
    for raw in raw_requests:
        rid = raw["request"]
        client = {e["event"]: e for e in clients if e["request"] == rid}
        response_id = client["client_first_token"]["response_id"]
        candidates = [
            e
            for e in scheduler
            if e["event"] == "enqueue"
            and re.fullmatch(
                re.escape(response_id) + r"-0-[0-9a-f]{8}", e["request_id"]
            )
        ]
        assert len(candidates) == 1, (rid, candidates)
        enqueue = candidates[0]
        engine_id = enqueue["request_id"]
        output = next(e for e in completes.values() if engine_id in e["output_counts"])
        chunks = [
            s
            for s in steps
            if engine_id in s["requests"]
            and s["step"] <= output["step"]
            and s["requests"][engine_id]["logical_start"] < raw["prompt_tokens"]
        ]
        assert chunks and chunks[-1]["step"] == output["step"]
        start = client["client_start"]["time_ns"]
        first_token = client["client_first_token"]["time_ns"]
        first_schedule, last_complete = chunks[0]["start_ns"], output["end_ns"]
        step_wall = sum(s["end_ns"] - s["start_ns"] for s in chunks)
        gaps = [(a["end_ns"], b["start_ns"]) for a, b in zip(chunks, chunks[1:])]
        restarts, previous = [], 0
        chunk_details = []
        for chunk in chunks:
            state = chunk["requests"][engine_id]
            if state["logical_start"] < previous:
                restarts.append(chunk["step"])
            previous = state["logical_start"] + state["q"]
            chunk_details.append(
                dict(
                    step=chunk["step"],
                    logical_start=state["logical_start"],
                    q=state["q"],
                    complete_ns=chunk["end_ns"],
                )
            )
        parts = dict(
            ingress=(enqueue["time_ns"] - start) / 1e6,
            initial_queue=(first_schedule - enqueue["time_ns"]) / 1e6,
            own_prefill_steps=step_wall / 1e6,
            between_chunks=(last_complete - first_schedule - step_wall) / 1e6,
            output_delivery=(first_token - last_complete) / 1e6,
        )
        ttft = (first_token - start) / 1e6
        assert abs(sum(parts.values()) - ttft) < 0.001
        assert abs(ttft / 1e3 - raw["ttft_s"]) < 0.005
        requests.append(
            dict(
                request=rid,
                engine_id=engine_id,
                prompt_tokens=raw["prompt_tokens"],
                ttft_ms=ttft,
                parts_ms=parts,
                initial_queue_occupancy_ms=occupying_work(
                    enqueue["time_ns"], first_schedule
                ),
                between_chunk_occupancy_ms={
                    k: sum(occupying_work(a, b).get(k, 0) for a, b in gaps)
                    for k in ("maintenance", "prefill_step", "decode_or_empty_step")
                },
                chunks=chunk_details,
                recompute_starts=restarts,
                prompt_tokens_computed=sum(
                    min(s["q"], raw["prompt_tokens"] - s["logical_start"])
                    for s in chunk_details
                ),
            )
        )

    reclaims = []
    for step, scheduled in timed.items():
        intent = scheduled["reclaim"]
        if intent is None:
            continue
        complete = completes[step]
        commits = [
            e
            for e in kv
            if e["event"] == "reclaim_commit"
            and e["request_id"] == intent["request_id"]
            and complete["start_ns"] <= e["time_ns"] <= complete["end_ns"]
        ]
        assert len(commits) == 1
        commit = commits[0]
        assert commit["source_effective_kv_len"] == intent["source_e"]
        assert commit["effective_kv_len"] == intent["target_e"]
        ranks = {}
        for rank in (0, 1):
            event = host[rank, step, "KVReclamationWorker.reclaim"]
            activity = [
                e
                for e in activities[rank, step]
                if e["phase"]
                in ("KVReclamationWorker.reclaim", "StableKVCompactor.compact")
            ]
            ack = [
                e for e in activity if e["category"] == "reclaim_completion_collective"
            ]
            assert len(ack) == 1, (rank, step, ack)
            gpu_end = max(e["end_ns"] for e in activity)
            assert gpu_end <= commit["time_ns"] + tolerance
            ranks[rank] = dict(
                host_start_ns=event["start_ns"],
                host_return_ns=event["end_ns"],
                first_gpu_start_ns=min(e["start_ns"] for e in activity),
                last_gpu_end_ns=gpu_end,
                gpu_busy_ms=union_ns((e["start_ns"], e["end_ns"]) for e in activity)
                / 1e6,
                completion_collective_ms=(ack[0]["end_ns"] - ack[0]["start_ns"]) / 1e6,
                gpu_end_to_commit_ms=(commit["time_ns"] - gpu_end) / 1e6,
            )
        gate = host[
            0,
            step,
            "AsyncOutput.get_output"
            if step_map[step]["q"]
            else "GPUModelRunner.execute_model",
        ]
        reclaims.append(
            dict(
                step=step,
                intent=intent,
                maintenance=not step_map[step]["q"],
                schedule_end_ns=scheduled["end_ns"],
                completion_gate_return_ns=gate["end_ns"],
                commit_ns=commit["time_ns"],
                released_blocks=commit["released_blocks"],
                ranks=ranks,
            )
        )

    aggregates = []
    for rank in (0, 1):
        for kind in ("prefill", "decode", "maintenance", "empty"):
            selected = [
                s
                for s in steps
                if (
                    "prefill"
                    if s["prefill"]
                    else "decode"
                    if s["q"]
                    else "maintenance"
                    if s["reclaim"]
                    else "empty"
                )
                == kind
            ]
            aggregates.append(
                dict(
                    rank=rank,
                    kind=kind,
                    steps=len(selected),
                    step_wall_ms=sum(s["end_ns"] - s["start_ns"] for s in selected)
                    / 1e6,
                    gpu_busy_ms=sum(s["ranks"][rank]["gpu_busy_ms"] for s in selected),
                    category_ms={
                        category: sum(
                            s["ranks"][rank]["category_ms"].get(category, 0)
                            for s in selected
                        )
                        for category in sorted(
                            {
                                c
                                for s in selected
                                for c in s["ranks"][rank]["category_ms"]
                            }
                        )
                    },
                    host_inclusive_ms={
                        phase: dict(
                            wall=sum(
                                s["ranks"][rank]["host"]
                                .get(phase, {})
                                .get("wall_ms", 0)
                                for s in selected
                            ),
                            thread=sum(
                                s["ranks"][rank]["host"]
                                .get(phase, {})
                                .get("thread_ms", 0)
                                for s in selected
                            ),
                        )
                        for phase in sorted(
                            {p for s in selected for p in s["ranks"][rank]["host"]}
                        )
                    },
                )
            )
    result = dict(
        identity_and_step_join="PASS",
        gpu_span_join="PASS",
        window_ns=[begin, end],
        clock_calibration=clock_check,
        clock_tolerance_ns=tolerance,
        graph_launches=len(graph_launches),
        graph_spans=len(seen_graphs),
        unjoined_gpu_activities=dict(unjoined),
        serving=window,
        requests=requests,
        reclaims=reclaims,
        aggregates=aggregates,
        steps=steps,
        limits=[
            "diagnostic only; not a formal performance result",
            "collective device time includes rank waiting and profiler perturbation",
            "opaque decode graphs do not expose internal communication kernels",
            "host phases are inclusive and overlap GPU activity; do not subtract sums",
            "busy unions cover traced project work, not other users' GPU activity",
            "own prefill step wall includes other requests sharing the batch",
            "non-output rank may finish ordinary postprocess after Core update starts",
            "cumulative returned blocks are not unique pages or simultaneous capacity",
        ],
    )
    (args.output or run / "ttft-analysis.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "identity_and_step_join": result["identity_and_step_join"],
                "gpu_span_join": result["gpu_span_join"],
                "requests": len(requests),
                "steps": len(steps),
                "reclaim_commits": len(reclaims),
                "graph_spans": len(seen_graphs),
            }
        )
    )


if __name__ == "__main__":
    main()
