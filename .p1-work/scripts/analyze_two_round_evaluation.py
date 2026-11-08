# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Recompute single-GPU, TP2 and DP2 metrics from two-round experiment artifacts."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from analyze_dp_work import distribution, events, read
from analyze_protected_evaluation import work_summary
from dp_serving import aggregate


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized_requests(items, dp):
    if dp:
        return items
    return [dict(r, finish_s=r["start_s"] + r["e2e_s"]) for r in items]


def analyze(run):
    protocol = read(run / "protocol.json")
    args = protocol["args"]
    dp = "scenario" in args
    guard = read(run / "result.json")
    assert guard["status"] == "PASS" and guard["exit_code"] == 0
    assert not guard["limit_triggered"]
    assert guard["peak_project_nvml_bytes"] < 44 * 2**30
    assert args["repeats"] == 3 and not args["trace"] and args["work_counters"]
    command = protocol["command"]
    compilation = json.loads(command[command.index("--compilation-config") + 1])
    assert compilation == {
        "mode": 0,
        "cudagraph_mode": "FULL_DECODE_ONLY",
        "cudagraph_capture_sizes": [1, 2, 4, 8],
    }
    assert protocol["slo"] == {"ttft_s": 2, "tpot_s": 0.08}
    assert protocol["prompt_sha256"] == [
        hashlib.sha256(json.dumps(p).encode()).hexdigest()
        for p in read(run / "prompts.json")
    ]
    if dp:
        assert args["scenario"] == "w1" and args["routing"] == "known_work"
        assert args["pool_gib"] == 4
        assert args["max_num_batched_tokens"] == 512
        assert not args["load_diagnostic"]
        assert protocol["prompt_tokens"] == [8192, 512] * 4
        assert protocol["arrivals_s"] == [i * 0.05 for i in range(8)]
    else:
        assert args["graph_mode"] == "full"
        assert args["workload"] == "mixed" and args["full_warmup"]
        assert args["pool_gib"] * args["tp"] == 1.5
        assert args["max_num_batched_tokens"] == 1024
        assert not args["prefix_caching"] and args["reserve"] == 0
        assert protocol["prompt_tokens"] == [4096, 8192, 128, 128] * 2
        assert protocol["arrivals_s"] == [0, 0.3, 0.6, 0.9, 5, 5.3, 5.6, 5.9]
    assert args["max_num_seqs"] == 8
    assert protocol["output_tokens"] == 128
    name = "dp-source-hashes.json" if dp else "runtime-source-hashes.json"
    hashes = read(run / name)
    for relative, expected in hashes.items():
        assert digest(run / "source" / relative) == expected
        assert digest(Path(relative)) == expected
    invocation = read(run / "invocation.json")
    devices = "0,1" if dp or args["tp"] == 2 else "1"
    assert invocation["environment"]["CUDA_VISIBLE_DEVICES"] == devices
    assert invocation["hard_limit_bytes"] == 45 * 2**30
    assert invocation["stop_threshold_bytes"] == 44 * 2**30
    for source, expected in invocation["source_files"].items():
        assert digest(run / "source" / Path(source).relative_to(invocation["cwd"])) == (
            expected
        )
    affinities = [
        read(run / f"worker-affinity-{i}.json")
        for i in range(2 if devices == "0,1" else 1)
    ]
    assert [v["affinity"] for v in affinities] == (
        [[20], [21]] if devices == "0,1" else [[21]]
    )
    work = [events(p) for p in sorted(run.glob("replica-*/work.jsonl"))]
    assert sum(map(len, work)) == 33
    all_items, timed_work, rounds = [], [], []
    summaries = read(run / "serving.json")
    assert [s["repeat"] for s in summaries] == [0, 1, 2]
    for summary in [read(run / "warmup-summary.json"), *summaries]:
        label = summary["repeat"]
        items = read(run / f"requests-{label}.json")
        assert [r["prompt_tokens"] for r in items] == protocol["prompt_tokens"]
        for item in items:
            assert len(item["token_ids"]) == item["usage"]["completion_tokens"] == 128
        observed = [
            [
                r
                for r in records
                if summary["start_monotonic_ns"]
                <= r["completed_ns"]
                <= summary["end_monotonic_ns"]
            ]
            for records in work
        ]
        completed = [r for records in observed for r in records]
        assert len(completed) == 8
        for record in completed:
            assert record["scheduled_tokens"] == (
                record["prompt_tokens"] + 127 + record["recomputed_tokens"]
            )
        counts = work_summary(completed)
        preemptions = sum(
            value
            for key, value in summary["counter_deltas"].items()
            if key.startswith("vllm:num_preemptions_total{")
        )
        assert preemptions == counts["preemptions"]
        calculated = aggregate(normalized_requests(items, dp), summary["elapsed_s"])
        stored = summary["aggregate"] if dp else summary
        for key in (
            "slo_good_requests",
            "slo_goodput_requests_per_s",
            "output_tokens_per_s",
            "ttft_s",
            "tpot_s",
            "itl_s",
        ):
            assert calculated[key] == stored[key], (run, label, key)
        if not dp and args["reclamation_counts"]:
            before = summary["reclamation_counts_before"][-1]["canonical_commits"]
            after = summary["reclamation_counts_after"][-1]["canonical_commits"]
            assert after - before == counts["reclaim_calls"]
        rounds.append(dict(repeat=label, metrics=calculated, work=counts))
        if label != "warmup":
            all_items.extend(items)
            timed_work.extend(completed)
    combined = aggregate(
        normalized_requests(all_items, dp), sum(r["elapsed_s"] for r in summaries)
    )
    combined.pop("active_span_s")
    by_length = {}
    for length in sorted({r["prompt_tokens"] for r in all_items}):
        group = [r for r in all_items if r["prompt_tokens"] == length]
        by_length[length] = dict(
            count=len(group),
            slo=sum(r["ttft_s"] <= 2 and r["tpot_s"] <= 0.08 for r in group),
            ttft_s=distribution([r["ttft_s"] for r in group]),
            tpot_s=distribution([r["tpot_s"] for r in group]),
        )
    answer = dict(
        run=str(run),
        config=args,
        metrics=combined,
        by_length=by_length,
        work=work_summary(timed_work),
        rounds=rounds,
        source_hashes=hashes,
        input_hashes=protocol["prompt_sha256"],
        devices=devices,
        affinity=affinities,
        guard=guard,
    )
    if dp:
        answer["requests_per_replica"] = dict(Counter(r["replica"] for r in all_items))
        answer["timed_work_per_replica"] = [
            work_summary(
                [
                    r
                    for r in records
                    if any(
                        s["start_monotonic_ns"]
                        <= r["completed_ns"]
                        <= s["end_monotonic_ns"]
                        for s in summaries
                    )
                ]
            )
            for records in work
        ]
    return answer


def analyze_diagnostic(run):
    protocol = read(run / "protocol.json")
    args, guard = protocol["args"], read(run / "result.json")
    assert args["trace"] and args["repeats"] == 1
    assert args["tp"] == 2 and args["decode_reclaim_rounds"] == 2
    assert guard["status"] == "PASS" and not guard["limit_triggered"]
    assert guard["peak_project_nvml_bytes"] < 44 * 2**30
    hashes = read(run / "runtime-source-hashes.json")
    for relative, expected in hashes.items():
        assert digest(run / "source" / relative) == digest(Path(relative)) == expected
    summary = read(run / "serving.json")[0]
    start, end = summary["start_monotonic_ns"], summary["end_monotonic_ns"]
    trace = [e for e in events(run / "kv-trace.jsonl") if start <= e["time_ns"] <= end]
    records = [
        r
        for r in events(run / "replica-0/work.jsonl")
        if start <= r["completed_ns"] <= end
    ]
    work = {r["request"]: r for r in records}
    checks = {}
    admissions = {}
    for event in trace:
        if event.get("request_status") != "WAITING":
            continue
        if event["event"] in ("allocation_check", "allocation_miss"):
            checks.setdefault(event["request_id"], event)
        if event["event"] == "allocation_done":
            admissions.setdefault(event["request_id"], event)
    assert len(checks) == len(admissions) == len(work) == 8
    assert [work[r]["prompt_tokens"] for r in checks] == protocol["prompt_tokens"]
    timelines, windows = [], []
    for index, (request, first_check) in enumerate(checks.items()):
        admitted = admissions[request]
        prefill = [
            e
            for e in trace
            if e["event"] == "allocation_done"
            and e["request_id"] == request
            and e["logical_tokens"] < work[request]["prompt_tokens"]
        ]
        timelines.append(
            dict(
                index=index,
                server_request=request,
                prompt_tokens=work[request]["prompt_tokens"],
                first_capacity_event_s=(first_check["time_ns"] - start) / 1e9,
                admission_s=(admitted["time_ns"] - start) / 1e9,
                last_prefill_allocation_s=(prefill[-1]["time_ns"] - start) / 1e9,
                prefill_allocation_span_s=(prefill[-1]["time_ns"] - admitted["time_ns"])
                / 1e9,
            )
        )
        plans = [
            e
            for e in trace
            if e["event"] == "protected_plan"
            and e["requester"] == request
            and e["time_ns"] < admitted["time_ns"]
        ]
        assert len(plans) <= 2 and all(e["plan"] for e in plans)
        if len(plans) != 2:
            continue
        rounds = []
        for plan, bound in zip(plans, [plans[1], admitted]):
            commits = [
                e
                for e in trace
                if e["event"] == "reclaim_commit"
                and plan["time_ns"] <= e["time_ns"] < bound["time_ns"]
            ]
            assert len(plan["plan"]) == len(commits) == 1
            candidate, commit = plan["plan"][0], commits[0]
            assert candidate["request_id"] == commit["request_id"]
            rounds.append(
                dict(
                    shortfall=plan["shortfall"],
                    plan_s=(plan["time_ns"] - start) / 1e9,
                    commit_s=(commit["time_ns"] - start) / 1e9,
                    released_pages=commit["released_blocks"],
                    copy_bytes=candidate["copy_bytes"],
                    settled_logical_tokens=commit["settled_logical_tokens"],
                )
            )
        assert rounds[1]["settled_logical_tokens"] > rounds[0]["settled_logical_tokens"]
        windows.append(
            dict(
                requester_index=index,
                rounds=rounds,
                plan_to_admission_ms=(admitted["time_ns"] - plans[0]["time_ns"]) / 1e6,
            )
        )
    counts = Counter(e["event"] for e in trace)
    totals = work_summary(records)
    assert counts["reclaim_commit"] == totals["reclaim_calls"] == 8
    assert counts["allocation_miss"] == totals["allocation_misses"] == 10
    assert totals["preemptions"] == totals["recomputed_tokens"] == 0
    items = read(run / "requests-0.json")
    metrics = aggregate(normalized_requests(items, False), summary["elapsed_s"])
    assert metrics["slo_good_requests"] == summary["slo_good_requests"] == 8
    assert metrics["output_tokens_per_s"] == summary["output_tokens_per_s"]
    return dict(
        run=str(run),
        metrics=metrics,
        work=totals,
        event_counts=dict(counts),
        server_timeline=timelines,
        two_round_windows=windows,
        guard=guard,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--diagnostic", action="store_true")
    args = parser.parse_args()
    for run in args.runs:
        answer = analyze_diagnostic(run) if args.diagnostic else analyze(run)
        filename = "diagnostic-audited.json" if args.diagnostic else "audited.json"
        (run / filename).write_text(json.dumps(answer, indent=2))
        print(
            json.dumps(
                {
                    k: answer[k]
                    for k in ("run", "metrics", "by_length", "work")
                    if k in answer
                },
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
