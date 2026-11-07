# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Audit final fixed workloads; separate correctness, micro and Serving claims."""

import argparse
import ast
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path

from analyze_dp_work import analyze_formal, compare_formal, distribution, events, read
from dp_serving import aggregate


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_run(run, hashes, *, devices="0,1", frozen_source=False):
    guard = read(run / "result.json")
    assert guard["status"] == "PASS" and guard["exit_code"] == 0
    assert not guard["limit_triggered"]
    assert guard["peak_project_nvml_bytes"] < 44 * 2**30
    for name, expected in hashes.items():
        assert digest(run / "source" / name) == expected, (run, name)
        if not frozen_source:
            assert digest(Path(name)) == expected, ("final source changed", run, name)
    invocation = read(run / "invocation.json")
    assert invocation["environment"]["CUDA_VISIBLE_DEVICES"] == devices
    assert invocation["hard_limit_bytes"] == 45 * 2**30
    assert invocation["stop_threshold_bytes"] == 44 * 2**30
    for name, expected in invocation["source_files"].items():
        relative = Path(name).relative_to(invocation["cwd"])
        assert digest(run / "source" / relative) == expected
    affinities = [read(run / f"worker-affinity-{rank}.json") for rank in (0, 1)]
    assert [a["affinity"] for a in affinities] == [[20], [21]]
    return guard


def work_summary(records):
    totals = Counter()
    for record in records:
        totals.update(
            {
                k: record[k]
                for k in (
                    "scheduled_tokens",
                    "prefill_tokens",
                    "recomputed_tokens",
                    "allocation_misses",
                    "reclaim_calls",
                    "released_pages",
                    "estimated_copy_bytes",
                    "zero_copy_calls",
                    "preemptions",
                )
            }
        )
    return dict(
        **totals,
        max_batch_victims=max(r["max_batch_victims"] for r in records),
        usage_min=min(r["usage_min"] for r in records),
        usage_max=max(r["usage_max"] for r in records),
    )


def analyze_tp(run, **audit_options):
    protocol = read(run / "protocol.json")
    args = protocol["args"]
    hashes = read(run / "runtime-source-hashes.json")
    guard = audit_run(run, hashes, **audit_options)
    assert args["tp"] == 2 and args["pool_gib"] == 0.75
    assert args["graph_mode"] == "full" and args["full_warmup"]
    assert args["repeats"] == 3 and args["work_counters"]
    assert not args["trace"]
    assert not args["prefix_caching"] and args["reserve"] == 0
    assert args["max_num_batched_tokens"] == 1024 and args["max_num_seqs"] == 8
    assert protocol["slo"] == {"ttft_s": 2, "tpot_s": 0.08}
    assert protocol["prompt_tokens"] == [4096, 8192] * 4
    assert protocol["arrivals_s"] == [0, 0.3, 0.6, 0.9, 5, 5.3, 5.6, 5.9]
    assert protocol["prompt_sha256"] == [
        hashlib.sha256(json.dumps(p).encode()).hexdigest()
        for p in read(run / "prompts.json")
    ]
    records = events(run / "replica-0/work.jsonl")
    assert len(records) == 33
    rounds, pooled, timed_work = [], [], []
    summaries = read(run / "serving.json")
    assert [s["repeat"] for s in summaries] == [0, 1, 2]
    for summary in [read(run / "warmup-summary.json"), *summaries]:
        label = summary["repeat"]
        items = read(run / f"requests-{label}.json")
        assert [r["prompt_tokens"] for r in items] == protocol["prompt_tokens"]
        assert all(r["output_tokens"] == len(r["token_ids"]) == 128 for r in items)
        observed = [
            r
            for r in records
            if summary["start_monotonic_ns"]
            <= r["completed_ns"]
            <= summary["end_monotonic_ns"]
        ]
        assert len(observed) == 8
        assert sorted(r["prompt_tokens"] for r in observed) == sorted(
            protocol["prompt_tokens"]
        )
        for record in observed:
            assert record["scheduled_tokens"] == (
                record["prompt_tokens"] + 127 + record["recomputed_tokens"]
            )
        counts = work_summary(observed)
        preemptions = sum(
            value
            for key, value in summary["counter_deltas"].items()
            if key.startswith("vllm:num_preemptions_total{")
        )
        assert preemptions == counts["preemptions"]
        calculated = aggregate(
            [dict(r, finish_s=r["start_s"] + r["e2e_s"]) for r in items],
            summary["elapsed_s"],
        )
        for key in (
            "slo_good_requests",
            "slo_goodput_requests_per_s",
            "output_tokens_per_s",
            "ttft_s",
            "tpot_s",
            "itl_s",
        ):
            assert calculated[key] == summary[key], (run, label, key)
        if args["reclamation_counts"]:
            before = summary["reclamation_counts_before"][-1]["canonical_commits"]
            after = summary["reclamation_counts_after"][-1]["canonical_commits"]
            assert after - before == counts["reclaim_calls"]
        else:
            assert "reclamation_counts_before" not in summary
            assert "reclamation_counts_after" not in summary
        rounds.append(dict(repeat=label, summary=summary, work=counts))
        if label != "warmup":
            pooled.extend(items)
            timed_work.extend(observed)
    elapsed = sum(s["elapsed_s"] for s in summaries)
    combined = aggregate(
        [dict(r, finish_s=r["start_s"] + r["e2e_s"]) for r in pooled], elapsed
    )
    combined.pop("active_span_s")
    return dict(
        run=str(run),
        protocol=protocol,
        guard=guard,
        source_hashes=hashes,
        source_audit=audit_options,
        reclamation_count_audit=(
            "Scheduler work records cross-checked with Worker canonical commits"
            if args["reclamation_counts"]
            else "Scheduler completed work records; no Worker counter RPC"
        ),
        aggregate=combined,
        rounds=rounds,
        work=work_summary(timed_work),
        max_ttft_s=max(r["ttft_s"] for r in pooled),
        max_tpot_s=max(r["tpot_s"] for r in pooled),
        request_slo=[
            dict(
                request=r["request"],
                ttft_s=r["ttft_s"],
                tpot_s=r["tpot_s"],
                passed=r["ttft_s"] <= 2 and r["tpot_s"] <= 0.08,
            )
            for r in pooled
        ],
    )


def compare_tp(runs, *, allow_source_change=False):
    if not runs:
        return []
    baseline = runs[0]
    helpers = ["dp_hooks.py", "tp_worker_hooks.py", "worker_affinity.py"]
    comparisons = []
    for run in runs[1:]:
        if not allow_source_change:
            assert run["source_hashes"] == baseline["source_hashes"]
        assert run["protocol"]["prompt_sha256"] == baseline["protocol"]["prompt_sha256"]
        configs = [dict(r["protocol"]["args"]) for r in (baseline, run)]
        for config in configs:
            config.pop("policy")
            config.pop("budget_mode")
        assert configs[0] == configs[1]
        inactive_helper_changes = []
        for helper in helpers:
            a = Path(baseline["run"]) / helper
            b = Path(run["run"]) / helper
            if digest(a) == digest(b):
                continue
            assert allow_source_change and helper == "tp_worker_hooks.py"
            trees = [ast.parse(path.read_text()) for path in (a, b)]
            # Only the unused correctness Worker class may differ. In particular,
            # imports, module statements and the imported members() must match.
            for tree in trees:
                tree.body = [
                    node
                    for node in tree.body
                    if not (
                        isinstance(node, ast.ClassDef) and node.name == "P1TPWorker"
                    )
                ]
            assert ast.dump(trees[0]) == ast.dump(trees[1])
            for root in (Path(baseline["run"]), Path(run["run"])):
                assert (
                    "from tp_worker_hooks import members"
                    in (root / "dp_hooks.py").read_text()
                )
                assert "P1TPWorker" not in (root / "dp_hooks.py").read_text()
                serving = root / "source/.p1-work/scripts/serving_reclamation.py"
                assert "P1TPWorker" not in serving.read_text()
            inactive_helper_changes.append(helper)
        comparisons.append(
            dict(
                A=baseline["run"],
                B=run["run"],
                changed_sources=[
                    name
                    for name in sorted(run["source_hashes"] | baseline["source_hashes"])
                    if run["source_hashes"].get(name)
                    != baseline["source_hashes"].get(name)
                ],
                inactive_diagnostic_helper_changes=inactive_helper_changes,
                throughput_change_percent=100
                * (
                    run["aggregate"]["output_tokens_per_s"]
                    / baseline["aggregate"]["output_tokens_per_s"]
                    - 1
                ),
            )
        )
    return comparisons


def analyze_dp(run, **audit_options):
    output = analyze_formal(run)
    audit_run(run, output["source_hashes"], **audit_options)
    records = [events(run / f"replica-{rank}/work.jsonl") for rank in (0, 1)]
    timed = read(run / "serving.json")
    output["timed_work"] = [
        work_summary(
            [
                r
                for r in rank_records
                if any(
                    s["start_monotonic_ns"]
                    <= r["completed_ns"]
                    <= s["end_monotonic_ns"]
                    for s in timed
                )
            ]
        )
        for rank_records in records
    ]
    output["usage_at_route"] = [
        distribution(
            [
                s["kv_cache_usage"]
                for route in events(run / "routing.jsonl")
                if "warmup" not in route["external"]
                for i, s in enumerate(route["snapshots"])
                if i == rank and s
            ]
        )
        for rank in (0, 1)
    ]
    output["request_slo"] = [
        dict(
            request=r["request"],
            replica=r["replica"],
            ttft_s=r["ttft_s"],
            tpot_s=r["tpot_s"],
            passed=r["ttft_s"] <= 2 and r["tpot_s"] <= 0.08,
        )
        for summary in timed
        for r in read(run / f"requests-{summary['repeat']}.json")
    ]
    output.pop("outputs")
    return output


def analyze_micro(run, *, devices="0,1"):
    hashes = read(run / "source-hashes.json")
    guard = read(run / "result.json")
    assert guard["status"] == "PASS" and not guard["limit_triggered"]
    assert guard["peak_project_nvml_bytes"] < 44 * 2**30
    invocation = read(run / "invocation.json")
    assert invocation["environment"]["CUDA_VISIBLE_DEVICES"] == devices
    for name, expected in hashes.items():
        assert digest(run / "source" / name) == digest(Path(name)) == expected
    cpu = read(run / "cpu-micro.json")
    ranks = [read(run / f"micro-rank-{rank}.json") for rank in (0, 1)]
    assert [r["cpu_affinity"] for r in ranks] == [[20], [21]]
    assert len({r["gpu_uuid"] for r in ranks}) == 2
    records = []
    for a, b in zip(ranks[0]["records"], ranks[1]["records"]):
        if not a.get("feasible", True):
            assert a == b
            records.append(a)
            continue
        assert {k: v for k, v in a.items() if k != "samples"} == {
            k: v for k, v in b.items() if k != "samples"
        }
        samples = [
            dict((key, max(x[key], y[key])) for key in x)
            for x, y in zip(a["samples"], b["samples"])
        ]
        records.append(
            {
                **{k: v for k, v in a.items() if k != "samples"},
                "max_rank_median": {
                    key: statistics.median(s[key] for s in samples)
                    for key in samples[0]
                },
            }
        )
    return dict(
        run=str(run),
        guard=guard,
        controller=[
            {k: v for k, v in r.items() if k not in ("plan", "decision_us")}
            for r in cpu["controller"]
        ],
        routing=[
            {k: v for k, v in r.items() if k != "route_us"} for r in cpu["routing"]
        ],
        worker=records,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp", nargs="*", type=Path, default=[])
    parser.add_argument("--dp", nargs="*", type=Path, default=[])
    parser.add_argument("--combo", type=Path)
    parser.add_argument("--micro", type=Path)
    parser.add_argument("--devices", default="0,1")
    parser.add_argument("--frozen-source-run", action="append", type=Path, default=[])
    parser.add_argument("--allow-source-change", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    frozen = {run.resolve() for run in args.frozen_source_run}

    def audit_options(run):
        return dict(devices=args.devices, frozen_source=run.resolve() in frozen)

    assert frozen <= {run.resolve() for run in [*args.tp, *args.dp, args.combo] if run}
    tp = [analyze_tp(run, **audit_options(run)) for run in args.tp]
    dp = [analyze_dp(run, **audit_options(run)) for run in args.dp]
    # compare_formal also audits token IDs; retain that result before projection.
    comparisons = compare_formal([analyze_formal(run) for run in args.dp])
    output = dict(
        tp=tp,
        tp_comparisons=compare_tp(tp, allow_source_change=args.allow_source_change),
        dp=dp,
        dp_comparisons=comparisons,
        combo=analyze_dp(args.combo, **audit_options(args.combo))
        if args.combo
        else None,
        micro=analyze_micro(args.micro, devices=args.devices) if args.micro else None,
    )
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                tp=[
                    dict(run=r["run"], aggregate=r["aggregate"], work=r["work"])
                    for r in output["tp"]
                ],
                dp=comparisons,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
