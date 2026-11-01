# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate the fixed TP2 matrix and keep diagnostic timing separate."""

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path("/data_ssd/qcrs/experiments/p1-v8/TP")


def read(path):
    return json.loads(path.read_text())


def quantiles(items, name):
    values = (
        [dt for item in items for dt in item[name]]
        if name == "itl_s"
        else [item[name] for item in items]
    )
    return dict(zip(("p50", "p95", "p99"), np.percentile(values, [50, 95, 99])))


def preemptions(row):
    return sum(
        value
        for key, value in row["counter_deltas"].items()
        if key.split("{")[0] == "vllm:num_preemptions_total"
    )


def engineering_summary(root, baseline_micro, fused_micro, output):
    runs, tokens = {}, {}
    for run in sorted(root.iterdir()):
        protocol = read(run / "protocol.json")
        rounds = read(run / "serving.json")
        items = [
            item
            for row in rounds
            for item in read(run / f"requests-{row['repeat']}.json")
        ]
        args = protocol["args"]
        count = 2 if args["workload"] == "short" else 8
        assert len(rounds) == 3 and len(items) == count * 3
        assert args["tp"] == 2 and args["policy"] == "triattention"
        assert args["worker_cpus"] == [20, 21] and args["full_warmup"]
        assert all(len(i["token_ids"]) == i["output_tokens"] == 128 for i in items)
        prompts = read(run / "prompts.json")
        assert protocol["prompt_sha256"] == [
            hashlib.sha256(json.dumps(p).encode()).hexdigest() for p in prompts
        ]
        for row in rounds:
            requests = read(run / f"requests-{row['repeat']}.json")
            assert math.isclose(
                count * 128 / row["elapsed_s"], row["output_tokens_per_s"]
            )
            assert row["slo_good_requests"] == sum(
                i["ttft_s"] <= 2 and i["tpot_s"] <= 0.08 for i in requests
            )
            row["preemptions"] = preemptions(row)
        elapsed = sum(r["elapsed_s"] for r in rounds)
        key = run.name[16:]
        tokens[key] = [i["token_ids"] for i in items]
        guard = read(run / "result.json")
        assert guard["exit_code"] == 0 and not guard["limit_triggered"]
        affinity = [read(run / f"worker-affinity-{rank}.json") for rank in (0, 1)]
        runs[key] = dict(
            directory=str(run),
            protocol=protocol,
            affinity=affinity,
            guard=guard,
            rounds=rounds,
            warmup_preemptions=preemptions(read(run / "warmup-summary.json")),
            aggregate=dict(
                output_tokens_per_s=len(items) * 128 / elapsed,
                request_throughput_per_s=len(items) / elapsed,
                slo_goodput_requests_per_s=sum(r["slo_good_requests"] for r in rounds)
                / elapsed,
                slo_good_requests=sum(r["slo_good_requests"] for r in rounds),
                preemptions=sum(r["preemptions"] for r in rounds),
                latencies={
                    name: quantiles(items, name) for name in ("ttft_s", "tpot_s")
                },
                max_ttft_s=max(i["ttft_s"] for i in items),
                max_tpot_s=max(i["tpot_s"] for i in items),
                outside_slo=[
                    i["request"] for i in items if i["ttft_s"] > 2 or i["tpot_s"] > 0.08
                ],
            ),
        )
    comparisons = []
    for suffix in ("batch1024", "batch1024-reverse", "short"):
        stem = "triattention-" + suffix.removesuffix("-reverse")
        tail = "-reverse" if suffix.endswith("-reverse") else ""
        left, right = stem + "-baseline" + tail, stem + "-fused" + tail
        lhs, rhs = runs[left], runs[right]
        assert lhs["protocol"]["prompt_sha256"] == rhs["protocol"]["prompt_sha256"]
        comparisons.append(
            dict(
                baseline=left,
                fused=right,
                throughput_change_percent=100
                * (
                    rhs["aggregate"]["output_tokens_per_s"]
                    / lhs["aggregate"]["output_tokens_per_s"]
                    - 1
                ),
                equal_token_outputs=sum(
                    a == b for a, b in zip(tokens[left], tokens[right])
                ),
                outputs=len(tokens[left]),
            )
        )

    micro = {}
    plans = {}
    for version, run in (("baseline", baseline_micro), ("fused", fused_micro)):
        records = [r for f in sorted(run.glob("runtime-rank-*.json")) for r in read(f)]
        groups = {}
        for record in records:
            shape = record["shape"]["E"]
            if record["part"] == "plan_identity":
                plans.setdefault(shape, set()).add(record["sha256"])
            if "median_ms" not in record:
                continue
            key = f"{shape}/{record['collective']}/{record['part']}"
            groups.setdefault(key, []).append(record)
        micro[version] = dict(
            directory=str(run),
            guard=read(run / "result.json"),
            shapes=read(run / "runtime-shapes.json"),
            stages={
                key: dict(
                    slowest_rank_median_ms=max(r["median_ms"] for r in rows),
                    per_rank=rows,
                )
                for key, rows in groups.items()
            },
        )
    assert all(len(hashes) == 1 for hashes in plans.values()), plans
    result = dict(
        runs=runs,
        comparisons=comparisons,
        micro=micro,
        micro_plan_hashes={str(k): next(iter(v)) for k, v in plans.items()},
    )
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(output=str(output), comparisons=comparisons), indent=2))


def main():
    expected = (
        {
            ("long", "full", pool, policy)
            for pool in (8, 0.75)
            for policy in ("full", "noop", "sink_recent")
        }
        | {
            ("long", "full", pool, policy)
            for pool in (0.75,)
            for policy in ("random", "triattention")
        }
        | {
            ("short", mode, 8, policy)
            for mode in ("full", "eager")
            for policy in ("full", "noop")
        }
    )
    runs, failed_runs = {}, []
    for run in sorted(ROOT.glob("*-http-*")):
        protocol = read(run / "protocol.json")
        args = protocol["args"]
        if args["repeats"] != 3 or args["trace"]:
            continue
        key = (args["workload"], args["graph_mode"], args["pool_gib"], args["policy"])
        result = read(run / "result.json")
        if result["status"] != "PASS":
            failed_runs.append(
                {"directory": run.name, "config": list(key), "guard": result}
            )
            continue
        assert key in expected and key not in runs, (run, key)
        rounds = read(run / "serving.json")
        prompts = read(run / "prompts.json")
        assert result["status"] == "PASS" and not result["limit_triggered"], run
        assert args["tp"] == 2
        assert not args["prefix_caching"] and not (run / "kv-trace.jsonl").exists()
        assert len(rounds) == 3 and (run / "warmup-summary.json").exists()
        lengths = [128, 256] if args["workload"] == "short" else [4096, 8192]
        count = 2 if args["workload"] == "short" else 8
        assert [len(p) for p in prompts] == lengths
        assert protocol["prompt_sha256"] == [
            hashlib.sha256(json.dumps(p).encode()).hexdigest() for p in prompts
        ]
        pooled = []
        for row in rounds:
            items = read(run / f"requests-{row['repeat']}.json")
            pooled.extend(items)
            assert len(items) == row["requests"] == count
            assert [i["prompt_tokens"] for i in items] == lengths * (count // 2)
            assert all(
                len(i["token_ids"]) == i["usage"]["completion_tokens"] == 128
                and i["usage"]["prompt_tokens"] == i["prompt_tokens"]
                for i in items
            )
            good = sum(i["ttft_s"] <= 2 and i["tpot_s"] <= 0.08 for i in items)
            assert good == row["slo_good_requests"]
            assert math.isclose(
                count * 128 / row["elapsed_s"], row["output_tokens_per_s"]
            )
            for name in ("ttft_s", "tpot_s", "itl_s"):
                assert np.allclose(
                    list(quantiles(items, name).values()), list(row[name].values())
                )
            row["preemptions"] = preemptions(row)
        lines = (run / "server.log").read_text().splitlines()
        metrics = [i for i, line in enumerate(lines) if '"GET /metrics HTTP' in line]
        assert len(metrics) == 8, run
        warnings = [
            {"line": line, "phase": "warmup" if i < metrics[2] else "timed"}
            for i, line in enumerate(lines)
            if "Triton kernel JIT compilation during inference" in line
        ]
        elapsed = sum(r["elapsed_s"] for r in rounds)
        runs[key] = {
            "directory": run.name,
            "config": list(key),
            "prompt_sha256": protocol["prompt_sha256"],
            "guard": result,
            "rounds": rounds,
            "aggregate": {
                "output_tokens_per_s": count * 3 * 128 / elapsed,
                "request_throughput_per_s": count * 3 / elapsed,
                "slo_goodput_requests_per_s": sum(
                    r["slo_good_requests"] for r in rounds
                )
                / elapsed,
                "slo_good_requests": sum(r["slo_good_requests"] for r in rounds),
                "preemptions": sum(r["preemptions"] for r in rounds),
                "pooled_latency_quantiles": {
                    name: quantiles(pooled, name)
                    for name in ("ttft_s", "tpot_s", "itl_s")
                },
            },
            "jit_warnings": warnings,
        }
    assert runs.keys() == expected, expected - runs.keys()
    identities, comparisons = [], []
    for key, native in runs.items():
        workload, mode, pool, policy = key
        if policy != "full":
            continue
        noop = runs[workload, mode, pool, "noop"]
        count = 2 if workload == "short" else 8
        equal = 0
        for repeat in range(3):
            lhs = read(ROOT / native["directory"] / f"requests-{repeat}.json")
            rhs = read(ROOT / noop["directory"] / f"requests-{repeat}.json")
            equal += sum(a["token_ids"] == b["token_ids"] for a, b in zip(lhs, rhs))
        assert equal == 3 * count
        identities.append({"config": list(key[:3]), "equal_outputs": equal})
        for candidate_key, candidate in runs.items():
            if candidate_key[:3] != key[:3] or candidate_key[3] == "full":
                continue
            assert candidate["prompt_sha256"] == native["prompt_sha256"]
            agg, base = candidate["aggregate"], native["aggregate"]
            comparisons.append(
                {
                    "config": list(candidate_key),
                    "throughput_change_percent": 100
                    * (agg["output_tokens_per_s"] / base["output_tokens_per_s"] - 1),
                    "noop_mean_overhead_percent": 100
                    * (
                        1
                        - np.mean(
                            [r["output_tokens_per_s"] for r in candidate["rounds"]]
                        )
                        / np.mean([r["output_tokens_per_s"] for r in native["rounds"]])
                    )
                    if candidate_key[3] == "noop"
                    else None,
                    "all_requests_within_slo": agg["slo_good_requests"] == 3 * count,
                    "goodput_improved": agg["slo_goodput_requests_per_s"]
                    > base["slo_goodput_requests_per_s"],
                }
            )
    diagnostics = []
    for run in sorted(ROOT.glob("*-http-*-diagnostic")):
        events = [
            json.loads(line)
            for line in (run / "kv-trace.jsonl").read_text().splitlines()
        ]
        for row in [read(run / "warmup-summary.json"), *read(run / "serving.json")]:
            phase = [
                e
                for e in events
                if row["start_monotonic_ns"] <= e["time_ns"] <= row["end_monotonic_ns"]
            ]
            released = [
                e["released_blocks"] for e in phase if e["event"] == "reclaim_commit"
            ]
            diagnostics.append(
                {
                    "directory": run.name,
                    "repeat": row["repeat"],
                    "events": dict(Counter(e["event"] for e in phase)),
                    "cumulative_released_blocks": sum(released),
                    "released_per_commit_range": [
                        min(released, default=0),
                        max(released, default=0),
                    ],
                }
            )
    summary = {
        "status": "PASS",
        "runs": list(runs.values()),
        "identities": identities,
        "comparisons": comparisons,
        "diagnostics": diagnostics,
        "failed_runs": failed_runs,
    }
    (ROOT / "serving-summary.json").write_text(json.dumps(summary, indent=2))
    print(
        json.dumps(
            {
                "validated_runs": len(runs),
                "identities": identities,
                "comparisons": comparisons,
                "diagnostics": diagnostics,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--engineering", type=Path)
    parser.add_argument("--baseline-micro", type=Path)
    parser.add_argument("--fused-micro", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.engineering:
        engineering_summary(
            args.engineering, args.baseline_micro, args.fused_micro, args.output
        )
    else:
        main()
