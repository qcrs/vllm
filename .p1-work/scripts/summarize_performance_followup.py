# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Recompute placement, ABBA and pressure evidence from the retained raw runs."""

import argparse
import hashlib
import json
import math
import pstats
import statistics
from collections import Counter
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(path.read_text())


def one(root, suffix):
    matches = list(root.glob("*-" + suffix))
    assert len(matches) == 1, (suffix, matches)
    return matches[0]


def quantiles(values):
    return dict(zip(("p50", "p95", "p99"), np.percentile(values, [50, 95, 99])))


def preemptions(path):
    return {
        key: float(value)
        for line in path.read_text().splitlines()
        if line.startswith("vllm:num_preemptions_total{")
        for key, value, *_ in [line.split()]
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
    }


def serving(run, protocol=None):
    protocol = protocol or read(run / "protocol.json")
    rows = read(run / "serving.json")
    assert [r["repeat"] for r in rows] == list(range(len(rows))), run
    rounds, requests = [], []
    for row in rows:
        repeat = row["repeat"]
        items = read(run / f"requests-{repeat}.json")
        assert len(items) == len(protocol["arrivals_s"]), run
        for i, item in enumerate(items):
            assert item["prompt_tokens"] == protocol["prompt_tokens"][i], run
            assert item["arrival_s"] == protocol["arrivals_s"][i], run
            assert len(item["token_ids"]) == item["output_tokens"] == 128, run
        recomputed = aggregate(items, row["elapsed_s"])
        recorded = row.get("aggregate", row)
        for key in ("output_tokens_per_s", "slo_goodput_requests_per_s"):
            assert math.isclose(recomputed[key], recorded[key]), (run, key)
        assert recomputed["slo_good_requests"] == recorded["slo_good_requests"], run
        before = preemptions(run / f"metrics-before-{repeat}.txt")
        after = preemptions(run / f"metrics-{repeat}.txt")
        deltas = {k: v - before[k] for k, v in after.items()}
        assert all(row["counter_deltas"][k] == v for k, v in deltas.items()), run
        rounds.append(
            {
                "repeat": repeat,
                "elapsed_s": row["elapsed_s"],
                **recomputed,
                "preemptions": sum(deltas.values()),
            }
        )
        requests.extend(items)
    elapsed = sum(r["elapsed_s"] for r in rounds)
    result = {
        "run": run.name,
        "rounds": rounds,
        "aggregate": aggregate(requests, elapsed),
        "mean_tokens_per_s": statistics.mean(r["output_tokens_per_s"] for r in rounds),
        "preemptions": sum(r["preemptions"] for r in rounds),
        "slo_status": "PASS"
        if all(i["ttft_s"] <= 2 and i["tpot_s"] <= 0.08 for i in requests)
        else "FAIL",
    }
    if "replica" in requests[0]:
        result["replicas"] = {
            str(rank): aggregate([i for i in requests if i["replica"] == rank], elapsed)
            for rank in range(2)
        }
    return result, requests


def measured_run(run):
    guard = read(run / "result.json")
    assert guard["status"] == "PASS" and not guard["limit_triggered"], run
    assert guard["peak_project_nvml_bytes"] < 44 * 2**30, run
    protocol = read(run / "protocol.json")
    assert protocol["slo"] == {"ttft_s": 2, "tpot_s": 0.08}, run
    prompts = read(run / "prompts.json")
    assert protocol["prompt_sha256"] == [
        hashlib.sha256(json.dumps(p).encode()).hexdigest() for p in prompts
    ], run
    result, requests = serving(run, protocol)
    assert len(result["rounds"]) == protocol["args"]["repeats"], run
    affinity = [read(p) for p in sorted(run.glob("worker-affinity-*.json"))]
    cpus = protocol["args"].get("worker_cpus")
    if cpus is not None:
        assert len(affinity) == len(cpus), run
        assert [a["affinity"] for a in affinity] == [[cpu] for cpu in cpus], run
    if "replica" in requests[0]:
        routes = [
            json.loads(line)
            for line in (run / "routing.jsonl").read_text().splitlines()
        ]
        for item in requests:
            assigned = [
                r["replica"]
                for r in routes
                if r["external"] == f"cmpl-{item['request']}-0"
            ]
            assert assigned == [item["replica"]], (run, item["request"])
    result.update(
        config=protocol["args"],
        guard=guard,
        affinity=[
            {k: a[k] for k in ("rank", "local_rank", "affinity")} for a in affinity
        ],
    )
    return result, requests


def compare(root, label):
    native_run = one(root, f"tp-eager-affinity-{label}-full")
    noop_run = one(root, f"tp-eager-affinity-{label}-noop")
    native, a = measured_run(native_run)
    noop, b = measured_run(noop_run)
    assert len(native["rounds"]) == len(noop["rounds"]) == 3
    for key in ("prompt_tokens", "arrivals_s", "prompt_sha256", "warmup"):
        assert (
            read(native_run / "protocol.json")[key]
            == read(noop_run / "protocol.json")[key]
        ), key
    assert native["config"] | {"policy": "noop"} == noop["config"]
    assert native["config"]["worker_cpus"] == [20, 21]
    assert native["config"]["graph_mode"] == "eager"
    assert native["config"]["full_warmup"]

    def common_command(run):
        command = read(run / "protocol.json")["command"]
        return [
            value
            for i, value in enumerate(command)
            if value not in ("--port", "--additional-config")
            and (i == 0 or command[i - 1] not in ("--port", "--additional-config"))
        ]

    assert common_command(native_run) == common_command(noop_run)
    noop_command = read(noop_run / "protocol.json")["command"]
    config = json.loads(noop_command[noop_command.index("--additional-config") + 1])
    assert config["kv_reclamation"]["retention"] == 1.0
    assert all(x["token_ids"] == y["token_ids"] for x, y in zip(a, b))
    overhead = 100 * (1 - noop["mean_tokens_per_s"] / native["mean_tokens_per_s"])
    return {
        "native": native,
        "noop": noop,
        "paired_outputs_equal": len(a),
        "throughput_overhead_percent": overhead,
        "budget_status": "PASS" if overhead <= 5 else "FAIL",
    }


def intervention(root, suffix, event_prefix):
    run = one(root, suffix)
    protocol = read(run / "protocol.json")
    events = [read(run / f"{event_prefix}-rank-{rank}.json") for rank in range(2)]
    phases = []
    for phase in sorted(p.parent for p in run.glob("*/serving.json")):
        summary, _ = serving(phase, protocol)
        windows = read(phase / "serving.json")
        ranks = []
        for rows in events:
            selected = [
                event
                for event in rows
                if any(
                    w["start_monotonic_ns"] <= event[0]
                    and event[1] <= w["end_monotonic_ns"]
                    for w in windows
                )
            ]
            ranks.append(
                {
                    "execute_calls": len(selected),
                    "wall_ms": quantiles([(e[1] - e[0]) / 1e6 for e in selected]),
                    "thread_cpu_ms": quantiles([e[2] / 1e6 for e in selected]),
                    "observed_cpus": sorted(
                        {e[3] for e in selected} | {e[4] for e in selected}
                    ),
                }
            )
        phases.append({"phase": phase.name, "serving": summary, "ranks": ranks})
    return {"run": run.name, "guard": read(run / "result.json"), "phases": phases}


def cprofile(root):
    result = []
    for policy, suffix in (
        ("full", "tp-eager-native-steady-cprofile"),
        ("noop", "tp-eager-noop-steady-cprofile"),
    ):
        run = one(root, suffix)
        for rank in range(2):
            stats = pstats.Stats(str(run / f"cpu-rank-{rank}.prof"))
            prepare = [
                v
                for k, v in stats.stats.items()
                if k[0].endswith("gpu/kv_reclamation.py") and k[2] == "prepare_attn"
            ]
            calls = sum(v[1] for v in prepare)
            cumulative = sum(v[3] for v in prepare)
            result.append(
                {
                    "run": run.name,
                    "policy": policy,
                    "rank": rank,
                    "total_self_cpu_s": stats.total_tt,
                    "total_calls": stats.total_calls,
                    "prepare_calls": calls,
                    "prepare_cumulative_s": cumulative,
                    "prepare_us_per_call": cumulative * 1e6 / calls if calls else None,
                }
            )
    return result


def reclamation_diagnostic(root, kind, suffix):
    run = one(root, suffix)
    guard = read(run / "result.json")
    assert guard["status"] == "PASS", run
    row = read(run / "serving.json")[0]
    paths = (
        [run / f"replica-{rank}" / "kv-trace.jsonl" for rank in range(2)]
        if kind == "dp"
        else [run / "kv-trace.jsonl"]
    )
    replicas = []
    for path in paths:
        events = [json.loads(line) for line in path.read_text().splitlines()]
        phase = [
            e
            for e in events
            if row["start_monotonic_ns"] <= e["time_ns"] <= row["end_monotonic_ns"]
        ]
        commits = [e for e in phase if e["event"] == "reclaim_commit"]
        assert all(
            e["released_blocks"] > 0
            and e["source_effective_kv_len"] > e["effective_kv_len"]
            for e in commits
        ), path
        replicas.append(
            {
                "replica": len(replicas),
                "events": dict(Counter(e["event"] for e in phase)),
                "commits": len(commits),
                "cumulative_released_blocks": sum(
                    e["released_blocks"] for e in commits
                ),
            }
        )
    assert replicas[0]["commits"] > 0, run
    if kind == "dp":
        assert replicas[1]["commits"] == 0, run
    return {"run": run.name, "guard": guard, "replicas": replicas}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "root",
        nargs="?",
        type=Path,
        default="/data_ssd/qcrs/experiments/p1-v8/Perf",
    )
    args = parser.parse_args()
    pairs = [compare(args.root, label) for label in ("a", "b")]
    rates = {
        policy: statistics.mean(
            r["output_tokens_per_s"] for pair in pairs for r in pair[policy]["rounds"]
        )
        for policy in ("native", "noop")
    }
    overhead = 100 * (1 - rates["noop"] / rates["native"])
    pressure = [
        measured_run(run)[0]
        for run in sorted(args.root.iterdir())
        if run.is_dir() and ("-tp-pressure-" in run.name or "-dp-skewed-" in run.name)
    ]
    references = {
        "tp": args.root.parent
        / "TP"
        / "20261031-120229-http-long-full-0.75-sink_recent",
        "dp": args.root.parent / "DP" / "20261031-125521-http-skewed-1.5-sink_recent",
    }
    for row in pressure:
        kind = "tp" if "-tp-pressure-" in row["run"] else "dp"
        protocol = read(args.root / row["run"] / "protocol.json")
        reference = read(references[kind] / "protocol.json")
        for key in (
            "prompt_tokens",
            "arrivals_s",
            "output_tokens",
            "slo",
            "prompt_sha256",
        ):
            assert protocol[key] == reference[key], (row["run"], key)
        for key in ("policy", "pool_gib"):
            assert protocol["args"][key] == reference["args"][key], (row["run"], key)
        assert not protocol["args"]["trace"], row["run"]
        command = protocol["command"]
        config = json.loads(command[command.index("--additional-config") + 1])[
            "kv_reclamation"
        ]
        assert config["retention"] == 0.5 and config["reserve_cap"] == 0, row["run"]
    result = {
        "evidence_status": "PASS",
        "tp_eager_abba": pairs,
        "tp_eager_pooled": {
            **rates,
            "throughput_overhead_percent": overhead,
            "budget_percent": 5,
            "budget_status": "PASS" if overhead <= 5 else "FAIL",
            "paired_outputs_equal": sum(p["paired_outputs_equal"] for p in pairs),
        },
        "pressure_including_pilots": pressure,
        "pressure_historical_references": {
            kind: measured_run(run)[0] for kind, run in references.items()
        },
        "placement_diagnostic": intervention(
            args.root, "tp-eager-core-placement-crossover", "placement"
        ),
        "toggle_diagnostic": intervention(
            args.root, "tp-eager-same-server-toggle", "toggle"
        ),
        "steady_cprofile_diagnostic": cprofile(args.root),
        "reclamation_diagnostics": [
            reclamation_diagnostic(args.root, "tp", "tp-batch1024-diagnostic"),
            reclamation_diagnostic(args.root, "dp", "dp-batch512-diagnostic"),
        ],
        "run_inventory": [
            {"run": p.parent.name, **read(p)}
            for p in sorted(args.root.glob("*/result.json"))
        ],
        "boundary": (
            "Evidence PASS is not a blanket SLO PASS. All pilots are retained. "
            "Formal ABBA excludes profilers and changes host placement only, "
            "equally for native and noop. CPU 20/21 is specific to this host. "
            "Pressure tuning keeps inputs, arrivals, SLO and retention unchanged."
        ),
    }
    output = args.root / "performance-followup-summary.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(output),
                "evidence_status": result["evidence_status"],
                "tp_eager_pooled": result["tp_eager_pooled"],
                "pressure_runs": len(pressure),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
