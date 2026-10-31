# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Verify DP2 HTTP evidence and aggregate by the observed native affinity."""

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
from dp_serving import aggregate
from serving_reclamation import quantiles

ROOT = Path("/data_ssd/qcrs/experiments/p1-v8/DP")


def read(path):
    return json.loads(path.read_text())


def preemptions(row, rank=None):
    return sum(
        value
        for key, value in row["counter_deltas"].items()
        if key.split("{")[0] == "vllm:num_preemptions_total"
        and (rank is None or f'engine="{rank}"' in key)
    )


def engineering_summary(root, diagnostic, output):
    runs = {}
    tokens = {}
    for run in sorted(root.iterdir()):
        rounds = read(run / "serving.json")
        requests = [
            item
            for row in rounds
            for item in read(run / f"requests-{row['repeat']}.json")
        ]
        protocol = read(run / "protocol.json")
        version = "d0" if "-d0-" in run.name else "d1"
        key = f"{version}-{protocol['args']['scenario']}"
        tokens[key] = [item["token_ids"] for item in requests]
        runs[key] = dict(
            directory=str(run),
            guard=read(run / "result.json"),
            rounds=rounds,
            aggregate=aggregate(requests, sum(r["elapsed_s"] for r in rounds)),
            replicas={
                str(rank): dict(
                    prompt_tokens=sum(
                        r["prompt_tokens"] for r in requests if r["replica"] == rank
                    ),
                    **aggregate(
                        [r for r in requests if r["replica"] == rank],
                        sum(r["elapsed_s"] for r in rounds),
                    ),
                )
                for rank in (0, 1)
            },
            requests=[
                {k: v for k, v in r.items() if k not in ("itl_s", "token_ids", "usage")}
                for r in requests
            ],
            max_ttft_s=max(r["ttft_s"] for r in requests),
            max_tpot_s=max(r["tpot_s"] for r in requests),
            preemptions=sum(preemptions(r) for r in rounds),
        )
        # Requests use round-relative clocks; only each round has an active span.
        runs[key]["aggregate"].pop("active_span_s")
        for replica in runs[key]["replicas"].values():
            replica.pop("active_span_s")

    events = sorted(
        (
            json.loads(line)
            for f in diagnostic.glob("dp-load-*.jsonl")
            for line in f.read_text().splitlines()
        ),
        key=lambda r: r["time_ns"],
    )
    routes = {
        r["request"]: r
        for r in map(
            json.loads, (diagnostic / "routing.jsonl").read_text().splitlines()
        )
    }
    publications, steps, frontend, replay = [], {}, None, []
    propagation, delivery = [], []
    seen_generation = set()
    changed_delivery = []
    for event in events:
        site, now = event["site"], event["time_ns"]
        if site == "coordinator_publish":
            publications.append(event)
        elif site == "scheduler_completed":
            steps[event["rank"]] = event
        elif site == "frontend_receive":
            matching = next(
                (p for p in reversed(publications) if p["counts"] == event["counts"]),
                None,
            )
            frontend = dict(event, publication=matching)
            if matching:
                delivery.append((now - matching["time_ns"]) / 1e6)
                propagation.extend(
                    (matching["time_ns"] - s["generated_ns"]) / 1e6
                    for s in matching["latest"].values()
                )
                for rank, state in matching["latest"].items():
                    identity = (rank, state["generated_ns"])
                    if identity not in seen_generation:
                        seen_generation.add(identity)
                        changed_delivery.append(
                            dict(
                                rank=rank,
                                counts=state["counts"],
                                generated_ns=state["generated_ns"],
                                generation_to_publish_ms=(
                                    matching["time_ns"] - state["generated_ns"]
                                )
                                / 1e6,
                                generation_to_frontend_ms=(now - state["generated_ns"])
                                / 1e6,
                            )
                        )
        elif site == "route" and len(steps) == 2:
            C = [max(4 * w + r, f) for (w, r), f in zip(event["counts"], event["F"])]
            scores = [
                (C[i] + steps[i]["U"] / 512) * (1 + steps[i]["usage"]) for i in (0, 1)
            ]
            order = [(event["start"] + i) % 2 for i in (0, 1)]
            chosen = routes[event["request"]]["replica"]
            assert min(order, key=lambda i: C[i]) == chosen
            latest = frontend["publication"]["latest"]
            replay.append(
                dict(
                    external=event["external"],
                    time_ns=now,
                    counts=event["counts"],
                    local_inflight=event["F"],
                    C=C,
                    D1_choice=chosen,
                    D2_score=scores,
                    D2_choice=min(order, key=lambda i: scores[i]),
                    U=[steps[i]["U"] for i in (0, 1)],
                    usage=[steps[i]["usage"] for i in (0, 1)],
                    poststep_age_ms=[(now - steps[i]["time_ns"]) / 1e6 for i in (0, 1)],
                    snapshot_generation_age_ms=[
                        (now - latest[str(i)]["generated_ns"]) / 1e6
                        if str(i) in latest
                        else None
                        for i in (0, 1)
                    ],
                    frontend_age_ms=(now - frontend["time_ns"]) / 1e6,
                )
            )
    result = dict(
        runs=runs,
        token_identity={
            s: sum(a == b for a, b in zip(tokens[f"d0-{s}"], tokens[f"d1-{s}"]))
            for s in ("native", "uniform")
        },
        diagnostic=dict(
            directory=str(diagnostic),
            serving=read(diagnostic / "serving.json"),
            coordinator_state_age_ms=quantiles(propagation),
            coordinator_to_frontend_ms=quantiles(delivery),
            changed_generation_first_delivery=changed_delivery,
            changed_generation_latency_ms={
                name: quantiles([item[name] for item in changed_delivery])
                for name in (
                    "generation_to_publish_ms",
                    "generation_to_frontend_ms",
                )
            },
            final_publication=publications[-1],
            final_frontend=frontend,
            final_poststep=steps,
            replay=replay,
            replay_scope="Last observed poststep U/usage, not simulated D2 dynamics",
        ),
    )
    output.write_text(json.dumps(result, indent=2))
    print(output)


def summarize_local():
    evidence, failures = {}, []
    for run in sorted(ROOT.glob("*-local-pressure-*")):
        guard = read(run / "result.json")
        if guard["status"] != "PASS":
            failures.append({"directory": run.name, "guard": guard})
            continue
        item = read(run / "evidence.json")
        assert item["status"] == "PASS" and item["free_final"] == [32, 32]
        policy = item["policy"]
        assert policy not in evidence, (policy, run)
        evidence[policy] = {"directory": run.name, "guard": guard, "evidence": item}
    assert evidence.keys() == {"full", "sink_recent"}
    native, reclaimed = (evidence[p]["evidence"] for p in ("full", "sink_recent"))
    peer = native["outputs"]["B"]["tokens"]
    assert len(peer) == 128 and peer == reclaimed["outputs"]["B"]["tokens"]
    result = {
        "status": "PASS",
        "peer_output_tokens_equal": len(peer),
        "peer_output_sha256": hashlib.sha256(json.dumps(peer).encode()).hexdigest(),
        "runs": list(evidence.values()),
        "failed_runs": failures,
    }
    (ROOT / "correctness-summary.json").write_text(json.dumps(result, indent=2))
    return result


def main():
    expected = (
        {
            ("native", pool, policy)
            for pool in (4, 1.5)
            for policy in ("full", "noop", "sink_recent")
        }
        | {("native", 1.5, p) for p in ("random", "triattention")}
        | {("skewed", 1.5, p) for p in ("full", "sink_recent")}
    )
    runs, failed_runs = {}, []
    for run in sorted(ROOT.glob("*-http-*")):
        protocol = read(run / "protocol.json")
        args = protocol["args"]
        if args["repeats"] != 3 or args["trace"]:
            continue
        key = (args["scenario"], args["pool_gib"], args["policy"])
        guard = read(run / "result.json")
        if guard["status"] != "PASS":
            failed_runs.append(
                {"directory": run.name, "config": list(key), "guard": guard}
            )
            continue
        assert key in expected and key not in runs, (run, key)
        rounds = read(run / "serving.json")
        assert guard["status"] == "PASS" and not guard["limit_triggered"]
        assert guard["peak_project_nvml_bytes"] < 44 * 2**30
        assert len(rounds) == 3 and (run / "warmup-summary.json").exists()
        lengths = [4096, 8192] * 4
        if key[0] == "skewed":
            lengths[-2:] = [128, 256]
        prompts = read(run / "prompts.json")
        assert [len(p) for p in prompts] == lengths
        assert protocol["prompt_sha256"] == [
            hashlib.sha256(json.dumps(p).encode()).hexdigest() for p in prompts
        ]
        assert protocol["arrivals_s"] == [0, 0.3, 0.6, 0.9, 5, 5.3, 5.6, 5.9]
        routes = [
            json.loads(line)
            for line in (run / "routing.jsonl").read_text().splitlines()
        ]
        assert len(routes) == 33  # One small warmup and four eight-request rounds.
        costs = read(run / "routing-costs.json")
        assert costs["calls"] == len(routes) == len(costs["wrapper_including_write_ns"])
        timed_route_ids = {
            f"cmpl-r{repeat}-{i}-0" for repeat in range(3) for i in range(8)
        }
        timed_costs = [
            (route["route_cpu_ns"], cost)
            for route, cost in zip(routes, costs["wrapper_including_write_ns"])
            if route["external"] in timed_route_ids
        ]
        assert len(timed_costs) == 24
        pooled = []
        for row in rounds:
            items = read(run / f"requests-{row['repeat']}.json")
            assert len(items) == 8 and [i["prompt_tokens"] for i in items] == lengths
            for item in items:
                assert item["replica"] in (0, 1)
                assert (
                    len(item["token_ids"]) == item["usage"]["completion_tokens"] == 128
                )
                assert item["usage"]["prompt_tokens"] == item["prompt_tokens"]
                route = [
                    r for r in routes if r["external"] == f"cmpl-{item['request']}-0"
                ]
                assert len(route) == 1 and route[0]["replica"] == item["replica"]
                assert route[0]["explicit"] == (
                    (0 if int(item["request"].split("-")[-1]) < 6 else 1)
                    if key[0] == "skewed"
                    else None
                )
            for rank in (None, 0, 1):
                selected = (
                    items
                    if rank is None
                    else [i for i in items if i["replica"] == rank]
                )
                calculated = aggregate(selected, row["elapsed_s"])
                saved = row["aggregate"] if rank is None else row["replicas"][str(rank)]
                assert saved == calculated
                saved["preemptions"] = preemptions(row, rank)
            assert math.isclose(
                sum(i["output_tokens_per_s"] for i in row["replicas"].values()),
                row["aggregate"]["output_tokens_per_s"],
            )
            pooled.extend(items)
        elapsed = sum(row["elapsed_s"] for row in rounds)
        aggregates = {}
        for rank in (None, 0, 1):
            selected = (
                pooled if rank is None else [i for i in pooled if i["replica"] == rank]
            )
            result = aggregate(selected, elapsed)
            result.pop("active_span_s")  # The per-round clocks have separate origins.
            result["prompt_tokens"] = sum(i["prompt_tokens"] for i in selected)
            result["preemptions"] = sum(preemptions(row, rank) for row in rounds)
            aggregates["all" if rank is None else str(rank)] = result
        lines = (run / "server.log").read_text().splitlines()
        metrics = [i for i, line in enumerate(lines) if '"GET /metrics HTTP' in line]
        assert len(metrics) == 8
        warnings = [
            {"line": line, "phase": "warmup" if i < metrics[2] else "timed"}
            for i, line in enumerate(lines)
            if "Triton kernel JIT compilation during inference" in line
        ]
        runs[key] = {
            "directory": run.name,
            "config": list(key),
            "guard": guard,
            "prompt_sha256": protocol["prompt_sha256"],
            "rounds": rounds,
            "aggregate": aggregates,
            "routing_cpu_us": quantiles([r["route_cpu_ns"] / 1e3 for r in routes]),
            "observation_including_route_us": quantiles(
                [c / 1e3 for c in costs["wrapper_including_write_ns"]]
            ),
            "timed_routing_calls": len(timed_costs),
            "timed_routing_cpu_us": quantiles([c[0] / 1e3 for c in timed_costs]),
            "timed_observation_including_route_us": quantiles(
                [c[1] / 1e3 for c in timed_costs]
            ),
            "jit_warnings": warnings,
        }
    assert runs.keys() == expected, expected - runs.keys()
    identities, comparisons = [], []
    for key, native in runs.items():
        scenario, pool, policy = key
        if policy != "full":
            continue
        if scenario == "native":
            noop = runs[scenario, pool, "noop"]
            equal = []
            for repeat in range(3):
                lhs = read(ROOT / native["directory"] / f"requests-{repeat}.json")
                rhs = read(ROOT / noop["directory"] / f"requests-{repeat}.json")
                equal.extend(a["token_ids"] == b["token_ids"] for a, b in zip(lhs, rhs))
            identities.append(
                {
                    "pool_gib": pool,
                    "equal_outputs": sum(equal),
                    "total": len(equal),
                    "status": "PASS" if all(equal) else "FAIL",
                }
            )
        for other_key, candidate in runs.items():
            if other_key[:2] != key[:2] or other_key[2] == "full":
                continue
            assert native["prompt_sha256"] == candidate["prompt_sha256"]
            base, result = native["aggregate"]["all"], candidate["aggregate"]["all"]
            noop_overhead = None
            if other_key[2] == "noop":
                noop_overhead = 100 * (
                    1
                    - np.mean(
                        [
                            r["aggregate"]["output_tokens_per_s"]
                            for r in candidate["rounds"]
                        ]
                    )
                    / np.mean(
                        [
                            r["aggregate"]["output_tokens_per_s"]
                            for r in native["rounds"]
                        ]
                    )
                )
            comparisons.append(
                {
                    "config": list(other_key),
                    "noop_mean_overhead_percent": noop_overhead,
                    "noop_budget_status": None
                    if noop_overhead is None
                    else ("PASS" if noop_overhead <= 5 else "FAIL"),
                    "throughput_change_percent": 100
                    * (result["output_tokens_per_s"] / base["output_tokens_per_s"] - 1),
                    "all_requests_within_slo": result["slo_good_requests"] == 24,
                    "goodput_improved": result["slo_goodput_requests_per_s"]
                    > base["slo_goodput_requests_per_s"],
                }
            )
    diagnostics, diagnostic_keys = [], set()
    expected_diagnostics = {
        (scenario, 1.5, policy)
        for scenario in ("native", "skewed")
        for policy in ("full", "sink_recent")
    } | {("native", 4, "sink_recent")}
    for run in sorted(ROOT.glob("*-http-*-diagnostic")):
        args = read(run / "protocol.json")["args"]
        key = (args["scenario"], args["pool_gib"], args["policy"])
        assert key in expected_diagnostics and key not in diagnostic_keys, run
        assert args["trace"] and args["repeats"] == 1
        guard = read(run / "result.json")
        assert guard["status"] == "PASS" and not guard["limit_triggered"], run
        assert guard["peak_project_nvml_bytes"] < 44 * 2**30
        rounds = read(run / "serving.json")
        assert len(rounds) == 1
        diagnostic_keys.add(key)
        for rank in range(2):
            events = [
                json.loads(line)
                for line in (run / f"replica-{rank}" / "kv-trace.jsonl")
                .read_text()
                .splitlines()
            ]
            scheduler = [
                json.loads(line)
                for line in (run / f"replica-{rank}" / "scheduler.jsonl")
                .read_text()
                .splitlines()
            ]
            identity = scheduler[0]
            assert identity["event"] == "identity" and identity["engine"] == rank
            for row in [read(run / "warmup-summary.json"), *rounds]:
                phase = [
                    e
                    for e in events
                    if row["start_monotonic_ns"]
                    <= e["time_ns"]
                    <= row["end_monotonic_ns"]
                ]
                released = [
                    e["released_blocks"]
                    for e in phase
                    if e["event"] == "reclaim_commit"
                ]
                samples = [
                    e
                    for e in scheduler
                    if e["event"] == "schedule"
                    and row["start_monotonic_ns"]
                    <= e["time_ns"]
                    <= row["end_monotonic_ns"]
                ]
                assert samples, (run, rank, row["repeat"])
                intents = sum(e["intent"] is not None for e in samples)
                if key[1] == 4:
                    assert not intents and not released
                    assert not any(
                        e["event"] in ("allocation_miss", "reclaim_select")
                        for e in phase
                    )
                diagnostics.append(
                    {
                        "directory": run.name,
                        "config": list(key),
                        "replica": rank,
                        "repeat": row["repeat"],
                        "events": dict(Counter(e["event"] for e in phase)),
                        "cumulative_released_blocks": sum(released),
                        "released_per_commit_range": [min(released), max(released)]
                        if released
                        else None,
                        "free_blocks_range": [
                            min(e["free"] for e in samples),
                            max(e["free"] for e in samples),
                        ],
                        "total_blocks_including_null": identity["total_blocks"],
                        "occupied_blocks_peak": identity["total_blocks"]
                        - 1
                        - min(e["free"] for e in samples),
                        "max_waiting": max(e["waiting"] for e in samples),
                        "max_running": max(e["running"] for e in samples),
                        "reclaim_intent_schedule_steps": intents,
                        "retry_schedule_steps": sum(e["retry"] for e in samples),
                    }
                )
    assert diagnostic_keys == expected_diagnostics
    summary = {
        "status": "PASS" if all(i["status"] == "PASS" for i in identities) else "FAIL",
        "status_scope": "Evidence completeness and native/noop output identity",
        "correctness": summarize_local(),
        "runs": list(runs.values()),
        "identities": identities,
        "comparisons": comparisons,
        "diagnostics": diagnostics,
        "failed_runs": failed_runs,
    }
    (ROOT / "serving-summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "runs"}, indent=2))
    assert summary["status"] == "PASS", "Native/noop output identity failed"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--engineering", type=Path)
    parser.add_argument("--diagnostic", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.engineering:
        engineering_summary(args.engineering, args.diagnostic, args.output)
    else:
        main()
