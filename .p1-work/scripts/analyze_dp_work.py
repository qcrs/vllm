# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Recompute fixed DP3 observations; shadow choices are not counterfactual runs."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from dp_serving import aggregate
from summarize_dp_serving import preemptions


def read(path):
    return json.loads(path.read_text())


def events(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def distribution(values):
    if not values:
        return {"n": 0}
    return dict(
        n=len(values),
        **dict(
            zip(("min", "p50", "p95", "max"), np.percentile(values, [0, 50, 95, 100]))
        ),
    )


def analyze(run):
    protocol, guard = read(run / "protocol.json"), read(run / "result.json")
    assert guard["exit_code"] == 0
    observations = sorted(
        [e for path in run.glob("dp-load-*.jsonl") for e in events(path)],
        key=lambda e: e["time_ns"],
    )
    routes = [e for e in observations if e["site"] == "route"]
    received = [e for e in observations if e["site"] == "frontend_receive"]
    seen, previous = set(), {}
    first_deliveries, load_only = [], Counter()
    for event in received:
        for rank, snapshot in enumerate(event["snapshots"]):
            if snapshot is None:
                continue
            identity = (rank, snapshot["sequence"])
            if identity in seen:
                continue
            seen.add(identity)
            generated = snapshot["generated_ns"]
            coordinator = snapshot["coordinator_received_ns"]
            published, frontend = event["published_ns"], event["received_ns"]
            assert generated <= coordinator <= published <= frontend
            old = previous.get(rank)
            counts = event["counts"][rank]
            if old is not None:
                assert snapshot["sequence"] > old["snapshot"]["sequence"]
                if counts == old["counts"]:
                    load_only[rank] += 1
            previous[rank] = dict(snapshot=snapshot, counts=counts)
            first_deliveries.append(
                dict(
                    rank=rank,
                    sequence=snapshot["sequence"],
                    engine_to_coordinator_ms=(coordinator - generated) / 1e6,
                    coordinator_hold_ms=(published - coordinator) / 1e6,
                    publish_to_frontend_ms=(frontend - published) / 1e6,
                    generation_to_frontend_ms=(frontend - generated) / 1e6,
                )
            )
    assert all(v["counts"] == [0, 0] for v in previous.values())
    assert all(v["snapshot"]["unfinished_tokens"] == 0 for v in previous.values())
    traces = [events(run / f"replica-{rank}" / "scheduler.jsonl") for rank in range(2)]
    rounds = []
    for summary in [read(run / "warmup-summary.json"), *read(run / "serving.json")]:
        label = summary["repeat"]
        requests = read(run / f"requests-{label}.json")
        selected_routes = [
            e for e in routes if e["external"].startswith(f"cmpl-r{label}-")
        ]
        assert len(selected_routes) == len(requests) == 8
        by_id = {f"cmpl-{r['request']}-0": r for r in requests}
        rows, work = [], [Counter(), Counter()]
        high_water = {}
        for rank, trace in enumerate(traces):
            for step in trace:
                if step["event"] != "schedule":
                    continue
                if not (
                    summary["start_monotonic_ns"]
                    <= step["time_ns"]
                    <= summary["end_monotonic_ns"]
                ):
                    continue
                for rid, q in step["scheduled"].items():
                    match = next(
                        (r for r in selected_routes if r["request"] == rid), None
                    )
                    assert match is not None
                    prompt = by_id[match["external"]]["prompt_tokens"]
                    end = min(step["requests"][rid]["L"], prompt)
                    start = min(step["requests"][rid]["L"] - q, prompt)
                    work[rank]["scheduled_tokens"] += q
                    work[rank]["prefill_tokens"] += end - start
                    work[rank]["recomputed_prefill_tokens"] += max(
                        0, min(end, high_water.get(rid, 0)) - start
                    )
                    high_water[rid] = max(end, high_water.get(rid, 0))
        for route in selected_routes:
            rank, rid = route["chosen"], route["request"]
            item = by_id[route["external"]]
            first = next(
                e
                for e in traces[rank]
                if e["event"] == "schedule" and rid in e["scheduled"]
            )
            state = []
            for replica in range(2):
                history = [
                    e
                    for e in observations
                    if e["site"] == "scheduler_completed"
                    and e["rank"] == replica
                    and e["time_ns"] <= route["time_ns"]
                ]
                state.append(history[-1] if history else None)
            qualifies = (
                route["fresh"]
                and max(route["C"]) - min(route["C"]) <= 1
                and max(route["U"]) - min(route["U"])
                >= protocol["args"]["max_num_batched_tokens"]
            )
            rows.append(
                dict(
                    request=item["request"],
                    internal=rid,
                    prompt=item["prompt_tokens"],
                    chosen=rank,
                    shadow=route["shadow"],
                    C=route["C"],
                    U=route["U"],
                    fresh=route["fresh"],
                    state_age_s=route["state_age_s"],
                    source_sequences=[
                        s["sequence"] if s else None for s in route["snapshots"]
                    ],
                    qualifies=qualifies,
                    fresh_shadow_difference=qualifies and rank != route["shadow"],
                    route_to_first_schedule_ms=(first["time_ns"] - route["time_ns"])
                    / 1e6,
                    actual_post_output_before_route=state,
                    ttft_s=item["ttft_s"],
                    tpot_s=item["tpot_s"],
                )
            )
        rounds.append(dict(summary=summary, routes=rows, work=work))
    paths = [
        run / "protocol.json",
        run / "prompts.json",
        run / "result.json",
        run / "serving.json",
        run / "warmup-summary.json",
        run / "dp-source-hashes.json",
        *run.glob("requests-*.json"),
        *run.glob("dp-load-*.jsonl"),
        *run.glob("replica-*/scheduler.jsonl"),
    ]
    return dict(
        run=str(run),
        guard=guard,
        protocol=protocol,
        message_counts=dict(Counter(e["site"] for e in observations)),
        load_only_first_deliveries=dict(load_only),
        final_snapshots=previous,
        first_delivery_latency_ms={
            key: distribution([r[key] for r in first_deliveries])
            for key in (
                "engine_to_coordinator_ms",
                "coordinator_hold_ms",
                "publish_to_frontend_ms",
                "generation_to_frontend_ms",
            )
        },
        route_snapshot_age_ms=distribution(
            [age * 1000 for r in routes for age in r["state_age_s"] if age is not None]
        ),
        rounds=rounds,
        source_hashes=read(run / "dp-source-hashes.json"),
        artifact_sha256={
            str(path.relative_to(run)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths
        },
    )


def analyze_formal(run):
    """Audit only the explicitly supplied fixed W1/W0 run, including warmup."""
    protocol, guard = read(run / "protocol.json"), read(run / "result.json")
    args = protocol["args"]
    assert guard["status"] == "PASS" and not guard["limit_triggered"]
    assert guard["peak_project_nvml_bytes"] < 44 * 2**30
    assert args["scenario"] in ("w1", "w0") and args["repeats"] == 3
    assert not args["trace"] and not args["load_diagnostic"] and args["work_counters"]
    assert args["pool_gib"] == 4 and args["policy"] == "sink_recent"
    assert args["max_num_batched_tokens"] == 512 and args["worker_cpus"] == [20, 21]
    lengths = [8192, 512] * 4 if args["scenario"] == "w1" else [4096] * 8
    assert protocol["prompt_tokens"] == lengths
    assert protocol["arrivals_s"] == [i * 0.05 for i in range(8)]
    assert protocol["prompt_sha256"] == [
        hashlib.sha256(json.dumps(p).encode()).hexdigest()
        for p in read(run / "prompts.json")
    ]
    routes = events(run / "routing.jsonl")
    work = [events(run / f"replica-{rank}/work.jsonl") for rank in range(2)]
    assert len(routes) == sum(map(len, work)) == 33
    assert len({r["request"] for r in routes}) == 33
    for route in routes:
        assert route["explicit"] is None
        scores = [max(4 * w + r, f) for (w, r), f in zip(route["counts"], route["F"])]
        if args["routing"] == "known_work":
            assert route["batch"] == 512
            scores = [
                c + (s["unfinished_tokens"] / 512 if s else 0)
                for c, s in zip(scores, route["snapshots"])
            ]
        else:
            assert route["batch"] is None
        order = [(route["scan_start"] + i) % 2 for i in range(2)]
        assert route["replica"] == min(order, key=lambda rank: scores[rank])
    rounds, pooled, outputs = [], [], {}
    for row in [read(run / "warmup-summary.json"), *read(run / "serving.json")]:
        items = read(run / f"requests-{row['repeat']}.json")
        assert [r["prompt_tokens"] for r in items] == lengths
        assert row["aggregate"] == aggregate(items, row["elapsed_s"])
        actual = [Counter(), Counter()]
        observations = []
        for item in items:
            assert len(item["token_ids"]) == item["usage"]["completion_tokens"] == 128
            route = next(
                r for r in routes if r["external"] == f"cmpl-{item['request']}-0"
            )
            rank = item["replica"]
            assert route["replica"] == rank
            recorded = [w for w in work[rank] if w["request"] == route["request"]]
            assert len(recorded) == 1
            record = recorded[0]
            assert record["prompt_tokens"] == item["prompt_tokens"]
            assert record["scheduled_tokens"] == (
                item["prompt_tokens"] + 127 + record["recomputed_tokens"]
            )
            actual[rank].update(
                {
                    k: record[k]
                    for k in ("scheduled_tokens", "prefill_tokens", "recomputed_tokens")
                }
            )
            observations.append(
                dict(
                    request=item["request"],
                    replica=rank,
                    C=[
                        max(4 * w + r, f)
                        for (w, r), f in zip(route["counts"], route["F"])
                    ],
                    U=[
                        s["unfinished_tokens"] if s else None
                        for s in route["snapshots"]
                    ],
                    age_ms=[
                        (route["used_ns"] - s["generated_ns"]) / 1e6 if s else None
                        for s in route["snapshots"]
                    ],
                    ttft_s=item["ttft_s"],
                    tpot_s=item["tpot_s"],
                )
            )
            outputs[item["request"]] = item["token_ids"]
        for rank in range(2):
            reported_prompt = sum(
                v
                for k, v in row["counter_deltas"].items()
                if k.startswith("vllm:prompt_tokens_total{") and f'engine="{rank}"' in k
            )
            if actual[rank]["recomputed_tokens"] == 0:
                assert reported_prompt == actual[rank]["prefill_tokens"]
        rounds.append(
            dict(
                summary=row,
                work=actual,
                preemptions=preemptions(row),
                routes=observations,
            )
        )
        if row["repeat"] != "warmup":
            pooled.extend(items)
    elapsed = sum(r["summary"]["elapsed_s"] for r in rounds[1:])
    combined = aggregate(pooled, elapsed)
    combined.pop("active_span_s")
    costs = read(run / "routing-costs.json")["wrapper_including_write_ns"]
    assert len(costs) == 33
    paths = (
        [
            run / name
            for name in (
                "protocol.json",
                "prompts.json",
                "result.json",
                "routing.jsonl",
                "routing-costs.json",
                "serving.json",
                "warmup-summary.json",
                "dp-source-hashes.json",
                "dp_hooks.py",
                "dp_server.py",
                "source/.p1-work/scripts/dp_serving.py",
            )
        ]
        + list(run.glob("requests-*.json"))
        + list(run.glob("replica-*/work.jsonl"))
    )
    return dict(
        run=str(run),
        guard=guard,
        protocol=protocol,
        rounds=rounds,
        aggregate=combined,
        max_ttft_s=max(r["ttft_s"] for r in pooled),
        max_tpot_s=max(r["tpot_s"] for r in pooled),
        routing_cpu_us=distribution([r["route_cpu_ns"] / 1e3 for r in routes]),
        observation_cpu_us=distribution([c / 1e3 for c in costs]),
        outputs=outputs,
        source_hashes=read(run / "dp-source-hashes.json"),
        artifact_sha256={
            str(p.relative_to(run)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in paths
        },
    )


def compare_formal(runs):
    pairs = {}
    for run in runs:
        args = run["protocol"]["args"]
        key = (args["scenario"], "reverse" in Path(run["run"]).name)
        pair = pairs.setdefault(key, {})
        assert args["routing"] not in pair
        pair[args["routing"]] = run
    comparisons = []
    for (scenario, reverse), pair in pairs.items():
        if len(pair) != 2:
            continue
        a, b = pair["requests"], pair["known_work"]
        assert a["source_hashes"] == b["source_hashes"]
        assert a["protocol"]["prompt_sha256"] == b["protocol"]["prompt_sha256"]
        ca, cb = (dict(r["protocol"]["args"]) for r in (a, b))
        ca.pop("routing")
        cb.pop("routing")
        assert ca == cb
        for helper in (
            "source/.p1-work/scripts/dp_serving.py",
            "dp_hooks.py",
            "dp_server.py",
        ):
            assert a["artifact_sha256"][helper] == b["artifact_sha256"][helper]
        comparisons.append(
            dict(
                scenario=scenario,
                reverse=reverse,
                A=a["run"],
                B=b["run"],
                throughput_change_percent=100
                * (
                    b["aggregate"]["output_tokens_per_s"]
                    / a["aggregate"]["output_tokens_per_s"]
                    - 1
                ),
                goodput_change_percent=100
                * (
                    b["aggregate"]["slo_goodput_requests_per_s"]
                    / a["aggregate"]["slo_goodput_requests_per_s"]
                    - 1
                ),
                rounds_throughput={
                    r["protocol"]["args"]["routing"]: [
                        v["summary"]["aggregate"]["output_tokens_per_s"]
                        for v in r["rounds"][1:]
                    ]
                    for r in (a, b)
                },
                identical_timed_outputs=sum(
                    a["outputs"][rid] == b["outputs"][rid]
                    for rid in a["outputs"]
                    if "warmup" not in rid
                ),
            )
        )
    return comparisons


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--formal", action="store_true")
    args = parser.parse_args()
    if args.formal:
        runs = [analyze_formal(run) for run in args.runs]
        result = dict(runs=runs, comparisons=compare_formal(runs))
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result["comparisons"], indent=2))
        return
    result = [analyze(run) for run in args.runs]
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for item in result:
        print(
            json.dumps(
                dict(
                    run=item["run"],
                    rounds=[
                        dict(
                            repeat=r["summary"]["repeat"],
                            routes=[
                                (v["chosen"], v["shadow"], v["qualifies"])
                                for v in r["routes"]
                            ],
                        )
                        for r in item["rounds"]
                    ],
                )
            )
        )


if __name__ == "__main__":
    main()
