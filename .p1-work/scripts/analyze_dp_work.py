# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Recompute fixed DP3 observations; shadow choices are not counterfactual runs."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
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
