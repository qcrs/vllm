# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Audit protected capacity and waiting from an explicitly selected trace run."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def pages(tokens):
    return (tokens + 15) // 16


def analyze(run):
    protocol = read(run / "protocol.json")
    args = protocol["args"]
    assert args["trace"] and args["budget_mode"] == "protected"
    assert args["tp"] == 2 and args["pool_gib"] == 0.75
    assert protocol["slo"] == {"ttft_s": 2, "tpot_s": 0.08}
    invocation = read(run / "invocation.json")
    assert invocation["environment"]["CUDA_VISIBLE_DEVICES"] == "1,2"
    guard = read(run / "result.json")
    assert guard["status"] == "PASS" and not guard["limit_triggered"]
    assert guard["peak_project_nvml_bytes"] < 44 * 2**30
    hashes = read(run / "runtime-source-hashes.json")
    for name, expected in hashes.items():
        assert hashlib.sha256((run / "source" / name).read_bytes()).hexdigest() == (
            expected
        )
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected
    trace = [
        json.loads(line) for line in (run / "kv-trace.jsonl").read_text().splitlines()
    ]
    rounds = []
    for summary in read(run / "serving.json"):
        start, end = summary["start_monotonic_ns"], summary["end_monotonic_ns"]
        events = [e for e in trace if start <= e["time_ns"] <= end]
        plans, rejected = [], []
        for event in events:
            if event["event"] != "protected_plan":
                continue
            retry = event["retry_forecast"]
            assert retry is not None
            allocations = {
                a["request_id"]: a
                for a in retry["allocations"] + retry["following_allocations"]
            }
            capacities = []
            for c in event["candidates"]:
                source, logical, prefix = (
                    c[k]
                    for k in (
                        "source_effective_kv_len",
                        "logical_tokens",
                        "protected_prefix_tokens",
                    )
                )
                config = c["policy_binding"]["config"]
                floor = prefix + max(
                    config["min_keep_tokens"],
                    config["recent_tokens"] + max(0, config["sink_tokens"] - prefix),
                    int(config["retention"] * (source - prefix)),
                    (4 * (source - prefix) + 4) // 5,
                    (max(0, logical - prefix) + 1) // 2,
                )
                assert c["keep_tokens"] == floor
                allocation = allocations.get(c["request_id"], {})
                net = (
                    len(c["source_block_ids"])
                    - pages(floor + allocation.get("num_tokens", 0))
                    + allocation.get("required_blocks", 0)
                )
                capacities.append(dict(request=c["request_id"], floor=floor, net=net))
            upper = sum(sorted((c["net"] for c in capacities), reverse=True)[:2])
            if not event["plan"]:
                rejected.append(
                    dict(
                        time_ns=event["time_ns"],
                        requester=event["requester"],
                        shortfall=event["shortfall"],
                        max_two_victim_net_pages=upper,
                        capacity_insufficient=upper < event["shortfall"],
                        candidates=capacities,
                        retry_forecast=retry,
                    )
                )
                continue
            assert len(event["plan"]) <= 2
            selected = []
            for c in event["plan"]:
                candidate = next(
                    x for x in capacities if x["request"] == c["request_id"]
                )
                assert c["keep_tokens"] >= candidate["floor"]
                released = len(c["source_block_ids"]) - pages(c["keep_tokens"])
                assert c["released_blocks"] == released
                allocation = allocations.get(c["request_id"], {})
                net = (
                    len(c["source_block_ids"])
                    - pages(c["keep_tokens"] + allocation.get("num_tokens", 0))
                    + allocation.get("required_blocks", 0)
                )
                prefix = max(
                    c["protected_prefix_tokens"],
                    c["policy_binding"]["config"]["sink_tokens"],
                )
                aligned = (
                    c["keep_tokens"]
                    - (c["keep_tokens"] - c["source_effective_kv_len"] - prefix) % 16
                )
                selected.append(
                    dict(
                        request=c["request_id"],
                        source=c["source_effective_kv_len"],
                        floor=candidate["floor"],
                        keep=c["keep_tokens"],
                        released_pages=released,
                        net_pages=net,
                        zero_copy=c["zero_copy"],
                        copy_bytes=c["copy_bytes"],
                        aligned_keep_below_floor=aligned < candidate["floor"],
                        aligned_keep=aligned,
                    )
                )
            assert sum(c["net_pages"] for c in selected) >= event["shortfall"]
            plans.append(dict(shortfall=event["shortfall"], selected=selected))
        misses = [e for e in events if e["event"] == "allocation_miss"]
        waiting = [e for e in misses if e["request_status"] == "WAITING"]
        waits = []
        for rid in sorted({e["request_id"] for e in waiting}):
            checks = [e for e in waiting if e["request_id"] == rid]
            admitted = next(
                e
                for e in events
                if e["event"] == "allocation_done"
                and e["request_id"] == rid
                and e["time_ns"] > checks[0]["time_ns"]
            )
            waits.append(
                dict(
                    request=rid,
                    rejected_checks=len(checks),
                    first_failed_required_pages=checks[0]["required"],
                    first_failed_available_pages=checks[0]["available"],
                    first_failed_check_to_allocation_ms=(
                        admitted["time_ns"] - checks[0]["time_ns"]
                    )
                    / 1e6,
                )
            )
        rounds.append(
            dict(
                repeat=summary["repeat"],
                diagnostic_only=True,
                event_counts=dict(Counter(e["event"] for e in events)),
                miss_status=dict(Counter(e["request_status"] for e in misses)),
                rejected_plans=len(rejected),
                capacity_insufficient_plans=sum(
                    e["capacity_insufficient"] for e in rejected
                ),
                first_rejected_plan=rejected[0] if rejected else None,
                selected_plans=plans,
                waiting=waits,
            )
        )
    paths = [run / n for n in ("protocol.json", "kv-trace.jsonl", "serving.json")]
    return dict(
        run=str(run),
        guard=guard,
        source_hashes=hashes,
        rounds=rounds,
        artifact_sha256={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = analyze(args.run)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(
        json.dumps(
            [
                {
                    "repeat": r["repeat"],
                    "capacity_insufficient_plans": r["capacity_insufficient_plans"],
                }
                for r in output["rounds"]
            ]
        )
    )


if __name__ == "__main__":
    main()
