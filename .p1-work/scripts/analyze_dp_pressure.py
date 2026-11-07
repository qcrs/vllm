# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Audit same-score usage evidence without treating trace as counterfactual runs."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def analyze(run):
    events = sorted(
        (
            json.loads(line)
            for path in run.glob("dp-load-*.jsonl")
            for line in path.read_text().splitlines()
        ),
        key=lambda e: e["time_ns"],
    )
    routes = [e for e in events if e["site"] == "route"]
    misses = [e for e in events if e["site"] == "allocation_miss"]
    actions = [e for e in events if e["site"] == "pressure_action"]
    audited = []
    for route in routes:
        scores = [
            c + (u + p) / 512
            for c, u, p in zip(route["C"], route["U"], route["pending_tokens"])
        ]
        assert scores == route["scores"]
        order = [(route["start"] + i) % 2 for i in range(2)]
        assert route["chosen"] == min(order, key=lambda i: scores[i])
        snapshots = route["snapshots"]
        valid = all(
            s is not None
            and 0
            < s["generated_ns"]
            <= s["coordinator_received_ns"]
            <= route["published_ns"]
            <= route["received_ns"]
            <= route["time_ns"]
            for s in snapshots
        )
        usage = [s["kv_cache_usage"] if s else None for s in snapshots]
        tied = route["tied"]
        high_usage = (
            valid
            and len(tied) > 1
            and usage[route["chosen"]] > min(usage[i] for i in tied)
        )
        following = [
            m
            for m in misses
            if m["request"] == route["request"] and m["time_ns"] >= route["time_ns"]
        ]
        audited.append(
            dict(
                external=route["external"],
                chosen=route["chosen"],
                scores=scores,
                usage=usage,
                valid_snapshot=valid,
                age_s=route["state_age_s"],
                recent_250ms_diagnostic=route["fresh_250ms_diagnostic"],
                tied=tied,
                high_usage_tie=high_usage,
                following_request_misses=len(following),
            )
        )
    for rank in range(2):
        completed = [
            e
            for e in events
            if e["site"] == "scheduler_completed" and e["rank"] == rank
        ]
        assert completed[-1]["counts"] == [0, 0]
        assert completed[-1]["snapshot"]["unfinished_tokens"] == 0
    summary = dict(
        run=str(run),
        routes=len(routes),
        sites=dict(Counter(e["site"] for e in events)),
        valid_snapshot_routes=sum(r["valid_snapshot"] for r in audited),
        score_ties=sum(len(r["tied"]) > 1 for r in audited),
        high_usage_ties=sum(r["high_usage_tie"] for r in audited),
        recent_high_usage_ties=sum(
            r["high_usage_tie"] and r["recent_250ms_diagnostic"] for r in audited
        ),
        allocation_misses=len(misses),
        pressure_actions=len(actions),
        decision="STOPPED" if not misses else "REQUIRES_CAUSAL_REVIEW",
        reason="No local miss; no repeated pressure-causing misroute evidence"
        if not misses
        else "Inspect same-score choices and contemporaneous peer capacity",
        freshness=(
            "Sequence/watermark consistency, no wall-age expiry; "
            "250ms is diagnostic only"
        ),
        routes_detail=audited,
    )
    summary["trace_sha256"] = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in run.glob("dp-load-*.jsonl")
    }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    output = analyze(args.run)
    (args.run / "pressure-analysis.json").write_text(json.dumps(output, indent=2))
    print(
        json.dumps(
            {
                k: v
                for k, v in output.items()
                if k not in ("routes_detail", "trace_sha256")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
