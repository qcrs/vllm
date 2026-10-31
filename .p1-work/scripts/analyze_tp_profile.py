# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Join diagnostic collective NVTX, CUDA launch and NCCL device activity."""

import argparse
import json
import sqlite3
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    db = sqlite3.connect(args.run / "trace.sqlite")
    marks = dict(
        db.execute("SELECT text,start FROM NVTX_EVENTS WHERE text LIKE 'metrics-%'")
    )
    timed = bool(marks)
    begin, end = (
        (marks["metrics-2-after"], marks["metrics-3-before"])
        if timed
        else (
            0,
            db.execute("SELECT max(end) FROM CUPTI_ACTIVITY_KIND_KERNEL").fetchone()[0],
        )
    )
    ranges = db.execute(
        "SELECT n.start,n.end,n.globalTid,coalesce(n.text,s.value) "
        "FROM NVTX_EVENTS n LEFT JOIN StringIds s ON n.textId=s.id "
        "WHERE coalesce(n.text,s.value) LIKE 'p1-tp|%' AND n.start>=? AND n.end<=?",
        (begin, end),
    ).fetchall()
    assert ranges, "No TP diagnostic annotations"
    grouped, details, missing = defaultdict(list), [], []
    for start, finish, tid, text in ranges:
        _, rank, sequence, kind, size = text.split("|")
        if kind == "reclaim":
            continue
        kernels = db.execute(
            "SELECT k.start,k.end,k.deviceId,s.value,k.correlationId "
            "FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.shortName=s.id "
            "JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON r.correlationId=k.correlationId "
            "AND (r.globalTid & -16777216)=k.globalPid "
            "WHERE r.globalTid=? AND r.start>=? AND r.end<=? "
            "AND lower(s.value) LIKE '%nccl%'",
            (tid, start, finish),
        ).fetchall()
        if not kernels:
            missing.append(text)
        item = {
            "rank": int(rank),
            "sequence": int(sequence),
            "kind": kind,
            "logical_payload_bytes": int(size),
            "host_ms": (finish - start) / 1e6,
            "nccl_device_ms": sum(b - a for a, b, *_ in kernels) / 1e6,
            "kernels": kernels,
        }
        details.append(item)
        grouped[rank, kind].append(item)
    assert {r for r, _ in grouped} == {"0", "1"}
    summary = []
    for (rank, kind), items in sorted(grouped.items()):
        summary.append(
            {
                "rank": int(rank),
                "kind": kind,
                "calls": len(items),
                "logical_payload_bytes": sum(i["logical_payload_bytes"] for i in items),
                "host_sum_ms": sum(i["host_ms"] for i in items),
                "nccl_device_sum_ms": sum(i["nccl_device_ms"] for i in items),
                "nccl_kernels": sum(len(i["kernels"]) for i in items),
            }
        )
    last = db.execute(
        "SELECT deviceId,max(end) FROM CUPTI_ACTIVITY_KIND_KERNEL "
        "WHERE start>=? AND end<=? GROUP BY deviceId",
        (begin, end),
    ).fetchall()
    assert len(last) == 2
    if timed:
        assert all(end - t < 250_000_000 for _, t in last), "CUDA tail truncated"
    launches = (
        db.execute(
            "SELECT r.correlationId,r.globalTid,r.start "
            "FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON r.nameId=s.id "
            "LEFT JOIN CUPTI_ACTIVITY_KIND_GRAPH_TRACE g "
            "ON r.correlationId=g.correlationId "
            "AND (r.globalTid & -16777216)=g.globalPid "
            "WHERE s.value LIKE 'cudaGraphLaunch%' AND r.start>=? AND r.end<=? "
            "AND g.start IS NULL",
            (begin, end),
        ).fetchall()
        if timed
        else []
    )
    result = {
        "status": "PARTIAL_GPU_SPANS" if missing or launches else "PASS",
        "window": "first timed Serving round" if timed else "entire trajectory",
        "window_ns": [begin, end],
        "collectives": summary,
        "missing_collective_spans": missing,
        "missing_graph_spans": launches,
        "last_kernel_to_end_ms": [(d, (end - t) / 1e6) for d, t in last],
        "details": details,
        "boundary": (
            "Logical bytes per rank, not bus traffic; device time includes rank wait"
        ),
    }
    (args.run / "profile-summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "details"}, indent=2))


if __name__ == "__main__":
    main()
