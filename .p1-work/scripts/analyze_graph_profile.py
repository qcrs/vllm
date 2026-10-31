# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Summarize an NSYS HTTP diagnostic inside its timed NVTX window."""

import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    connection = sqlite3.connect(args.run / "trace.sqlite")
    tables = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }
    for suffix in ("RUNTIME", "KERNEL", "GRAPH_TRACE", "MEMCPY"):
        assert "CUPTI_ACTIVITY_KIND_" + suffix in tables, suffix
    marks = dict(
        connection.execute(
            "SELECT text,start FROM NVTX_EVENTS WHERE text LIKE 'metrics-%'"
        )
    )
    start, end = marks["metrics-2-after"], marks["metrics-3-before"]
    runtime = connection.execute(
        "SELECT s.value,count(*),sum(r.end-r.start)/1e6 "
        "FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON r.nameId=s.id "
        "WHERE r.start>=? AND r.end<=? GROUP BY s.value ORDER BY 3 DESC",
        (start, end),
    ).fetchall()
    kernels = connection.execute(
        "SELECT s.value,count(*),sum(k.end-k.start)/1e6 "
        "FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.shortName=s.id "
        "WHERE k.start>=? AND k.end<=? GROUP BY s.value ORDER BY 3 DESC",
        (start, end),
    ).fetchall()
    copies = connection.execute(
        "SELECT copyKind,count(*),sum(bytes),sum(end-start)/1e6 "
        "FROM CUPTI_ACTIVITY_KIND_MEMCPY WHERE start>=? AND end<=? GROUP BY copyKind",
        (start, end),
    ).fetchall()
    intervals = []
    activities = {}
    for name in ("KERNEL", "GRAPH_TRACE", "MEMCPY", "MEMSET"):
        table = "CUPTI_ACTIVITY_KIND_" + name
        rows = (
            connection.execute(
                f"SELECT start,end FROM {table} "
                "WHERE start>=? AND end<=? ORDER BY start",
                (start, end),
            ).fetchall()
            if table in tables
            else []
        )
        activities[name] = {
            "count": len(rows),
            "sum_ms": sum(b - a for a, b in rows) / 1e6,
        }
        intervals.extend(rows)
    merged = []
    for a, b in sorted(intervals):
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    gaps = [merged[i][0] - merged[i - 1][1] for i in range(1, len(merged))]
    busy = sum(b - a for a, b in merged)
    last_activity = max(b for a, b in intervals)
    assert end - last_activity < 250_000_000, "CUDA trace ends before workload"
    graph_launches = next(
        n for name, n, _ in runtime if name.startswith("cudaGraphLaunch")
    )
    missing_graphs = connection.execute(
        "SELECT r.correlationId,r.start FROM CUPTI_ACTIVITY_KIND_RUNTIME r "
        "JOIN StringIds s ON r.nameId=s.id "
        "LEFT JOIN CUPTI_ACTIVITY_KIND_GRAPH_TRACE g "
        "ON r.correlationId=g.correlationId "
        "WHERE s.value LIKE 'cudaGraphLaunch%' AND r.start>=? AND r.end<=? "
        "AND g.start IS NULL",
        (start, end),
    ).fetchall()
    summary = {
        "status": "PASS" if not missing_graphs else "PARTIAL_GPU_SPANS",
        "missing_graph_spans_correlation_start": missing_graphs,
        "graph_launch_api_count": graph_launches,
        "last_activity_to_window_end_ms": (end - last_activity) / 1e6,
        "diagnostics": connection.execute(
            "SELECT text FROM DIAGNOSTIC_EVENT WHERE text LIKE '%CUDA%'"
        ).fetchall(),
        "window_ns": [start, end],
        "window_ms": (end - start) / 1e6,
        "runtime_name_count_sum_ms": runtime,
        "kernel_name_count_sum_ms": kernels,
        "memcpy_kind_count_bytes_sum_ms": copies,
        "activities": activities,
        "gpu_activity_union_ms": busy / 1e6,
        "outside_activity_union_ms": (end - start - busy) / 1e6,
        "inter_activity_gap_ms": dict(
            zip(
                ("p50", "p95", "p99", "max"),
                np.percentile(gaps, [50, 95, 99, 100]) / 1e6,
            )
        ),
        "boundary": "GPU union is a lower bound; gaps include arrivals/CPU/HTTP wait",
    }
    (args.run / "profile-summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
