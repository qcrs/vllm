# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inspect kernel execution and rank waiting in a short eager HTTP trace."""

import argparse
import json
import sqlite3
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    db = sqlite3.connect(args.run / "trace.sqlite")
    marks = dict(
        db.execute("SELECT text,start FROM NVTX_EVENTS WHERE text LIKE 'metrics-%'")
    )
    start, end = marks["metrics-2-after"], marks["metrics-3-before"]
    kernels = db.execute(
        "SELECT k.deviceId,s.value,count(*),sum(k.end-k.start)/1e6 "
        "FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.shortName=s.id "
        "WHERE k.start>=? AND k.end<=? GROUP BY 1,2 ORDER BY 4 DESC",
        (start, end),
    ).fetchall()
    runtime = db.execute(
        "SELECT s.value,count(*),sum(r.end-r.start)/1e6 "
        "FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON r.nameId=s.id "
        "WHERE r.start>=? AND r.end<=? GROUP BY s.value ORDER BY 3 DESC",
        (start, end),
    ).fetchall()
    devices = []
    for rank in (0, 1):
        spans = db.execute(
            "SELECT start,end FROM CUPTI_ACTIVITY_KIND_KERNEL "
            "WHERE start>=? AND end<=? AND deviceId=? ORDER BY start",
            (start, end, rank),
        ).fetchall()
        merged = []
        for begin, finish in spans:
            if merged and begin <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], finish)
            else:
                merged.append([begin, finish])
        assert spans and end - merged[-1][1] < 250_000_000
        devices.append(
            {
                "device": rank,
                "kernels": len(spans),
                "kernel_sum_ms": sum(b - a for a, b in spans) / 1e6,
                "kernel_union_ms": sum(b - a for a, b in merged) / 1e6,
                "tail_ms": (end - merged[-1][1]) / 1e6,
            }
        )
    result = {
        "directory": args.run.name,
        "window_ms": (end - start) / 1e6,
        "devices": devices,
        "kernels": kernels,
        "runtime": runtime,
        "boundary": "Kernel time includes collective waiting, not just useful work",
    }
    (args.run / "eager-summary.json").write_text(json.dumps(result, indent=2))
    print(
        json.dumps({k: v for k, v in result.items() if k not in ("kernels", "runtime")})
    )


if __name__ == "__main__":
    main()
