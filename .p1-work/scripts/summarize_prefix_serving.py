# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Summarize the fixed Prefix HTTP matrix without merging diagnostic timings."""

import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path("/data_ssd/qcrs/experiments/p1-v8/Prefix")


def read(path):
    return json.loads(path.read_text())


def counter(row, name, source=None):
    return sum(
        value
        for key, value in row["counter_deltas"].items()
        if key.split("{")[0] == "vllm:" + name + "_total"
        and (source is None or f'source="{source}"' in key)
    )


def forward_ledger(events, row):
    live, segments, first_hits, final_lengths = {}, [], {}, {}
    peak_owned_requests = 0
    for event in events:
        rid = event["request_id"]
        if event["event"] == "allocation_done" and event["request_status"] in (
            "WAITING",
            "PREEMPTED",
        ):
            assert rid not in live and event["settled_logical_tokens"] == 0
            hit = 16 * (event["owned_blocks"][0] - (event["q"] + 15) // 16)
            assert hit >= 0
            live[rid] = {
                "request_id": rid,
                "admission_status": event["request_status"],
                "cached_tokens": hit,
            }
            peak_owned_requests = max(peak_owned_requests, len(live))
            if rid not in first_hits:
                first_hits[rid] = hit
        elif event["event"] == "free_before":
            segment = live.pop(rid)
            hit = segment["cached_tokens"]
            length = event["settled_logical_tokens"]
            assert length >= hit
            segment.update(
                settled_logical_tokens=length,
                completed_forward_tokens=length - hit,
                end_status=event["request_status"],
            )
            segments.append(segment)
            final_lengths[rid] = length
    assert not live
    assert len(first_hits) == 8
    assert sum(first_hits.values()) == row["cached_prompt_tokens"]
    assert sorted(final_lengths.values()) == sorted([4096 + 127, 8192 + 127] * 4)
    assert len(segments) == 8 + counter(row, "num_preemptions")
    baseline = 49152 - sum(first_hits.values()) + 8 * 127
    completed = sum(segment["completed_forward_tokens"] for segment in segments)
    return {
        "ownership_segments": len(segments),
        "peak_owned_requests": peak_owned_requests,
        "completed_forward_tokens": completed,
        "without_recompute_forward_tokens": baseline,
        "extra_forward_tokens": completed - baseline,
        "segments": segments,
    }


def main():
    runs = []
    for run in sorted(ROOT.glob("*-http-apc-*")):
        protocol = read(run / "protocol.json")
        result = read(run / "result.json")
        rounds = read(run / "serving.json")
        prompts = read(run / "prompts.json")
        assert result["status"] == "PASS", run
        assert len(rounds) == protocol["args"]["repeats"] == 3, run
        assert protocol["args"]["prefix_caching"] and not protocol["trace"]
        assert not (run / "kv-trace.jsonl").exists(), run
        assert [len(prompt) for prompt in prompts] == [4096, 8192] * 4
        assert all(prompt[:256] == prompts[0][:256] for prompt in prompts)
        assert len({prompt[256] for prompt in prompts}) == 8
        assert protocol["prompt_sha256"] == [
            hashlib.sha256(json.dumps(prompt).encode()).hexdigest()
            for prompt in prompts
        ]
        pooled = []
        for row in rounds:
            requests = read(run / f"requests-{row['repeat']}.json")
            pooled.extend(requests)
            assert len(requests) == 8
            assert all(len(item["token_ids"]) == 128 for item in requests)
            assert all(
                item["usage"]["prompt_tokens"] == len(prompt)
                and item["usage"]["completion_tokens"] == 128
                for item, prompt in zip(requests, prompts)
            )
            assert row["cached_prompt_tokens_by_request"] == [0] + [256] * 7
            assert row["cached_prompt_tokens"] == 1792
            good = sum(
                item["ttft_s"] <= 2 and item["tpot_s"] <= 0.08 for item in requests
            )
            assert good == row["slo_good_requests"]
            assert math.isclose(1024 / row["elapsed_s"], row["output_tokens_per_s"])
            for name in ("ttft_s", "tpot_s", "itl_s"):
                values = (
                    [dt for item in requests for dt in item[name]]
                    if name == "itl_s"
                    else [item[name] for item in requests]
                )
                assert np.allclose(
                    np.percentile(values, [50, 95, 99]), list(row[name].values())
                )
            row["preemptions"] = counter(row, "num_preemptions")
            row["initial_prefill_compute_reported"] = counter(
                row, "prompt_tokens_by_source", "local_compute"
            )
        log_lines = (run / "server.log").read_text().splitlines()
        resets = [
            i
            for i, line in enumerate(log_lines)
            if "POST /reset_prefix_cache HTTP" in line
        ]
        assert len(resets) == 4
        runs.append(
            {
                "directory": run.name,
                "pool_gib": protocol["args"]["pool_gib"],
                "policy": protocol["args"]["policy"],
                "prompt_sha256": protocol["prompt_sha256"],
                "result": result,
                "rounds": rounds,
                "aggregate": {
                    "output_tokens_per_s": 3072 / sum(r["elapsed_s"] for r in rounds),
                    "request_throughput_per_s": 24
                    / sum(r["elapsed_s"] for r in rounds),
                    "slo_goodput_requests_per_s": sum(
                        r["slo_good_requests"] for r in rounds
                    )
                    / sum(r["elapsed_s"] for r in rounds),
                    "slo_good_requests": sum(r["slo_good_requests"] for r in rounds),
                    "preemptions": sum(r["preemptions"] for r in rounds),
                    "pooled_latency_quantiles": {
                        name: dict(
                            zip(
                                ("p50", "p95", "p99"),
                                np.percentile(
                                    [dt for item in pooled for dt in item[name]]
                                    if name == "itl_s"
                                    else [item[name] for item in pooled],
                                    [50, 95, 99],
                                ),
                            )
                        )
                        for name in ("ttft_s", "tpot_s", "itl_s")
                    },
                },
                "jit_warnings": [
                    {
                        "line": line,
                        "phase": "warmup" if i < resets[1] else "timed",
                    }
                    for i, line in enumerate(log_lines)
                    if "Triton kernel JIT compilation during inference" in line
                ],
            }
        )
    assert len(runs) == 12
    assert {(r["pool_gib"], r["policy"]) for r in runs} == {
        (pool, policy)
        for pool in (8, 6, 3, 1.5)
        for policy in ("full", "noop", "sink_recent")
    }
    assert all(run["prompt_sha256"] == runs[0]["prompt_sha256"] for run in runs)
    identity = []
    comparisons = []
    for pool in (8, 6, 3, 1.5):
        native = next(
            r for r in runs if r["pool_gib"] == pool and r["policy"] == "full"
        )
        noop = next(r for r in runs if r["pool_gib"] == pool and r["policy"] == "noop")
        sink = next(
            r for r in runs if r["pool_gib"] == pool and r["policy"] == "sink_recent"
        )
        comparisons.append(
            {
                "pool_gib": pool,
                "noop_mean_throughput_overhead_percent": 100
                * (
                    1
                    - np.mean([r["output_tokens_per_s"] for r in noop["rounds"]])
                    / np.mean([r["output_tokens_per_s"] for r in native["rounds"]])
                ),
                "sink_throughput_change_percent": 100
                * (
                    sink["aggregate"]["output_tokens_per_s"]
                    / native["aggregate"]["output_tokens_per_s"]
                    - 1
                ),
                "sink_all_requests_within_slo": sink["aggregate"]["slo_good_requests"]
                == 24,
            }
        )
        for repeat in range(3):
            lhs = read(ROOT / native["directory"] / f"requests-{repeat}.json")
            rhs = read(ROOT / noop["directory"] / f"requests-{repeat}.json")
            identity.append(
                {
                    "pool_gib": pool,
                    "repeat": repeat,
                    "equal_outputs": sum(
                        a["token_ids"] == b["token_ids"] for a, b in zip(lhs, rhs)
                    ),
                }
            )
    assert all(item["equal_outputs"] == 8 for item in identity)
    diagnostics = []
    for run in sorted(ROOT.glob("*-http-diagnostic-*")):
        assert read(run / "result.json")["status"] == "PASS", run
        events = [
            json.loads(line)
            for line in (run / "kv-trace.jsonl").read_text().splitlines()
        ]
        for row in [read(run / "warmup-summary.json"), *read(run / "serving.json")]:
            phase = [
                event
                for event in events
                if row["start_monotonic_ns"]
                <= event["time_ns"]
                <= row["end_monotonic_ns"]
            ]
            plans = [event for event in phase if event["event"] == "reclaim_select"]
            released = [
                event["released_blocks"]
                for event in phase
                if event["event"] == "reclaim_commit"
            ]
            diagnostics.append(
                {
                    "directory": run.name,
                    "repeat": row["repeat"],
                    "events": dict(Counter(event["event"] for event in phase)),
                    "protected_prefix_tokens": sorted(
                        {p["protected_prefix_tokens"] for p in plans}
                    ),
                    "cumulative_released_blocks": sum(released),
                    "released_blocks_per_commit_range": [
                        min(released, default=0),
                        max(released, default=0),
                    ],
                    "counter_scope_correction": (
                        "prefix query/hit excludes preempted queries; prompt source "
                        "counters describe initial prefill, not all recomputation"
                    ),
                    "forward_ledger": forward_ledger(phase, row),
                }
            )
    summary = {
        "runs": runs,
        "native_noop_output_comparison": identity,
        "comparisons": comparisons,
        "diagnostics": diagnostics,
    }
    (ROOT / "serving-summary.json").write_text(json.dumps(summary, indent=2))
    table = [
        "| Pool/GiB | 策略 | token/s（3轮） | TTFT p99/s | TPOT p99/ms | "
        "SLO-good/8 | preempt | 峰值/GiB |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for run in runs:
        rows = run["rounds"]
        cells = [
            str(run["pool_gib"]),
            run["policy"],
            " / ".join(f"{row['output_tokens_per_s']:.2f}" for row in rows),
            " / ".join(f"{row['ttft_s']['p99']:.3f}" for row in rows),
            " / ".join(f"{row['tpot_s']['p99'] * 1000:.2f}" for row in rows),
            " / ".join(str(row["slo_good_requests"]) for row in rows),
            " / ".join(str(int(row["preemptions"])) for row in rows),
            f"{run['result']['peak_project_nvml_bytes'] / 2**30:.3f}",
        ]
        table.append("| " + " | ".join(cells) + " |")
    (ROOT / "serving-table.md").write_text("\n".join(table) + "\n")
    print("\n".join(table))
    print(json.dumps(identity))


if __name__ == "__main__":
    main()
