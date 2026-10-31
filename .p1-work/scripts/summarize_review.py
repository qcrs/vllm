# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Recompute final review comparisons without replacing historical evidence."""

import argparse
import json
import statistics
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def serving(root, name):
    run = root / name
    guard = read(run / "result.json")
    assert guard["status"] == "PASS", name
    rounds = read(run / "serving.json")
    assert [r["repeat"] for r in rounds] == [0, 1, 2], name
    requests = [read(run / f"requests-{r}.json") for r in range(3)]
    assert all(len(items) == 2 for items in requests), name
    assert all(
        len(item["token_ids"]) == item["output_tokens"] == 128
        for items in requests
        for item in items
    ), name
    rates = [r["output_tokens_per_s"] for r in rounds]
    return {
        "run": name,
        "tokens_per_s": rates,
        "mean_tokens_per_s": statistics.mean(rates),
        "slo_good_requests": sum(r["slo_good_requests"] for r in rounds),
        "requests": sum(len(items) for items in requests),
        "peak_project_nvml_bytes": guard["peak_project_nvml_bytes"],
    }, requests


def compare(root, native_name, noop_name):
    native, native_requests = serving(root, native_name)
    noop, noop_requests = serving(root, noop_name)
    assert read(root / native_name / "prompts.json") == read(
        root / noop_name / "prompts.json"
    )
    paired = 0
    for native_round, noop_round in zip(native_requests, noop_requests):
        for a, b in zip(native_round, noop_round):
            assert a["token_ids"] == b["token_ids"], (a["request"], b["request"])
            paired += 1
    overhead = 100 * (1 - noop["mean_tokens_per_s"] / native["mean_tokens_per_s"])
    return {
        "native": native,
        "noop": noop,
        "paired_outputs_equal": paired,
        "throughput_overhead_percent": overhead,
        "budget_percent": 5,
        "budget_status": "PASS" if overhead <= 5 else "FAIL",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "root", type=Path, nargs="?", default="/data_ssd/qcrs/experiments/p1-v8"
    )
    args = parser.parse_args()
    review = args.root / "Review"
    historical = compare(
        args.root / "TP",
        "20261031-121030-http-short-eager-8-full",
        "20261031-121131-http-short-eager-8-noop",
    )
    before = compare(
        review,
        "20261031-143243-tp-eager-native-before",
        "20261031-143405-tp-eager-noop-before",
    )
    after_a = compare(
        review,
        "20261031-144209-tp-eager-after-a-full",
        "20261031-144310-tp-eager-after-a-noop",
    )
    after_b = compare(
        review,
        "20261031-144514-tp-eager-after-b-full",
        "20261031-144413-tp-eager-after-b-noop",
    )
    native = statistics.mean(
        after_a["native"]["tokens_per_s"] + after_b["native"]["tokens_per_s"]
    )
    noop = statistics.mean(
        after_a["noop"]["tokens_per_s"] + after_b["noop"]["tokens_per_s"]
    )
    pooled_overhead = 100 * (1 - noop / native)
    result = {
        "evidence_status": "PASS",
        "historical": historical,
        "before": before,
        "after_abba": [after_a, after_b],
        "after_pooled": {
            "native_mean_tokens_per_s": native,
            "noop_mean_tokens_per_s": noop,
            "throughput_overhead_percent": pooled_overhead,
            "budget_status": "PASS" if pooled_overhead <= 5 else "FAIL",
            "paired_outputs_equal": 12,
        },
        "boundary": (
            "Independent servers, full workload warmup and three timed rounds each. "
            "Keep both ABBA pairs; host/rank scheduling varies across servers. "
            "Profiler runs are excluded. No causal claim that fusion fixes 14.33%."
        ),
    }
    output = review / "serving-review-summary.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
