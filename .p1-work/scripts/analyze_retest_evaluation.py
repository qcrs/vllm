# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Audit bounded baseline and arrival-pressure retests on a specified GPU pair."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import regex as re
from analyze_dp_work import distribution, events, read
from analyze_prefill_evaluation import digest, length_summary, request_stages
from analyze_protected_evaluation import work_summary
from analyze_two_round_evaluation import normalized_requests
from dp_serving import aggregate


def audit_protocol(run, device_pair):
    protocol, guard = read(run / "protocol.json"), read(run / "result.json")
    args, command = protocol["args"], protocol["command"]
    dp = "scenario" in args
    assert guard["status"] == "PASS" and guard["exit_code"] == 0
    assert not guard["limit_triggered"]
    assert guard["peak_project_nvml_bytes"] < 44 * 2**30
    assert args["repeats"] == 3 and not args["trace"] and args["work_counters"]
    assert args["max_num_seqs"] == 8
    assert args["long_prefill_token_threshold"] == 512
    assert command[command.index("--long-prefill-token-threshold") + 1] == "512"
    assert json.loads(command[command.index("--compilation-config") + 1]) == {
        "mode": 0,
        "cudagraph_mode": "FULL_DECODE_ONLY",
        "cudagraph_capture_sizes": [1, 2, 4, 8],
    }
    assert protocol["slo"] == {"ttft_s": 2, "tpot_s": 0.08}
    assert protocol["output_tokens"] == 128
    if args["budget_mode"] == "protected":
        assert args["decode_reclaim_rounds"] == args["admission_bypass_limit"] == 2
    else:
        assert args["budget_mode"] == "fixed"
    if dp:
        assert args["scenario"] in ("w1", "w0")
        assert args["routing"] in ("requests", "known_work")
        assert args["pool_gib"] in (1.5, 4)
        assert args["max_num_batched_tokens"] == 512
        assert args["policy"] == "sink_recent" and not args["load_diagnostic"]
        assert args["arrival_scale"] == 1
        lengths = [8192, 512] * 4 if args["scenario"] == "w1" else [4096] * 8
        arrivals = [i * 0.05 for i in range(8)]
    else:
        assert args["tp"] in (1, 2) and args["pool_gib"] * args["tp"] == 1.5
        assert args["graph_mode"] == "full" and args["full_warmup"]
        assert args["workload"] == "mixed" and args["reserve"] == 0
        assert not args["prefix_caching"] and args["reclamation_counts"]
        assert args["max_num_batched_tokens"] == 1024
        assert args["policy"] in ("full", "sink_recent")
        assert args["arrival_scale"] in (1, 0.1)
        lengths = [4096, 8192, 128, 128] * 2
        arrivals = [
            t * args["arrival_scale"] for t in (0, 0.3, 0.6, 0.9, 5, 5.3, 5.6, 5.9)
        ]
    assert protocol["prompt_tokens"] == lengths
    assert protocol["arrivals_s"] == arrivals
    assert protocol["prompt_sha256"] == [
        hashlib.sha256(json.dumps(p).encode()).hexdigest()
        for p in read(run / "prompts.json")
    ]
    name = "dp-source-hashes.json" if dp else "runtime-source-hashes.json"
    hashes = read(run / name)
    for relative, expected in hashes.items():
        assert digest(run / "source" / relative) == expected, (run, relative)
    invocation = read(run / "invocation.json")
    physical_devices = device_pair if dp or args["tp"] == 2 else device_pair[:1]
    devices = ",".join(map(str, physical_devices))
    assert invocation["environment"]["CUDA_VISIBLE_DEVICES"] == devices
    assert invocation["hard_limit_bytes"] == 45 * 2**30
    assert invocation["stop_threshold_bytes"] == 44 * 2**30
    for source, expected in invocation["source_files"].items():
        assert digest(run / "source" / Path(source).relative_to(invocation["cwd"])) == (
            expected
        )
    affinities = [
        read(run / f"worker-affinity-{i}.json") for i in range(len(physical_devices))
    ]
    assert [v["affinity"] for v in affinities] == (
        [[20], [21]] if len(physical_devices) == 2 else [[20]]
    )
    placements = {}
    for sample in events(run / "nvml.jsonl"):
        for process in sample["processes"]:
            assert process["gpu"] in physical_devices, (run, process)
            placements.setdefault(process["pid"], set()).add(process["gpu"])
    for rank, affinity in enumerate(affinities):
        assert placements[affinity["pid"]] == {physical_devices[rank]}
    helpers = {
        name: digest(run / name) for name in ("dp_hooks.py", "tp_worker_hooks.py")
    }
    return protocol, guard, hashes, invocation, affinities, helpers


def audit_routes(run, routing):
    routes = events(run / "routing.jsonl")
    assert len(routes) == len({r["request"] for r in routes}) == 33
    for route in routes:
        assert route["explicit"] is None
        scores = [max(4 * w + r, f) for (w, r), f in zip(route["counts"], route["F"])]
        if routing == "known_work":
            assert route["batch"] == 512
            scores = [
                c + ((s["unfinished_tokens"] if s else 0) + p) / 512
                for c, s, p in zip(scores, route["snapshots"], route["pending_tokens"])
            ]
            for snapshot, adds, tokens in zip(
                route["snapshots"], route["pending_adds"], route["pending_tokens"]
            ):
                watermark = (
                    snapshot.get("add_watermarks", {}).get(
                        str(route["client_index"]), 0
                    )
                    if snapshot
                    else 0
                )
                assert tokens == sum(n for _, n in adds.values())
                assert all(seq > watermark for seq, _ in adds.values())
        else:
            assert route["batch"] is None
        order = [(route["scan_start"] + i) % 2 for i in range(2)]
        assert route["replica"] == min(order, key=lambda rank: scores[rank])
    return routes


def server_stages(item, record):
    """DP clients lack monotonic first-token stamps; report server stages only."""
    stamps = [
        record[name]
        for name in (
            "received_ns",
            "first_scheduled_ns",
            "prefill_completed_ns",
            "first_token_ns",
            "completed_ns",
        )
    ]
    assert stamps == sorted(stamps), (item["request"], stamps)
    return dict(
        request=item["request"],
        server_request=record["request"],
        prompt_tokens=item["prompt_tokens"],
        ttft_s=item["ttft_s"],
        tpot_s=item["tpot_s"],
        queue_to_first_schedule_s=(stamps[1] - stamps[0]) / 1e9,
        prefill_until_result_consumed_s=(stamps[2] - stamps[1]) / 1e9,
        prefill_to_sample_s=(stamps[3] - stamps[2]) / 1e9,
    )


def observed_runtime(run, config):
    log = (run / "server.log").read_text()
    pools = [
        int(value.replace(",", ""))
        for value in re.findall(r"GPU KV cache size: ([\d,]+) tokens", log)
    ]
    expected = 29120 if config.get("scenario") and config["pool_gib"] == 4 else 10912
    assert pools and all(value == expected for value in pools), (run, pools)
    captures = [
        line
        for line in log.splitlines()
        if "Capturing CUDA graphs (FULL): 100%" in line and "4/4" in line
    ]
    assert captures, (run, "missing completed Graph capture")
    return dict(
        pool_tokens=pools,
        blocks_per_pool=expected // 16,
        completed_full_capture_logs=len(captures),
    )


def analyze(run, device_pair):
    protocol, guard, hashes, invocation, affinities, helpers = audit_protocol(
        run, device_pair
    )
    args = protocol["args"]
    dp = "scenario" in args
    routes = audit_routes(run, args["routing"]) if dp else []
    work = [events(p) for p in sorted(run.glob("replica-*/work.jsonl"))]
    assert sum(map(len, work)) == 33
    summaries = read(run / "serving.json")
    assert [s["repeat"] for s in summaries] == [0, 1, 2]
    all_items, timed_work, timelines, rounds, timed_routes = [], [], [], [], []
    for summary in [read(run / "warmup-summary.json"), *summaries]:
        label = summary["repeat"]
        items = read(run / f"requests-{label}.json")
        assert [r["prompt_tokens"] for r in items] == protocol["prompt_tokens"]
        assert [r["arrival_s"] for r in items] == protocol["arrivals_s"]
        completed = [
            r
            for records in work
            for r in records
            if summary["start_monotonic_ns"]
            <= r["completed_ns"]
            <= summary["end_monotonic_ns"]
        ]
        assert len(completed) == 8
        stage_rows, round_routes = [], []
        for item in items:
            assert len(item["token_ids"]) == item["usage"]["completion_tokens"] == 128
            assert len(item["itl_s"]) == 127
            assert abs(sum(item["itl_s"]) / 127 - item["tpot_s"]) < 1e-9
            matches = [
                r
                for r in completed
                if r["request"].startswith(f"cmpl-{item['request']}-0-")
            ]
            assert len(matches) == 1, (item["request"], matches)
            record = matches[0]
            assert record["prompt_tokens"] == item["prompt_tokens"]
            assert record["scheduled_tokens"] == (
                record["prompt_tokens"] + 127 + record["recomputed_tokens"]
            )
            stage_rows.append((server_stages if dp else request_stages)(item, record))
            if dp:
                (route,) = [r for r in routes if r["request"] == record["request"]]
                assert route["replica"] == item["replica"]
                assert record in work[item["replica"]]
                round_routes.append(route)
        counts = work_summary(completed)
        preemptions = sum(
            value
            for key, value in summary["counter_deltas"].items()
            if key.startswith("vllm:num_preemptions_total{")
        )
        assert preemptions == counts["preemptions"]
        calculated = aggregate(normalized_requests(items, dp), summary["elapsed_s"])
        stored = summary["aggregate"] if dp else summary
        for key in (
            "slo_good_requests",
            "slo_goodput_requests_per_s",
            "output_tokens_per_s",
            "ttft_s",
            "tpot_s",
            "itl_s",
        ):
            assert calculated[key] == stored[key], (run, label, key)
        if not dp:
            before = summary["reclamation_counts_before"][-1]["canonical_commits"]
            after = summary["reclamation_counts_after"][-1]["canonical_commits"]
            assert after - before == counts["reclaim_calls"]
        rounds.append(
            dict(
                repeat=label,
                metrics=calculated,
                work=counts,
                by_length=length_summary(items, stage_rows),
            )
        )
        if label != "warmup":
            all_items.extend(items)
            timed_work.extend(completed)
            timelines.extend(stage_rows)
            timed_routes.extend(round_routes)
    combined = aggregate(
        normalized_requests(all_items, dp), sum(r["elapsed_s"] for r in summaries)
    )
    combined.pop("active_span_s")
    answer = dict(
        run=str(run),
        config=args,
        metrics=combined,
        by_length=length_summary(all_items, timelines),
        work=work_summary(timed_work),
        rounds=rounds,
        timelines=timelines,
        source_hashes=hashes,
        helper_hashes=helpers,
        input_hashes=protocol["prompt_sha256"],
        affinity=affinities,
        guard=guard,
        commit=invocation["commit"],
        physical_devices=invocation["environment"]["CUDA_VISIBLE_DEVICES"],
        observed_runtime=observed_runtime(run, args),
        slo_failures=dict(
            ttft=sum(r["ttft_s"] > 2 for r in all_items),
            tpot=sum(r["tpot_s"] > 0.08 for r in all_items),
            both=sum(r["ttft_s"] > 2 and r["tpot_s"] > 0.08 for r in all_items),
        ),
        max_ttft_s=max(r["ttft_s"] for r in all_items),
        max_tpot_s=max(r["tpot_s"] for r in all_items),
    )
    if dp:
        answer["requests_per_replica"] = dict(Counter(r["replica"] for r in all_items))
        answer["timed_work_per_replica"] = [
            work_summary([r for r in records if r in timed_work]) for records in work
        ]
        answer["route_cpu_us"] = distribution(
            [r["route_cpu_ns"] / 1000 for r in timed_routes]
        )
        answer["snapshot_age_ms"] = distribution(
            [
                (r["used_ns"] - s["generated_ns"]) / 1e6
                for r in timed_routes
                for s in r["snapshots"]
                if s
            ]
        )
        answer["snapshot_usage"] = [
            [s["kv_cache_usage"] if s else None for s in r["snapshots"]]
            for r in timed_routes
        ]
        answer["routes"] = timed_routes
    return answer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device-pair", choices=("0,1", "1,2"), default="0,1")
    args = parser.parse_args()
    device_pair = [int(device) for device in args.device_pair.split(",")]
    answer = [analyze(run, device_pair) for run in args.runs]
    sources, helpers, inputs = {}, {}, {}
    for row in answer:
        for observed, expected in (
            (row["source_hashes"], sources),
            (row["helper_hashes"], helpers),
        ):
            for name, value in observed.items():
                assert expected.setdefault(name, value) == value, (row["run"], name)
        config = row["config"]
        family = config.get("scenario", "mixed")
        assert inputs.setdefault(family, row["input_hashes"]) == row["input_hashes"]
    args.output.write_text(json.dumps(answer, indent=2) + "\n")
    for row in answer:
        metrics, work = row["metrics"], row["work"]
        print(
            Path(row["run"]).name,
            f"SLO={metrics['slo_good_requests']}/24",
            f"tokens/s={metrics['output_tokens_per_s']:.3f}",
            f"preempt={work['preemptions']} recompute={work['recomputed_tokens']}",
            f"reclaim={work['reclaim_calls']}",
        )


if __name__ == "__main__":
    main()
