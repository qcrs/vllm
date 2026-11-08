# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Audit same-GPU prefill comparisons and split first-token latency by stage."""

import argparse
import hashlib
import json
from pathlib import Path

from analyze_dp_work import distribution, events, read
from analyze_protected_evaluation import work_summary
from dp_serving import aggregate


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def request_stages(item, record):
    stamps = [
        item["client_started_ns"],
        record["received_ns"],
        record["first_scheduled_ns"],
        record["prefill_completed_ns"],
        record["first_token_ns"],
        item["client_first_token_ns"],
    ]
    assert stamps == sorted(stamps), (item["request"], stamps)
    stages = dict(
        zip(
            (
                "send_to_scheduler_s",
                "queue_to_first_schedule_s",
                "prefill_until_result_consumed_s",
                "prefill_to_sample_s",
                "sample_to_client_s",
            ),
            [(right - left) / 1e9 for left, right in zip(stamps, stamps[1:])],
        )
    )
    assert abs(sum(stages.values()) - item["ttft_s"]) < 0.001
    return dict(
        request=item["request"],
        server_request=record["request"],
        prompt_tokens=item["prompt_tokens"],
        timestamps_ns=dict(
            zip(
                (
                    "client_sent",
                    "scheduler_received",
                    "first_scheduled",
                    "prefill_result_consumed",
                    "first_sampled",
                    "client_received",
                ),
                stamps,
            )
        ),
        ttft_s=item["ttft_s"],
        tpot_s=item["tpot_s"],
        slo=item["ttft_s"] <= 2 and item["tpot_s"] <= 0.08,
        **stages,
    )


def length_summary(items, timelines):
    answer = {}
    for length in sorted({r["prompt_tokens"] for r in items}):
        group = [r for r in items if r["prompt_tokens"] == length]
        stages = [r for r in timelines if r["prompt_tokens"] == length]
        answer[length] = dict(
            count=len(group),
            slo=sum(r["ttft_s"] <= 2 and r["tpot_s"] <= 0.08 for r in group),
            ttft_s=distribution([r["ttft_s"] for r in group]),
            tpot_s=distribution([r["tpot_s"] for r in group]),
            stages={
                key: distribution([r[key] for r in stages])
                for key in stages[0]
                if key.endswith("_s") and key not in ("ttft_s", "tpot_s")
            },
        )
    return answer


def analyze(run):
    protocol, guard = read(run / "protocol.json"), read(run / "result.json")
    args, command = protocol["args"], protocol["command"]
    assert guard["status"] == "PASS" and guard["exit_code"] == 0
    assert not guard["limit_triggered"]
    assert guard["peak_project_nvml_bytes"] < 44 * 2**30
    assert args["tp"] == 1 and args["pool_gib"] == 1.5
    assert args["repeats"] == 3 and not args["trace"] and args["work_counters"]
    assert args["graph_mode"] == "full" and args["full_warmup"]
    assert args["workload"] == "mixed" and args["reserve"] == 0
    assert not args["prefix_caching"] and args["max_num_seqs"] == 8
    assert args["max_num_batched_tokens"] == 1024
    assert args["long_prefill_token_threshold"] in (0, 512)
    assert command[command.index("--long-prefill-token-threshold") + 1] == str(
        args["long_prefill_token_threshold"]
    )
    assert json.loads(command[command.index("--compilation-config") + 1]) == {
        "mode": 0,
        "cudagraph_mode": "FULL_DECODE_ONLY",
        "cudagraph_capture_sizes": [1, 2, 4, 8],
    }
    if args["budget_mode"] == "protected":
        assert args["decode_reclaim_rounds"] == args["admission_bypass_limit"] == 2
    assert protocol["slo"] == {"ttft_s": 2, "tpot_s": 0.08}
    assert protocol["prompt_tokens"] == [4096, 8192, 128, 128] * 2
    assert protocol["arrivals_s"] == [0, 0.3, 0.6, 0.9, 5, 5.3, 5.6, 5.9]
    assert protocol["output_tokens"] == 128
    assert protocol["prompt_sha256"] == [
        hashlib.sha256(json.dumps(p).encode()).hexdigest()
        for p in read(run / "prompts.json")
    ]
    hashes = read(run / "runtime-source-hashes.json")
    for relative, expected in hashes.items():
        assert digest(run / "source" / relative) == expected, (run, relative)
    invocation = read(run / "invocation.json")
    assert invocation["environment"]["CUDA_VISIBLE_DEVICES"] == "0"
    assert invocation["hard_limit_bytes"] == 45 * 2**30
    assert invocation["stop_threshold_bytes"] == 44 * 2**30
    for source, expected in invocation["source_files"].items():
        assert digest(run / "source" / Path(source).relative_to(invocation["cwd"])) == (
            expected
        )
    affinity = read(run / "worker-affinity-0.json")
    assert affinity["affinity"] == [20]
    helpers = {
        name: digest(run / name) for name in ("dp_hooks.py", "tp_worker_hooks.py")
    }
    work = events(run / "replica-0/work.jsonl")
    assert len(work) == 33
    summaries = read(run / "serving.json")
    assert [s["repeat"] for s in summaries] == [0, 1, 2]
    all_items, timed_work, timelines, rounds = [], [], [], []
    for summary in [read(run / "warmup-summary.json"), *summaries]:
        label = summary["repeat"]
        items = read(run / f"requests-{label}.json")
        assert [r["prompt_tokens"] for r in items] == protocol["prompt_tokens"]
        assert [r["arrival_s"] for r in items] == protocol["arrivals_s"]
        completed = [
            r
            for r in work
            if summary["start_monotonic_ns"]
            <= r["completed_ns"]
            <= summary["end_monotonic_ns"]
        ]
        assert len(completed) == 8
        stage_rows = []
        for item in items:
            assert len(item["token_ids"]) == item["usage"]["completion_tokens"] == 128
            matches = [
                r
                for r in completed
                if r["request"].startswith(f"cmpl-{item['request']}-0-")
            ]
            assert len(matches) == 1, (item["request"], matches)
            record = matches[0]
            assert record["scheduled_tokens"] == (
                record["prompt_tokens"] + 127 + record["recomputed_tokens"]
            )
            stage_rows.append(request_stages(item, record))
        counts = work_summary(completed)
        preemptions = sum(
            value
            for key, value in summary["counter_deltas"].items()
            if key.startswith("vllm:num_preemptions_total{")
        )
        assert preemptions == counts["preemptions"]
        normalized = [dict(r, finish_s=r["start_s"] + r["e2e_s"]) for r in items]
        calculated = aggregate(normalized, summary["elapsed_s"])
        for key in (
            "slo_good_requests",
            "slo_goodput_requests_per_s",
            "output_tokens_per_s",
            "ttft_s",
            "tpot_s",
            "itl_s",
        ):
            assert calculated[key] == summary[key], (run, label, key)
        before = summary["reclamation_counts_before"][-1]["canonical_commits"]
        after = summary["reclamation_counts_after"][-1]["canonical_commits"]
        assert after - before == counts["reclaim_calls"]
        rounds.append(
            dict(
                repeat=label,
                metrics=calculated,
                work=counts,
                by_length=length_summary(items, stage_rows),
                short_prefill_overlap=[
                    dict(
                        request=stage_rows[index]["request"],
                        long_request=stage_rows[long_index]["request"],
                        received_before_long_finished=(
                            stage_rows[index]["timestamps_ns"]["scheduler_received"]
                            < stage_rows[long_index]["timestamps_ns"][
                                "prefill_result_consumed"
                            ]
                        ),
                        scheduled_before_long_finished=(
                            stage_rows[index]["timestamps_ns"]["first_scheduled"]
                            < stage_rows[long_index]["timestamps_ns"][
                                "prefill_result_consumed"
                            ]
                        ),
                    )
                    for index, long_index in ((2, 1), (3, 1), (6, 5), (7, 5))
                ],
            )
        )
        if label != "warmup":
            all_items.extend(normalized)
            timed_work.extend(completed)
            timelines.extend(stage_rows)
    combined = aggregate(all_items, sum(r["elapsed_s"] for r in summaries))
    combined.pop("active_span_s")
    return dict(
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
        affinity=affinity,
        guard=guard,
        commit=invocation["commit"],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    answer = [analyze(run) for run in args.runs]
    first = answer[0]
    for row in answer[1:]:
        for key in ("source_hashes", "helper_hashes", "input_hashes"):
            assert row[key] == first[key], (row["run"], key)
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
