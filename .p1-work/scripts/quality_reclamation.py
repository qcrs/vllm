# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Matched local first-turn evaluation after a forced newline and settled reclaim."""

import argparse
import collections
import hashlib
import json
import os
import string
import time
from pathlib import Path

import pyarrow as pa
import regex as re
import torch
from audit_datasets import first_turn
from transformers import AutoTokenizer

from vllm import EngineArgs, LLMEngine, SamplingParams
from vllm.v1.core.sched.output import SchedulerOutput


def normalize(text):
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def select_records(manifest, per_split):
    selected, counts = [], collections.Counter()
    for line in manifest.read_text().splitlines():
        record = json.loads(line)
        if not record["eligible_first_turn"]:
            continue
        key = (record["dataset"], record["subset"], record["split"])
        limit = per_split if key[0] == "babilong" else 8
        if counts[key] < limit:
            selected.append(record)
            counts[key] += 1
    return selected


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--policy",
        choices=["full", "sink_recent", "random", "triattention"],
        required=True,
    )
    parser.add_argument("--per-split", type=int, default=5)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--stats")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "/data_ssd/qcrs/experiments/p1-v8/M0/dataset-audit-20261030-2248-resume/manifest.jsonl"
        ),
    )
    args = parser.parse_args()
    run = Path(os.environ["P1_RUN_DIR"])
    selected = select_records(args.manifest, args.per_split)
    if args.limit:
        selected = selected[: args.limit]
    (run / "selection.json").write_text(json.dumps(selected, indent=2))
    protocol = {
        "policy": args.policy,
        "retention": 1 if args.policy == "full" else 0.5,
        "sink": 16,
        "recent": 16,
        "min_keep": 128,
        "max_tokens": 128,
        "temperature": 0,
        "seed": 0,
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "selection_sha256": hashlib.sha256(json.dumps(selected).encode()).hexdigest(),
        "scope": (
            "first-turn; forced identical newline before any scored answer; "
            "q=0 after full prefill"
        ),
        "metrics": (
            "normalized exact match, target token subsequence, token F1; "
            "not official task scores"
        ),
        "quality_budget": "exploratory; no default-enable claim from this sample",
    }
    (run / "protocol.json").write_text(json.dumps(protocol, indent=2))
    model = "/data/models/Qwen3-8B"
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    (newline,) = tokenizer.encode("\n", add_special_tokens=False)
    options = EngineArgs(
        model=model,
        dtype="bfloat16",
        enforce_eager=True,
        attention_backend="FLASH_ATTN",
        block_size=16,
        enable_prefix_caching=False,
        async_scheduling=False,
        disable_cascade_attn=True,
        scheduler_reserve_full_isl=False,
        max_model_len=40960,
        max_num_seqs=1,
        max_num_batched_tokens=2048,
        kv_cache_memory_bytes=6 * 2**30,
        gpu_memory_utilization=0.4,
        seed=0,
        disable_log_stats=True,
    )
    if args.policy != "full":
        options.additional_config = {
            "kv_reclamation": {
                "policy": args.policy,
                "stats_path": args.stats,
            }
        }
    engine = LLMEngine.from_engine_args(options)
    core = engine.engine_core.engine_core
    scheduler = core.scheduler
    runner = core.model_executor.driver_worker.worker.model_runner
    original_schedule = scheduler.schedule
    pending = None

    def schedule(*schedule_args, **schedule_kwargs):
        nonlocal pending
        if pending is None:
            return original_schedule(*schedule_args, **schedule_kwargs)
        request = pending
        pending = None
        output = SchedulerOutput.make_empty()
        output.reclaim_request = scheduler.make_reclaim_request(
            request, output, max(128, request.effective_kv_len // 2)
        )
        return output

    scheduler.schedule = schedule
    tables, results = {}, []
    for number, record in enumerate(selected):
        source = record["source"]
        if source not in tables:
            tables = {source: pa.ipc.open_stream(pa.memory_map(source)).read_all()}
            assert (
                hashlib.sha256(Path(source).read_bytes()).hexdigest()
                == record["source_sha256"]
            )
        row = tables[source].slice(record["row"], 1).to_pylist()[0]
        content, target = first_turn(
            row, record["dataset"] == "babilong", record["subset"]
        )
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        assert hashlib.sha256(text.encode()).hexdigest() == record["prompt_sha256"]
        ids = tokenizer.encode(text, add_special_tokens=False)
        assert len(ids) == record["prompt_tokens"] and len(ids) + 128 <= 40960
        params = SamplingParams(
            temperature=0, max_tokens=128, seed=0, allowed_token_ids=[newline]
        )
        started = time.perf_counter()
        rid = engine.add_request(str(number), {"prompt_token_ids": ids}, params)
        output = None
        while output is None or not output.outputs[0].token_ids:
            batch = engine.step()
            if batch:
                (output,) = batch
        assert list(output.outputs[0].token_ids) == [newline]
        request = scheduler.requests[rid]
        assert (
            request.num_computed_tokens == len(ids) and not request.num_in_flight_tokens
        )
        request.sampling_params.allowed_token_ids = None
        index = runner.req_states.req_id_to_index[rid]
        with torch.cuda.stream(runner.main_stream):
            runner.sampler.logit_bias_state.add_request(
                index, len(ids), request.sampling_params
            )
            runner.sampler.logit_bias_state.apply_staged_writes()
        released = 0
        reclaim_ms = None
        if args.policy != "full":
            assert request.effective_kv_len == len(ids)
            before = scheduler.kv_cache_manager.block_pool.get_num_free_blocks()
            pending = request
            reclaim_start = time.perf_counter()
            assert engine.step() == []
            reclaim_ms = (time.perf_counter() - reclaim_start) * 1000
            assert request.effective_kv_len == max(128, len(ids) // 2)
            assert request.num_computed_tokens == len(ids)
            released = (
                scheduler.kv_cache_manager.block_pool.get_num_free_blocks() - before
            )
            assert released > 0
        while engine.has_unfinished_requests():
            batch = engine.step()
            if batch:
                (output,) = batch
        predicted = output.outputs[0].text
        answer, expected = normalize(predicted), normalize(target)
        a, b = answer.split(), expected.split()
        common = sum((collections.Counter(a) & collections.Counter(b)).values())
        item = {
            **record,
            "policy": args.policy,
            "target": target,
            "prediction": predicted,
            "output_ids": list(output.outputs[0].token_ids),
            "exact_match": answer == expected,
            "target_subsequence": bool(
                re.search(r"(?<!\w)" + re.escape(expected) + r"(?!\w)", answer)
            ),
            "token_f1": 2 * common / (len(a) + len(b)) if a or b else 1,
            "released_blocks": released,
            "maintenance_host_ms": reclaim_ms,
            "elapsed_s": time.perf_counter() - started,
        }
        results.append(item)
        with (run / "answers.jsonl").open("a") as stream:
            stream.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(
            json.dumps(
                {
                    "sample": number,
                    "subset": record["subset"],
                    "split": record["split"],
                    "target": target,
                    "prediction": predicted,
                    "exact_match": item["exact_match"],
                }
            ),
            flush=True,
        )
    grouped = collections.defaultdict(list)
    for item in results:
        grouped[item["dataset"] + "/" + item["subset"]].append(item)
    summary = {}
    for key, items in grouped.items():
        summary[key] = {
            "samples": len(items),
            **{
                metric: sum(float(i[metric]) for i in items) / len(items)
                for metric in ("exact_match", "target_subsequence", "token_f1")
            },
        }
    (run / "quality.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"status": "PASS", "quality": summary}), flush=True)


if __name__ == "__main__":
    main()
