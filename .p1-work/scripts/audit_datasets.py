# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Audit existing Arrow data using Qwen3-8B chat tokens without copying data."""

import argparse
import collections
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pyarrow as pa
from transformers import AutoTokenizer


def digest(path):
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 2**20), b""):
            sha.update(chunk)
    return sha.hexdigest()


def first_turn(row, babi, subset):
    if babi:
        return row["input"] + "\n\n" + row["question"] + "\nAnswer:", row["target"]
    turn = row["multi_turns"][0]
    if subset == "scbench_mf":
        # Match SCBench's list rendering and repeated first-turn instruction.
        assert isinstance(row["context"], list)
        content = "{prefix}\n\n{context}\n\n{input}".format(
            prefix=turn["input"], context=row["context"], input=turn["input"]
        )
        assert isinstance(turn["answer"], int)
        return content, str(turn["answer"])
    context = row["input"] if subset == "scbench_vt" else row["context"]
    assert isinstance(context, str) and isinstance(turn["input"], str)
    target = turn["answer"]
    if subset == "scbench_vt":
        assert isinstance(target, list) and all(isinstance(x, str) for x in target)
        target = ", ".join(target)
    assert isinstance(target, str)
    return context + "\n\n" + turn["input"], target


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    model = Path("/data/models/Qwen3-8B")
    tokenizer = AutoTokenizer.from_pretrained(str(model), local_files_only=True)
    cache = Path("/data_ssd/qcrs/hf_cache/datasets")
    sources = sorted(cache.glob("RMT-team___babilong-1k-samples/[138]*k/*/*/*.arrow"))
    sources = [p for p in sources if p.parts[-4] in ("8k", "16k", "32k")]
    sources += sorted(cache.glob("microsoft___sc_bench/*/*/*/*.arrow"))
    summary = {
        "model": str(model),
        "max_position_embeddings": 40960,
        "generation_reserve": 256,
        "enable_thinking": False,
        "command": __import__("sys").argv,
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "model_files": {
            p.name: digest(p)
            for p in [
                model / "config.json",
                model / "tokenizer_config.json",
                model / "tokenizer.json",
            ]
        },
        "prompt_contract": {
            "babilong": "user: input + two newlines + question + newline + Answer:",
            "scbench": "user: context + two newlines + first turn input",
            "template": (
                "apply_chat_template(add_generation_prompt=True, enable_thinking=False)"
            ),
            "scbench_scope": "first turn; multi-turn validation remains separate",
            "truncation": "none; reject prompt+generation_reserve above 40960",
        },
        "sources": [],
    }
    resumed = collections.defaultdict(list)
    if args.resume_from:
        for line in (args.resume_from / "manifest.jsonl").open():
            record = json.loads(line)
            resumed[record["source"]].append(record)
        summary["resume_from"] = str(args.resume_from)
    summary["prompt_contract"]["scbench_mf"] = (
        "first input + two newlines + formatted integer list "
        "+ two newlines + first input"
    )
    summary["prompt_contract"]["scbench_vt"] = (
        "top-level input as context; answer list preserved as comma-separated labels"
    )
    with (args.out / "manifest.jsonl").open("w") as manifest:
        for path in sources:
            babi = "babilong" in str(path)
            table = pa.ipc.open_stream(pa.memory_map(str(path))).read_all()
            info = {
                "path": str(path),
                "sha256": digest(path),
                "rows": table.num_rows,
                "columns": table.column_names,
                "schema": str(table.schema),
            }
            metadata = json.loads((path.parent / "dataset_info.json").read_text())
            info["dataset_revision"] = path.parent.name
            info["declared_splits"] = metadata["splits"]
            lengths = []
            eligible = invalid = 0
            row_index = 0
            prior = resumed.get(str(path), [])
            if prior:
                assert len(prior) == table.num_rows
                assert [r["row"] for r in prior] == list(range(table.num_rows))
                assert all(r["source_sha256"] == info["sha256"] for r in prior)
                for record in prior:
                    manifest.write(json.dumps(record) + "\n")
                lengths = [r["prompt_tokens"] for r in prior]
                eligible = sum(r["eligible_first_turn"] for r in prior)
                invalid = sum(r["reason"] == "invalid fields" for r in prior)
                info["reused_manifest_rows"] = len(prior)
            for batch in [] if prior else table.to_batches(max_chunksize=16):
                rows = batch.to_pylist()
                contents, targets = zip(
                    *(first_turn(row, babi, path.parts[-4]) for row in rows)
                )
                texts = [
                    tokenizer.apply_chat_template(
                        [{"role": "user", "content": c}],
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )
                    for c in contents
                ]
                encoded = tokenizer(texts, add_special_tokens=False)["input_ids"]
                for row, text, ids, content, target in zip(
                    rows, texts, encoded, contents, targets
                ):
                    valid = bool(target.strip()) and bool(content.strip())
                    n = len(ids)
                    accepted = valid and n + 256 <= 40960
                    eligible += accepted
                    invalid += not valid
                    lengths.append(n)
                    manifest.write(
                        json.dumps(
                            {
                                "source": str(path),
                                "source_sha256": info["sha256"],
                                "row": row_index,
                                "dataset": "babilong" if babi else "scbench",
                                "subset": path.parts[-4] if babi else path.parts[-4],
                                "split": path.stem,
                                "record_id": row.get("id", row_index),
                                "prompt_tokens": n,
                                "eligible_first_turn": accepted,
                                "num_turns": 1 if babi else len(row["multi_turns"]),
                                "prompt_sha256": hashlib.sha256(
                                    text.encode()
                                ).hexdigest(),
                                "status": "PASS" if accepted else "BLOCKED",
                                "reason": None
                                if accepted
                                else "invalid fields or exceeds context limit",
                            }
                        )
                        + "\n"
                    )
                    row_index += 1
            info.update(
                {
                    "eligible_first_turn": eligible,
                    "invalid": invalid,
                    "tokens_min_p50_p95_max": np.percentile(
                        lengths, [0, 50, 95, 100]
                    ).tolist(),
                }
            )
            summary["sources"].append(info)
            manifest.flush()
            (args.out / "summary.partial.json").write_text(
                json.dumps(summary, indent=2)
            )
            print(
                json.dumps(
                    {
                        k: info[k]
                        for k in [
                            "path",
                            "rows",
                            "eligible_first_turn",
                            "tokens_min_p50_p95_max",
                        ]
                    }
                ),
                flush=True,
            )
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
