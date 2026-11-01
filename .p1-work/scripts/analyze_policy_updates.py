# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Recompute prepare, publication and retirement ordering from saved TP runs."""

import argparse
import hashlib
import json
from pathlib import Path


def analyze(run):
    evidence = json.loads((run / "evidence.json").read_text())
    guard = json.loads((run / "result.json").read_text())
    assert evidence["status"] == guard["status"] == "PASS"
    assert guard["exit_code"] == 0 and not guard["limit_triggered"]
    assert guard["peak_project_nvml_bytes"] < 44 * 2**30
    assert evidence["args"]["policy_updates"]
    assert evidence["free_final"] == 32
    assert all(item["finished"] for item in evidence["outputs"].values())
    updates = evidence["policy_events"]
    assert [u["binding"]["version"] for u in updates] == [1, 2, 3]
    assert [u["binding"]["config"]["policy"] for u in updates] == [
        "random",
        "triattention",
        "full",
    ]
    assert all(u["observed_old"] for u in updates[:2])
    assert [e["intent"]["policy_binding"]["version"] for e in evidence["events"]] == [
        0,
        1,
        2,
    ]
    paths = [run / "evidence.json", run / "result.json", run / "invocation.json"]
    ranks = []
    for rank in evidence["ranks"]:
        assert rank["policy_versions"] == [3]
        assert rank["counts"]["policy_prepares"] == 3
        assert rank["counts"]["fp32_attention"] > 0 and rank["rewritten"]
        directory = run / f"rank-{rank['rank']}"

        def marker(name, directory=directory):
            path = directory / name
            paths.append(path)
            return int(path.read_text())

        prepared, retired = [], []
        for version, update in enumerate(updates, 1):
            start = marker(f"prepare-start-{version}")
            end = marker(f"prepare-end-{version}")
            assert start < end <= update["published_ns"]
            prepared.append(
                dict(
                    version=version,
                    start_ns=start,
                    end_ns=end,
                    published_ns=update["published_ns"],
                )
            )
        for version, event in enumerate(evidence["events"]):
            done = marker(f"local-done-{version + 1}")
            retirement = marker(f"retire-{version}")
            assert done <= event["ack_ns"] < retirement
            results = event["rank_results"]
            assert all(
                (r["keep"], r["row"], r["E"])
                == (results[0]["keep"], results[0]["row"], results[0]["E"])
                for r in results
            )
            retired.append(
                dict(
                    version=version,
                    local_done_ns=done,
                    ack_ns=event["ack_ns"],
                    retired_ns=retirement,
                )
            )
        tri = next(m for m in rank["retire_memory"] if m["versions"] == [2])
        assert tri["before"] > tri["after"]
        ranks.append(
            dict(
                rank=rank["rank"],
                prepared=prepared,
                retired=retired,
                versions_final=rank["policy_versions"],
                tri_released_allocated_bytes=tri["before"] - tri["after"],
                policy_memory=rank["policy_memory"],
                retire_memory=rank["retire_memory"],
            )
        )
    assert len(ranks) == evidence["args"]["tp"]
    return dict(
        run=str(run),
        status="PASS",
        guard=guard,
        ranks=ranks,
        updates_s=[u["update_seconds"] for u in updates],
        cold_steps=evidence["cold_steps"],
        artifact_sha256={
            str(path.relative_to(run)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = [analyze(run) for run in args.runs]
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps([dict(run=r["run"], status=r["status"]) for r in result]))


if __name__ == "__main__":
    main()
