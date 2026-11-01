# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reproduce the finite TP2b Serving comparison from archived rounds."""

import json
from pathlib import Path

ROOT = Path("/data_ssd/qcrs/experiments/p1-v8/Engineering-Next-TP-Formal")
RUNS = {
    "A": "20261101-103008-fused-layer-counted-pressure",
    "B": "20261101-104017-grouped-counted-pressure",
    "B_reverse": "20261101-104240-grouped-counted-reverse",
    "A_reverse": "20261101-104407-layered-tp2a-counted-reverse",
}


def main():
    result = {}
    for label, name in RUNS.items():
        run = ROOT / name
        result[label] = {
            "run": str(run),
            "guard": json.loads((run / "result.json").read_text()),
            "warmup": json.loads((run / "warmup-summary.json").read_text()),
            "rounds": json.loads((run / "serving.json").read_text()),
        }
    result["token_comparison"] = {}
    for a, b in (("A", "B"), ("A_reverse", "B_reverse")):
        equal = []
        for repeat in range(3):
            rows = [
                json.loads((ROOT / RUNS[k] / f"requests-{repeat}.json").read_text())
                for k in (a, b)
            ]
            equal.extend(
                x["token_ids"] == y["token_ids"] for x, y in zip(*rows, strict=True)
            )
        result["token_comparison"][f"{a}/{b}"] = {
            "equal": sum(equal),
            "total": len(equal),
            "quality": "NOT_PROVEN",
        }
    (ROOT / "tp2b-serving-summary.json").write_text(json.dumps(result, indent=2))
    for label in RUNS:
        print(
            label,
            [
                (r["output_tokens_per_s"], r["slo_good_requests"])
                for r in result[label]["rounds"]
            ],
        )
    print(result["token_comparison"])


if __name__ == "__main__":
    main()
