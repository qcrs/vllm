# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
from pathlib import Path

import torch

from vllm.v1.worker.gpu.kv_scoring import (
    _grouped_normalize,
    _normalize_reduce_scores,
    protected_topk,
)

out = []
for n in (2, 36):
    torch.manual_seed(11)
    raw = torch.randn(n, 9, 131, device="cuda")
    raw[:, :, 64:68] = raw[:, :, 64:65]
    raw[:, :, 68] = torch.nextafter(raw[:, :, 64], raw[:, :, 64] + 1)
    a = torch.zeros(131, device="cuda")
    b = torch.zeros_like(a)
    c = torch.zeros_like(a)
    tmp = torch.empty_like(a)
    vm = 0
    for first in range(0, n, 4):
        group = raw[first : first + 4]
        v, m = torch.var_mean(group[:, :, 3:128], dim=2, correction=0)
        for i in range(len(group)):
            lv, lm = torch.var_mean(group[i, :, 3:128], dim=1, correction=0)
            vm = max(vm, float((lv - v[i]).abs().max()), float((lm - m[i]).abs().max()))
            _normalize_reduce_scores[(2,)](
                group[i], lv, lm, tmp, 131, 131, 3, 3, 128, enable_fp_fusion=False
            )
            a.add_(tmp, alpha=1 / n)
            _normalize_reduce_scores[(2,)](
                group[i],
                v[i],
                m[i],
                c,
                131,
                131,
                3,
                3,
                128,
                1 / n,
                True,
                enable_fp_fusion=False,
            )
        _grouped_normalize[(2,)](
            group, v, m, b, 131, 131, 3, 3, len(group), 1 / n, enable_fp_fusion=False
        )
    out.append(
        dict(
            layers=n,
            statistics_max_difference=vm,
            score_max_difference=float((a - b).abs().max()),
            group_vs_individual_same_statistics=float((b - c).abs().max()),
            plan_equal=torch.equal(
                protected_topk(a, 68, 3, 128), protected_topk(b, 68, 3, 128)
            ),
        )
    )
Path(os.environ["P1_RUN_DIR"], "rounding.json").write_text(json.dumps(out, indent=2))
print(out)
