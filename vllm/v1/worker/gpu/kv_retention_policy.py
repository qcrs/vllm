# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Controlled retention adapters; KV writes and TP protocol stay in the worker."""

from collections.abc import Callable
from dataclasses import dataclass

import torch

from vllm.v1.worker.gpu.kv_scoring import GroupedTriAttentionScorer, protected_topk


@dataclass
class RetentionPolicy:
    local_scores: Callable | None
    sum_scores: bool
    root_plan: Callable
    scorer: GroupedTriAttentionScorer | None = None
    generator: torch.Generator | None = None


def _sink(config, model, device, tp, compactor):
    def plan(scores, source, keep, prefix, recent):
        return torch.cat(
            (
                torch.arange(prefix, dtype=torch.int32, device=device),
                torch.arange(recent, source, dtype=torch.int32, device=device),
            )
        )

    return RetentionPolicy(None, False, plan)


def _topk(scores, source, keep, prefix, recent):
    return protected_topk(scores, keep, prefix, recent)


def _random(config, model, device, tp, compactor):
    generator = torch.Generator(device=device).manual_seed(0)

    def scores(layers, row, logical, source, prefix, recent):
        return torch.rand(source, device=device, generator=generator)

    return RetentionPolicy(scores, False, _topk, generator=generator)


def _triattention(config, model, device, tp, compactor):
    scorer = GroupedTriAttentionScorer.from_file(
        config.stats_path,
        model,
        model.max_model_len,
        device,
        tp.world_size,
        tp.rank_in_group,
    )
    scorer.initialize_groups(compactor)
    return RetentionPolicy(scorer.shared_scores, True, _topk, scorer=scorer)


POLICY_FACTORIES = {
    "sink_recent": _sink,
    "random": _random,
    "triattention": _triattention,
}
