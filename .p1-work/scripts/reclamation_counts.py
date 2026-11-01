# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in interval counters queried outside measured HTTP workload rounds."""


def install():
    import os
    from functools import wraps

    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.engine.core import EngineCore
    from vllm.v1.worker.gpu.kv_scoring import (
        GroupedTriAttentionScorer,
        TriAttentionScorer,
    )
    from vllm.v1.worker.gpu_worker import Worker

    counts = {"scorer_calls": 0, "canonical_commits": 0}
    score = TriAttentionScorer.shared_scores
    grouped_score = GroupedTriAttentionScorer.shared_scores
    if os.environ.get("P1_LAYERED_SCORER") == "1":
        grouped_score = score
    commit = KVCacheManager.commit_reclamation
    rpc = EngineCore.collective_rpc

    def counted_score(self, *args, **kwargs):
        counts["scorer_calls"] += 1
        return score(self, *args, **kwargs)

    def counted_grouped_score(self, *args, **kwargs):
        counts["scorer_calls"] += 1
        return grouped_score(self, *args, **kwargs)

    def counted_commit(self, *args, **kwargs):
        result = commit(self, *args, **kwargs)
        counts["canonical_commits"] += 1
        return result

    def worker_counts(self):
        return {"rank": self.rank, "scorer_calls": counts["scorer_calls"]}

    @wraps(rpc)
    def counted_rpc(self, method, *args, **kwargs):
        result = rpc(self, method, *args, **kwargs)
        if method == "p1_reclamation_counts":
            result.append({"canonical_commits": counts["canonical_commits"]})
        return result

    TriAttentionScorer.shared_scores = counted_score
    GroupedTriAttentionScorer.shared_scores = counted_grouped_score
    KVCacheManager.commit_reclamation = counted_commit
    Worker.p1_reclamation_counts = worker_counts
    EngineCore.collective_rpc = counted_rpc
