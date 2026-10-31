# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Independent complete-GQA reference against real two-GPU NCCL score/plan."""

import json
import os
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tests.v1.worker.kv_compaction_utils import scalar_retained_kv
from tests.v1.worker.test_kv_compaction import dense_frequency_reference
from vllm.v1.worker.gpu.kv_scoring import TriAttentionScorer, protected_topk


def worker(rank, run):
    torch.accelerator.set_device_index(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        "nccl",
        init_method="file://" + str(run / "store"),
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
        device_id=device,
    )
    results = []
    try:
        for kv_heads, ties in ((4, False), (1, False), (1, True)):
            generator = torch.Generator().manual_seed(142)
            source, logical, prefix, recent, keep = 47, 17031, 3, 41, 23
            row = torch.tensor([3, 1, 0], dtype=torch.int32, device=device)
            clock = torch.tensor([logical], dtype=torch.int32, device=device)
            stats = torch.randn(2, 8, 3, 64, generator=generator) * 0.2
            stats[:, :, 2] = stats[:, :, :2].square().sum(2).sqrt() + 0.3
            if ties:
                stats.zero_()
            cpu = [
                torch.randn(4, kv_heads, 16, 256, generator=generator).to(
                    torch.bfloat16
                )
                for _ in range(2)
            ]
            if ties:
                for value in cpu:
                    value.fill_(1)
            expected = []
            for layer, cache in enumerate(cpu):
                dense, _ = scalar_retained_kv(
                    [cache], row.cpu().tolist(), list(range(source))
                )
                raw = dense_frequency_reference(dense[0], stats[layer], logical, 1e6)
                var, mean = torch.var_mean(raw[:, prefix:recent], dim=1, correction=0)
                std = var.sqrt()
                norm = (raw - mean[:, None]) / torch.where(std > 0, std, 1)[:, None]
                expected.append(
                    norm.view(kv_heads, 8 // kv_heads, source).amax(1).mean(0)
                )
            reference = torch.stack(expected).mean(0)
            if ties:
                assert reference.count_nonzero() == 0
            local = [
                v.to(device)
                if kv_heads == 1
                else v[:, rank * 2 : (rank + 1) * 2].to(device)
                for v in cpu
            ]
            scorer = TriAttentionScorer(stats, kv_heads, 1e6, 64, 2, rank, device)
            scores = scorer.shared_scores(local, row, clock, source, prefix, recent)
            before = scores.clone()
            dist.all_reduce(scores)
            plan = (
                protected_topk(scores, keep, prefix, recent)
                if rank == 0
                else torch.empty(keep, dtype=torch.int32, device=device)
            )
            dist.broadcast(plan, src=0)
            received = [torch.empty_like(plan) for _ in range(2)]
            dist.all_gather(received, plan)
            torch.testing.assert_close(scores.cpu(), reference, atol=2e-3, rtol=1e-3)
            ranked = sorted(
                range(prefix, recent), key=lambda i: (-float(reference[i]), i)
            )
            selected = sorted(
                [
                    *range(prefix),
                    *ranked[: keep - prefix - (source - recent)],
                    *range(recent, source),
                ]
            )
            assert all(p.tolist() == selected for p in received)
            if ties:
                assert selected == [*range(17), *range(recent, source)]
            if kv_heads == 1 and rank == 1:
                assert before.count_nonzero() == 0 and scorer.stats.shape[1] == 0
            results.append(
                {
                    "kv_heads": kv_heads,
                    "ties": ties,
                    "scoring_owner": scorer.scoring_owner,
                    "loaded_query_heads": scorer.stats.shape[1],
                    "max_score_error": (scores.cpu() - reference).abs().max().item(),
                    "score_payload_bytes": 4 * source,
                    "plan_payload_bytes": 4 * keep,
                    "plan": selected,
                }
            )
        (run / f"rank-{rank}.json").write_text(json.dumps(results, indent=2))
    finally:
        dist.destroy_process_group()


def main():
    run = Path(os.environ["P1_RUN_DIR"])
    mp.spawn(worker, args=(run,), nprocs=2, join=True)
    print(json.dumps({"status": "PASS", "cases": 3, "ranks": 2}), flush=True)


if __name__ == "__main__":
    main()
