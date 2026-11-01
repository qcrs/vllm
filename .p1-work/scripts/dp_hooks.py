# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Diagnostic-only replica-local scheduler and payload observations."""

import json
import os
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import torch
from tp_worker_hooks import members

from vllm.distributed import get_tp_group
from vllm.v1.core.kv_cache_metrics import KVCacheTrace
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_reclamation import sink_recent_ranges


def directory(config):
    path = Path(os.environ["P1_RUN_DIR"]) / (
        f"replica-{config.parallel_config.data_parallel_index}"
    )
    path.mkdir(exist_ok=True)
    return path


class P1DPScheduler(Scheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        path = directory(self.vllm_config)
        self.kv_cache_manager.trace = KVCacheTrace(str(path / "kv-trace.jsonl"))
        self._p1_trace = KVCacheTrace(str(path / "scheduler.jsonl"))
        self._p1_trace.record(
            "identity",
            pid=os.getpid(),
            engine=self.parallel_config.data_parallel_index,
            worker_dp_size=self.parallel_config.data_parallel_size,
            pool_id=id(self.kv_cache_manager.block_pool),
            total_blocks=self.kv_cache_config.num_blocks,
            free=self.kv_cache_manager.block_pool.get_num_free_blocks(),
        )

    def schedule(self, *args, **kwargs):
        output = super().schedule(*args, **kwargs)
        self._p1_trace.record(
            "schedule",
            free=self.kv_cache_manager.block_pool.get_num_free_blocks(),
            waiting=len(self.waiting),
            running=len(self.running),
            retry=self.reclaim_retry_pending,
            scheduled=output.num_scheduled_tokens,
            intent=asdict(output.reclaim_request) if output.reclaim_request else None,
            requests={
                rid: {
                    "L": r.num_computed_tokens,
                    "E": r.effective_kv_len,
                    "row": self.kv_cache_manager.get_block_ids(rid)[0],
                }
                for rid, r in self.requests.items()
            },
        )
        return output

    def update_from_output(self, *args, **kwargs):
        result = super().update_from_output(*args, **kwargs)
        self._p1_trace.record(
            "settled",
            free=self.kv_cache_manager.block_pool.get_num_free_blocks(),
            requests=list(self.requests),
        )
        return result


class P1DPWorker:
    def p1_install(self):
        runner = self.model_runner
        path = directory(self.vllm_config)
        group = get_tp_group()
        assert group.world_size == self.parallel_config.data_parallel_size == 1
        assert not self.model_config.is_moe
        worker = runner.reclamation
        context = self.vllm_config.compilation_config.static_forward_context
        layers = (
            worker.layers
            if worker
            else [
                layer.kv_cache
                for layer in context.values()
                if isinstance(getattr(layer, "kv_cache", None), torch.Tensor)
            ]
        )
        assert len(layers) == 36 and all(v.shape[1] == 8 for v in layers)
        identity = {
            "pid": os.getpid(),
            "engine": self.parallel_config.data_parallel_index,
            "worker_dp_size": self.parallel_config.data_parallel_size,
            "tp_size": group.world_size,
            "kv_heads": [v.shape[1] for v in layers],
            "kv_pointers": [v.data_ptr() for v in layers],
            "device": str(runner.device),
            "gpu_uuid": str(torch.cuda.get_device_properties(runner.device).uuid),
            "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        }
        (path / "worker.json").write_text(json.dumps(identity, indent=2))
        counts, references, released, rewritten = Counter(), {}, set(), set()
        rewritten_by_request = {}
        trace = KVCacheTrace(str(path / "worker-trace.jsonl"))
        pending = set()
        awaiting_rewrite = set()
        before_write = {}
        original_prepare = worker.prepare_attn if worker else None
        original_execute = runner.execute_model

        if worker:
            original_reclaim, original_compact = (
                worker.reclaim,
                worker.compactor.compact,
            )
            plan = None

            def compact(row, keep, **kwargs):
                nonlocal plan
                plan = keep.cpu().tolist()
                return original_compact(row, keep, **kwargs)

            def reclaim(intent, tables):
                nonlocal plan
                prefix, recent = sink_recent_ranges(
                    intent.source_effective_kv_len,
                    intent.target_keep_tokens,
                    worker.config.sink_tokens,
                    intent.protected_prefix_tokens,
                )
                plan = [*range(prefix), *range(recent, intent.source_effective_kv_len)]
                source = [
                    members(
                        v, intent.source_block_ids, intent.source_effective_kv_len
                    ).cpu()
                    for v in layers
                ]
                result = original_reclaim(intent, tables)
                expected = [value[plan] for value in source]
                for view, value in zip(layers, expected):
                    assert torch.equal(
                        members(view, result.retained_block_ids, len(plan)).cpu(), value
                    )
                references[intent.request_id] = (result.retained_block_ids, expected)
                removed = set(intent.source_block_ids) - set(result.retained_block_ids)
                pending.update(removed)
                awaiting_rewrite.update(removed)
                released.update(removed)
                counts["reclaim"] += 1
                trace.record(
                    "reclaim",
                    request_id=intent.request_id,
                    removed=sorted(removed),
                    retained=result.retained_block_ids,
                    E=result.new_effective_kv_len,
                    keep=plan,
                    local_done=worker.local_reclaim_done.query(),
                )
                return result

            worker.reclaim, worker.compactor.compact = reclaim, compact

            def prepare(batch):
                before_write.clear()
                if pending:
                    index = torch.tensor(sorted(pending), device=runner.device)
                    for view in layers:
                        view[index] = float("nan")
                    pending.clear()
                physical = original_prepare(batch)
                offset = 0
                for i, rid in enumerate(batch.req_ids):
                    idx, q = batch.idx_mapping_np[i], int(batch.num_scheduled_tokens[i])
                    logical = int(runner.req_states.num_computed_tokens_np[idx])
                    effective = int(worker.effective_kv_len_np[idx])
                    assert batch.positions[offset : offset + q].cpu().tolist() == list(
                        range(logical, logical + q)
                    )
                    assert physical.positions[
                        offset : offset + q
                    ].cpu().tolist() == list(range(effective, effective + q))
                    assert int(physical.seq_lens[i]) == effective + q
                    row = runner.block_tables.block_tables[0].gpu[idx].cpu().tolist()
                    for p in range(effective, effective + q):
                        page, slot = row[p // 16], p % 16
                        if page in awaiting_rewrite:
                            before_write[rid, page, slot] = [
                                v[page, :, slot, :].clone() for v in layers
                            ]
                    offset += q
                return physical

            worker.prepare_attn = prepare

        def execute(*args, **kwargs):
            answer = original_execute(*args, **kwargs)
            current = runner.execute_model_state
            if current is None:
                return answer
            for (rid, page, slot), values in before_write.items():
                for view, previous in zip(layers, values):
                    value = view[page, :, slot, :]
                    assert torch.isfinite(value).all()
                    assert not torch.equal(
                        value.view(torch.int16), previous.view(torch.int16)
                    )
                rewritten.add(page)
                rewritten_by_request.setdefault(rid, set()).add(page)
            awaiting_rewrite.difference_update(page for _, page, _ in before_write)
            active = runner.req_states.req_id_to_index
            peer_verified = []
            for rid, (row, values) in references.items():
                if rid in active:
                    for view, value in zip(layers, values):
                        assert torch.equal(members(view, row, len(value)).cpu(), value)
                    counts["retained_forward"] += 1
                    if rid.startswith("B-"):
                        peer_verified.append(rid)
                        counts["peer_retained_forward"] += 1
            batch = current.input_batch
            for i, rid in enumerate(batch.req_ids):
                if not rid.startswith("B-"):
                    continue
                idx = batch.idx_mapping_np[i]
                logical = int(runner.req_states.num_computed_tokens_np[idx])
                if logical + int(batch.num_scheduled_tokens[i]) < 128:
                    continue
                if rid not in references:
                    row = runner.block_tables.block_tables[0].gpu[idx, :8].tolist()
                    references[rid] = (
                        row,
                        [members(v, row, 128).cpu() for v in layers],
                    )
            counts["forward"] += 1
            trace.record("forward", requests=batch.req_ids, peer_verified=peer_verified)
            (path / "payload.json").write_text(
                json.dumps(
                    {
                        "counts": dict(counts),
                        "released": sorted(released),
                        "rewritten": sorted(rewritten),
                        "rewritten_by_request": {
                            rid: sorted(pages)
                            for rid, pages in rewritten_by_request.items()
                        },
                        "time_ns": time.monotonic_ns(),
                    }
                )
            )
            return answer

        runner.execute_model = execute
        return identity
