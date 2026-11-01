# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker execution mirror and post-forward physical KV publication."""

from dataclasses import replace

import numpy as np
import torch

from vllm.distributed import get_tp_group
from vllm.v1.kv_reclamation import (
    PolicyBinding,
    ReclaimRequest,
    ReclaimResult,
    ReclamationConfig,
    sink_recent_ranges,
    whole_block_row,
)
from vllm.v1.worker.gpu.buffer_utils import StagedWriteTensor
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.kv_compaction import StableKVCompactor
from vllm.v1.worker.gpu.kv_retention_policy import POLICY_FACTORIES, RetentionPolicy


class KVReclamationWorker:
    def __init__(
        self, config: ReclamationConfig, req_states, input_buffers, model_config
    ):
        self.config = config
        self.model_config = model_config
        self.policies: dict[int, tuple[PolicyBinding, RetentionPolicy | None]] = {}
        self.slot_bindings: list[PolicyBinding | None] = [
            None
        ] * req_states.max_num_reqs
        self.req_states = req_states
        self.effective_kv_len = StagedWriteTensor(
            req_states.max_num_reqs, dtype=torch.int32, device=req_states.device
        )
        self.effective_kv_len_np = np.zeros(req_states.max_num_reqs, dtype=np.int32)
        self.cache_positions = torch.zeros_like(input_buffers.positions)
        self.kv_seq_lens = torch.zeros_like(input_buffers.seq_lens)
        self.tp_group = get_tp_group()
        self.local_reclaim_done = torch.cuda.Event()
        self.completion_token = torch.zeros(
            (), dtype=torch.int32, device=req_states.device
        )

    def initialize_cache(self, cache_config, layers, attn_layers) -> None:
        self.config.validate_cache(cache_config)
        for name in layers:
            if attn_layers[name].impl.alibi_slopes is not None:
                raise ValueError("KV reclamation does not support ALiBi position bias")
        self.layers = list(layers.values())
        for cache in self.layers:
            if cache.ndim != 4 or cache.dtype != torch.bfloat16:
                raise ValueError("KV reclamation requires packed 4D BF16 layer views")
            if cache.shape[2] != 16 or cache.stride(3) != 1:
                raise ValueError("KV reclamation requires B16 with contiguous channels")
        if len({cache.data_ptr() for cache in self.layers}) != len(self.layers):
            raise ValueError("KV-sharing layers are not supported by reclamation")
        self.compactor = StableKVCompactor(self.layers)
        self.initialize_policy(0, self.config)

    def initialize_policy(self, version: int, config: ReclamationConfig) -> dict:
        if version in self.policies:
            raise ValueError("KV policy version is already prepared")
        policy = (
            POLICY_FACTORIES[config.policy](
                config,
                self.model_config,
                self.req_states.device,
                self.tp_group,
                self.compactor,
            )
            if config.policy != "full"
            else None
        )
        fingerprint = (
            policy.scorer.calibration_sha256
            if policy is not None and policy.scorer is not None
            else None
        )
        self.policies[version] = (PolicyBinding(version, config, fingerprint), policy)
        return self.policy_ack(version)

    def prepare_policy(self, version: int, config: ReclamationConfig) -> dict:
        self.initialize_policy(version, config)
        policy = self.policies[version][1]
        if policy is not None:
            from vllm.v1.worker.gpu.kv_scoring import protected_topk

            length = min(128, self.model_config.max_model_len)
            scores = torch.zeros(length, device=self.req_states.device)
            if policy.scorer is not None:
                row = torch.zeros(
                    (length + 15) // 16,
                    dtype=torch.int32,
                    device=self.req_states.device,
                )
                logical = torch.full(
                    (1,), length, dtype=torch.int32, device=self.req_states.device
                )
                scores = policy.scorer.shared_scores(
                    self.layers, row, logical, length, 0, length
                )
            protected_topk(scores, length // 2, 0, length)
        # Preparation never writes KV or consumes the Random request RNG.
        torch.accelerator.synchronize()
        return self.policy_ack(version)

    def retire_policies(self, versions: list[int]) -> dict:
        # The engine has no logical references and prior output is committed.
        # Complete every device stream before dropping the version's resources.
        torch.accelerator.synchronize()
        for version in versions:
            del self.policies[version]
        return {"rank": self.tp_group.rank_in_group, "versions": versions}

    def policy_ack(self, version: int) -> dict:
        return {
            "rank": self.tp_group.rank_in_group,
            "binding": self.policies[version][0],
        }

    def add_request(
        self, index: int, computed: int, binding: PolicyBinding | None
    ) -> None:
        assert binding is not None and self.policies[binding.version][0] == binding
        self.slot_bindings[index] = binding
        self.effective_kv_len_np[index] = computed
        self.effective_kv_len.stage_write_elem(index, computed)
        self.effective_kv_len.apply_write()

    def remove_request(self, index: int) -> None:
        self.slot_bindings[index] = None

    def prepare_attn(self, batch: InputBatch) -> InputBatch:
        upper = np.zeros(batch.num_reqs_after_padding, dtype=np.int32)
        upper[: batch.num_reqs] = (
            self.effective_kv_len_np[batch.idx_mapping_np] + batch.num_scheduled_tokens
        )
        return replace(
            batch,
            positions=self.cache_positions[: batch.num_tokens_after_padding],
            seq_lens=self.kv_seq_lens[: batch.num_reqs_after_padding],
            seq_lens_cpu_upper_bound=torch.from_numpy(upper),
        )

    def advance(self, batch: InputBatch) -> None:
        self.effective_kv_len_np[batch.idx_mapping_np] += batch.num_scheduled_tokens

    def prepare_dummy_attn(self, batch: InputBatch) -> InputBatch:
        self.kv_seq_lens.zero_()
        self.kv_seq_lens[: batch.num_reqs].copy_(batch.seq_lens)
        return replace(batch, seq_lens=self.kv_seq_lens[: batch.num_reqs_after_padding])

    def reclaim(self, intent: ReclaimRequest, block_tables) -> ReclaimResult:
        idx = self.req_states.req_id_to_index[intent.request_id]
        binding = intent.policy_binding
        assert binding is not None and binding == self.slot_bindings[idx]
        prepared, policy = self.policies[binding.version]
        assert prepared == binding and policy is not None
        config = binding.config
        assert self.effective_kv_len_np[idx] == intent.source_effective_kv_len
        prefix, recent = sink_recent_ranges(
            intent.source_effective_kv_len,
            intent.target_keep_tokens,
            config.sink_tokens,
            intent.protected_prefix_tokens,
        )
        row = None
        if config.policy == "sink_recent":
            row = whole_block_row(
                intent.source_block_ids,
                intent.source_effective_kv_len,
                intent.target_keep_tokens,
                prefix,
                recent,
                block_tables.block_sizes[0],
            )
        if row is None:
            source = intent.source_effective_kv_len
            if config.policy != "sink_recent":
                recent = max(prefix, source - config.recent_tokens)
            scores = None
            if policy.local_scores is not None and (
                policy.sum_scores or self.tp_group.rank_in_group == 0
            ):
                scores = policy.local_scores(
                    self.layers,
                    block_tables.block_tables[0].gpu[idx],
                    self.req_states.num_computed_tokens.gpu[idx : idx + 1],
                    source,
                    prefix,
                    recent,
                )
                if policy.sum_scores and self.tp_group.world_size > 1:
                    torch.distributed.all_reduce(
                        scores, group=self.tp_group.device_group
                    )
            if self.tp_group.rank_in_group == 0:
                keep = policy.root_plan(
                    scores, source, intent.target_keep_tokens, prefix, recent
                )
            else:
                keep = torch.empty(
                    intent.target_keep_tokens,
                    dtype=torch.int32,
                    device=self.req_states.device,
                )
            if self.tp_group.world_size > 1:
                self.tp_group.broadcast(keep, src=0)
            self.compactor.compact(block_tables.block_tables[0].gpu[idx], keep)
            b = block_tables.block_sizes[0]
            row = intent.source_block_ids[: (intent.target_keep_tokens + b - 1) // b]
        block_tables.append_block_ids(idx, (row,), overwrite=True)
        block_tables.apply_staged_writes()
        self.effective_kv_len_np[idx] = intent.target_keep_tokens
        self.effective_kv_len.stage_write_elem(idx, intent.target_keep_tokens)
        self.effective_kv_len.apply_write()
        self.local_reclaim_done.record()
        if self.tp_group.world_size > 1:
            # ProcessGroup orders NCCL after local writes and gates the ACK stream.
            torch.distributed.all_reduce(
                self.completion_token, group=self.tp_group.device_group
            )
        return ReclaimResult(intent.request_id, intent.target_keep_tokens, row, binding)
