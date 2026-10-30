# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker execution mirror and post-forward physical KV publication."""

from dataclasses import replace

import numpy as np
import torch

from vllm.triton_utils import tl, triton
from vllm.v1.kv_reclamation import (
    ReclaimRequest,
    ReclaimResult,
    ReclamationConfig,
    sink_recent_ranges,
    whole_block_row,
)
from vllm.v1.worker.gpu.buffer_utils import StagedWriteTensor
from vllm.v1.worker.gpu.input_batch import InputBatch, prepare_pos_seq_lens
from vllm.v1.worker.gpu.kv_compaction import StableKVCompactor
from vllm.v1.worker.gpu.kv_scoring import TriAttentionScorer, protected_topk


@triton.jit
def _advance_effective(lengths, indices, starts, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    mask = i < N
    idx = tl.load(indices + i, mask, other=0)
    q = tl.load(starts + i + 1, mask, other=0) - tl.load(starts + i, mask, other=0)
    value = tl.load(lengths + idx, mask, other=0)
    tl.store(lengths + idx, value + q, mask)


class KVReclamationWorker:
    def __init__(
        self, config: ReclamationConfig, req_states, input_buffers, model_config
    ):
        self.config = config
        self.req_states = req_states
        self.effective_kv_len = StagedWriteTensor(
            req_states.max_num_reqs, dtype=torch.int32, device=req_states.device
        )
        self.effective_kv_len_np = np.zeros(req_states.max_num_reqs, dtype=np.int32)
        self.cache_positions = torch.zeros_like(input_buffers.positions)
        self.kv_seq_lens = torch.zeros_like(input_buffers.seq_lens)
        self.scorer = (
            TriAttentionScorer.from_file(
                config.stats_path,
                model_config,
                model_config.max_model_len,
                req_states.device,
            )
            if config.policy == "triattention"
            else None
        )
        self.generator = torch.Generator(device=req_states.device).manual_seed(0)

    def initialize_cache(self, cache_config, layers) -> None:
        self.config.validate_cache(cache_config)
        self.layers = list(layers.values())
        for cache in self.layers:
            if cache.ndim != 4 or cache.dtype != torch.bfloat16:
                raise ValueError("KV reclamation requires packed 4D BF16 layer views")
            if cache.shape[2] != 16 or cache.stride(3) != 1:
                raise ValueError("KV reclamation requires B16 with contiguous channels")
        if len({cache.data_ptr() for cache in self.layers}) != len(self.layers):
            raise ValueError("KV-sharing layers are not supported by reclamation")
        self.compactor = StableKVCompactor(self.layers)

    def add_request(self, index: int, computed: int) -> None:
        self.effective_kv_len_np[index] = computed
        self.effective_kv_len.stage_write_elem(index, computed)
        self.effective_kv_len.apply_write()

    def update_requests(self, scheduler_output) -> None:
        for rid, value in (scheduler_output.effective_kv_lens or {}).items():
            idx = self.req_states.req_id_to_index[rid]
            assert self.effective_kv_len_np[idx] == value

    def prepare_attn(self, batch: InputBatch) -> InputBatch:
        prepare_pos_seq_lens(
            batch.idx_mapping,
            batch.query_start_loc,
            self.effective_kv_len.gpu,
            self.cache_positions,
            self.kv_seq_lens,
        )
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
        _advance_effective[(1,)](
            self.effective_kv_len.gpu,
            batch.idx_mapping,
            batch.query_start_loc,
            batch.num_reqs,
            triton.next_power_of_2(batch.num_reqs),
        )
        self.effective_kv_len_np[batch.idx_mapping_np] += batch.num_scheduled_tokens

    def reclaim(self, intent: ReclaimRequest, block_tables) -> ReclaimResult:
        idx = self.req_states.req_id_to_index[intent.request_id]
        assert self.effective_kv_len_np[idx] == intent.source_effective_kv_len
        prefix, recent = sink_recent_ranges(
            intent.source_effective_kv_len,
            intent.target_keep_tokens,
            self.config.sink_tokens,
            intent.protected_prefix_tokens,
        )
        row = None
        if self.config.policy == "sink_recent":
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
            if self.config.policy == "sink_recent":
                keep = torch.cat(
                    (
                        torch.arange(
                            prefix, dtype=torch.int32, device=self.req_states.device
                        ),
                        torch.arange(
                            recent,
                            source,
                            dtype=torch.int32,
                            device=self.req_states.device,
                        ),
                    )
                )
            else:
                recent = max(prefix, source - self.config.recent_tokens)
                if self.scorer is not None:
                    keep = self.scorer.plan(
                        self.layers,
                        block_tables.block_tables[0].gpu[idx],
                        self.req_states.num_computed_tokens.gpu[idx : idx + 1],
                        source,
                        intent.target_keep_tokens,
                        prefix,
                        recent,
                    )
                else:
                    scores = torch.rand(
                        source, device=self.req_states.device, generator=self.generator
                    )
                    keep = protected_topk(
                        scores, intent.target_keep_tokens, prefix, recent
                    )
            self.compactor.compact(block_tables.block_tables[0].gpu[idx], keep)
            b = block_tables.block_sizes[0]
            row = intent.source_block_ids[: (intent.target_keep_tokens + b - 1) // b]
        block_tables.append_block_ids(idx, (row,), overwrite=True)
        block_tables.apply_staged_writes()
        self.effective_kv_len_np[idx] = intent.target_keep_tokens
        self.effective_kv_len.stage_write_elem(idx, intent.target_keep_tokens)
        self.effective_kv_len.apply_write()
        return ReclaimResult(intent.request_id, intent.target_keep_tokens, row)
