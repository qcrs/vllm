# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import torch

from vllm.v1.attention.backends.fa_utils import (
    flash_attn_varlen_func,
    reshape_and_cache_flash,
)
from vllm.v1.attention.backends.ragged_layout import (
    RaggedStepViews,
    as_virtual_block_view,
)


def _virtual_kv_cache(
    physical_cache: torch.Tensor,
    page_group_size: int,
    head_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    '''
    把 vLLM 内部存储的 physical KV cache，转换成 Attention kernel 使用的 K/V 两个 view。
    physical_cache: torch.Tensor, [P Hp B 2D]
    '''
    virtual_cache = as_virtual_block_view(physical_cache, page_group_size)
    # [P*Hp,1,B,2D] 交换维度
    return virtual_cache.transpose(1, 2).split(head_size, dim=-1)


def ragged_kv_cache_update(
    key: torch.Tensor,
    value: torch.Tensor,
    physical_cache: torch.Tensor,
    views: RaggedStepViews,
    *,
    layer_idx: int,
    num_kv_heads: int,
    kv_cache_dtype: str,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
) -> None:
    """Write current-step K/V through the existing FA cache kernel."""
    '''
    将当前 step 的 K/V 按照 ragged layout 的 slot mapping 写入 paged KV cache。
    '''
    head_size = key.shape[-1]
    key_cache, value_cache = _virtual_kv_cache(
        physical_cache, views.page_group_size, head_size
    )
    member_start = layer_idx * num_kv_heads
    member_end = member_start + num_kv_heads
    layer_slots = views.member_slot_mapping[:, member_start:member_end]
    # [T*Hkv,1,D]
    reshape_and_cache_flash(
        key.reshape(-1, 1, head_size),
        value.reshape(-1, 1, head_size),
        key_cache,
        value_cache,
        layer_slots.reshape(-1).to(dtype=torch.long),
        kv_cache_dtype,
        k_scale,
        v_scale,
    )


def ragged_attention_forward(
    query: torch.Tensor,
    physical_cache: torch.Tensor,
    output: torch.Tensor,
    views: RaggedStepViews,
    *,
    layer_idx: int,
    num_kv_heads: int,
    softmax_scale: float,
    fa_version: int = 2,
) -> None:
    """Run focused FA2 Ragged decode or prefill/mixed attention."""
    '''
    当前层产生的Query
    T  = 当前 step 实际 query token 数
    Hq = query head 数
    D  = head_size

    physical_cache
    `=
    [P, Hp, B, 2D]` 包含所有 Ragged physical KV storage
    output.shape = [T,Hq,D]
    views RaggedStepViews 是最关键的 metadata。
    “当前这一步 Ragged attention 所需要的地址说明书”
    '''
    head_size = query.shape[-1]
    num_query_heads = query.shape[1]
    queries_per_kv_head = num_query_heads // num_kv_heads
    key_cache, value_cache = _virtual_kv_cache(
        physical_cache, views.page_group_size, head_size
    )

    member_start = layer_idx * num_kv_heads
    member_end = member_start + num_kv_heads
    layer_block_table = views.member_block_table[:, member_start:member_end]
    layer_seq_lens = views.member_seq_lens[:, member_start:member_end]
    num_requests = layer_block_table.shape[0]

    if views.max_query_len == 1:
        packed_query = query.reshape(
            num_requests * num_kv_heads,
            queries_per_kv_head,
            head_size,
        )
        packed_output = output.reshape_as(packed_query)
        block_table = layer_block_table.reshape(
            num_requests * num_kv_heads, -1
        )
        seq_lens = layer_seq_lens.reshape(-1)
        query_start_loc = torch.arange(
            num_requests * num_kv_heads + 1,
            dtype=torch.int32,
            device=query.device,
        )
        max_query_len = 1
    else:
        packed_query = (
            query.reshape(
                views.num_actual_tokens,
                num_kv_heads,
                queries_per_kv_head,
                head_size,
            )
            .permute(1, 0, 2, 3)
            .contiguous()
            .reshape(
                num_kv_heads * views.num_actual_tokens,
                queries_per_kv_head,
                head_size,
            )
        )
        packed_output = torch.empty_like(packed_query)
        block_table = (
            layer_block_table.permute(1, 0, 2)
            .contiguous()
            .reshape(num_kv_heads * num_requests, -1)
        )
        seq_lens = layer_seq_lens.transpose(0, 1).contiguous().reshape(-1)
        head_offsets = (
            torch.arange(
                num_kv_heads,
                dtype=views.query_start_loc.dtype,
                device=query.device,
            )
            * views.num_actual_tokens
        )
        query_start_loc = torch.cat(
            (
                (
                    views.query_start_loc[:-1].unsqueeze(0)
                    + head_offsets.unsqueeze(1)
                ).reshape(-1),
                views.query_start_loc.new_tensor(
                    [num_kv_heads * views.num_actual_tokens]
                ),
            )
        )
        max_query_len = views.max_query_len

    flash_attn_varlen_func(
        q=packed_query,
        k=key_cache,
        v=value_cache,
        out=packed_output,
        cu_seqlens_q=query_start_loc,
        max_seqlen_q=max_query_len,
        seqused_k=seq_lens,
        max_seqlen_k=views.max_kv_len,
        softmax_scale=softmax_scale,
        causal=True,
        block_table=block_table,
        fa_version=fa_version,
    )

    if views.max_query_len > 1:
        output.copy_(
            packed_output.view(
                num_kv_heads,
                views.num_actual_tokens,
                queries_per_kv_head,
                head_size,
            )
            .permute(1, 0, 2, 3)
            .reshape_as(output)
        )
