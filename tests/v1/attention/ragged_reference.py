# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import torch

from vllm.v1.ragged_kv_layout import MemberPlacementMap, resolve_kv_address


def write_history_to_physical_cache(
    physical_cache: torch.Tensor,
    cluster_rows: torch.Tensor,
    placement: MemberPlacementMap,
    *,
    layer_idx: int,
    kv_head_idx: int,
    block_size: int,
    keys: torch.Tensor,
    values: torch.Tensor,
) -> None:
    """Populate logical history using only the scalar frozen address oracle."""
    active_row = tuple(tuple(row) for row in cluster_rows.cpu().tolist())
    head_size = keys.shape[-1]
    for position in range(keys.shape[0]):
        address = resolve_kv_address(
            layer_idx,
            kv_head_idx,
            position,
            block_size,
            active_row,
            placement,
        )
        physical_cache[
            address.physical_page_id,
            address.column_index,
            address.block_offset,
            :head_size,
        ] = keys[position]
        physical_cache[
            address.physical_page_id,
            address.column_index,
            address.block_offset,
            head_size:,
        ] = values[position]


def torch_ragged_attention_reference(
    query: torch.Tensor,
    current_key: torch.Tensor,
    current_value: torch.Tensor,
    histories: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]],
    query_start_loc: torch.Tensor,
    *,
    num_kv_heads: int,
    softmax_scale: float,
) -> torch.Tensor:
    """Compute causal GQA attention directly from logical member histories."""
    output = torch.empty_like(query)
    queries_per_kv_head = query.shape[1] // num_kv_heads
    starts = query_start_loc.cpu().tolist()
    for request_idx, (start, end) in enumerate(zip(starts[:-1], starts[1:])):
        for kv_head_idx in range(num_kv_heads):
            history_key, history_value = histories[(request_idx, kv_head_idx)]
            for local_idx, token_idx in enumerate(range(start, end)):
                visible_key = torch.cat(
                    (history_key, current_key[start : token_idx + 1, kv_head_idx]),
                    dim=0,
                ).float()
                visible_value = torch.cat(
                    (
                        history_value,
                        current_value[start : token_idx + 1, kv_head_idx],
                    ),
                    dim=0,
                ).float()
                for query_head_offset in range(queries_per_kv_head):
                    query_head_idx = (
                        kv_head_idx * queries_per_kv_head + query_head_offset
                    )
                    scores = (
                        query[token_idx, query_head_idx].float() @ visible_key.T
                    ) * softmax_scale
                    output[token_idx, query_head_idx] = (
                        torch.softmax(scores, dim=-1) @ visible_value
                    ).to(dtype=query.dtype)
    return output
