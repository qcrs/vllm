# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import torch

from vllm.v1.ragged_kv_layout import MemberPlacementMap


def ragged_physical_cache_shape(
    num_pages: int,
    page_group_size: int,
    block_size: int,
    head_size: int,
) -> tuple[int, int, int, int]:
    """Return the contiguous v0.26-native Ragged physical cache shape."""
    return (num_pages, page_group_size, block_size, 2 * head_size)


def as_virtual_block_view(
    physical_cache: torch.Tensor, page_group_size: int
) -> torch.Tensor:
    """View ``[P, Hp, B, 2D]`` as ``[P*Hp, 1, B, 2D]`` without copying."""
    if physical_cache.ndim != 4:
        raise ValueError("Ragged physical cache must have four dimensions")
    if physical_cache.shape[1] != page_group_size:
        raise ValueError("Ragged physical cache has the wrong page width")
    if not physical_cache.is_contiguous():
        raise ValueError("Ragged physical cache must be contiguous")
    return physical_cache.view(
        physical_cache.shape[0] * page_group_size,
        1,
        physical_cache.shape[2],
        physical_cache.shape[3],
    )


def placement_to_tensors(
    placement: MemberPlacementMap,
    *,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Materialize placement tuples as derived execution tensors."""
    return (
        torch.tensor(placement.member_to_cluster, dtype=torch.long, device=device),
        torch.tensor(placement.member_to_column, dtype=torch.long, device=device),
    )


def member_virtual_block_table(
    physical_cluster_table: torch.Tensor,
    placement: MemberPlacementMap,
) -> torch.Tensor:
    """Expand ``[R, C, MaxPages]`` physical rows to ``[R, M, MaxPages]``."""
    member_to_cluster, member_to_column = placement_to_tensors(
        placement, device=physical_cluster_table.device
    )
    cluster_table = physical_cluster_table.index_select(1, member_to_cluster)
    column = member_to_column.to(dtype=physical_cluster_table.dtype).view(1, -1, 1)
    virtual_table = cluster_table * placement.page_group_size + column
    return torch.where(cluster_table == 0, 0, virtual_table)


def member_virtual_slots(
    physical_slots: torch.Tensor,
    placement: MemberPlacementMap,
    block_size: int,
) -> torch.Tensor:
    """Expand ``[Q, C]`` physical slots to member-major virtual slots."""
    member_to_cluster, member_to_column = placement_to_tensors(
        placement, device=physical_slots.device
    )
    member_slots = physical_slots.index_select(1, member_to_cluster)
    page = torch.div(member_slots, block_size, rounding_mode="floor")
    offset = torch.remainder(member_slots, block_size)
    column = member_to_column.to(dtype=physical_slots.dtype).view(1, -1)
    virtual_slots = (page * placement.page_group_size + column) * block_size + offset
    return torch.where(member_slots == -1, -1, virtual_slots)


def member_seq_lens(
    physical_seq_lens: torch.Tensor,
    placement: MemberPlacementMap,
) -> torch.Tensor:
    """Gather cluster sequence lengths into member-major order."""
    member_to_cluster, _ = placement_to_tensors(
        placement, device=physical_seq_lens.device
    )
    return physical_seq_lens.index_select(1, member_to_cluster)
