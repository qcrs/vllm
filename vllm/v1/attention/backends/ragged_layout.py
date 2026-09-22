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
    '''
    Ragged 真正 physical KV layout
    =
    [P, Hp, B, 2D]

    P  = physical pages
    Hp = 一个 physical page 横向放多少 member
    B  = 每个 member 每页多少 token
    2D = K + V
    '''
    return (num_pages, page_group_size, block_size, 2 * head_size)


def as_virtual_block_view(
    physical_cache: torch.Tensor, page_group_size: int
) -> torch.Tensor:
    '''
    physical:
    [P, Hp, B, 2D]

        ↓ zero-copy view

    virtual:
    [P*Hp, 1, B, 2D]
    '''
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

# 转化为Tensor 需要Tensor 计算
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
    '''
     physical_cluster_table 这个是在说我们的token有多个需要多少个页来存储
    [R, C, MaxPages]
    R        = requests 数
    C        = clusters 数
    MaxPages = 每个 cluster table padding 后最大 page 数
    physical_cluster_table.shape = [1,2,2]

    physical_cluster_table =

    request0:
        cluster0 → [10,12]
        cluster1 → [20,25]
    [
    [
        [10,12],   # cluster0
        [20,25],   # cluster1
    ]
    ]
    '''
    """Expand ``[R, C, MaxPages]`` physical rows to ``[R, M, MaxPages]``."""
    member_to_cluster, member_to_column = placement_to_tensors(
        placement, device=physical_cluster_table.device
    )
    '''
    它根据 member_to_cluster 提供的索引，
    从 physical_cluster_table 的第 1 维（通常是“列”）中把对应的数据抽取出来，拼成一张全新的表。
    member_to_cluster = [0, 0, 1, 1]
    member_to_column  = [0, 1, 0, 1]
    R = 1 request
    C = 2 clusters
    M = 4 members
    Hp = 2
    MaxPages = 3
    member0 → cluster0,col0
    member1 → cluster0,col1

    member2 → cluster1,col0
    member3 → cluster1,col1
    physical_cluster_table = [
    [
        [10, 12, 0],   # cluster0
        [20, 25, 0],   # cluster1
    ]
    ]
    cluster0:
    depth0 → page10
    depth1 → page12
    depth2 → padding

    cluster1:
    depth0 → page20
    depth1 → page25
    depth2 → padding
    '''
    '''
    在指定维度 dim 上，按照 index 给出的下标重新取数据。
    所以是按照维度1来找
    cluster_table = [
    [
        [10, 12, 0],   # member0
        [10, 12, 0],   # member1
        [20, 25, 0],   # member2
        [20, 25, 0],   # member3
    ]
    ]

    '''
    cluster_table = physical_cluster_table.index_select(1, member_to_cluster)
    '''
    member_to_column
        =
        tensor([0,1,0,1])
    转化为  [1 M 1]
    '''
    column = member_to_column.to(dtype=physical_cluster_table.dtype).view(1, -1, 1)
    # 得到虚拟的table 维持 一个 head 一个 block 所以 要拆分 转化成 member这样
    virtual_table = cluster_table * placement.page_group_size + column
    # 问题在于
    '''
    [21,25,1] 这里的 1 是错的 所以 要转换
    '''
    return torch.where(cluster_table == 0, 0, virtual_table)

#把 cluster-major metadata 展开成 member-major metadata。
'''
Hp = 2
B  = 16

C = 2 clusters
M = 4 members

member0 → cluster0, column0
member1 → cluster0, column1
member2 → cluster1, column0
member3 → cluster1, column1
member_to_cluster = [0, 0, 1, 1]
member_to_column  = [0, 1, 0, 1]
'''
def member_virtual_slots(
    physical_slots: torch.Tensor,
    placement: MemberPlacementMap,
    block_size: int,
) -> torch.Tensor:
    """Expand ``[Q, C]`` physical slots to member-major virtual slots."""
    '''
    当前 step 里要写入 KV 的若干 token positions。
    [1,2]
    Q C
    physical_slots = [
        [165, 325]
    ]
    cluster0 → physical slot 165
    cluster1 → physical slot 325
    physical_slot
    =
    physical_page_id * B + offset
    B = 16 这里的 
    '''
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
    '''
    输入：[R,C]
    physical_seq_lens = [
    [32, 20]
    ]
    request0

    cluster0 effective KV len = 32
    cluster1 effective KV len = 20
    '''
    """Gather cluster sequence lengths into member-major order."""
    member_to_cluster, _ = placement_to_tensors(
        placement, device=physical_seq_lens.device
    )
    return physical_seq_lens.index_select(1, member_to_cluster)
