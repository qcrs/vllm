# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass

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
    member_to_cluster: torch.Tensor,
    member_to_column: torch.Tensor,
    page_group_size: int,
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
    '''
    它根据 member_to_cluster 提供的索引，
    从 physical_cluster_table 的第 1 维（通常是“列”）中把对应的数据抽取出来，
    拼成一张全新的表。
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
    virtual_table = cluster_table * page_group_size + column
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
vllm 原生接收得格式是 二维得 一页 然后这一页得位置
'''
def member_virtual_slots(
    physical_slots: torch.Tensor,
    member_to_cluster: torch.Tensor,
    member_to_column: torch.Tensor,
    page_group_size: int,
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
    member_slots = physical_slots.index_select(1, member_to_cluster)
    page = torch.div(member_slots, block_size, rounding_mode="floor")
    offset = torch.remainder(member_slots, block_size)
    column = member_to_column.to(dtype=physical_slots.dtype).view(1, -1)
    virtual_slots = (page * page_group_size + column) * block_size + offset
    return torch.where(member_slots == -1, -1, virtual_slots)


def member_seq_lens(
    physical_seq_lens: torch.Tensor,
    member_to_cluster: torch.Tensor,
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
    return physical_seq_lens.index_select(1, member_to_cluster)


@dataclass(frozen=True)
class RaggedStepViews:
    """Derived Ragged execution metadata for one scheduler step."""

    cluster_block_table: torch.Tensor
    member_block_table: torch.Tensor
    member_slot_mapping: torch.Tensor
    member_seq_lens: torch.Tensor
    query_start_loc: torch.Tensor
    num_actual_tokens: int
    page_group_size: int
    block_size: int
    max_query_len: int
    max_kv_len: int

'''
group_physical_slots() 基于 source effective length（当前 KV frontier）+
flattened query token 与 request 的映射关系 + cluster page table，
将每个新 token 在每个 cluster 上的逻辑位置解析成真实的 physical slot。
'''
def group_physical_slots(
    cluster_rows: torch.Tensor,
    source_effective_lens: torch.Tensor,
    query_start_loc: torch.Tensor,
    block_size: int,
    num_actual_tokens: int,
) -> torch.Tensor:
    """Build ``[Q, C]`` physical slots from source effective frontiers."""
    '''
    返回 [Q C] physical_slots
    flattened token q0/q1/q2 分别属于哪个 request、它在这个 request 内部是第几个新 token？
    '''
    '''
    query_start_loc = [0,2,3]
    request0:
    [0,2) → 2 tokens

    request1:
    [2,3) → 1 token
  `query_lens = [2,1]
    '''
    '''
    [1:] 从1 开始取到最后 [:-1] 从0开始取到 倒数 第二个 交差相减 得到结果
    query_lens.shape[0] 请求长度 request id
    第一个元素重复 query_lens[0] 次，第二个元素重复 query_lens[1] 次。
    就是说清楚 输入的token 是哪个请求 [0,query_lens]id 然后  query_lens.to(dtype=torch.long), 长度
    torch.repeat_interleave(x, 2) 每个元素分别重复
    '''
    query_lens = query_start_loc[1:] - query_start_loc[:-1]
    request_indices = torch.repeat_interleave(
        torch.arange(
            query_lens.shape[0],
            device=query_start_loc.device,
            dtype=torch.long,
        ),
        query_lens.to(dtype=torch.long),
        output_size=num_actual_tokens,
    )
    '''
    [0,1,2]
    '''
    token_indices = torch.arange(
        num_actual_tokens,
        device=query_start_loc.device,
        dtype=query_start_loc.dtype,
    )
    '''
    request_indices = [0,0,1]

    query_start_loc[:-1]
    = [0,2]
    [0,0,2]
    token_indices
    `[0,1,2]

    -
    request starts
    [0,0,2]

    =

    local_offsets
    [0,1,0]`
    Q0 = req0 第0个新 token
    Q1 = req0 第1个新 token
    Q2 = req1 第0个新 token
    假设是 [0,2,3] 得到[0,2] 就是每个request的起始位置 
    index_select 在某个维度上 给出 下表去元素

    [0,2].index_select(
    dim=0,
    index=[0,0,1],
    )
    [0,2].index_select(
    dim=0,
    index=[0,0,1],
    )
    得到他的启示位置

    ① 算每个 request 这轮有几个 query token

    ② 给 flattened 的每个 token 标记：
    “你属于哪个 request”

    ③ 给 flattened token 编全局编号：
    0,1,2,...

    ④ 找出每个 token 所属 request 的起始位置

    ⑤ 全局 token index - request 起点
    得到它在 request 内部的 local offset
    '''
    local_offsets = token_indices - query_start_loc[:-1].index_select(
        0, request_indices
    )
    physical_positions = source_effective_lens.index_select(
        0, request_indices
    ) + local_offsets.unsqueeze(1)
    page_depths = torch.div(physical_positions, block_size, rounding_mode="floor")
    block_offsets = torch.remainder(physical_positions, block_size)
    '''
    cluster_rows
    shape = [R, C, MaxPages]
    cluster_rows = [
        # req0
        [
            [10,12,14],   # cluster0
            [20,25,28],   # cluster1
        ],

        # req1
        [
            [30,31,32],   # cluster0
            [40,41,42],   # cluster1
        ],
    ]
    就是把cluster 转化为token row 原本是基于请求的 现在是基于 token的
    '''
    '''
    unsqueeze：增加一个维度（加一个轴）
    squeeze：删除一个长度为 1 的维度（去掉一个轴）
    '''
    token_rows = cluster_rows.index_select(0, request_indices)
    physical_pages = token_rows.gather(
        2, page_depths.to(dtype=torch.long).unsqueeze(2)
    ).squeeze(2)
    return physical_pages * block_size + block_offsets

'''
把一个 scheduler step 所需要的所有 Ragged execution metadata 一次性构造出来，
并封装成 immutable execution view
'''
def build_ragged_step_views(
    cluster_rows: torch.Tensor,
    source_effective_lens: torch.Tensor,
    query_start_loc: torch.Tensor,
    member_to_cluster: torch.Tensor,
    member_to_column: torch.Tensor,
    page_group_size: int,
    block_size: int,
    *,
    num_actual_tokens: int,
    max_query_len: int,
    max_kv_len: int,
) -> RaggedStepViews:
    '''
    输入分为两类： layout/state 输入
    execution shape
    '''
    """Derive reusable write/read views without rebuilding placement tensors."""
    physical_slots = group_physical_slots(
        cluster_rows,
        source_effective_lens,
        query_start_loc,
        block_size,
        num_actual_tokens,
    )
    query_lens = query_start_loc[1:] - query_start_loc[:-1]
    post_write_group_seq_lens = source_effective_lens + query_lens.unsqueeze(1)
    return RaggedStepViews(
        cluster_block_table=cluster_rows,
        member_block_table=member_virtual_block_table(
            cluster_rows,
            member_to_cluster,
            member_to_column,
            page_group_size,
        ),
        member_slot_mapping=member_virtual_slots(
            physical_slots,
            member_to_cluster,
            member_to_column,
            page_group_size,
            block_size,
        ),
        member_seq_lens=member_seq_lens(
            post_write_group_seq_lens,
            member_to_cluster,
        ),
        query_start_loc=query_start_loc,
        num_actual_tokens=num_actual_tokens,
        page_group_size=page_group_size,
        block_size=block_size,
        max_query_len=max_query_len,
        max_kv_len=max_kv_len,
    )
