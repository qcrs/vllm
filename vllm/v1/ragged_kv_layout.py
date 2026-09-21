# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class MemberPlacementMap:
    """Static mapping from semantic KV members to physical page slots."""
    '''
    L   = 当前 Ragged layout 涵盖的 layer 数
    Hkv = 每层 local KV head 数
    Hp  = 一个 physical page 容纳几个 member
    '''
    num_layers: int
    num_kv_heads: int
    page_group_size: int
    # 实际mapping
    '''
    member_to_cluster = (
    0, 0,
    1, 1,
    2, 2,
    3, 3,
    )

    member_to_column = (
        0, 1,
        0, 1,
        0, 1,
        0, 1,
    )
    '''
    member_to_cluster: tuple[int, ...]
    member_to_column: tuple[int, ...]

    def __post_init__(self) -> None:
        if min(self.num_layers, self.num_kv_heads, self.page_group_size) <= 0:
            raise ValueError("Ragged placement dimensions must be positive")
        if self.num_members % self.page_group_size != 0:
            raise ValueError("Ragged member count must be divisible by page_group_size")
        if len(self.member_to_cluster) != self.num_members:
            raise ValueError("member_to_cluster has the wrong member count")
        if len(self.member_to_column) != self.num_members:
            raise ValueError("member_to_column has the wrong member count")
        '''
        组合成
        member_to_cluster = (0,0,1,1)
        member_to_column  = (0,1,0,1)
        '''
        slots = tuple(zip(self.member_to_cluster, self.member_to_column))
        if any(cluster < 0 or cluster >= self.num_clusters for cluster, _ in slots):
            raise ValueError("Ragged placement cluster is out of range")
        if any(column < 0 or column >= self.page_group_size for _, column in slots):
            raise ValueError("Ragged placement column is out of range")
        if len(set(slots)) != self.num_members:
            raise ValueError("Ragged placement contains duplicate physical slots")
        expected_slots = {
            (cluster, column)
            for cluster in range(self.num_clusters)
            for column in range(self.page_group_size)
        }
        if set(slots) != expected_slots:
            raise ValueError("Ragged placement must fill every physical slot")

    @property
    def num_members(self) -> int:
        # M = L × Hkv 当前 rank 上一共有多少份独立 semantic KV member。
        return self.num_layers * self.num_kv_heads

    @property
    def num_clusters(self) -> int:
        # 有多少clusters
        return self.num_members // self.page_group_size

    def flat_member_index(self, layer_idx: int, kv_head_idx: int) -> int:
        # 把二维的 (layer_idx, kv_head_idx) 转成一个一维的 member id。
        if not 0 <= layer_idx < self.num_layers:
            raise IndexError("layer_idx is out of range")
        if not 0 <= kv_head_idx < self.num_kv_heads:
            raise IndexError("kv_head_idx is out of range")
        return layer_idx * self.num_kv_heads + kv_head_idx

    def placement_for(self, layer_idx: int, kv_head_idx: int) -> tuple[int, int]:
        # 给我一个 (layer, head)，告诉我这个 member 被放在哪个 (cluster, column)。
        '''
        member_to_cluster = (
            0, 0,
            1, 1,
            2, 2,
            3, 3,
        )

        member_to_column = (
            0, 1,
            0, 1,
            0, 1,
            0, 1,
        )
        '''
        member = self.flat_member_index(layer_idx, kv_head_idx)
        return self.member_to_cluster[member], self.member_to_column[member]
    # 方法类 构造这方法
    # 后面的参数必须用关键字形式传递，不能按位置传。
    '''
    MemberPlacementMap.identity(
    num_layers=2,
    num_kv_heads=4,
    page_group_size=2,
    )
    forward reference（前向引用）
    返回值类型注解 类再函数内部还在创建 所以 类本身可能还没有完全绑定完成
    返回类型就是这个类。
    '''
    @classmethod
    def identity(
        cls,
        *,
        num_layers: int,
        num_kv_heads: int,
        page_group_size: int,
    ) -> "MemberPlacementMap":
        if page_group_size <= 0 or num_kv_heads % page_group_size != 0:
            raise ValueError("num_kv_heads must be divisible by page_group_size")
        '''
        num_layers = 2
        num_kv_heads = 4
        page_group_size = 2
        groups_per_layer
        = 4 // 2
        = 2
        layer0:
        head0, head1 -> cluster0
        head2, head3 -> cluster1

        layer1:
        head0, head1 -> cluster2
        head2, head3 -> cluster3

        (layer0, head0) -> cluster0, column0
        (layer0, head1) -> cluster0, column1
        (layer0, head2) -> cluster1, column0
        (layer0, head3) -> cluster1, column1

        (layer1, head0) -> cluster2, column0
        (layer1, head1) -> cluster2, column1
        (layer1, head2) -> cluster3, column0
        (layer1, head3) -> cluster3, column1
        '''
        groups_per_layer = num_kv_heads // page_group_size
        clusters = []
        columns = []
        for layer_idx in range(num_layers):
            for kv_head_idx in range(num_kv_heads):
                clusters.append(
                    layer_idx * groups_per_layer
                    + kv_head_idx // page_group_size
                )
                columns.append(kv_head_idx % page_group_size)
        return cls(
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            page_group_size=page_group_size,
            member_to_cluster=tuple(clusters),
            member_to_column=tuple(columns),
        )


@dataclass(frozen=True)
class ResolvedKVAddress:
    """Reference address for one semantic KV member and token position."""

    member_index: int
    cluster_index: int
    column_index: int
    page_depth: int
    block_offset: int
    physical_page_id: int
    virtual_block_id: int
    virtual_slot: int


def resolve_kv_address(
    layer_idx: int,
    kv_head_idx: int,
    physical_position: int,
    block_size: int,
    active_row: Sequence[Sequence[int]],
    placement: MemberPlacementMap,
) -> ResolvedKVAddress:
    """Resolve a physical token position through the canonical placement map."""
    member_index = placement.flat_member_index(layer_idx, kv_head_idx)
    if physical_position < 0:
        raise ValueError("physical_position must be non-negative")

    cluster_index = placement.member_to_cluster[member_index]
    column_index = placement.member_to_column[member_index]
    page_depth, block_offset = divmod(physical_position, block_size)
    cluster_row = active_row[cluster_index]
    if page_depth >= len(cluster_row):
        raise IndexError("physical_position is outside the supplied active row")

    physical_page_id = cluster_row[page_depth]
    if physical_page_id == 0:
        raise ValueError("resolved physical page cannot be the NULL page")

    virtual_block_id = physical_page_id * placement.page_group_size + column_index
    virtual_slot = virtual_block_id * block_size + block_offset
    return ResolvedKVAddress(
        member_index=member_index,
        cluster_index=cluster_index,
        column_index=column_index,
        page_depth=page_depth,
        block_offset=block_offset,
        physical_page_id=physical_page_id,
        virtual_block_id=virtual_block_id,
        virtual_slot=virtual_slot,
    )
