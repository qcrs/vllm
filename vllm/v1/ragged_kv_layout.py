# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass


@dataclass(frozen=True)
class MemberPlacementMap:
    """Static mapping from semantic KV members to physical page slots."""

    num_layers: int
    num_kv_heads: int
    page_group_size: int
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
        return self.num_layers * self.num_kv_heads

    @property
    def num_clusters(self) -> int:
        return self.num_members // self.page_group_size

    def flat_member_index(self, layer_idx: int, kv_head_idx: int) -> int:
        if not 0 <= layer_idx < self.num_layers:
            raise IndexError("layer_idx is out of range")
        if not 0 <= kv_head_idx < self.num_kv_heads:
            raise IndexError("kv_head_idx is out of range")
        return layer_idx * self.num_kv_heads + kv_head_idx

    def placement_for(self, layer_idx: int, kv_head_idx: int) -> tuple[int, int]:
        member = self.flat_member_index(layer_idx, kv_head_idx)
        return self.member_to_cluster[member], self.member_to_column[member]

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
