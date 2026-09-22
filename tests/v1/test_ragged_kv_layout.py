# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.v1.attention.backends.ragged_layout import (
    as_virtual_block_view,
    member_seq_lens,
    member_virtual_block_table,
    member_virtual_slots,
    placement_to_tensors,
)
from vllm.v1.ragged_kv_layout import (
    MemberPlacementMap,
    ResolvedKVAddress,
    resolve_kv_address,
)

pytestmark = pytest.mark.cpu_test


def test_identity_placement_derives_complete_cluster_topology():
    placement = MemberPlacementMap.identity(
        num_layers=2,
        num_kv_heads=4,
        page_group_size=2,
    )

    assert placement.num_members == 8
    assert placement.num_clusters == 4
    assert [
        placement.placement_for(layer, head)
        for layer in range(2)
        for head in range(4)
    ] == [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
        (2, 0),
        (2, 1),
        (3, 0),
        (3, 1),
    ]


def test_custom_valid_bijection_is_not_rewritten_as_identity():
    placement = MemberPlacementMap(
        num_layers=1,
        num_kv_heads=4,
        page_group_size=2,
        member_to_cluster=(1, 0, 1, 0),
        member_to_column=(1, 0, 0, 1),
    )

    assert placement.num_clusters == 2
    assert placement.placement_for(0, 0) == (1, 1)
    assert placement.placement_for(0, 3) == (0, 1)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {
                "member_to_cluster": (0, 0, 1, 1),
                "member_to_column": (0, 0, 0, 1),
            },
            "duplicate",
        ),
        (
            {
                "member_to_cluster": (0, 0, 0, 0),
                "member_to_column": (0, 1, 0, 1),
            },
            "duplicate",
        ),
        (
            {
                "member_to_cluster": (0, 0, 1, 2),
                "member_to_column": (0, 1, 0, 1),
            },
            "cluster is out of range",
        ),
        (
            {
                "member_to_cluster": (0, 0, 1, 1),
                "member_to_column": (0, 1, 0, 2),
            },
            "column is out of range",
        ),
        (
            {
                "member_to_cluster": (0, 0, 1),
                "member_to_column": (0, 1, 0),
            },
            "wrong member count",
        ),
    ],
)
def test_invalid_placement_is_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        MemberPlacementMap(
            num_layers=1,
            num_kv_heads=4,
            page_group_size=2,
            **kwargs,
        )


def test_non_divisible_topology_is_rejected():
    with pytest.raises(ValueError, match="divisible"):
        MemberPlacementMap(
            num_layers=1,
            num_kv_heads=3,
            page_group_size=2,
            member_to_cluster=(0, 0, 1),
            member_to_column=(0, 1, 0),
        )


def test_scalar_address_oracle_resolves_physical_position():
    placement = MemberPlacementMap.identity(
        num_layers=2,
        num_kv_heads=4,
        page_group_size=2,
    )
    address = resolve_kv_address(
        layer_idx=1,
        kv_head_idx=2,
        physical_position=21,
        block_size=16,
        active_row=((), (), (), (40, 41)),
        placement=placement,
    )

    assert address == ResolvedKVAddress(
        member_index=6,
        cluster_index=3,
        column_index=0,
        page_depth=1,
        block_offset=5,
        physical_page_id=41,
        virtual_block_id=82,
        virtual_slot=1317,
    )


@pytest.mark.parametrize("physical_position", [0, 15, 16, 31])
def test_scalar_address_oracle_block_boundaries(physical_position):
    placement = MemberPlacementMap.identity(
        num_layers=1,
        num_kv_heads=2,
        page_group_size=2,
    )
    address = resolve_kv_address(
        0,
        1,
        physical_position,
        16,
        ((10, 11),),
        placement,
    )
    expected_page = 10 if physical_position < 16 else 11
    assert address.physical_page_id == expected_page


def test_scalar_address_oracle_rejects_position_outside_owned_row():
    placement = MemberPlacementMap.identity(
        num_layers=1,
        num_kv_heads=2,
        page_group_size=2,
    )
    with pytest.raises(IndexError):
        resolve_kv_address(0, 0, 32, 16, ((10, 11),), placement)


def test_scalar_address_oracle_uses_custom_placement():
    placement = MemberPlacementMap(
        num_layers=1,
        num_kv_heads=4,
        page_group_size=2,
        member_to_cluster=(1, 0, 1, 0),
        member_to_column=(1, 0, 0, 1),
    )
    address = resolve_kv_address(
        0,
        0,
        16,
        16,
        ((7, 0), (11, 13)),
        placement,
    )
    assert address.member_index == 0
    assert address.cluster_index == 1
    assert address.column_index == 1
    assert address.physical_page_id == 13
    assert address.virtual_block_id == 27
    assert address.virtual_slot == 432


@pytest.mark.parametrize("page_group_size", [1, 2, 4])
def test_virtual_block_view_is_zero_copy(page_group_size):
    physical = torch.arange(
        3 * page_group_size * 4 * 6, dtype=torch.float32
    ).view(3, page_group_size, 4, 6)
    virtual = as_virtual_block_view(physical, page_group_size)

    assert virtual.shape == (3 * page_group_size, 1, 4, 6)
    assert physical.stride() == (page_group_size * 4 * 6, 4 * 6, 6, 1)
    assert virtual.stride() == (4 * 6, 4 * 6, 6, 1)
    assert virtual.data_ptr() == physical.data_ptr()
    assert virtual.untyped_storage().data_ptr() == physical.untyped_storage().data_ptr()
    physical[2, page_group_size - 1, 3, 5] = -123
    assert virtual[2 * page_group_size + page_group_size - 1, 0, 3, 5] == -123


def test_virtual_block_view_rejects_non_contiguous_input():
    physical = torch.zeros((3, 4, 2, 6), dtype=torch.float32).transpose(1, 2)
    with pytest.raises(ValueError, match="contiguous"):
        as_virtual_block_view(physical, 2)


def _scalar_member_virtual_block_table(
    physical_table: torch.Tensor,
    placement: MemberPlacementMap,
) -> torch.Tensor:
    rows, _, max_pages = physical_table.shape
    expected = torch.empty(
        (rows, placement.num_members, max_pages),
        dtype=physical_table.dtype,
        device=physical_table.device,
    )
    for row in range(rows):
        for member, (cluster, column) in enumerate(
            zip(placement.member_to_cluster, placement.member_to_column)
        ):
            for depth in range(max_pages):
                page = physical_table[row, cluster, depth].item()
                expected[row, member, depth] = (
                    0 if page == 0 else page * placement.page_group_size + column
                )
    return expected


def _scalar_member_virtual_slots(
    physical_slots: torch.Tensor,
    placement: MemberPlacementMap,
    block_size: int,
) -> torch.Tensor:
    queries, _ = physical_slots.shape
    expected = torch.empty(
        (queries, placement.num_members),
        dtype=physical_slots.dtype,
        device=physical_slots.device,
    )
    for query in range(queries):
        for member, (cluster, column) in enumerate(
            zip(placement.member_to_cluster, placement.member_to_column)
        ):
            slot = physical_slots[query, cluster].item()
            if slot == -1:
                expected[query, member] = -1
                continue
            page, offset = divmod(slot, block_size)
            expected[query, member] = (
                (page * placement.page_group_size + column) * block_size + offset
            )
    return expected


def test_member_metadata_transforms_match_scalar_addressing():
    placement = MemberPlacementMap.identity(
        num_layers=2,
        num_kv_heads=2,
        page_group_size=2,
    )
    physical_table = torch.tensor(
        [[[7, 8, 0], [11, 0, 0]], [[13, 0, 0], [17, 19, 0]]],
        dtype=torch.int32,
    )
    member_to_cluster, member_to_column = placement_to_tensors(placement)
    member_table = member_virtual_block_table(
        physical_table,
        member_to_cluster,
        member_to_column,
        placement.page_group_size,
    )
    torch.testing.assert_close(
        member_table,
        _scalar_member_virtual_block_table(physical_table, placement),
    )

    physical_slots = torch.tensor([[7 * 16 + 5, 11 * 16 + 15], [-1, 19 * 16]],
                                  dtype=torch.int32)
    member_slots = member_virtual_slots(
        physical_slots,
        member_to_cluster,
        member_to_column,
        placement.page_group_size,
        16,
    )
    torch.testing.assert_close(
        member_slots,
        _scalar_member_virtual_slots(physical_slots, placement, 16),
    )
    assert member_slots[1, 0].item() == -1
    assert member_slots[1, 1].item() == -1

    physical_lens = torch.tensor([[17, 32], [9, 48]], dtype=torch.int32)
    expected_lens = torch.tensor(
        [
            [
                physical_lens[row, cluster].item()
                for cluster in placement.member_to_cluster
            ]
            for row in range(physical_lens.shape[0])
        ],
        dtype=physical_lens.dtype,
    )
    torch.testing.assert_close(
        member_seq_lens(physical_lens, member_to_cluster), expected_lens
    )


def test_member_metadata_transforms_follow_custom_placement():
    placement = MemberPlacementMap(
        num_layers=1,
        num_kv_heads=4,
        page_group_size=2,
        member_to_cluster=(1, 0, 1, 0),
        member_to_column=(1, 0, 0, 1),
    )
    physical_table = torch.tensor([[[7, 0], [11, 13]]], dtype=torch.int32)
    member_to_cluster, member_to_column = placement_to_tensors(placement)
    torch.testing.assert_close(
        member_virtual_block_table(
            physical_table,
            member_to_cluster,
            member_to_column,
            placement.page_group_size,
        ),
        _scalar_member_virtual_block_table(physical_table, placement),
    )
