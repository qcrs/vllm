# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.v1.ragged_kv_layout import MemberPlacementMap

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
