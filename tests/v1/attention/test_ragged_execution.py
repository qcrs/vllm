# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import math

import pytest
import torch

from tests.v1.attention.ragged_reference import (
    torch_ragged_attention_reference,
    write_history_to_physical_cache,
)
from vllm.v1.attention.backends.ragged_layout import (
    build_ragged_step_views,
    group_physical_slots,
    placement_to_tensors,
)
from vllm.v1.ragged_kv_layout import MemberPlacementMap, resolve_kv_address

BLOCK_SIZE = 16
PAGE_GROUP_SIZE = 2
NUM_KV_HEADS = 4
NUM_QUERY_HEADS = 8
HEAD_SIZE = 64
DTYPE = torch.bfloat16


def _query_start_loc(query_lens: list[int], device: torch.device) -> torch.Tensor:
    starts = [0]
    for query_len in query_lens:
        starts.append(starts[-1] + query_len)
    return torch.tensor(starts, dtype=torch.int32, device=device)


def _cluster_rows(
    num_requests: int,
    num_clusters: int,
    max_kv_len: int,
    device: torch.device,
) -> tuple[torch.Tensor, int]:
    max_pages = math.ceil(max_kv_len / BLOCK_SIZE)
    num_pages = num_requests * num_clusters * max_pages + 1
    rows = torch.arange(
        1,
        num_pages,
        dtype=torch.int32,
        device=device,
    ).view(num_requests, num_clusters, max_pages)
    return rows, num_pages


def _identity_placement() -> MemberPlacementMap:
    return MemberPlacementMap.identity(
        num_layers=1,
        num_kv_heads=NUM_KV_HEADS,
        page_group_size=PAGE_GROUP_SIZE,
    )


def _custom_placement() -> MemberPlacementMap:
    return MemberPlacementMap(
        num_layers=1,
        num_kv_heads=NUM_KV_HEADS,
        page_group_size=PAGE_GROUP_SIZE,
        member_to_cluster=(1, 0, 1, 0),
        member_to_column=(1, 0, 0, 1),
    )


def _forward_ops():
    from vllm.v1.attention.backends.ragged_forward import (
        ragged_attention_forward,
        ragged_kv_cache_update,
    )

    return ragged_attention_forward, ragged_kv_cache_update


def test_group_physical_slots_crosses_block_boundary_from_source_e():
    cluster_rows = torch.tensor([[[10, 11, 12], [20, 21, 22]]], dtype=torch.int32)
    source_e = torch.tensor([[15, 31]], dtype=torch.int32)
    query_start_loc = torch.tensor([0, 2], dtype=torch.int32)

    slots = group_physical_slots(
        cluster_rows,
        source_e,
        query_start_loc,
        BLOCK_SIZE,
        num_actual_tokens=2,
    )

    expected = torch.tensor(
        [[10 * 16 + 15, 21 * 16 + 15], [11 * 16, 22 * 16]],
        dtype=torch.int32,
    )
    torch.testing.assert_close(slots, expected)


def test_step_views_reuse_cached_placement_and_post_write_lens():
    placement = _custom_placement()
    member_to_cluster, member_to_column = placement_to_tensors(placement)
    cluster_rows = torch.tensor(
        [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]],
        dtype=torch.int32,
    )
    source_e = torch.tensor([[15, 31], [17, 5]], dtype=torch.int32)

    for query_lens in ([1, 1], [1, 3]):
        query_start_loc = _query_start_loc(query_lens, torch.device("cpu"))
        views = build_ragged_step_views(
            cluster_rows,
            source_e,
            query_start_loc,
            member_to_cluster,
            member_to_column,
            PAGE_GROUP_SIZE,
            BLOCK_SIZE,
            num_actual_tokens=sum(query_lens),
            max_query_len=max(query_lens),
            max_kv_len=max(
                source_e[request_idx].max().item() + query_len
                for request_idx, query_len in enumerate(query_lens)
            ),
        )
        expected_group_lens = source_e + torch.tensor(
            query_lens, dtype=source_e.dtype
        ).unsqueeze(1)
        torch.testing.assert_close(
            views.member_seq_lens,
            expected_group_lens.index_select(1, member_to_cluster),
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires real CUDA")
def test_real_cuda_write_hits_exact_cells_with_custom_placement():
    _, ragged_kv_cache_update = _forward_ops()
    device = torch.device("cuda:0")
    placement = _custom_placement()
    member_to_cluster, member_to_column = placement_to_tensors(
        placement, device=device
    )
    query_lens = [2, 1]
    source_e = torch.tensor([[15, 31], [17, 5]], dtype=torch.int32, device=device)
    query_start_loc = _query_start_loc(query_lens, device)
    max_kv_len = 33
    cluster_rows, num_pages = _cluster_rows(2, 2, max_kv_len, device)
    views = build_ragged_step_views(
        cluster_rows,
        source_e,
        query_start_loc,
        member_to_cluster,
        member_to_column,
        PAGE_GROUP_SIZE,
        BLOCK_SIZE,
        num_actual_tokens=3,
        max_query_len=2,
        max_kv_len=max_kv_len,
    )
    physical_cache = torch.zeros(
        (num_pages, PAGE_GROUP_SIZE, BLOCK_SIZE, 2 * HEAD_SIZE),
        dtype=DTYPE,
        device=device,
    )
    values = torch.arange(
        3 * NUM_KV_HEADS * HEAD_SIZE,
        dtype=torch.float32,
        device=device,
    ).view(3, NUM_KV_HEADS, HEAD_SIZE)
    key = (values / 1000).to(DTYPE)
    value = (-values / 2000).to(DTYPE)
    scale = torch.tensor(1.0, dtype=torch.float32, device=device)

    ragged_kv_cache_update(
        key,
        value,
        physical_cache,
        views,
        layer_idx=0,
        num_kv_heads=NUM_KV_HEADS,
        kv_cache_dtype="auto",
        k_scale=scale,
        v_scale=scale,
    )
    torch.cuda.synchronize()

    rows = cluster_rows.cpu().tolist()
    starts = query_start_loc.cpu().tolist()
    observed = []
    for request_idx, (start, end) in enumerate(zip(starts[:-1], starts[1:])):
        for token_idx in range(start, end):
            local_offset = token_idx - start
            for kv_head_idx in range(NUM_KV_HEADS):
                cluster_idx = placement.member_to_cluster[kv_head_idx]
                address = resolve_kv_address(
                    0,
                    kv_head_idx,
                    int(source_e[request_idx, cluster_idx]) + local_offset,
                    BLOCK_SIZE,
                    tuple(tuple(row) for row in rows[request_idx]),
                    placement,
                )
                torch.testing.assert_close(
                    physical_cache[
                        address.physical_page_id,
                        address.column_index,
                        address.block_offset,
                        :HEAD_SIZE,
                    ],
                    key[token_idx, kv_head_idx],
                    rtol=0,
                    atol=0,
                )
                torch.testing.assert_close(
                    physical_cache[
                        address.physical_page_id,
                        address.column_index,
                        address.block_offset,
                        HEAD_SIZE:,
                    ],
                    value[token_idx, kv_head_idx],
                    rtol=0,
                    atol=0,
                )
                observed.append(
                    (
                        address.physical_page_id,
                        address.column_index,
                        address.block_offset,
                    )
                )
    print(f"WRITE_EXACT_CELL_PASS addresses={observed}")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires real CUDA")
@pytest.mark.parametrize(
    "query_lens,source_e",
    [
        ([1], [[16, 31]]),
        ([1, 1], [[15, 31], [17, 5]]),
        ([3], [[15, 31]]),
        ([3, 2], [[15, 31], [17, 5]]),
        ([1, 3], [[15, 31], [17, 5]]),
    ],
)
def test_real_cuda_ragged_attention_matches_independent_reference(
    query_lens: list[int], source_e: list[list[int]]
):
    ragged_attention_forward, ragged_kv_cache_update = _forward_ops()
    device = torch.device("cuda:0")
    torch.manual_seed(7)
    placement = _identity_placement()
    member_to_cluster, member_to_column = placement_to_tensors(
        placement, device=device
    )
    source_effective_lens = torch.tensor(
        source_e, dtype=torch.int32, device=device
    )
    query_start_loc = _query_start_loc(query_lens, device)
    num_actual_tokens = sum(query_lens)
    max_kv_len = max(
        max(row) + query_len for row, query_len in zip(source_e, query_lens)
    )
    cluster_rows, num_pages = _cluster_rows(
        len(query_lens), 2, max_kv_len, device
    )
    views = build_ragged_step_views(
        cluster_rows,
        source_effective_lens,
        query_start_loc,
        member_to_cluster,
        member_to_column,
        PAGE_GROUP_SIZE,
        BLOCK_SIZE,
        num_actual_tokens=num_actual_tokens,
        max_query_len=max(query_lens),
        max_kv_len=max_kv_len,
    )
    physical_cache = torch.zeros(
        (num_pages, PAGE_GROUP_SIZE, BLOCK_SIZE, 2 * HEAD_SIZE),
        dtype=DTYPE,
        device=device,
    )
    histories: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}
    for request_idx in range(len(query_lens)):
        for kv_head_idx in range(NUM_KV_HEADS):
            cluster_idx = placement.member_to_cluster[kv_head_idx]
            history_len = source_e[request_idx][cluster_idx]
            history_key = (
                torch.randn(history_len, HEAD_SIZE, device=device) * 0.2
            ).to(DTYPE)
            history_value = (
                torch.randn(history_len, HEAD_SIZE, device=device) * 0.2
            ).to(DTYPE)
            histories[(request_idx, kv_head_idx)] = (
                history_key,
                history_value,
            )
            write_history_to_physical_cache(
                physical_cache,
                cluster_rows[request_idx],
                placement,
                layer_idx=0,
                kv_head_idx=kv_head_idx,
                block_size=BLOCK_SIZE,
                keys=history_key,
                values=history_value,
            )

    key = (
        torch.randn(num_actual_tokens, NUM_KV_HEADS, HEAD_SIZE, device=device)
        * 0.2
    ).to(DTYPE)
    value = (
        torch.randn(num_actual_tokens, NUM_KV_HEADS, HEAD_SIZE, device=device)
        * 0.2
    ).to(DTYPE)
    query = (
        torch.randn(num_actual_tokens, NUM_QUERY_HEADS, HEAD_SIZE, device=device)
        * 0.2
    ).to(DTYPE)
    scale_tensor = torch.tensor(1.0, dtype=torch.float32, device=device)
    softmax_scale = HEAD_SIZE**-0.5
    ragged_kv_cache_update(
        key,
        value,
        physical_cache,
        views,
        layer_idx=0,
        num_kv_heads=NUM_KV_HEADS,
        kv_cache_dtype="auto",
        k_scale=scale_tensor,
        v_scale=scale_tensor,
    )
    output = torch.empty_like(query)
    ragged_attention_forward(
        query,
        physical_cache,
        output,
        views,
        layer_idx=0,
        num_kv_heads=NUM_KV_HEADS,
        softmax_scale=softmax_scale,
        fa_version=2,
    )
    torch.cuda.synchronize()

    expected = torch_ragged_attention_reference(
        query,
        key,
        value,
        histories,
        query_start_loc,
        num_kv_heads=NUM_KV_HEADS,
        softmax_scale=softmax_scale,
    )
    max_error = (output.float() - expected.float()).abs().max().item()
    torch.testing.assert_close(output, expected, rtol=0.03, atol=0.03)
    mode = "decode" if max(query_lens) == 1 else "prefill_mixed"
    print(
        f"ATTENTION_REFERENCE_PASS mode={mode} query_lens={query_lens} "
        f"source_e={source_e} max_error={max_error:.6f}"
    )

@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires real CUDA")
def test_real_cuda_multilayer_layer1_write_and_decode():
    """
    D closure smoke:
    prove non-zero layer_idx uses the correct global member slice.

    L=2, Hkv=4, Hp=2:
      layer0 -> members 0..3 -> clusters 0..1
      layer1 -> members 4..7 -> clusters 2..3

    This test exercises layer_idx=1 through:
      1. RaggedStepViews global member metadata
      2. exact-cell CUDA KV write
      3. FA2 decode read
      4. independent PyTorch reference
    """
    ragged_attention_forward, ragged_kv_cache_update = _forward_ops()

    device = torch.device("cuda:0")
    torch.manual_seed(17)

    num_layers = 2
    layer_idx = 1

    placement = MemberPlacementMap.identity(
        num_layers=num_layers,
        num_kv_heads=NUM_KV_HEADS,
        page_group_size=PAGE_GROUP_SIZE,
    )

    member_to_cluster, member_to_column = placement_to_tensors(
        placement,
        device=device,
    )

    # Two decode requests.
    query_lens = [1, 1]
    query_start_loc = _query_start_loc(query_lens, device)

    # [R, C], where C=4 for L=2,Hkv=4,Hp=2.
    #
    # layer0 owns clusters 0,1
    # layer1 owns clusters 2,3
    #
    # Make layer1 frontiers deliberately non-uniform and near block boundaries.
    source_e = torch.tensor(
        [
            [3, 7, 15, 31],
            [5, 9, 17, 5],
        ],
        dtype=torch.int32,
        device=device,
    )

    assert placement.num_clusters == 4

    num_actual_tokens = sum(query_lens)
    max_kv_len = int(source_e.max().item()) + 1

    cluster_rows, num_pages = _cluster_rows(
        num_requests=2,
        num_clusters=placement.num_clusters,
        max_kv_len=max_kv_len,
        device=device,
    )

    views = build_ragged_step_views(
        cluster_rows,
        source_e,
        query_start_loc,
        member_to_cluster,
        member_to_column,
        PAGE_GROUP_SIZE,
        BLOCK_SIZE,
        num_actual_tokens=num_actual_tokens,
        max_query_len=1,
        max_kv_len=max_kv_len,
    )

    physical_cache = torch.zeros(
        (
            num_pages,
            PAGE_GROUP_SIZE,
            BLOCK_SIZE,
            2 * HEAD_SIZE,
        ),
        dtype=DTYPE,
        device=device,
    )

    # ------------------------------------------------------------------
    # Populate layer1 history through the independent scalar C1 oracle.
    # ------------------------------------------------------------------
    histories: dict[
        tuple[int, int],
        tuple[torch.Tensor, torch.Tensor],
    ] = {}

    for request_idx in range(2):
        for kv_head_idx in range(NUM_KV_HEADS):
            member_idx = layer_idx * NUM_KV_HEADS + kv_head_idx
            cluster_idx = placement.member_to_cluster[member_idx]
            history_len = int(source_e[request_idx, cluster_idx].item())

            history_key = (
                torch.randn(history_len, HEAD_SIZE, device=device) * 0.2
            ).to(DTYPE)
            history_value = (
                torch.randn(history_len, HEAD_SIZE, device=device) * 0.2
            ).to(DTYPE)

            histories[(request_idx, kv_head_idx)] = (
                history_key,
                history_value,
            )

            write_history_to_physical_cache(
                physical_cache,
                cluster_rows[request_idx],
                placement,
                layer_idx=layer_idx,
                kv_head_idx=kv_head_idx,
                block_size=BLOCK_SIZE,
                keys=history_key,
                values=history_value,
            )

    # ------------------------------------------------------------------
    # Current-step layer1 K/V and query.
    # ------------------------------------------------------------------
    key = (
        torch.randn(
            num_actual_tokens,
            NUM_KV_HEADS,
            HEAD_SIZE,
            device=device,
        )
        * 0.2
    ).to(DTYPE)

    value = (
        torch.randn(
            num_actual_tokens,
            NUM_KV_HEADS,
            HEAD_SIZE,
            device=device,
        )
        * 0.2
    ).to(DTYPE)

    query = (
        torch.randn(
            num_actual_tokens,
            NUM_QUERY_HEADS,
            HEAD_SIZE,
            device=device,
        )
        * 0.2
    ).to(DTYPE)

    scale_tensor = torch.tensor(
        1.0,
        dtype=torch.float32,
        device=device,
    )

    # ------------------------------------------------------------------
    # Real CUDA KV write for layer_idx=1.
    # ------------------------------------------------------------------
    ragged_kv_cache_update(
        key,
        value,
        physical_cache,
        views,
        layer_idx=layer_idx,
        num_kv_heads=NUM_KV_HEADS,
        kv_cache_dtype="auto",
        k_scale=scale_tensor,
        v_scale=scale_tensor,
    )

    torch.cuda.synchronize()

    # Verify current-step K/V landed in layer1's physical clusters,
    # not layer0's member slice.
    rows = cluster_rows.cpu().tolist()
    starts = query_start_loc.cpu().tolist()

    for request_idx, (start, end) in enumerate(
        zip(starts[:-1], starts[1:])
    ):
        for token_idx in range(start, end):
            local_offset = token_idx - start

            for kv_head_idx in range(NUM_KV_HEADS):
                member_idx = layer_idx * NUM_KV_HEADS + kv_head_idx
                cluster_idx = placement.member_to_cluster[member_idx]

                physical_position = (
                    int(source_e[request_idx, cluster_idx].item())
                    + local_offset
                )

                address = resolve_kv_address(
                    layer_idx=layer_idx,
                    kv_head_idx=kv_head_idx,
                    physical_position=physical_position,
                    block_size=BLOCK_SIZE,
                    active_row=tuple(
                        tuple(row) for row in rows[request_idx]
                    ),
                    placement=placement,
                )

                torch.testing.assert_close(
                    physical_cache[
                        address.physical_page_id,
                        address.column_index,
                        address.block_offset,
                        :HEAD_SIZE,
                    ],
                    key[token_idx, kv_head_idx],
                    rtol=0,
                    atol=0,
                )

                torch.testing.assert_close(
                    physical_cache[
                        address.physical_page_id,
                        address.column_index,
                        address.block_offset,
                        HEAD_SIZE:,
                    ],
                    value[token_idx, kv_head_idx],
                    rtol=0,
                    atol=0,
                )

    # ------------------------------------------------------------------
    # Real FA2 decode for layer_idx=1.
    # ------------------------------------------------------------------
    output = torch.empty_like(query)
    softmax_scale = HEAD_SIZE**-0.5

    ragged_attention_forward(
        query,
        physical_cache,
        output,
        views,
        layer_idx=layer_idx,
        num_kv_heads=NUM_KV_HEADS,
        softmax_scale=softmax_scale,
        fa_version=2,
    )

    torch.cuda.synchronize()

    expected = torch_ragged_attention_reference(
        query,
        key,
        value,
        histories,
        query_start_loc,
        num_kv_heads=NUM_KV_HEADS,
        softmax_scale=softmax_scale,
    )

    max_error = (
        output.float() - expected.float()
    ).abs().max().item()

    torch.testing.assert_close(
        output,
        expected,
        rtol=0.03,
        atol=0.03,
    )

    print(
        "MULTILAYER_LAYER1_PASS "
        f"layer_idx={layer_idx} "
        f"num_clusters={placement.num_clusters} "
        f"max_error={max_error:.6f}"
    )