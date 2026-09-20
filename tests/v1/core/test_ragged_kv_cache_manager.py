# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import replace

import pytest
import torch

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.ragged_kv_cache_manager import RaggedAttentionManager
from vllm.v1.kv_cache_interface import RaggedAttentionSpec

pytestmark = pytest.mark.cpu_test


def make_manager(
    *, num_blocks: int = 64, num_clusters: int = 4
) -> tuple[RaggedAttentionManager, BlockPool]:
    block_size = 16
    pool = BlockPool(
        num_gpu_blocks=num_blocks,
        enable_caching=False,
        hash_block_size=block_size,
    )
    spec = RaggedAttentionSpec(
        block_size=block_size,
        num_kv_heads=4,
        head_size=8,
        dtype=torch.float32,
        page_group_size=2,
    )
    manager = RaggedAttentionManager(
        kv_cache_spec=spec,
        block_pool=pool,
        enable_caching=False,
        kv_cache_group_id=0,
        scheduler_block_size=block_size,
        num_clusters=num_clusters,
    )
    return manager, pool


def reserve_and_commit(
    manager: RaggedAttentionManager, request_id: str, effective_lens: list[int]
) -> None:
    plan = manager.plan_capacity(request_id, effective_lens)
    manager.apply_capacity_plan(plan)
    manager.commit_effective_lens(
        request_id, plan.source_effective_lens, effective_lens
    )


def test_non_uniform_plan_is_pure_and_reserve_does_not_advance_frontier():
    manager, pool = make_manager()
    free_before = pool.get_num_free_blocks()

    plan = manager.plan_capacity("req", [32, 56, 20, 48])

    assert plan.source_effective_lens == (0, 0, 0, 0)
    assert plan.source_page_counts == (0, 0, 0, 0)
    assert plan.required_page_counts == (2, 4, 2, 3)
    assert plan.appended_page_counts == (2, 4, 2, 3)
    assert plan.total_new_pages == 11
    assert "req" not in manager.req_to_ragged_state
    assert pool.get_num_free_blocks() == free_before

    delta = manager.apply_capacity_plan(plan)
    state = manager.get_state("req")
    assert state.page_counts == (2, 4, 2, 3)
    assert state.effective_lens == (0, 0, 0, 0)
    assert delta.expected_source_effective_lens == (0, 0, 0, 0)
    assert delta.appended_page_counts == (2, 4, 2, 3)
    assert len(delta.flat_new_page_ids) == 11


def test_non_uniform_incremental_scatter_and_frontier_commit():
    manager, _ = make_manager()
    reserve_and_commit(manager, "req", [32, 56, 20, 48])
    source = manager.get_state("req")

    plan = manager.plan_capacity("req", [33, 57, 21, 49])
    assert plan.appended_page_counts == (1, 0, 0, 1)
    delta = manager.apply_capacity_plan(plan)
    reserved = manager.get_state("req")

    assert reserved.page_counts == (3, 4, 2, 4)
    assert reserved.effective_lens == source.effective_lens
    assert reserved.page_rows[0][-1].block_id == delta.flat_new_page_ids[0]
    assert reserved.page_rows[3][-1].block_id == delta.flat_new_page_ids[1]

    manager.commit_effective_lens(
        "req", source.effective_lens, [33, 57, 21, 49]
    )
    assert manager.get_state("req").effective_lens == (33, 57, 21, 49)


def test_stale_or_malformed_plan_has_zero_pool_and_state_mutation():
    manager, pool = make_manager()
    stale = manager.plan_capacity("req", [16, 16, 16, 16])
    reserve_and_commit(manager, "req", [16, 0, 0, 0])
    state_before = manager.get_state("req")
    free_before = pool.get_num_free_blocks()

    with pytest.raises(ValueError, match="stale"):
        manager.apply_capacity_plan(stale)
    assert manager.get_state("req") == state_before
    assert pool.get_num_free_blocks() == free_before

    valid = manager.plan_capacity("req", [17, 0, 0, 0])
    malformed = replace(valid, appended_page_counts=(0, 0, 0, 0))
    with pytest.raises(ValueError, match="inconsistent"):
        manager.apply_capacity_plan(malformed)
    assert manager.get_state("req") == state_before
    assert pool.get_num_free_blocks() == free_before


def test_insufficient_pool_has_zero_canonical_mutation():
    manager, pool = make_manager(num_blocks=3)
    free_before = pool.get_num_free_blocks()
    plan = manager.plan_capacity("req", [16, 16, 16, 0])

    with pytest.raises(ValueError, match="Cannot get"):
        manager.apply_capacity_plan(plan)

    assert "req" not in manager.req_to_ragged_state
    assert pool.get_num_free_blocks() == free_before


def test_compaction_detaches_scheduler_owned_pages_and_reuses_them():
    manager, pool = make_manager()
    reserve_and_commit(manager, "req-a", [64, 64, 64, 64])
    old_state = manager.get_state("req-a")
    free_before = pool.get_num_free_blocks()

    detached_ids = manager.reconcile_compaction(
        "req-a",
        expected_source_effective_lens=[64, 64, 64, 64],
        expected_source_page_counts=[4, 4, 4, 4],
        new_effective_lens=[32, 48, 16, 64],
        new_page_counts=[2, 3, 1, 4],
    )

    assert manager.get_state("req-a").page_counts == (2, 3, 1, 4)
    assert set(detached_ids) == {
        block.block_id
        for row, count in zip(old_state.page_rows, (2, 3, 1, 4))
        for block in row[count:]
    }
    assert pool.get_num_free_blocks() == free_before + 6

    reserve_and_commit(manager, "req-b", [16, 16, 16, 16])
    reused = {
        block.block_id for row in manager.get_state("req-b").page_rows for block in row
    }
    assert reused.intersection(detached_ids)


def test_stale_compaction_has_zero_pool_and_state_mutation():
    manager, pool = make_manager()
    reserve_and_commit(manager, "req", [32, 48, 16, 64])
    state_before = manager.get_state("req")
    free_before = pool.get_num_free_blocks()

    with pytest.raises(ValueError, match="stale"):
        manager.reconcile_compaction(
            "req",
            expected_source_effective_lens=[32, 32, 16, 64],
            expected_source_page_counts=[2, 3, 1, 4],
            new_effective_lens=[16, 32, 16, 48],
            new_page_counts=[1, 2, 1, 3],
        )

    assert manager.get_state("req") == state_before
    assert pool.get_num_free_blocks() == free_before


def test_request_free_removes_ownership_before_pool_reuse():
    manager, pool = make_manager()
    reserve_and_commit(manager, "req-a", [16, 32, 16, 48])
    owned_ids = {
        block.block_id for row in manager.get_state("req-a").page_rows for block in row
    }
    free_before = pool.get_num_free_blocks()

    released = manager.free_request("req-a")

    assert "req-a" not in manager.req_to_ragged_state
    assert set(released) == owned_ids
    assert pool.get_num_free_blocks() == free_before + len(owned_ids)


def test_two_requests_keep_independent_cluster_rows():
    manager, _ = make_manager()
    reserve_and_commit(manager, "req-a", [16, 64, 32, 48])
    reserve_and_commit(manager, "req-b", [48, 16, 64, 32])

    ids_a = {
        block.block_id
        for row in manager.get_state("req-a").page_rows
        for block in row
    }
    ids_b = {
        block.block_id
        for row in manager.get_state("req-b").page_rows
        for block in row
    }
    assert ids_a.isdisjoint(ids_b)
    assert manager.get_state("req-a").page_counts == (1, 4, 2, 3)
    assert manager.get_state("req-b").page_counts == (3, 1, 4, 2)


def test_prefix_caching_and_dense_scalar_apis_fail_closed():
    manager, pool = make_manager()
    with pytest.raises(RuntimeError, match="vector capacity API"):
        manager.get_num_blocks_to_allocate()
    with pytest.raises(RuntimeError, match="vector capacity API"):
        manager.allocate_new_blocks()

    spec = manager.kv_cache_spec
    with pytest.raises(ValueError, match="prefix caching"):
        RaggedAttentionManager(
            kv_cache_spec=spec,
            block_pool=pool,
            enable_caching=True,
            kv_cache_group_id=0,
            scheduler_block_size=16,
            num_clusters=4,
        )
