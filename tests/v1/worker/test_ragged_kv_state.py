# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import replace

import numpy as np
import pytest
import torch

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.ragged_kv_cache_manager import RaggedAttentionManager
from vllm.v1.core.sched.output import (
    RaggedCompactionResultData,
    RaggedKVUpdateData,
    RaggedPageAllocationDeltaData,
    RaggedRequestStateSnapshotData,
    SchedulerOutput,
)
from vllm.v1.kv_cache_interface import RaggedAttentionSpec
from vllm.v1.ragged_kv_layout import MemberPlacementMap
from vllm.v1.worker.gpu.ragged_kv_state import RaggedWorkerPhysicalState

pytestmark = pytest.mark.cpu_test


def make_placement() -> MemberPlacementMap:
    return MemberPlacementMap.identity(
        num_layers=2,
        num_kv_heads=4,
        page_group_size=2,
    )


def make_control_plane():
    block_size = 16
    placement = make_placement()
    pool = BlockPool(
        num_gpu_blocks=64,
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
    scheduler = RaggedAttentionManager(
        kv_cache_spec=spec,
        block_pool=pool,
        enable_caching=False,
        kv_cache_group_id=0,
        scheduler_block_size=block_size,
        placement=placement,
    )
    worker = RaggedWorkerPhysicalState(
        max_num_reqs=4,
        placement=placement,
        max_pages_per_cluster=8,
        block_size=block_size,
    )
    return scheduler, worker


def snapshot(
    *,
    version: int = 3,
    effective_lens: tuple[int, ...] = (16, 0, 0, 0),
    page_counts: tuple[int, ...] = (1, 0, 0, 0),
    page_ids: tuple[int, ...] = (7,),
) -> RaggedRequestStateSnapshotData:
    return RaggedRequestStateSnapshotData(
        request_id="req",
        state_version=version,
        effective_lens=effective_lens,
        page_counts=page_counts,
        flat_page_ids=page_ids,
    )


def rows_from_snapshot(value: RaggedRequestStateSnapshotData) -> list[list[int]]:
    rows = []
    offset = 0
    for count in value.page_counts:
        rows.append(list(value.flat_page_ids[offset : offset + count]))
        offset += count
    return rows


def assert_worker_matches_snapshot(
    worker: RaggedWorkerPhysicalState,
    req_index: int,
    value: RaggedRequestStateSnapshotData,
) -> None:
    expected_rows = rows_from_snapshot(value)
    assert tuple(worker.counts[req_index]) == value.page_counts
    assert tuple(worker.effective_lens[req_index]) == value.effective_lens
    assert int(worker.state_versions[req_index]) == value.state_version
    for cluster, row in enumerate(expected_rows):
        assert worker.rows[req_index, cluster, : len(row)].tolist() == row


def test_scheduler_wire_worker_non_uniform_end_to_end_and_readd():
    scheduler, worker = make_control_plane()
    req_id = "req-a"
    req_index = 2
    initial_e = [32, 56, 20, 48]

    initial_plan = scheduler.plan_capacity(req_id, initial_e)
    initial_delta = scheduler.apply_capacity_plan(initial_plan)
    scheduler.commit_effective_lens(
        req_id, initial_delta.new_state_version, [0, 0, 0, 0], initial_e
    )
    initial_snapshot = scheduler.export_snapshot(req_id)
    output = SchedulerOutput.make_empty()
    output.ragged_kv_updates = RaggedKVUpdateData(
        snapshots={req_id: initial_snapshot}, allocations={}
    )

    worker.apply_snapshot(req_index, initial_snapshot)
    assert initial_snapshot.page_counts == (2, 4, 2, 3)
    assert_worker_matches_snapshot(worker, req_index, initial_snapshot)

    next_e = [33, 57, 21, 49]
    plan = scheduler.plan_capacity(req_id, next_e)
    delta = scheduler.apply_capacity_plan(plan)
    assert plan.appended_page_counts == (1, 0, 0, 1)
    assert delta.new_state_version == initial_snapshot.state_version + 1
    output.ragged_kv_updates = RaggedKVUpdateData(
        snapshots={}, allocations={req_id: delta}
    )
    worker.apply_allocation_delta(req_index, delta)

    assert scheduler.get_state(req_id).effective_lens == tuple(initial_e)
    assert tuple(worker.counts[req_index]) == (3, 4, 2, 4)
    assert tuple(worker.effective_lens[req_index]) == tuple(initial_e)
    assert int(worker.state_versions[req_index]) == delta.new_state_version

    scheduler.commit_effective_lens(
        req_id, delta.new_state_version, initial_e, next_e
    )
    worker.commit_effective_lens(
        req_index, delta.new_state_version, initial_e, next_e
    )
    committed_snapshot = scheduler.export_snapshot(req_id)
    assert_worker_matches_snapshot(worker, req_index, committed_snapshot)

    worker.remove_request(req_index)
    assert not worker.rows[req_index].any()
    assert worker.state_versions[req_index] == -1
    worker.apply_snapshot(req_index, committed_snapshot)
    assert_worker_matches_snapshot(worker, req_index, committed_snapshot)


@pytest.mark.parametrize(
    "invalid",
    [
        snapshot(page_counts=(1, 0, 0, 0), page_ids=()),
        snapshot(effective_lens=(17, 0, 0, 0)),
        snapshot(page_ids=(0,)),
        snapshot(
            effective_lens=(16, 16, 0, 0),
            page_counts=(1, 1, 0, 0),
            page_ids=(7, 7),
        ),
        snapshot(version=-1),
    ],
)
def test_invalid_snapshot_is_atomic(invalid):
    worker = RaggedWorkerPhysicalState(2, make_placement(), 4, 16)
    worker.apply_snapshot(0, snapshot())
    before = (
        worker.rows.copy(),
        worker.counts.copy(),
        worker.effective_lens.copy(),
        worker.state_versions.copy(),
    )

    with pytest.raises(ValueError):
        worker.apply_snapshot(0, invalid)

    for actual, expected in zip(
        (worker.rows, worker.counts, worker.effective_lens, worker.state_versions),
        before,
    ):
        np.testing.assert_array_equal(actual, expected)


def valid_delta() -> RaggedPageAllocationDeltaData:
    return RaggedPageAllocationDeltaData(
        request_id="req",
        expected_source_state_version=3,
        new_state_version=4,
        expected_source_effective_lens=(16, 0, 0, 0),
        expected_source_page_counts=(1, 0, 0, 0),
        appended_page_counts=(1, 0, 0, 0),
        flat_new_page_ids=(8,),
    )


@pytest.mark.parametrize(
    "delta",
    [
        replace(valid_delta(), expected_source_state_version=2),
        replace(valid_delta(), new_state_version=5),
        replace(valid_delta(), expected_source_effective_lens=(0, 0, 0, 0)),
        replace(valid_delta(), expected_source_page_counts=(0, 0, 0, 0)),
        replace(valid_delta(), flat_new_page_ids=()),
        replace(
            valid_delta(),
            appended_page_counts=(1, 1, 0, 0),
            flat_new_page_ids=(8, 8),
        ),
        replace(valid_delta(), flat_new_page_ids=(7,)),
        replace(valid_delta(), flat_new_page_ids=(0,)),
    ],
)
def test_invalid_or_stale_delta_is_atomic(delta):
    worker = RaggedWorkerPhysicalState(2, make_placement(), 4, 16)
    worker.apply_snapshot(0, snapshot())
    before = (
        worker.rows.copy(),
        worker.counts.copy(),
        worker.effective_lens.copy(),
        worker.state_versions.copy(),
    )

    with pytest.raises(ValueError):
        worker.apply_allocation_delta(0, delta)

    for actual, expected in zip(
        (worker.rows, worker.counts, worker.effective_lens, worker.state_versions),
        before,
    ):
        np.testing.assert_array_equal(actual, expected)


def test_equal_shape_different_generation_rejects_stale_delta():
    worker = RaggedWorkerPhysicalState(2, make_placement(), 4, 16)
    worker.apply_snapshot(0, snapshot(version=9))
    stale = valid_delta()

    with pytest.raises(ValueError, match="version is stale"):
        worker.apply_allocation_delta(0, stale)

    assert tuple(worker.counts[0]) == stale.expected_source_page_counts
    assert tuple(worker.effective_lens[0]) == stale.expected_source_effective_lens
    assert worker.state_versions[0] == 9


def test_gather_returns_cluster_major_copies_and_versions():
    worker = RaggedWorkerPhysicalState(4, make_placement(), 3, 16)
    worker.apply_snapshot(
        3,
        snapshot(
            version=5,
            effective_lens=(16, 32, 0, 0),
            page_counts=(1, 2, 0, 0),
            page_ids=(1, 2, 3),
        ),
    )
    worker.apply_snapshot(
        1,
        snapshot(
            version=7,
            effective_lens=(0, 16, 0, 0),
            page_counts=(0, 1, 0, 0),
            page_ids=(4,),
        ),
    )

    view = worker.gather([3, 1])

    assert view.cluster_rows.shape == (2, 4, 3)
    assert view.page_counts.tolist() == [[1, 2, 0, 0], [0, 1, 0, 0]]
    assert view.state_versions.tolist() == [5, 7]
    view.cluster_rows.fill(0)
    assert worker.rows[3, 0, 0] == 1


def test_frontier_commit_checks_version_and_capacity_atomically():
    worker = RaggedWorkerPhysicalState(2, make_placement(), 4, 16)
    worker.apply_snapshot(0, snapshot())
    before = worker.effective_lens.copy()

    with pytest.raises(ValueError, match="version is stale"):
        worker.commit_effective_lens(0, 2, [16, 0, 0, 0], [16, 0, 0, 0])
    with pytest.raises(ValueError, match="exceeds"):
        worker.commit_effective_lens(0, 3, [16, 0, 0, 0], [17, 0, 0, 0])

    np.testing.assert_array_equal(worker.effective_lens, before)
    assert worker.state_versions[0] == 3


def test_compaction_result_reports_versioned_shape_without_freed_page_ids():
    result = RaggedCompactionResultData(
        request_id="req",
        expected_source_state_version=7,
        expected_source_effective_lens=(64, 64),
        expected_source_page_counts=(4, 4),
        new_effective_lens=(32, 48),
        new_page_counts=(2, 3),
    )
    assert result.expected_source_state_version == 7
    assert not hasattr(result, "freed_page_ids")
    assert not hasattr(result, "retained_page_ids")


def test_ragged_update_rejects_snapshot_and_delta_for_same_request():
    with pytest.raises(ValueError, match="both"):
        RaggedKVUpdateData(
            snapshots={"req": snapshot()},
            allocations={"req": valid_delta()},
        )


def test_ragged_update_allows_different_request_channels():
    updates = RaggedKVUpdateData(
        snapshots={"snapshot-req": snapshot()},
        allocations={"delta-req": valid_delta()},
    )
    assert set(updates.snapshots) == {"snapshot-req"}
    assert set(updates.allocations) == {"delta-req"}


def test_scheduler_output_dense_default_has_no_ragged_updates():
    assert SchedulerOutput.make_empty().ragged_kv_updates is None
