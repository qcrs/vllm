# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import pytest

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
from vllm.v1.worker.gpu.ragged_kv_state import RaggedWorkerPhysicalState

pytestmark = pytest.mark.cpu_test


def make_control_plane():
    import torch

    block_size = 16
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
        num_clusters=4,
    )
    worker = RaggedWorkerPhysicalState(
        max_num_reqs=4,
        num_clusters=4,
        max_pages_per_cluster=8,
        block_size=block_size,
    )
    return scheduler, worker


def rows_from_snapshot(snapshot: RaggedRequestStateSnapshotData) -> list[list[int]]:
    rows = []
    offset = 0
    for count in snapshot.page_counts:
        rows.append(list(snapshot.flat_page_ids[offset : offset + count]))
        offset += count
    return rows


def assert_worker_matches_snapshot(
    worker: RaggedWorkerPhysicalState,
    req_index: int,
    snapshot: RaggedRequestStateSnapshotData,
) -> None:
    expected_rows = rows_from_snapshot(snapshot)
    assert tuple(worker.counts[req_index]) == snapshot.page_counts
    assert tuple(worker.effective_lens[req_index]) == snapshot.effective_lens
    for cluster, row in enumerate(expected_rows):
        assert worker.rows[req_index, cluster, : len(row)].tolist() == row


def test_scheduler_wire_worker_non_uniform_end_to_end_and_readd():
    scheduler, worker = make_control_plane()
    req_id = "req-a"
    req_index = 2
    initial_e = [32, 56, 20, 48]

    initial_plan = scheduler.plan_capacity(req_id, initial_e)
    scheduler.apply_capacity_plan(initial_plan)
    scheduler.commit_effective_lens(req_id, [0, 0, 0, 0], initial_e)
    snapshot = scheduler.export_snapshot(req_id)
    output = SchedulerOutput.make_empty()
    output.ragged_kv_updates = RaggedKVUpdateData(
        snapshots={req_id: snapshot}, allocations={}
    )

    assert output.ragged_kv_updates is not None
    worker.apply_snapshot(req_index, output.ragged_kv_updates.snapshots[req_id])
    assert snapshot.page_counts == (2, 4, 2, 3)
    assert_worker_matches_snapshot(worker, req_index, snapshot)

    next_e = [33, 57, 21, 49]
    plan = scheduler.plan_capacity(req_id, next_e)
    assert plan.appended_page_counts == (1, 0, 0, 1)
    delta = scheduler.apply_capacity_plan(plan)
    output.ragged_kv_updates = RaggedKVUpdateData(
        snapshots={}, allocations={req_id: delta}
    )
    worker.apply_allocation_delta(
        req_index, output.ragged_kv_updates.allocations[req_id]
    )

    assert scheduler.get_state(req_id).page_counts == (3, 4, 2, 4)
    assert scheduler.get_state(req_id).effective_lens == tuple(initial_e)
    assert tuple(worker.counts[req_index]) == (3, 4, 2, 4)
    assert tuple(worker.effective_lens[req_index]) == tuple(initial_e)

    scheduler.commit_effective_lens(req_id, initial_e, next_e)
    worker.commit_effective_lens(req_index, initial_e, next_e)
    committed_snapshot = scheduler.export_snapshot(req_id)
    assert_worker_matches_snapshot(worker, req_index, committed_snapshot)

    worker.remove_request(req_index)
    assert not worker.rows[req_index].any()
    worker.apply_snapshot(req_index, scheduler.export_snapshot(req_id))
    assert_worker_matches_snapshot(worker, req_index, committed_snapshot)


@pytest.mark.parametrize(
    "snapshot",
    [
        RaggedRequestStateSnapshotData("req", (16, 0), (1, 0), ()),
        RaggedRequestStateSnapshotData("req", (17, 0), (1, 0), (1,)),
        RaggedRequestStateSnapshotData("req", (16, 0), (1, 0), (0,)),
        RaggedRequestStateSnapshotData("req", (16, 16), (1, 1), (1, 1)),
    ],
)
def test_invalid_snapshot_is_atomic(snapshot):
    worker = RaggedWorkerPhysicalState(2, 2, 4, 16)
    baseline = RaggedRequestStateSnapshotData("req", (16, 0), (1, 0), (7,))
    worker.apply_snapshot(0, baseline)
    rows_before = worker.rows.copy()
    counts_before = worker.counts.copy()
    effective_before = worker.effective_lens.copy()

    with pytest.raises(ValueError):
        worker.apply_snapshot(0, snapshot)

    np.testing.assert_array_equal(worker.rows, rows_before)
    np.testing.assert_array_equal(worker.counts, counts_before)
    np.testing.assert_array_equal(worker.effective_lens, effective_before)


@pytest.mark.parametrize(
    "delta",
    [
        RaggedPageAllocationDeltaData(
            "req", (16, 0), (0, 0), (1, 0), (8,)
        ),
        RaggedPageAllocationDeltaData(
            "req", (16, 0), (1, 0), (1, 0), ()
        ),
        RaggedPageAllocationDeltaData(
            "req", (16, 0), (1, 0), (1, 1), (8, 8)
        ),
        RaggedPageAllocationDeltaData(
            "req", (16, 0), (1, 0), (1, 0), (7,)
        ),
        RaggedPageAllocationDeltaData(
            "req", (16, 0), (1, 0), (1, 0), (0,)
        ),
    ],
)
def test_invalid_or_stale_delta_is_atomic(delta):
    worker = RaggedWorkerPhysicalState(2, 2, 4, 16)
    worker.apply_snapshot(
        0, RaggedRequestStateSnapshotData("req", (16, 0), (1, 0), (7,))
    )
    rows_before = worker.rows.copy()
    counts_before = worker.counts.copy()
    effective_before = worker.effective_lens.copy()

    with pytest.raises(ValueError):
        worker.apply_allocation_delta(0, delta)

    np.testing.assert_array_equal(worker.rows, rows_before)
    np.testing.assert_array_equal(worker.counts, counts_before)
    np.testing.assert_array_equal(worker.effective_lens, effective_before)


def test_gather_returns_cluster_major_copies():
    worker = RaggedWorkerPhysicalState(4, 2, 3, 16)
    worker.apply_snapshot(
        3,
        RaggedRequestStateSnapshotData("a", (16, 32), (1, 2), (1, 2, 3)),
    )
    worker.apply_snapshot(
        1,
        RaggedRequestStateSnapshotData("b", (0, 16), (0, 1), (4,)),
    )

    view = worker.gather([3, 1])

    assert view.cluster_rows.shape == (2, 2, 3)
    assert view.page_counts.tolist() == [[1, 2], [0, 1]]
    assert view.effective_lens.tolist() == [[16, 32], [0, 16]]
    view.cluster_rows.fill(0)
    assert worker.rows[3, 0, 0] == 1


def test_frontier_commit_over_capacity_is_atomic():
    worker = RaggedWorkerPhysicalState(2, 2, 4, 16)
    worker.apply_snapshot(
        0, RaggedRequestStateSnapshotData("req", (16, 0), (1, 0), (7,))
    )
    effective_before = worker.effective_lens.copy()

    with pytest.raises(ValueError, match="exceeds"):
        worker.commit_effective_lens(0, [16, 0], [17, 0])

    np.testing.assert_array_equal(worker.effective_lens, effective_before)


def test_compaction_result_reports_shape_without_freed_page_ids():
    result = RaggedCompactionResultData(
        request_id="req",
        expected_source_effective_lens=(64, 64),
        expected_source_page_counts=(4, 4),
        new_effective_lens=(32, 48),
        new_page_counts=(2, 3),
        state_version=7,
    )
    assert result.new_page_counts == (2, 3)
    assert not hasattr(result, "freed_page_ids")


def test_scheduler_output_dense_default_has_no_ragged_updates():
    assert SchedulerOutput.make_empty().ragged_kv_updates is None
