# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tests.v1.core.test_scheduler import (
    create_requests_with_priority,
    create_scheduler_with_priority,
)
from vllm.config import VllmConfig
from vllm.v1.attention.backends.ragged_layout import placement_to_tensors
from vllm.v1.core.sched.output import (
    RaggedKVUpdateData,
    RaggedPageAllocationDeltaData,
    RaggedRequestStateSnapshotData,
)
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import RaggedAttentionSpec
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.ragged_kv_layout import MemberPlacementMap
from vllm.v1.request import RequestStatus
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.gpu.ragged_kv_state import RaggedWorkerPhysicalState

pytestmark = pytest.mark.cpu_test


def test_ragged_config_rejects_mrv1_runner():
    with pytest.raises(ValueError, match="requires the MRV2 model runner"):
        VllmConfig.validate_ragged_core(
            SimpleNamespace(
                cache_config=SimpleNamespace(page_group_size=2),
                use_v2_model_runner=False,
            )
        )


def _snapshot(
    req_id: str,
    version: int,
    effective_lens: tuple[int, int],
    page_counts: tuple[int, int],
    page_ids: tuple[int, ...],
) -> RaggedRequestStateSnapshotData:
    return RaggedRequestStateSnapshotData(
        request_id=req_id,
        state_version=version,
        effective_lens=effective_lens,
        page_counts=page_counts,
        flat_page_ids=page_ids,
    )


def _runner() -> GPUModelRunner:
    placement = MemberPlacementMap.identity(
        num_layers=1,
        num_kv_heads=4,
        page_group_size=2,
    )
    runner = object.__new__(GPUModelRunner)
    runner.device = torch.device("cpu")
    runner.kv_cache_config = SimpleNamespace(
        kv_cache_groups=[
            SimpleNamespace(
                kv_cache_spec=RaggedAttentionSpec(
                    block_size=16,
                    num_kv_heads=4,
                    head_size=8,
                    dtype=torch.float32,
                    page_group_size=2,
                )
            )
        ],
        num_blocks=64,
    )
    runner.ragged_worker_state = RaggedWorkerPhysicalState(
        max_num_reqs=8,
        placement=placement,
        max_pages_per_cluster=4,
        block_size=16,
    )
    (
        runner.ragged_member_to_cluster,
        runner.ragged_member_to_column,
    ) = placement_to_tensors(placement, device=runner.device)
    runner.ragged_layer_indices = {"layer.0": 0}
    return runner


def test_mrv2_transport_delta_reserves_capacity_without_advancing_frontier():
    runner = _runner()
    state_a = _snapshot("a", 3, (15, 15), (1, 1), (11, 21))
    state_b = _snapshot("b", 5, (7, 7), (1, 1), (12, 22))
    runner.req_states = SimpleNamespace(req_id_to_index={"a": 7, "b": 2})
    runner._apply_ragged_kv_updates(
        SimpleNamespace(
            ragged_kv_updates=RaggedKVUpdateData(
                snapshots={"a": state_a, "b": state_b}, allocations={}
            ),
            num_scheduled_tokens={"a": 1, "b": 1},
            scheduled_new_reqs=[
                SimpleNamespace(req_id="a"),
                SimpleNamespace(req_id="b"),
            ],
        )
    )
    assert tuple(runner.ragged_worker_state.counts[7]) == (1, 1)
    assert runner.ragged_worker_state.state_versions[7] == 3
    assert tuple(runner.ragged_worker_state.counts[2]) == (1, 1)
    assert runner.ragged_worker_state.state_versions[2] == 5

    delta = RaggedPageAllocationDeltaData(
        request_id="a",
        expected_source_state_version=3,
        new_state_version=4,
        expected_source_effective_lens=(15, 15),
        expected_source_page_counts=(1, 1),
        appended_page_counts=(1, 1),
        flat_new_page_ids=(13, 23),
    )
    runner._apply_ragged_kv_updates(
        SimpleNamespace(
            ragged_kv_updates=RaggedKVUpdateData(
                snapshots={}, allocations={"a": delta}
            ),
            num_scheduled_tokens={"a": 1},
            scheduled_new_reqs=[],
        )
    )

    assert tuple(runner.ragged_worker_state.effective_lens[7]) == (15, 15)
    assert tuple(runner.ragged_worker_state.counts[7]) == (2, 2)
    assert runner.ragged_worker_state.state_versions[7] == 4
    assert tuple(runner.ragged_worker_state.effective_lens[2]) == (7, 7)


def test_mrv2_step_gathers_and_commits_noncontiguous_request_slots():
    runner = _runner()
    snapshot_a = _snapshot("a", 3, (15, 31), (1, 2), (11, 21, 22))
    snapshot_b = _snapshot("b", 5, (7, 20), (1, 2), (12, 23, 24))
    runner.ragged_worker_state.apply_snapshot(7, snapshot_a)
    runner.ragged_worker_state.apply_snapshot(2, snapshot_b)
    input_batch = SimpleNamespace(
        idx_mapping_np=np.array([7, 2], dtype=np.int32),
        num_reqs=2,
        num_scheduled_tokens=np.array([1, 2], dtype=np.int32),
        query_start_loc=torch.tensor([0, 1, 3], dtype=torch.int32),
        num_tokens=3,
    )

    views, source = runner._prepare_ragged_step(input_batch)

    assert views.cluster_block_table[:, :, 0].tolist() == [[11, 21], [12, 23]]
    assert views.member_seq_lens.tolist() == [
        [16, 16, 32, 32],
        [9, 9, 22, 22],
    ]
    runner._commit_ragged_step(input_batch, source)

    np.testing.assert_array_equal(
        runner.ragged_worker_state.effective_lens[[7, 2]],
        np.array([[16, 32], [9, 22]], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        runner.ragged_worker_state.effective_lens[[0, 1]],
        np.zeros((2, 2), dtype=np.int32),
    )


def test_mrv2_ragged_transport_fails_closed_when_missing():
    runner = _runner()
    runner.req_states = SimpleNamespace(req_id_to_index={})

    with pytest.raises(RuntimeError, match="missing state transport"):
        runner._apply_ragged_kv_updates(
            SimpleNamespace(
                ragged_kv_updates=None,
                num_scheduled_tokens={},
                scheduled_new_reqs=[],
            )
        )


def test_scheduler_preempt_resume_materializes_full_snapshot_at_new_req_idx():
    """A resumed request is re-added at its current MRV2 slot via a snapshot."""
    spec = RaggedAttentionSpec(
        block_size=16,
        num_kv_heads=4,
        head_size=1,
        dtype=torch.float32,
        page_group_size=2,
    )
    scheduler: Scheduler = create_scheduler_with_priority(
        model="/data/models/Qwen3-0.6B",
        max_num_seqs=2,
        max_num_batched_tokens=200,
        num_blocks=9,
        block_size=16,
        use_v2_model_runner=True,
        kv_cache_spec=spec,
    )
    runner = _runner()
    runner.req_states = SimpleNamespace(req_id_to_index={"A": 7})

    def model_output(output):
        req_ids = list(output.num_scheduled_tokens)
        return ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index={req_id: i for i, req_id in enumerate(req_ids)},
            sampled_token_ids=[[100] for _ in req_ids],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        )

    request_a = create_requests_with_priority(
        1, [1], num_tokens=30, starting_idx=0, req_ids=["A"], max_tokens=100
    )[0]
    scheduler.add_request(request_a)
    first = scheduler.schedule()
    first_snapshot = first.ragged_kv_updates.snapshots["A"]
    runner._apply_ragged_kv_updates(first)
    scheduler.update_from_output(first, model_output(first))
    runner.ragged_worker_state.commit_effective_lens(7, 1, (0, 0), (30, 30))

    request_b = create_requests_with_priority(
        1, [0], num_tokens=32, starting_idx=1, req_ids=["B"], max_tokens=100
    )[0]
    scheduler.add_request(request_b)
    second = scheduler.schedule()
    runner.req_states.req_id_to_index["B"] = 6
    runner._apply_ragged_kv_updates(second)
    scheduler.update_from_output(second, model_output(second))
    runner.ragged_worker_state.commit_effective_lens(7, 1, (30, 30), (31, 31))
    runner.ragged_worker_state.commit_effective_lens(6, 1, (0, 0), (32, 32))

    third = scheduler.schedule()
    assert third.preempted_req_ids == {"A"}
    assert "A" not in third.num_scheduled_tokens
    runner.ragged_worker_state.remove_request(7)
    runner.req_states.req_id_to_index.pop("A")
    runner._apply_ragged_kv_updates(third)
    scheduler.update_from_output(third, model_output(third))

    scheduler.finish_requests("B", RequestStatus.FINISHED_STOPPED)
    fourth = scheduler.schedule()
    assert [req.req_id for req in fourth.scheduled_new_reqs] == ["A"]
    assert fourth.scheduled_cached_reqs.num_reqs == 0
    assert set(fourth.ragged_kv_updates.snapshots) == {"A"}
    assert fourth.ragged_kv_updates.allocations == {}

    resumed_snapshot = fourth.ragged_kv_updates.snapshots["A"]
    assert resumed_snapshot.state_version > first_snapshot.state_version
    assert resumed_snapshot.page_counts == (2, 2)
    assert resumed_snapshot.effective_lens == (0, 0)
    assert resumed_snapshot.flat_page_ids != first_snapshot.flat_page_ids

    # MRV2 add_requests assigns the current persistent slot; the old slot is
    # deliberately not reused as the source of the resumed materialization.
    runner.req_states.req_id_to_index["A"] = 2
    runner._apply_ragged_kv_updates(fourth)
    assert np.all(runner.ragged_worker_state.rows[7] == 0)
    assert tuple(runner.ragged_worker_state.counts[7]) == (0, 0)
    assert tuple(runner.ragged_worker_state.effective_lens[2]) == (0, 0)
    assert tuple(runner.ragged_worker_state.counts[2]) == (2, 2)
    assert runner.ragged_worker_state.state_versions[2] == resumed_snapshot.state_version
    np.testing.assert_array_equal(
        runner.ragged_worker_state.rows[2, :, :2].reshape(-1),
        np.asarray(resumed_snapshot.flat_page_ids, dtype=np.int32),
    )


def test_scheduler_frees_ragged_physical_pages_for_later_allocation():
    """Finishing A releases canonical pages back to the shared BlockPool."""
    spec = RaggedAttentionSpec(
        block_size=16,
        num_kv_heads=4,
        head_size=1,
        dtype=torch.float32,
        page_group_size=2,
    )
    scheduler: Scheduler = create_scheduler_with_priority(
        model="/data/models/Qwen3-0.6B",
        max_num_seqs=2,
        max_num_batched_tokens=200,
        num_blocks=12,
        block_size=16,
        use_v2_model_runner=True,
        kv_cache_spec=spec,
    )
    manager = scheduler.kv_cache_manager.ragged_manager
    assert manager is not None
    pool = scheduler.kv_cache_manager.block_pool
    initial_free = pool.get_num_free_blocks()

    request_a = create_requests_with_priority(
        1, [0], num_tokens=32, starting_idx=0, req_ids=["A"], max_tokens=64
    )[0]
    scheduler.add_request(request_a)
    output_a = scheduler.schedule()
    pages_a = output_a.ragged_kv_updates.snapshots["A"].flat_page_ids
    assert pages_a
    state_a = manager.req_to_ragged_state["A"]
    owned_a = tuple(page.block_id for row in state_a.page_rows for page in row)
    assert owned_a == pages_a
    free_with_a = pool.get_num_free_blocks()
    assert free_with_a == initial_free - len(pages_a)

    scheduler.finish_requests("A", RequestStatus.FINISHED_STOPPED)
    free_after_a = pool.get_num_free_blocks()
    a_released = "A" not in manager.req_to_ragged_state
    assert a_released
    assert free_after_a == initial_free

    request_b = create_requests_with_priority(
        1, [0], num_tokens=16, starting_idx=100, req_ids=["B"], max_tokens=64
    )[0]
    scheduler.add_request(request_b)
    output_b = scheduler.schedule()
    pages_b = output_b.ragged_kv_updates.snapshots["B"].flat_page_ids
    assert pages_b
    state_b = manager.req_to_ragged_state["B"]
    owned_b = tuple(page.block_id for row in state_b.page_rows for page in row)
    assert owned_b == pages_b
    assert pool.get_num_free_blocks() == free_after_a - len(pages_b)
    print(
        "PHYSICAL_PAGE_EVIDENCE "
        f"initial_free={initial_free} a_owned={owned_a} free_with_a={free_with_a} "
        f"a_absent_after_finish={a_released} "
        f"free_after_a={free_after_a} b_owned={owned_b} "
        f"free_after_b={pool.get_num_free_blocks()} "
        f"natural_id_overlap={sorted(set(owned_a) & set(owned_b))}"
    )
