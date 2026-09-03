# SPDX-License-Identifier: Apache-2.0

import pytest

from tests.v1.core.utils import create_requests, create_scheduler
from vllm.v1.outputs import ModelRunnerOutput


LOCAL_MODEL = "/data/models/Qwen3-0.6B"


def _model_output(request_id: str, sampled_token_ids: list[int]) -> ModelRunnerOutput:
    return ModelRunnerOutput(
        req_ids=[request_id],
        req_id_to_index={request_id: 0},
        sampled_token_ids=[sampled_token_ids],
    )


def _create_reclaim_scheduler():
    return create_scheduler(
        model=LOCAL_MODEL,
        skip_tokenizer_init=True,
        max_num_seqs=2,
        max_num_batched_tokens=94,
        max_model_len=128,
        num_blocks=8,
        block_size=16,
        use_v2_model_runner=True,
    )


def _schedule_reclaim_step(scheduler):
    (request,) = create_requests(
        num_requests=1,
        num_tokens=97,
        req_ids=["request-a"],
        max_tokens=8,
        block_size=16,
    )
    scheduler.add_request(request)

    first_output = scheduler.schedule()
    assert first_output.num_scheduled_tokens[request.request_id] == 94
    scheduler.update_from_output(first_output, _model_output(request.request_id, []))

    old_row = scheduler.kv_cache_manager.get_block_ids(request.request_id)[0]
    assert len(old_row) == 6
    scheduler._set_prepared_reclaim_plan(request.request_id, (0, 1, 4, 5), 62)
    reclaim_output = scheduler.schedule()
    assert reclaim_output.num_scheduled_tokens[request.request_id] == 3
    transition = reclaim_output.scheduled_cached_reqs.reclaim_transitions[0]
    assert transition is not None
    new_ids = reclaim_output.scheduled_cached_reqs.new_block_ids[0]
    assert new_ids is not None and len(new_ids[0]) == 1
    return request, old_row, new_ids[0][0], reclaim_output


def test_reclaim_commit_releases_and_reuses_dense_blocks() -> None:
    scheduler = _create_reclaim_scheduler()
    request, old_row, new_block_id, reclaim_output = _schedule_reclaim_step(scheduler)
    pool = scheduler.kv_cache_manager.block_pool
    free_before_commit = pool.get_num_free_blocks()

    # The schedule/transport phase does not release the removed candidates.
    assert scheduler.kv_cache_manager.get_block_ids(request.request_id)[0] == [
        *old_row,
        new_block_id,
    ]
    assert pool.get_num_free_blocks() == free_before_commit
    assert request.effective_kv_len is None

    scheduler.update_from_output(reclaim_output, _model_output(request.request_id, [7]))

    removed_ids = {old_row[2], old_row[3]}
    final_row = [old_row[0], old_row[1], old_row[4], old_row[5], new_block_id]
    assert request.effective_kv_len == 65
    assert scheduler.kv_cache_manager.get_block_ids(request.request_id)[0] == final_row
    assert pool.get_num_free_blocks() == free_before_commit + 2
    assert all(pool.blocks[block_id].ref_cnt == 0 for block_id in removed_ids)

    next_output = scheduler.schedule()
    cached_index = next_output.scheduled_cached_reqs.req_ids.index(request.request_id)
    assert next_output.num_scheduled_tokens[request.request_id] == 1
    assert next_output.scheduled_cached_reqs.new_block_ids[cached_index] is None
    assert scheduler.kv_cache_manager.get_block_ids(request.request_id)[0] == final_row
    scheduler.update_from_output(next_output, _model_output(request.request_id, [8]))
    assert request.effective_kv_len == 66

    (other_request,) = create_requests(
        num_requests=1,
        num_tokens=1,
        req_ids=["request-b"],
        block_size=16,
    )
    scheduler.add_request(other_request)
    other_output = scheduler.schedule()
    allocated_to_other = scheduler.kv_cache_manager.get_block_ids(
        other_request.request_id
    )[0]
    assert other_output.num_scheduled_tokens[other_request.request_id] == 1
    assert len(allocated_to_other) == 1
    assert allocated_to_other[0] in removed_ids


def test_reclaim_validation_failure_is_fail_closed() -> None:
    scheduler = _create_reclaim_scheduler()
    request, old_row, new_block_id, reclaim_output = _schedule_reclaim_step(scheduler)
    pool = scheduler.kv_cache_manager.block_pool
    free_before_commit = pool.get_num_free_blocks()
    transition = reclaim_output.scheduled_cached_reqs.reclaim_transitions[0]
    assert transition is not None
    transition.expected_old_num_blocks += 1

    with pytest.raises(ValueError, match="Canonical block count"):
        scheduler.update_from_output(
            reclaim_output, _model_output(request.request_id, [7])
        )

    assert request.effective_kv_len is None
    assert scheduler.kv_cache_manager.get_block_ids(request.request_id)[0] == [
        *old_row,
        new_block_id,
    ]
    assert pool.get_num_free_blocks() == free_before_commit
    assert pool.blocks[old_row[2]].ref_cnt == 1
    assert pool.blocks[old_row[3]].ref_cnt == 1


def test_preemption_after_commit_frees_only_current_dense_ownership() -> None:
    scheduler = _create_reclaim_scheduler()
    request, old_row, _, reclaim_output = _schedule_reclaim_step(scheduler)
    scheduler.update_from_output(reclaim_output, _model_output(request.request_id, [7]))
    pool = scheduler.kv_cache_manager.block_pool
    removed_ids = {old_row[2], old_row[3]}
    assert all(pool.blocks[block_id].ref_cnt == 0 for block_id in removed_ids)

    scheduler.running.remove(request)
    scheduler._preempt_request(request, 0.0)

    assert request.effective_kv_len is None
    assert scheduler.kv_cache_manager.get_block_ids(request.request_id)[0] == []
    assert pool.get_num_free_blocks() == pool.num_gpu_blocks - 1
    assert all(pool.blocks[block_id].ref_cnt == 0 for block_id in removed_ids)


def test_normal_request_keeps_logical_allocation_path() -> None:
    scheduler = _create_reclaim_scheduler()
    (request,) = create_requests(
        num_requests=1,
        num_tokens=17,
        req_ids=["normal"],
        block_size=16,
    )
    scheduler.add_request(request)
    output = scheduler.schedule()

    assert request.effective_kv_len is None
    assert output.num_scheduled_tokens[request.request_id] == 17
    assert len(scheduler.kv_cache_manager.get_block_ids(request.request_id)[0]) == 2
