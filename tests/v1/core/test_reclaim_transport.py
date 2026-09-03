# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from tests.v1.core.utils import create_requests, create_scheduler
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.core.sched.output import CachedRequestData


LOCAL_MODEL = "/data/models/Qwen3-0.6B"


def test_cached_request_reclaim_transport_is_aligned_after_allocation() -> None:
    scheduler = create_scheduler(
        model=LOCAL_MODEL,
        skip_tokenizer_init=True,
        max_num_seqs=1,
        max_num_batched_tokens=94,
        max_model_len=128,
        num_blocks=8,
        block_size=16,
        use_v2_model_runner=True,
    )
    (request,) = create_requests(
        num_requests=1,
        num_tokens=97,
        req_ids=["request-a"],
        block_size=16,
    )
    scheduler.add_request(request)

    first_output = scheduler.schedule()
    assert first_output.num_scheduled_tokens[request.request_id] == 94
    old_row = scheduler.kv_cache_manager.get_block_ids(request.request_id)[0]
    assert len(old_row) == 6

    scheduler._set_prepared_reclaim_plan(request.request_id, (0, 1, 4, 5), 62)
    second_output = scheduler.schedule()
    cached = second_output.scheduled_cached_reqs

    assert cached.req_ids == [request.request_id]
    assert len(cached.reclaim_transitions) == 1
    transition = cached.reclaim_transitions[0]
    assert transition is not None
    assert transition.retained_block_ids == [
        old_row[0],
        old_row[1],
        old_row[4],
        old_row[5],
    ]
    assert transition.expected_old_num_blocks == 6
    assert transition.new_effective_kv_len == 62

    new_ids = cached.new_block_ids[0]
    assert new_ids is not None
    assert len(new_ids[0]) == 1
    assert new_ids[0][0] not in transition.retained_block_ids

    canonical_row = scheduler.kv_cache_manager.get_block_ids(request.request_id)[0]
    assert canonical_row == [*old_row, new_ids[0][0]]
    assert old_row[2] in canonical_row and old_row[3] in canonical_row


def test_unscheduled_prepared_reclaim_plan_is_not_transport_state() -> None:
    scheduler = create_scheduler(
        model=LOCAL_MODEL,
        skip_tokenizer_init=True,
        max_num_seqs=1,
        max_num_batched_tokens=16,
        max_model_len=128,
        num_blocks=8,
        block_size=16,
        use_v2_model_runner=True,
    )
    (request,) = create_requests(
        num_requests=1,
        num_tokens=20,
        req_ids=["request-a"],
        block_size=16,
    )
    scheduler.add_request(request)
    first_output = scheduler.schedule()
    assert first_output.num_scheduled_tokens[request.request_id] == 16
    scheduler.update_from_output(
        first_output,
        ModelRunnerOutput(
            req_ids=[request.request_id],
            req_id_to_index={request.request_id: 0},
            sampled_token_ids=[[0]],
        ),
    )
    scheduler._set_prepared_reclaim_plan(request.request_id, (0,), 16)
    scheduler.kv_cache_manager.allocate_slots = lambda *args, **kwargs: None
    second_output = scheduler.schedule()
    assert second_output.scheduled_cached_reqs.req_ids == []
    assert request.request_id not in scheduler._prepared_reclaim_plans


def test_empty_and_normal_cached_transport_have_no_transition() -> None:
    assert CachedRequestData.make_empty().reclaim_transitions == []
    scheduler = create_scheduler(
        model=LOCAL_MODEL,
        skip_tokenizer_init=True,
        max_num_seqs=1,
        max_num_batched_tokens=16,
        max_model_len=64,
        num_blocks=8,
        block_size=16,
        use_v2_model_runner=True,
    )
    (request,) = create_requests(
        num_requests=1,
        num_tokens=20,
        req_ids=["request-a"],
        max_tokens=8,
        block_size=16,
    )
    scheduler.add_request(request)
    first_output = scheduler.schedule()
    scheduler.update_from_output(
        first_output,
        ModelRunnerOutput(
            req_ids=[request.request_id],
            req_id_to_index={request.request_id: 0},
            sampled_token_ids=[[0]],
        ),
    )
    output = scheduler.schedule()
    assert output.scheduled_cached_reqs.req_ids == [request.request_id]
    assert output.scheduled_cached_reqs.reclaim_transitions == [None]
