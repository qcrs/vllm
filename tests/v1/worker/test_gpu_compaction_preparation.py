from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.v1.core.sched.output import CompactionPlanData
from vllm.v1.worker.gpu import model_runner as model_runner_module
from vllm.v1.worker.gpu.model_runner import GPUModelRunner


def _runner(*, num_groups: int = 1, source_blocks: int = 3):
    persistent_effective_kv_len = torch.tensor([4, 0, 7, 0], dtype=torch.int32)
    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.req_states = SimpleNamespace(
        req_id_to_index={"request-a": 2, "request-b": 0},
        effective_kv_len=SimpleNamespace(gpu=persistent_effective_kv_len),
    )
    block_rows = torch.zeros((4, 6), dtype=torch.int32)
    block_rows[2, :3] = torch.tensor([5, 1, 3], dtype=torch.int32)
    num_blocks = np.zeros((num_groups, 4), dtype=np.int32)
    num_blocks[0, 2] = source_blocks
    runner.block_tables = SimpleNamespace(
        num_kv_cache_groups=num_groups,
        block_sizes=[4] * num_groups,
        num_blocks=SimpleNamespace(np=num_blocks),
        block_tables=[SimpleNamespace(gpu=block_rows)] * num_groups,
    )
    runner.kv_caches = [
        torch.zeros((8, 2, 4, 3), dtype=torch.float32),
        torch.ones((8, 2, 4, 3), dtype=torch.float32),
    ]
    return runner


def _batch(source_effective_kv_len: int = 10):
    return SimpleNamespace(
        req_ids=["request-a", "request-b"],
        idx_mapping_np=np.array([2, 0], dtype=np.int32),
        effective_kv_seq_lens=torch.tensor(
            [source_effective_kv_len, 4], dtype=torch.int32
        ),
    )


def _plan(
    keep_member_indices: list[int] | None = None,
    *,
    request_id: str = "request-a",
    source_effective_kv_len: int = 10,
    source_num_blocks: int = 3,
    step_seq: int | None = None,
):
    return CompactionPlanData(
        request_id=request_id,
        keep_member_indices=(
            [0, 2, 5, 8, 9]
            if keep_member_indices is None
            else keep_member_indices
        ),
        expected_source_effective_kv_len=source_effective_kv_len,
        expected_source_num_blocks=source_num_blocks,
        step_seq=step_seq,
    )


def test_prepare_v2_compaction_resolves_distinct_index_domains_without_mutation():
    runner = _runner()
    batch = _batch()
    plans = {"request-a": _plan()}
    block_table_before = runner.block_tables.block_tables[0].gpu.clone()
    effective_kv_len_before = runner.req_states.effective_kv_len.gpu.clone()
    kv_before = [cache.clone() for cache in runner.kv_caches]

    prepared = runner._prepare_v2_compactions(plans, batch)

    assert set(prepared) == {"request-a"}
    execution = prepared["request-a"]
    assert execution.req_state_idx == 2
    assert execution.batch_idx == 0
    assert execution.source_effective_kv_len == 10
    assert execution.source_num_blocks == 3
    assert execution.block_ids == (5, 1, 3)
    assert execution.block_size == 4
    assert execution.keep_member_indices == (0, 2, 5, 8, 9)
    assert execution.new_effective_kv_len == 5
    assert execution.new_num_blocks == 2
    assert torch.equal(runner.block_tables.block_tables[0].gpu, block_table_before)
    assert torch.equal(
        runner.req_states.effective_kv_len.gpu, effective_kv_len_before
    )
    assert all(
        torch.equal(cache, before)
        for cache, before in zip(runner.kv_caches, kv_before)
    )


def test_prepare_v2_compaction_keep_all_is_valid():
    runner = _runner()

    prepared = runner._prepare_v2_compactions(
        {"request-a": _plan(list(range(10)))}, _batch()
    )

    execution = prepared["request-a"]
    assert execution.new_effective_kv_len == 10
    assert execution.new_num_blocks == 3


@pytest.mark.parametrize(
    ("plans", "batch", "match"),
    [
        (
            {"missing": _plan(request_id="missing")},
            _batch(),
            "request is missing",
        ),
        (
            {"request-a": _plan(source_effective_kv_len=9)},
            _batch(),
            "effective KV length mismatch",
        ),
        (
            {"request-a": _plan(source_num_blocks=2)},
            _batch(),
            "source block count mismatch",
        ),
        ({"request-a": _plan([])}, _batch(), "non-empty"),
        ({"request-a": _plan([0, 2, 1])}, _batch(), "strictly increasing"),
        ({"request-a": _plan([0, 2, 2])}, _batch(), "strictly increasing"),
        ({"request-a": _plan([-1, 2])}, _batch(), "non-negative"),
        ({"request-a": _plan([0, 10])}, _batch(), "exceed the source"),
    ],
)
def test_prepare_v2_compaction_rejects_invalid_fences(plans, batch, match):
    with pytest.raises(ValueError, match=match):
        _runner()._prepare_v2_compactions(plans, batch)


def test_prepare_v2_compaction_rejects_request_outside_current_batch():
    runner = _runner()
    runner.req_states.req_id_to_index["request-c"] = 1

    with pytest.raises(ValueError, match="not in the current batch"):
        runner._prepare_v2_compactions(
            {"request-c": _plan(request_id="request-c")}, _batch()
        )


def test_prepare_v2_compaction_rejects_stale_index_mapping():
    batch = _batch()
    batch.idx_mapping_np[0] = 1

    with pytest.raises(ValueError, match="index mapping is stale"):
        _runner()._prepare_v2_compactions({"request-a": _plan()}, batch)


def test_prepare_v2_compaction_rejects_insufficient_source_capacity():
    with pytest.raises(ValueError, match="exact page count"):
        _runner(source_blocks=2)._prepare_v2_compactions(
            {"request-a": _plan(source_num_blocks=2)}, _batch()
        )


def test_prepare_v2_compaction_rejects_multiple_kv_groups():
    with pytest.raises(ValueError, match="exactly one KV cache group"):
        _runner(num_groups=2)._prepare_v2_compactions(
            {"request-a": _plan()}, _batch()
        )


def test_prepare_v2_compaction_no_plan_is_a_noop_for_unsupported_layout():
    assert _runner(num_groups=2)._prepare_v2_compactions({}, _batch()) == {}


def test_prepare_v2_compaction_prevalidates_every_layer():
    runner = _runner()
    runner.kv_caches[1] = torch.zeros((8, 2, 8, 3), dtype=torch.float32)

    with pytest.raises(ValueError, match="layer 1 block size mismatch"):
        runner._prepare_v2_compactions({"request-a": _plan()}, _batch())


def test_execute_v2_compaction_compacts_all_layers_and_stages_next_state(
    monkeypatch,
):
    runner = _runner()
    runner.device = torch.device("cpu")
    batch = _batch()
    batch.num_reqs = len(batch.req_ids)
    staged = []

    def append_block_ids(req_state_idx, new_block_ids, overwrite):
        staged.append((req_state_idx, new_block_ids, overwrite))

    runner.block_tables.append_block_ids = append_block_ids
    calls = []

    def fake_compact(kv_cache, block_ids, source_effective_kv_len, keep):
        calls.append(
            (kv_cache, block_ids.clone(), source_effective_kv_len, keep.clone())
        )
        kv_cache[5, :, 0, :] = 9
        return int(keep.numel()), 2

    monkeypatch.setattr(
        model_runner_module, "compact_paged_kv_triton_2d", fake_compact
    )

    prepared = runner._prepare_v2_compactions(
        {"request-a": _plan(step_seq=17)}, batch
    )
    override_valid, override_value, results = runner._execute_v2_compactions(
        prepared, batch
    )

    assert len(calls) == 2
    assert all(call[1].tolist() == [5, 1, 3] for call in calls)
    assert all(call[2] == 10 for call in calls)
    assert all(call[3].tolist() == [0, 2, 5, 8, 9] for call in calls)
    assert staged == [(2, ([5, 1],), True)]
    assert override_valid.tolist() == [True, False]
    assert override_value.tolist() == [5, 0]
    assert results[0].request_id == "request-a"
    assert results[0].new_effective_kv_len == 5
    assert results[0].new_num_blocks == 2
    assert results[0].step_seq == 17
    assert all(torch.all(cache[5, :, 0, :] == 9) for cache in runner.kv_caches)


def test_execute_v2_compaction_does_not_publish_success_after_layer_failure(
    monkeypatch,
):
    runner = _runner()
    runner.device = torch.device("cpu")
    batch = _batch()
    batch.num_reqs = len(batch.req_ids)
    staged = []
    runner.block_tables.append_block_ids = lambda *args, **kwargs: staged.append(
        (args, kwargs)
    )
    calls = 0

    def failing_compact(kv_cache, block_ids, source_effective_kv_len, keep):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected M4 failure")
        kv_cache[5, :, 0, :] = 9
        return int(keep.numel()), 2

    monkeypatch.setattr(
        model_runner_module, "compact_paged_kv_triton_2d", failing_compact
    )

    prepared = runner._prepare_v2_compactions({"request-a": _plan()}, batch)
    with pytest.raises(RuntimeError, match="injected M4 failure"):
        runner._execute_v2_compactions(prepared, batch)

    assert calls == 2
    assert staged == []
    assert torch.all(runner.kv_caches[0][5, :, 0, :] == 9)
    assert torch.all(runner.kv_caches[1][5, :, 0, :] == 1)


def test_execute_v2_compaction_payload_failure_does_not_stage_prior_request(
    monkeypatch,
):
    runner = _runner()
    runner.device = torch.device("cpu")
    runner.block_tables.block_tables[0].gpu[0, 0] = 7
    runner.block_tables.num_blocks.np[0, 0] = 1
    batch = _batch()
    batch.num_reqs = len(batch.req_ids)
    batch.effective_kv_seq_lens = torch.tensor([10, 4], dtype=torch.int32)
    before_num_blocks = runner.block_tables.num_blocks.np.copy()
    staged = []
    runner.block_tables.append_block_ids = lambda *args, **kwargs: staged.append(
        (args, kwargs)
    )
    calls = 0

    def failing_second_request(kv_cache, block_ids, source_effective_kv_len, keep):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("injected request-B failure")
        return int(keep.numel()), (int(keep.numel()) + 3) // 4

    monkeypatch.setattr(
        model_runner_module,
        "compact_paged_kv_triton_2d",
        failing_second_request,
    )

    plans = {
        "request-a": _plan(step_seq=1),
        "request-b": _plan(
            request_id="request-b",
            keep_member_indices=[0, 2, 3],
            source_effective_kv_len=4,
            source_num_blocks=1,
            step_seq=1,
        ),
    }
    prepared = runner._prepare_v2_compactions(plans, batch)
    with pytest.raises(RuntimeError, match="request-B failure"):
        runner._execute_v2_compactions(prepared, batch)

    assert calls == 3
    assert staged == []
    np.testing.assert_array_equal(runner.block_tables.num_blocks.np, before_num_blocks)


def test_execute_v2_compaction_preserves_result_identity_for_multiple_requests(
    monkeypatch,
):
    runner = _runner()
    runner.device = torch.device("cpu")

    # request-b:
    # req_state_idx=0, source_E=4, source_num_blocks=1, physical block=[7]
    runner.block_tables.block_tables[0].gpu[0, 0] = 7
    runner.block_tables.num_blocks.np[0, 0] = 1

    batch = _batch()
    batch.num_reqs = len(batch.req_ids)

    staged = []

    def append_block_ids(req_state_idx, new_block_ids, overwrite):
        staged.append((req_state_idx, new_block_ids, overwrite))

    runner.block_tables.append_block_ids = append_block_ids

    calls = []

    def fake_compact(kv_cache, block_ids, source_effective_kv_len, keep):
        calls.append(
            (
                block_ids.clone(),
                source_effective_kv_len,
                keep.clone(),
            )
        )
        new_effective_kv_len = int(keep.numel())
        new_num_blocks = (new_effective_kv_len + 3) // 4
        return new_effective_kv_len, new_num_blocks

    monkeypatch.setattr(
        model_runner_module,
        "compact_paged_kv_triton_2d",
        fake_compact,
    )

    plans = {
        "request-a": _plan(step_seq=17),
        "request-b": _plan(
            request_id="request-b",
            keep_member_indices=[0, 2, 3],
            source_effective_kv_len=4,
            source_num_blocks=1,
            step_seq=18,
        ),
    }

    prepared = runner._prepare_v2_compactions(plans, batch)

    override_valid, override_value, results = runner._execute_v2_compactions(
        prepared,
        batch,
    )

    # Two requests × two KV layers.
    assert len(calls) == 4

    # request-a keeps 5 members => 2 pages.
    # request-b keeps 3 members => 1 page.
    assert staged == [
        (2, ([5, 1],), True),
        (0, ([7],), True),
    ]

    assert override_valid.tolist() == [True, True]
    assert override_value.tolist() == [5, 3]

    # Regression: Phase 2 must use prepared.request_id rather than the stale
    # request_id left behind by the Phase 1 loop.
    assert [result.request_id for result in results] == [
        "request-a",
        "request-b",
    ]

    assert [result.new_effective_kv_len for result in results] == [5, 3]
    assert [result.new_num_blocks for result in results] == [2, 1]
    assert [result.step_seq for result in results] == [17, 18]
