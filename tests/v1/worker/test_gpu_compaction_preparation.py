from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.v1.core.sched.output import CompactionPlanData
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
    with pytest.raises(ValueError, match="do not cover source extent"):
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
