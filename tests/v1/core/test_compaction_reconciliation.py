from types import SimpleNamespace

import pytest

from vllm.v1.core.kv_cache_utils import KVCacheBlock
from vllm.v1.core.sched.output import (
    CompactionPlanData,
    CompactionResultData,
    SchedulerOutput,
)
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.single_type_kv_cache_manager import SingleTypeKVCacheManager


def _manager():
    class TestManager(SingleTypeKVCacheManager):
        def find_longest_cache_hit(self, *args, **kwargs):
            raise NotImplementedError

        def get_num_common_prefix_blocks(self, *args, **kwargs):
            raise NotImplementedError

    manager = TestManager.__new__(TestManager)
    manager.req_to_blocks = {"a": [KVCacheBlock(i, ref_cnt=1) for i in range(4)]}
    manager.num_cached_block = {"a": 4}
    return manager


def test_reconcile_compacted_blocks_detaches_prefix_without_freeing():
    manager = _manager()
    removed = manager.reconcile_compacted_blocks("a", 4, 2)
    assert [b.block_id for b in manager.req_to_blocks["a"]] == [0, 1]
    assert [b.block_id for b in removed] == [2, 3]
    assert manager.num_cached_block["a"] == 2
    assert all(b.ref_cnt == 1 for b in removed)


def test_reconcile_reclaimed_blocks_detaches_without_manager_free():
    manager = _manager()
    manager.req_to_blocks["a"] = [KVCacheBlock(i, ref_cnt=1) for i in range(4)]
    manager.block_pool = SimpleNamespace(
        free_blocks=lambda _blocks: pytest.fail("manager must not free detached blocks")
    )

    removed = manager.reconcile_reclaimed_blocks("a", [0, 2], 4, [])

    assert [b.block_id for b in manager.req_to_blocks["a"]] == [0, 2]
    assert [b.block_id for b in removed] == [1, 3]


@pytest.mark.parametrize("source,new", [(4, 0), (4, 5), (3, 2)])
def test_reconcile_compacted_blocks_rejects_invalid_shape(source, new):
    manager = _manager()
    with pytest.raises(ValueError):
        manager.reconcile_compacted_blocks("a", source, new)


def _scheduler_for_admission():
    scheduler = Scheduler.__new__(Scheduler)
    request = SimpleNamespace(effective_kv_len=32, num_computed_tokens=32,
                              is_finished=lambda: False)
    scheduler.requests = {"a": request}
    scheduler.block_size = 16
    scheduler.kv_cache_manager = SimpleNamespace(
        get_block_ids=lambda _request_id: ([0, 1, 2],),
    )
    return scheduler


def _output(plans, tokens=16):
    output = SchedulerOutput.make_empty()
    output.compaction_plans = plans
    output.num_scheduled_tokens = {"a": tokens}
    return output


def test_active_plan_without_result_is_rejected_before_mutation():
    scheduler = _scheduler_for_admission()
    plan = CompactionPlanData("a", [0, 1], 48, 3)
    before = dict(scheduler.requests)
    with pytest.raises(ValueError, match="missing result"):
        scheduler._prepare_compaction_reconciliations(_output({"a": plan}), [])
    assert scheduler.requests == before


def test_result_without_plan_is_rejected():
    scheduler = _scheduler_for_admission()
    result = CompactionResultData("a", 48, 3, 20, 2)
    with pytest.raises(ValueError, match="no matching plan"):
        scheduler._prepare_compaction_reconciliations(_output({}), [result])


def test_scheduler_releases_detached_blocks_without_deferred_fence():
    released = []
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.kv_cache_manager = SimpleNamespace(
        block_pool=SimpleNamespace(
            free_blocks=lambda blocks: released.extend(blocks),
        )
    )
    scheduler.deferred_frees = []
    removed = [KVCacheBlock(2, ref_cnt=1), KVCacheBlock(3, ref_cnt=1)]

    scheduler._release_reconciled_blocks(removed)

    assert [block.block_id for block in released] == [3, 2]
    assert scheduler.deferred_frees == []
