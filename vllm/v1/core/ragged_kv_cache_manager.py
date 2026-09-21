# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence
from dataclasses import dataclass

from vllm.utils.math_utils import cdiv
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import BlockHashList, KVCacheBlock
from vllm.v1.core.sched.output import (
    RaggedPageAllocationDeltaData,
    RaggedRequestStateSnapshotData,
)
from vllm.v1.core.single_type_kv_cache_manager import SingleTypeKVCacheManager
from vllm.v1.kv_cache_interface import KVCacheSpec, RaggedAttentionSpec
from vllm.v1.ragged_kv_layout import MemberPlacementMap


@dataclass(frozen=True)
class RaggedRequestPhysicalState:
    """
    Scheduler-authoritative physical ownership for one request.
    此刻的真实状态是什么
    某个 request 当前在 Scheduler 侧真实持有的 Ragged KV 物理状态。
    """
    # 长度为c的vector
    '''
    canonical physical ownership axis
    简单解释就是 分层的细粒度 类似于 几个 KV head
    effective_lens = (32, 56, 20, 48)
    cluster 0 当前有效 KV 长度 = 32
    cluster 1 当前有效 KV 长度 = 56
    cluster 2 当前有效 KV 长度 = 20
    cluster 3 当前有效 KV 长度 = 48
    page_rows = (
    (B10, B11),
    (B20, B21, B22, B23),
    (B30, B31),
    (B40, B41, B42),
    )
    cluster0 → [B10, B11]
    cluster1 → [B20, B21, B22, B23]
    cluster2 → [B30, B31]
    cluster3 → [B40, B41, B42]
    page_rows[C][variable depth]
    每个 cluster 的 page 数可以不同。
    Dense  request
    → [B10, B11, B12, B13] 只有一条 row。
    Ragged C条row 每条深度不同
    list 是可以改变的 重新赋值的 要求不能改变
    '''
    '''
    state_version: int
    Scheduler 眼里某个 request 当前真实的 physical ownership。
    request A

    state_version = 4

    effective_lens =
    [32, 48]

    page_rows =
    [
    [B10, B11],
    [B20, B21, B22]
    ]
    state_version
    = 这是第几代 ownership

    effective_lens
    = 每个 cluster 当前有多少有效 KV token

    page_rows
    = 每个 cluster 实际拥有哪些 physical pages
    '''
    state_version: int
    effective_lens: tuple[int, ...]
    page_rows: tuple[tuple[KVCacheBlock, ...], ...]

    @property
    def page_counts(self) -> tuple[int, ...]:
        return tuple(len(row) for row in self.page_rows)


@dataclass(frozen=True)
class RaggedCapacityPlan:
    '''
    从当前状态走到目标状态时，计划要扩多少 page。
    '''
    request_id: str
    source_state_version: int
    source_effective_lens: tuple[int, ...]
    source_page_counts: tuple[int, ...]
    target_effective_lens: tuple[int, ...]
    required_page_counts: tuple[int, ...]
    appended_page_counts: tuple[int, ...]
    total_new_pages: int


class RaggedAttentionManager(SingleTypeKVCacheManager):
    """Ragged physical-page authority; not registered for production yet."""
    """
    把 vLLM 原来“单一 KV 类型 + 单条 Dense block row”的管理器，
    扩展成“单一 Ragged KV 类型 + C 条非均匀 physical rows”的管理器。
    Coordinator 模型中又多种spec 统一协调
    所以 SingleTypeKVCacheManager 更像一个统一接口 + 基础能力层。
    """
    def __init__(
        self,
        kv_cache_spec: RaggedAttentionSpec,
        block_pool: BlockPool,
        enable_caching: bool,
        kv_cache_group_id: int,
        scheduler_block_size: int,
        placement: MemberPlacementMap,
    ) -> None:
        if enable_caching:
            raise ValueError("RaggedAttentionManager does not support prefix caching")
        if placement.num_kv_heads != kv_cache_spec.num_kv_heads:
            raise ValueError("Placement KV-head count does not match the cache spec")
        if placement.page_group_size != kv_cache_spec.page_group_size:
            raise ValueError("Placement page width does not match the cache spec")
        super().__init__(
            kv_cache_spec=kv_cache_spec,
            block_pool=block_pool,
            enable_caching=enable_caching,
            kv_cache_group_id=kv_cache_group_id,
            scheduler_block_size=scheduler_block_size,
        )
        '''
        request_id
            ↓
        RaggedRequestPhysicalState
            ├─ effective_lens[C]
            └─ page_rows[C][variable depth]
        '''
        self.placement = placement
        self.num_clusters = placement.num_clusters
        self.req_to_ragged_state: dict[str, RaggedRequestPhysicalState] = {}
        self._last_state_versions: dict[str, int] = {}

    def _empty_state(self, request_id: str) -> RaggedRequestPhysicalState:
        '''
        构造一个全空的合法初始状态
        初始状态
        '''
        return RaggedRequestPhysicalState(
            state_version=self._last_state_versions.get(request_id, -1) + 1,
            #(0,)*4 = (0,0,0,0)
            effective_lens=(0,) * self.num_clusters,
            #() for _ in range(4) 依次生成()
            page_rows=tuple(() for _ in range(self.num_clusters)),
        )

    def _validate_vector(self, values: Sequence[int], name: str) -> tuple[int, ...]:
        # 所有“按 cluster 表达的 vector”是否合法。
        result = tuple(values)
        if len(result) != self.num_clusters:
            raise ValueError(f"{name} must have {self.num_clusters} entries")
        if any(value < 0 for value in result):
            raise ValueError(f"{name} entries must be non-negative")
        return result

    def _validate_state(self, state: RaggedRequestPhysicalState) -> None:
        if state.state_version < 0:
            raise ValueError("state_version must be non-negative")
        # vector 合法
        effective_lens = self._validate_vector(
            state.effective_lens, "effective_lens"
        )
        if len(state.page_rows) != self.num_clusters:
            raise ValueError("page_rows has the wrong cluster count")
        '''
        page_rows = (
            (B10, B11),
            (B20, B21, B22),
        )
        page_ids = [
            10,
            11,
            20,
            21,
            22,
        ]
        '''
        page_ids = [block.block_id for row in state.page_rows for block in row]
        if any(
            block.is_null or block.block_id == 0
            for row in state.page_rows
            for block in row
        ):
            raise ValueError("Ragged ownership cannot contain the NULL page")
        if len(page_ids) != len(set(page_ids)):
            raise ValueError("Ragged request page IDs must be unique")
        if any(
            effective_len > len(row) * self.block_size
            for effective_len, row in zip(effective_lens, state.page_rows)
        ):
            '''
            page_rows = (
                (B10, B11),
                (B20, B21, B22, B23),
                (B30, B31),
                (B40, B41, B42),
            )
            effective_lens = (
                32,
                56,
                20,
                48,
            )
            32 ↔ (B10,B11)

            56 ↔ (B20,B21,B22,B23)

            20 ↔ (B30,B31)

            48 ↔ (B40,B41,B42)
            '''
            raise ValueError("effective_lens cannot exceed owned capacity")

    def get_state(self, request_id: str) -> RaggedRequestPhysicalState:
        try:
            return self.req_to_ragged_state[request_id]
        except KeyError as exc:
            raise ValueError(f"Request {request_id!r} has no Ragged state") from exc

    def plan_capacity(
        self, request_id: str, target_effective_lens: Sequence[int]
    ) -> RaggedCapacityPlan:
        '''
        lan_capacity()：只算，不申请 page
        '''
        target = self._validate_vector(target_effective_lens, "target_effective_lens")
        # 存在返回 不存在初始化
        state = self.req_to_ragged_state.get(request_id)
        if state is None:
            state = self._empty_state(request_id)
        source_counts = state.page_counts
        required_counts = tuple(cdiv(value, self.block_size) for value in target)
        appended_counts = tuple(
            max(required - current, 0)
            for required, current in zip(required_counts, source_counts)
        )
        return RaggedCapacityPlan(
            request_id=request_id,
            source_state_version=state.state_version,
            source_effective_lens=state.effective_lens,
            source_page_counts=source_counts,
            target_effective_lens=target,
            required_page_counts=required_counts,
            appended_page_counts=appended_counts,
            total_new_pages=sum(appended_counts),
        )

    def apply_capacity_plan(
        self, plan: RaggedCapacityPlan
    ) -> RaggedPageAllocationDeltaData:
        current = self.req_to_ragged_state.get(plan.request_id)
        if current is None:
            current = self._empty_state(plan.request_id)
        if (
            current.state_version != plan.source_state_version
            or current.effective_lens != plan.source_effective_lens
            or current.page_counts != plan.source_page_counts
        ):
            raise ValueError("Ragged capacity plan is stale")
        target = self._validate_vector(
            plan.target_effective_lens, "target_effective_lens"
        )
        required = self._validate_vector(
            plan.required_page_counts, "required_page_counts"
        )
        appended = self._validate_vector(
            plan.appended_page_counts, "appended_page_counts"
        )
        expected_required = tuple(cdiv(value, self.block_size) for value in target)
        expected_appended = tuple(
            max(required_count - current_count, 0)
            for required_count, current_count in zip(
                expected_required, current.page_counts
            )
        )
        if required != expected_required or appended != expected_appended:
            raise ValueError("Ragged capacity plan fields are inconsistent")
        if sum(appended) != plan.total_new_pages:
            raise ValueError("Ragged capacity plan total is inconsistent")

        new_blocks = self.block_pool.get_new_blocks(plan.total_new_pages)
        '''
        rows:

        c0 = (B10,B11)
        c1 = (B20,B21,B22,B23)
        c2 = (B30,B31)
        c3 = (B40,B41,B42)

        appended =
        (1,0,0,1)

        new_blocks =
        [B101,B102]
        '''
        candidate_rows: list[tuple[KVCacheBlock, ...]] = []
        offset = 0
        for row, count in zip(current.page_rows, appended):
            '''
            这里的 * 是 iterable unpacking，序列展开。
            row = (B10, B11)
            new_blocks = [B101, B102]
            offset = 0
            count = 1
            new_blocks[0:1]# [B101]
            tuple(row) + tuple(new_blocks_slice)
            '''
            candidate_rows.append((*row, *new_blocks[offset : offset + count]))
            offset += count
        assert offset == len(new_blocks)
        candidate = RaggedRequestPhysicalState(
            state_version=current.state_version + bool(new_blocks),
            # 先不加
            effective_lens=current.effective_lens,
            page_rows=tuple(candidate_rows),
        )
        self._validate_state(candidate)
        self.req_to_ragged_state[plan.request_id] = candidate
        self._last_state_versions[plan.request_id] = candidate.state_version
        return RaggedPageAllocationDeltaData(
            request_id=plan.request_id,
            expected_source_state_version=current.state_version,
            new_state_version=candidate.state_version,
            expected_source_effective_lens=plan.source_effective_lens,
            expected_source_page_counts=plan.source_page_counts,
            appended_page_counts=appended,
            # 新块
            flat_new_page_ids=tuple(block.block_id for block in new_blocks),
        )

    def commit_effective_lens(
        self,
        request_id: str,
        expected_state_version: int,
        expected_source_effective_lens: Sequence[int],
        new_effective_lens: Sequence[int],
    ) -> None:
        state = self.get_state(request_id)
        expected = self._validate_vector(
            expected_source_effective_lens, "expected_source_effective_lens"
        )
        new = self._validate_vector(new_effective_lens, "new_effective_lens")
        if state.state_version != expected_state_version:
            raise ValueError("Ragged physical state version is stale")
        if state.effective_lens != expected:
            raise ValueError("Ragged effective frontier is stale")
        if any(new_value < old_value for old_value, new_value in zip(expected, new)):
            raise ValueError("Normal effective frontier commit cannot shrink")
        candidate = RaggedRequestPhysicalState(
            state_version=state.state_version,
            effective_lens=new,
            page_rows=state.page_rows,
        )
        self._validate_state(candidate)
        self.req_to_ragged_state[request_id] = candidate

    def export_snapshot(self, request_id: str) -> RaggedRequestStateSnapshotData:
        '''
        E =
        (33,57,21,49)

        rows =
        (
        (B10,B11,B101),
        (B20,B21,B22,B23),
        (B30,B31),
        (B40,B41,B42,B102)
        )
        effective_lens =
        (33,57,21,49)

        page_counts =
        (3,4,2,4)

        flat_page_ids =
        (
        10,11,101,
        20,21,22,23,
        30,31,
        40,41,42,102
        )
        '''
        state = self.get_state(request_id)
        return RaggedRequestStateSnapshotData(
            request_id=request_id,
            state_version=state.state_version,
            effective_lens=state.effective_lens,
            page_counts=state.page_counts,
            flat_page_ids=tuple(
                block.block_id for row in state.page_rows for block in row
            ),
        )

    def reconcile_compaction(
        self,
        request_id: str,
        expected_source_state_version: int,
        expected_source_effective_lens: Sequence[int],
        expected_source_page_counts: Sequence[int],
        new_effective_lens: Sequence[int],
        new_page_counts: Sequence[int],
    ) -> tuple[int, ...]:
        '''
        Worker/compaction 路径告诉 Scheduler：
        “这个 request 压缩前是什么 shape，压缩后变成什么 shape。”
        Scheduler 自己根据 canonical page_rows 决定哪些 page 被保留、
        哪些 page detached，然后真正 free。
        '''
        state = self.get_state(request_id)
        expected_e = self._validate_vector(
            expected_source_effective_lens, "expected_source_effective_lens"
        )
        expected_counts = self._validate_vector(
            expected_source_page_counts, "expected_source_page_counts"
        )
        new_e = self._validate_vector(new_effective_lens, "new_effective_lens")
        new_counts = self._validate_vector(new_page_counts, "new_page_counts")
        if state.state_version != expected_source_state_version:
            raise ValueError("Ragged compaction source version is stale")
        if state.effective_lens != expected_e or state.page_counts != expected_counts:
            raise ValueError("Ragged compaction source state is stale")
        if any(value > source for value, source in zip(new_e, expected_e)):
            raise ValueError("Ragged compaction cannot grow effective frontiers")
        if any(value > source for value, source in zip(new_counts, expected_counts)):
            raise ValueError("Ragged compaction cannot grow page counts")

        candidate_rows = tuple(
            row[:count] for row, count in zip(state.page_rows, new_counts)
        )
        candidate = RaggedRequestPhysicalState(
            state_version=state.state_version + 1,
            effective_lens=new_e,
            page_rows=candidate_rows,
        )
        self._validate_state(candidate)
        # 找到删去的 白遍历所有的sate 然后 匹配新的 梳理 然后写入新的
        detached = [
            block
            for row, count in zip(state.page_rows, new_counts)
            for block in row[count:]
        ]
        self.req_to_ragged_state[request_id] = candidate
        self._last_state_versions[request_id] = candidate.state_version
        self.block_pool.free_blocks(reversed(detached))
        return tuple(block.block_id for block in detached)

    def free_request(self, request_id: str) -> tuple[int, ...]:
        state = self.req_to_ragged_state.pop(request_id, None)
        if state is None:
            return ()
        self._last_state_versions[request_id] = state.state_version
        blocks = [block for row in state.page_rows for block in row]
        self.block_pool.free_blocks(reversed(blocks))
        return tuple(block.block_id for block in blocks)

    def free(self, request_id: str) -> None:
        self.free_request(request_id)

    def get_num_blocks_to_allocate(self, *args, **kwargs) -> int:
        raise RuntimeError("Ragged manager requires vector capacity API")

    def allocate_new_blocks(self, *args, **kwargs) -> list[KVCacheBlock]:
        raise RuntimeError("Ragged manager requires vector capacity API")

    def get_num_common_prefix_blocks(self, running_request_id: str) -> int:
        return 0
    # 继承了接口
    @classmethod
    def find_longest_cache_hit(
        cls,
        block_hashes: BlockHashList,
        max_length: int,
        kv_cache_group_ids: list[int],
        block_pool: BlockPool,
        kv_cache_spec: KVCacheSpec,
        drop_eagle_block: bool,
        alignment_tokens: int,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
    ) -> tuple[tuple[list[KVCacheBlock], ...], int]:
        return tuple([] for _ in kv_cache_group_ids), 0
