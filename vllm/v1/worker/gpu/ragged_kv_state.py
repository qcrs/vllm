# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from vllm.v1.core.sched.output import (
    RaggedPageAllocationDeltaData,
    RaggedRequestStateSnapshotData,
)


@dataclass(frozen=True)
class RaggedClusterStepView:
    '''
    它不是 request 的持久化状态，
    而是从 RaggedWorkerPhysicalState 里，按本轮 active requests gather 出来的“本轮输入视图”。
    request0:
    c0 [10,11,0,0]
    c1 [20,21,22,23]
    c2 [30,31,0,0]
    c3 [40,41,42,0]

    page_counts: [R,C]
    effective_lens: [R,C]
    '''
    cluster_rows: npt.NDArray[np.int32]
    page_counts: npt.NDArray[np.int32]
    effective_lens: npt.NDArray[np.int32]


class RaggedWorkerPhysicalState:
    """Discardable CPU mirror of Scheduler-owned Ragged physical state."""
    '''
    Worker 侧对 Scheduler Ragged physical state 的可丢弃镜像。
    '''
    '''
    max_num_reqs = 256
    num_clusters = 4
    max_pages_per_cluster = 128
    block_size = 16
    最多同时保存 256 个 request slot

    每个 request 有 4 个 cluster

    每个 cluster 最多记录 128 个 physical pages

    每个 page 能承载 16 个 effective token slots
    '''

    def __init__(
        self,
        max_num_reqs: int,
        num_clusters: int,
        max_pages_per_cluster: int,
        block_size: int,
    ) -> None:
        if min(max_num_reqs, num_clusters, max_pages_per_cluster, block_size) <= 0:
            raise ValueError("Ragged Worker dimensions must be positive")
        self.max_num_reqs = max_num_reqs
        self.num_clusters = num_clusters
        self.max_pages_per_cluster = max_pages_per_cluster
        self.block_size = block_size
        # 【R C N】
        self.rows = np.zeros(
            (max_num_reqs, num_clusters, max_pages_per_cluster), dtype=np.int32
        )
        self.counts = np.zeros((max_num_reqs, num_clusters), dtype=np.int32)
        self.effective_lens = np.zeros(
            (max_num_reqs, num_clusters), dtype=np.int32
        )

    def _validate_req_index(self, req_index: int) -> None:
        if not 0 <= req_index < self.max_num_reqs:
            raise IndexError("req_index is out of range")

    def _validate_vector(self, values: Sequence[int], name: str) -> tuple[int, ...]:
        result = tuple(values)
        if len(result) != self.num_clusters:
            raise ValueError(f"{name} must have {self.num_clusters} entries")
        if any(value < 0 for value in result):
            raise ValueError(f"{name} entries must be non-negative")
        return result

    @staticmethod
    def _validate_page_ids(page_ids: Sequence[int]) -> tuple[int, ...]:
        # 这个专门检查从 Scheduler 传到 Worker 的 physical page IDs。
        result = tuple(page_ids)
        if any(page_id <= 0 for page_id in result):
            raise ValueError("Ragged ownership requires positive non-NULL page IDs")
        if len(result) != len(set(result)):
            raise ValueError("Ragged page IDs must be unique")
        return result
    '''
    把 Scheduler 传来的 Ragged physical state，同步到 Worker 本地的 RaggedWorkerPhysicalState
    apply_snapshot()
    = 全量覆盖

    apply_allocation_delta()
    = 基于旧状态做增量追加
    '''
    def apply_snapshot(
        self, req_index: int, snapshot: RaggedRequestStateSnapshotData
    ) -> None:
        self._validate_req_index(req_index)
        effective_lens = self._validate_vector(
            snapshot.effective_lens, "effective_lens"
        )
        counts = self._validate_vector(snapshot.page_counts, "page_counts")
        page_ids = self._validate_page_ids(snapshot.flat_page_ids)
        if sum(counts) != len(page_ids):
            raise ValueError("Snapshot flat page length does not match page_counts")
        if any(count > self.max_pages_per_cluster for count in counts):
            raise ValueError("Snapshot exceeds Worker row capacity")
        if any(
            effective_len > count * self.block_size
            for effective_len, count in zip(effective_lens, counts)
        ):
            raise ValueError("Snapshot effective_lens exceed capacity")

        candidate_rows = np.zeros(
            (self.num_clusters, self.max_pages_per_cluster), dtype=np.int32
        )
        offset = 0
        for cluster, count in enumerate(counts):
            candidate_rows[cluster, :count] = page_ids[offset : offset + count]
            offset += count
        self.rows[req_index] = candidate_rows
        self.counts[req_index] = counts
        self.effective_lens[req_index] = effective_lens

    def apply_allocation_delta(
        self, req_index: int, delta: RaggedPageAllocationDeltaData
    ) -> None:
        self._validate_req_index(req_index)
        expected_e = self._validate_vector(
            delta.expected_source_effective_lens,
            "expected_source_effective_lens",
        )
        expected_counts = self._validate_vector(
            delta.expected_source_page_counts,
            "expected_source_page_counts",
        )
        appended_counts = self._validate_vector(
            delta.appended_page_counts, "appended_page_counts"
        )
        current_e = tuple(int(value) for value in self.effective_lens[req_index])
        current_counts = tuple(int(value) for value in self.counts[req_index])
        if current_e != expected_e or current_counts != expected_counts:
            raise ValueError("Ragged allocation delta source state is stale")
        new_page_ids = self._validate_page_ids(delta.flat_new_page_ids)
        if sum(appended_counts) != len(new_page_ids):
            raise ValueError("Delta flat page length does not match appended counts")
        active_ids = {
            int(page_id)
            # 遍历一个序列时，同时拿到“下标”和“元素值”。
            for cluster, count in enumerate(current_counts)
            for page_id in self.rows[req_index, cluster, :count]
        }
        # s是否存在
        if active_ids.intersection(new_page_ids):
            raise ValueError("Delta page IDs already belong to the request")
        new_counts = tuple(
            current + appended
            for current, appended in zip(current_counts, appended_counts)
        )
        if any(count > self.max_pages_per_cluster for count in new_counts):
            raise ValueError("Delta exceeds Worker row capacity")
        # 一次成功
        candidate_rows = self.rows[req_index].copy()
        offset = 0
        # 就是 找到 当前的clustre 然后 从 current_counts 到 appended_counts
        for cluster, (start, count) in enumerate(
            zip(current_counts, appended_counts)
        ):
            candidate_rows[cluster, start : start + count] = new_page_ids[
                offset : offset + count
            ]
            offset += count
        self.rows[req_index] = candidate_rows
        self.counts[req_index] = new_counts

    def commit_effective_lens(
        self,
        req_index: int,
        expected_source_effective_lens: Sequence[int],
        new_effective_lens: Sequence[int],
    ) -> None:
        # forward / KV write 成功后，把 Worker 本地这个 request 的 effective_lens 从旧值推进到新值。
        self._validate_req_index(req_index)
        expected = self._validate_vector(
            expected_source_effective_lens, "expected_source_effective_lens"
        )
        new = self._validate_vector(new_effective_lens, "new_effective_lens")
        current = tuple(int(value) for value in self.effective_lens[req_index])
        if current != expected:
            raise ValueError("Ragged effective frontier is stale")
        if any(new_value < old_value for old_value, new_value in zip(expected, new)):
            raise ValueError("Normal effective frontier commit cannot shrink")
        counts = tuple(int(value) for value in self.counts[req_index])
        if any(
            effective_len > count * self.block_size
            for effective_len, count in zip(new, counts)
        ):
            raise ValueError("Effective frontier exceeds Worker capacity")
        self.effective_lens[req_index] = new

    def remove_request(self, req_index: int) -> None:
        self._validate_req_index(req_index)
        self.rows[req_index].fill(0)
        self.counts[req_index].fill(0)
        self.effective_lens[req_index].fill(0)

    def gather(self, req_indices: Sequence[int]) -> RaggedClusterStepView:
        # 把任意 Sequence[int] 统一转换成 NumPy ndarray。
        indices = np.asarray(req_indices, dtype=np.int64)
        if indices.ndim != 1:
            raise ValueError("req_indices must be one-dimensional")
        if np.any(indices < 0) or np.any(indices >= self.max_num_reqs):
            raise IndexError("req_indices contain an out-of-range index")
        return RaggedClusterStepView(
            cluster_rows=self.rows[indices].copy(),
            page_counts=self.counts[indices].copy(),
            effective_lens=self.effective_lens[indices].copy(),
        )
