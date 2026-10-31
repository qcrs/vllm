# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Physical KV reclamation contracts shared by scheduler and worker."""

from dataclasses import dataclass
from functools import cmp_to_key
from typing import TYPE_CHECKING

from vllm import envs
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig

if TYPE_CHECKING:
    from vllm.config import VllmConfig


@dataclass(frozen=True)
class ReclamationConfig:
    policy: str = "sink_recent"
    stats_path: str | None = None
    pressure: bool = False
    retention: float = 0.5
    sink_tokens: int = 16
    recent_tokens: int = 16
    min_keep_tokens: int = 128
    reserve_cap: int = 0

    @classmethod
    def from_vllm_config(cls, config: "VllmConfig") -> "ReclamationConfig | None":
        options = (
            config.additional_config.get("kv_reclamation")
            if isinstance(config.additional_config, dict)
            else None
        )
        if options is None:
            return None
        result = cls(**options)
        if result.policy not in ("sink_recent", "random", "triattention"):
            raise ValueError("Unknown KV reclamation policy")
        if result.policy == "triattention" and result.stats_path is None:
            raise ValueError("TriAttention requires calibrated frequency stats")
        if not 0 < result.retention <= 1:
            raise ValueError("KV reclamation retention must be in (0, 1]")
        if min(result.sink_tokens, result.recent_tokens, result.reserve_cap) < 0:
            raise ValueError("KV reclamation protection/reserve must be nonnegative")
        if result.min_keep_tokens < result.sink_tokens + result.recent_tokens:
            raise ValueError("min_keep_tokens must cover sink and recent protection")
        parallel = config.parallel_config
        if (
            not envs.VLLM_USE_V2_MODEL_RUNNER
            or config.scheduler_config.async_scheduling
            or config.max_concurrent_batches != 1
            or config.speculative_config is not None
            or config.kv_transfer_config is not None
            or config.scheduler_config.scheduler_reserve_full_isl
            or parallel.tensor_parallel_size != 1
            or parallel.pipeline_parallel_size != 1
            or parallel.decode_context_parallel_size != 1
            or parallel.prefill_context_parallel_size != 1
            or config.model_config.is_encoder_decoder
            or config.attention_config.backend != AttentionBackendEnum.FLASH_ATTN
            or config.attention_config.use_non_causal
            or not config.model_config.disable_cascade_attn
            or config.cache_config.cache_dtype != "auto"
        ):
            raise ValueError(
                "KV reclamation requires the validated MRV2 Core configuration"
            )
        return result

    def validate_cache(self, cache: KVCacheConfig) -> None:
        if len(cache.kv_cache_groups) != 1 or not isinstance(
            cache.kv_cache_groups[0].kv_cache_spec, FullAttentionSpec
        ):
            raise ValueError("KV reclamation requires one full-attention KV group")
        if cache.kv_cache_groups[0].kv_cache_spec.block_size != 16:
            raise ValueError("KV reclamation requires B16")


@dataclass(frozen=True)
class ReclaimRequest:
    request_id: str
    source_effective_kv_len: int
    source_block_ids: list[int]
    target_keep_tokens: int
    protected_prefix_tokens: int = 0


@dataclass(frozen=True)
class ReclaimResult:
    request_id: str
    new_effective_kv_len: int
    retained_block_ids: list[int]


@dataclass(frozen=True)
class ReclaimCandidate:
    request_id: str
    keep_tokens: int
    released_blocks: int
    copy_bytes: int
    zero_copy: bool
    arrival_time: float
    protected_prefix_tokens: int = 0


def choose_reclaim_candidate(
    candidates: list[ReclaimCandidate], goal_blocks: int
) -> ReclaimCandidate | None:
    """Prefer coverage, then zero copy and exact cost-per-released-page order."""
    if not candidates:
        return None
    covering = [c for c in candidates if c.released_blocks >= goal_blocks]

    def compare(a: ReclaimCandidate, b: ReclaimCandidate) -> int:
        if not covering and a.released_blocks != b.released_blocks:
            return -1 if a.released_blocks > b.released_blocks else 1
        if a.zero_copy != b.zero_copy:
            return -1 if a.zero_copy else 1
        ratio = a.copy_bytes * b.released_blocks - b.copy_bytes * a.released_blocks
        if ratio:
            return -1 if ratio < 0 else 1
        a_key = (-a.released_blocks, a.arrival_time, a.request_id)
        b_key = (-b.released_blocks, b.arrival_time, b.request_id)
        return (a_key > b_key) - (a_key < b_key)

    return min(covering or candidates, key=cmp_to_key(compare))


def sink_recent_ranges(
    source_len: int, keep_len: int, sink_tokens: int, protected_prefix: int = 0
) -> tuple[int, int]:
    """Return prefix length and recent start in the current member domain."""
    assert 0 < keep_len <= source_len
    prefix = min(keep_len, max(sink_tokens, protected_prefix))
    return prefix, source_len - (keep_len - prefix)


def whole_block_row(
    source_row: list[int],
    source_len: int,
    keep_len: int,
    prefix: int,
    recent: int,
    block_size: int,
) -> list[int] | None:
    """Recognize exact page/offset representability of SinkRecent intervals."""
    if keep_len == source_len:
        return source_row.copy()
    if prefix == keep_len:
        return source_row[: (keep_len + block_size - 1) // block_size]
    if prefix % block_size or recent % block_size:
        return None
    return (
        source_row[: prefix // block_size]
        + source_row[recent // block_size : (source_len + block_size - 1) // block_size]
    )
