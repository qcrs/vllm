# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Physical KV reclamation contracts shared by scheduler and worker."""

from dataclasses import dataclass, replace
from fractions import Fraction
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
    budget_mode: str = "fixed"

    def __post_init__(self) -> None:
        if self.budget_mode not in (
            "fixed",
            "demand",
            "retry",
            "retry_pass",
            "prefill_fixed",
            "protected",
        ):
            raise ValueError("Unknown KV reclamation budget mode")
        if self.policy not in ("full", "sink_recent", "random", "triattention"):
            raise ValueError("Unknown KV reclamation policy")
        if self.policy == "triattention" and self.stats_path is None:
            raise ValueError("TriAttention requires calibrated frequency stats")
        if not 0 < self.retention <= 1:
            raise ValueError("KV reclamation retention must be in (0, 1]")
        if min(self.sink_tokens, self.recent_tokens, self.reserve_cap) < 0:
            raise ValueError("KV reclamation protection/reserve must be nonnegative")
        if self.min_keep_tokens < self.sink_tokens + self.recent_tokens:
            raise ValueError("min_keep_tokens must cover sink and recent protection")

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
        parallel = config.parallel_config
        if (
            not envs.VLLM_USE_V2_MODEL_RUNNER
            or config.scheduler_config.async_scheduling
            or config.max_concurrent_batches != 1
            or config.speculative_config is not None
            or config.kv_transfer_config is not None
            or config.scheduler_config.scheduler_reserve_full_isl
            or parallel.tensor_parallel_size not in (1, 2)
            or parallel.pipeline_parallel_size != 1
            or parallel.decode_context_parallel_size != 1
            or parallel.prefill_context_parallel_size != 1
            or config.model_config.is_encoder_decoder
            or config.model_config.runner_type != "generate"
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
        spec = cache.kv_cache_groups[0].kv_cache_spec
        if (
            spec.sliding_window is not None
            or spec.attention_chunk_size is not None
            or spec.non_causal
        ):
            raise ValueError("KV reclamation requires unrestricted causal attention")
        if spec.block_size != 16:
            raise ValueError("KV reclamation requires B16")


@dataclass(frozen=True)
class PolicyBinding:
    version: int
    config: ReclamationConfig
    calibration_sha256: str | None = None


def validate_policy_acks(
    version: int,
    config: ReclamationConfig,
    acknowledgements: list[dict],
    tp_size: int,
) -> PolicyBinding:
    """A prepared version is usable only after every TP rank agrees."""
    if len(acknowledgements) != tp_size or {
        ack["rank"] for ack in acknowledgements
    } != set(range(tp_size)):
        raise ValueError("KV policy preparation requires every TP rank")
    binding = acknowledgements[0]["binding"]
    if (
        binding.version != version
        or binding.config != config
        or any(ack["binding"] != binding for ack in acknowledgements)
        or config.policy == "triattention"
        and not binding.calibration_sha256
    ):
        raise ValueError("KV policy preparation ACKs disagree")
    return binding


@dataclass(frozen=True)
class ReclaimRequest:
    request_id: str
    source_effective_kv_len: int
    source_block_ids: list[int]
    target_keep_tokens: int
    protected_prefix_tokens: int = 0
    policy_binding: PolicyBinding | None = None


@dataclass(frozen=True)
class ReclaimResult:
    request_id: str
    new_effective_kv_len: int
    retained_block_ids: list[int]
    policy_binding: PolicyBinding | None = None


@dataclass(frozen=True)
class ReclaimCandidate:
    request_id: str
    keep_tokens: int
    released_blocks: int
    copy_bytes: int
    zero_copy: bool
    arrival_time: float
    protected_prefix_tokens: int = 0
    source_effective_kv_len: int = 0
    source_block_ids: tuple[int, ...] = ()
    logical_tokens: int = 0
    q: int = 0
    policy_binding: PolicyBinding | None = None


@dataclass(frozen=True)
class ReclaimAllocation:
    request_id: str
    num_tokens: int
    required_blocks: int


@dataclass(frozen=True)
class ReclaimRetry:
    free_blocks: int
    allocations: tuple[ReclaimAllocation, ...]
    following_allocations: tuple[ReclaimAllocation, ...] = ()
    completion_headroom_blocks: int = 0


def protected_reclaim_floor(
    source: int, logical: int, prefix: int, config: ReclamationConfig
) -> int:
    """Keep the legacy E-domain floor, at least 80% of E and 50% of L privately."""
    private = source - prefix
    return max(
        prefix
        + max(
            config.min_keep_tokens,
            config.recent_tokens + max(0, config.sink_tokens - prefix),
            int(config.retention * private),
        ),
        prefix + (4 * private + 4) // 5,
        prefix + (max(0, logical - prefix) + 1) // 2,
    )


def protected_reclaim_plan(
    candidates: list[ReclaimCandidate],
    shortfall: int,
    config: ReclamationConfig,
    block_size: int,
    member_bytes: int,
    retry: ReclaimRetry | None = None,
) -> tuple[ReclaimCandidate, ...]:
    """Cover hard demand, then amortize within the same protected victims."""
    plan = _protected_reclaim_plan(
        candidates, shortfall, config, block_size, member_bytes, retry
    )
    if not plan or retry is None or retry.completion_headroom_blocks == 0:
        return plan
    chosen = {c.request_id for c in plan}
    selected = [c for c in candidates if c.request_id in chosen]
    allocations = {
        a.request_id: a for a in (*retry.allocations, *retry.following_allocations)
    }
    capacity = 0
    for candidate in selected:
        allocation = allocations.get(
            candidate.request_id, ReclaimAllocation(candidate.request_id, 0, 0)
        )
        capacity += (
            len(candidate.source_block_ids)
            - (candidate.keep_tokens + allocation.num_tokens + block_size - 1)
            // block_size
            + allocation.required_blocks
        )
    goal = min(shortfall + retry.completion_headroom_blocks, capacity)
    return _protected_reclaim_plan(
        selected, goal, config, block_size, member_bytes, retry
    )


def _protected_reclaim_plan(
    candidates: list[ReclaimCandidate],
    shortfall: int,
    config: ReclamationConfig,
    block_size: int,
    member_bytes: int,
    retry: ReclaimRetry | None,
) -> tuple[ReclaimCandidate, ...]:
    """Cover net demand, including victims' own post-reclaim allocations."""
    if shortfall <= 0:
        return ()
    allocations = (
        {a.request_id: a for a in (*retry.allocations, *retry.following_allocations)}
        if retry is not None
        else {}
    )

    def demand(candidate: ReclaimCandidate) -> ReclaimAllocation:
        return allocations.get(
            candidate.request_id, ReclaimAllocation(candidate.request_id, 0, 0)
        )

    def capacity(candidate: ReclaimCandidate) -> int:
        allocation = demand(candidate)
        return (
            len(candidate.source_block_ids)
            - (candidate.keep_tokens + allocation.num_tokens + block_size - 1)
            // block_size
            + allocation.required_blocks
        )

    def target(candidate: ReclaimCandidate, pages: int) -> ReclaimCandidate | None:
        allocation = demand(candidate)
        source_pages = len(candidate.source_block_ids)
        keep = min(
            candidate.source_effective_kv_len - 1,
            block_size * (source_pages - 1),
            block_size * (source_pages + allocation.required_blocks - pages)
            - allocation.num_tokens,
        )
        if keep < candidate.keep_tokens:
            return None
        policy = candidate.policy_binding.config if candidate.policy_binding else config
        if policy.policy == "sink_recent":
            prefix, _ = sink_recent_ranges(
                candidate.source_effective_kv_len,
                keep,
                policy.sink_tokens,
                candidate.protected_prefix_tokens,
            )
            aligned = (
                keep - (keep - candidate.source_effective_kv_len - prefix) % block_size
            )
            if (
                prefix % block_size == 0
                and aligned >= max(prefix, candidate.keep_tokens)
                and (aligned + allocation.num_tokens + block_size - 1) // block_size
                == (keep + allocation.num_tokens + block_size - 1) // block_size
            ):
                keep = aligned
        return retarget_reclaim_candidate(
            candidate, keep, policy, block_size, member_bytes
        )

    def ratio(candidate: ReclaimCandidate) -> Fraction:
        return Fraction(
            candidate.keep_tokens - candidate.protected_prefix_tokens,
            max(1, candidate.logical_tokens - candidate.protected_prefix_tokens),
        )

    covering = [c for candidate in candidates if (c := target(candidate, shortfall))]
    if covering:
        return (
            min(
                covering,
                key=lambda c: (
                    -ratio(c),
                    not c.zero_copy,
                    c.copy_bytes,
                    c.arrival_time,
                    c.request_id,
                ),
            ),
        )

    best: tuple[ReclaimCandidate, ...] = ()
    best_key = None
    ordered = sorted(candidates, key=lambda c: (c.arrival_time, c.request_id))
    for i, first in enumerate(ordered):
        for second in ordered[i + 1 :]:
            lower = max(1, shortfall - capacity(second))
            upper = min(capacity(first), shortfall - 1)
            if lower > upper:
                continue
            # Page remapping makes the ratios piecewise, but still monotone
            # with opposite slopes. Find the crossing without scanning pages.
            left, right = lower, upper
            while left < right:
                middle = (left + right) // 2
                a = target(first, middle)
                b = target(second, shortfall - middle)
                assert a is not None and b is not None
                if ratio(a) > ratio(b):
                    left = middle + 1
                else:
                    right = middle
            for first_pages in sorted({max(lower, left - 1), left}):
                a = target(first, first_pages)
                b = target(second, shortfall - first_pages)
                if a is None or b is None:
                    continue
                pair = (a, b)
                ratios = [ratio(c) for c in pair]
                key = (
                    -min(ratios),
                    abs(ratios[0] - ratios[1]),
                    sum(not c.zero_copy for c in pair),
                    sum(c.copy_bytes for c in pair),
                    tuple((c.arrival_time, c.request_id) for c in pair),
                )
                if best_key is None or key < best_key:
                    best, best_key = pair, key
    return best


def retry_reclaim_candidates(
    candidates: list[ReclaimCandidate],
    retry: ReclaimRetry,
    reserve: int,
    config: ReclamationConfig,
    block_size: int,
    member_bytes: int,
) -> list[ReclaimCandidate]:
    """Maximize K for the retry, optionally covering its known remaining pass.

    For a victim visited before (or at) the requester, release minus its next
    allocation is N - ceil((K + q_next) / B). A victim visited later has q_next=0.
    The full pass is a soft target, subordinate to the retention floor. This
    includes tail slack without reserving an unconditional extra page.
    """
    targets = [retry.allocations]
    if (
        config.budget_mode in ("retry_pass", "prefill_fixed")
        and retry.following_allocations
    ):
        targets.insert(0, retry.allocations + retry.following_allocations)
    for allocations in targets:
        total = sum(a.required_blocks for a in allocations)
        by_id = {a.request_id: a for a in allocations}
        for headroom in dict.fromkeys((reserve, 0)):
            covering = []
            for candidate in candidates:
                allocation = by_id.get(candidate.request_id)
                other = total - (allocation.required_blocks if allocation else 0)
                keep = min(
                    candidate.source_effective_kv_len - 1,
                    block_size * (len(candidate.source_block_ids) - 1),
                    block_size
                    * (
                        retry.free_blocks
                        + len(candidate.source_block_ids)
                        - other
                        - headroom
                    )
                    - (allocation.num_tokens if allocation else 0),
                )
                if keep >= candidate.keep_tokens:
                    covering.append(
                        retarget_reclaim_candidate(
                            candidate,
                            keep,
                            candidate.policy_binding.config
                            if candidate.policy_binding is not None
                            else config,
                            block_size,
                            member_bytes,
                        )
                    )
            if covering:
                return covering
    return candidates


def retarget_reclaim_candidate(
    candidate: ReclaimCandidate,
    keep: int,
    config: ReclamationConfig,
    block_size: int,
    member_bytes: int,
) -> ReclaimCandidate:
    prefix, recent = sink_recent_ranges(
        candidate.source_effective_kv_len,
        keep,
        config.sink_tokens,
        candidate.protected_prefix_tokens,
    )
    zero_copy = (
        config.policy == "sink_recent"
        and whole_block_row(
            list(candidate.source_block_ids),
            candidate.source_effective_kv_len,
            keep,
            prefix,
            recent,
            block_size,
        )
        is not None
    )
    return replace(
        candidate,
        keep_tokens=keep,
        released_blocks=len(candidate.source_block_ids)
        - (keep + block_size - 1) // block_size,
        copy_bytes=0 if zero_copy else 2 * (keep - prefix) * member_bytes,
        zero_copy=zero_copy,
    )


def demand_reclaim_candidates(
    candidates: list[ReclaimCandidate],
    shortfall: int,
    goal: int,
    config: ReclamationConfig,
    block_size: int,
    member_bytes: int,
) -> list[ReclaimCandidate]:
    """Maximize K at G, then D; retain fixed floors if neither is feasible."""
    for target in dict.fromkeys((goal, shortfall)):
        if target <= 0:
            continue
        covering = []
        for candidate in candidates:
            keep = min(
                candidate.source_effective_kv_len - 1,
                block_size * (len(candidate.source_block_ids) - target),
            )
            if keep >= candidate.keep_tokens:
                covering.append(
                    retarget_reclaim_candidate(
                        candidate,
                        keep,
                        candidate.policy_binding.config
                        if candidate.policy_binding is not None
                        else config,
                        block_size,
                        member_bytes,
                    )
                )
        if covering:
            return covering
    return candidates


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
