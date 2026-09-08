"""P1 V2 — 单 request Paged KV Token-Level Compaction 数据面实现。

本文件包含三层内容：

1. PyTorch Reference
   - 作为 correctness oracle。
   - 根据 block_ids / keep_member_indices 构造 compact 后的正确答案。
   - 不修改 runtime ownership，不释放 block。

2. Triton Gather
   - 从 paged KV 中读取需要保留的 members。
   - 输出独立 contiguous scratch[K,H,C]。

3. Triton Writeback
   - 将 scratch[K,H,C] 写回当前 request 的 physical block-row prefix。
   - compact 后未使用的最后一页 tail slot 写 0。

FA2 当前逻辑 KV shape：

    kv_cache: [B, H, N, C]

其中：

    B = 全局 physical block 数
    H = num_kv_heads
    N = block_size
    C = 2 * head_size

本文件只负责 KV payload transformation，不负责：

    - Worker BlockTable trim
    - effective_kv_len runtime commit
    - Scheduler canonical ownership reconcile
    - BlockPool free / deferred free
    - retention policy
"""


from __future__ import annotations

import torch

from vllm.triton_utils import tl, triton


# ---------------------------------------------------------------------------
# 1D / 2D Triton 实现思路
# ---------------------------------------------------------------------------
#
# Variant A — NHD-only 1D
#
#   grid = (K,)              # Gather
#   grid = (dst_capacity,)   # Writeback
#
#   一个 Triton program 负责一个完整 token 的 [H,C] payload。
#
#   之所以能这样做，是因为 NHD physical layout 为：
#
#       [B,N,H,C]
#
#   固定 physical_block 和 token_offset 后，
#   整个 [H,C] 在内存中连续，因此可以 flatten 成 H*C 一次搬运。
#
#
# Variant B — stride-safe 2D
#
#   grid = (K,H)                  # Gather
#   grid = (dst_capacity,H)       # Writeback
#
#   一个 Triton program 只负责一个 token 的一个 KV head，即 [C]。
#
#   source / destination 地址通过真实 stride：
#
#       stride_b
#       stride_h
#       stride_n
#       stride_c
#
#   计算，因此可以同时支持 HND / NHD。
#
#
# 两种 variant 的语义完全相同，区别只在 GPU work decomposition：
#
#   1D：一个 program 搬完整 token [H,C]
#   2D：一个 token 拆成 H 个 program，每个 program 搬 [C]
#
# 当前 full production candidate 使用 stride-safe 2D。
# ---------------------------------------------------------------------------


# PyTorch / Triton index tensor 当前只接受 int32 / int64。
_INDEX_DTYPES = {
    torch.int32,
    torch.int64,
}


def _validate_inputs(
    kv_cache: torch.Tensor,
    block_ids: torch.Tensor,
    source_effective_kv_len: int,
    keep_member_indices: torch.Tensor,
) -> int:
    """校验 Gather / Compaction 的公共输入，并返回 block_size。

    输入：
        kv_cache:
            当前 layer 的完整 paged KV pool，逻辑 shape [B,H,N,C]。

        block_ids:
            当前 request 的 active physical block row，shape [P]。
            例如 [7,2,11] 表示 request page 0/1/2 分别映射到 B7/B2/B11。

        source_effective_kv_len:
            当前真正有效的 physical KV member 数 E_source。

        keep_member_indices:
            需要保留的 current physical member indices，shape [K]。
            索引范围为 [0, E_source)，且必须严格递增。

    输出：
        block_size:
            kv_cache.shape[2]，即一个 physical page 可容纳的 token 数。

    约束：
        - E_source > 0
        - block_ids 必须精确覆盖当前 active source pages
        - block_ids 合法且不能重复
        - keep 非空、严格递增、不能越界
    """
    if kv_cache.ndim != 4:
        raise ValueError("kv_cache must have shape [B, H, N, C]")
    if block_ids.ndim != 1:
        raise ValueError("block_ids must be one-dimensional")
    if keep_member_indices.ndim != 1:
        raise ValueError("keep_member_indices must be one-dimensional")
    if block_ids.dtype not in _INDEX_DTYPES:
        raise TypeError("block_ids must have an integral dtype")
    if keep_member_indices.dtype not in _INDEX_DTYPES:
        raise TypeError("keep_member_indices must have an integral dtype")
    if not isinstance(source_effective_kv_len, int):
        raise TypeError("source_effective_kv_len must be an int")
    if source_effective_kv_len <= 0:
        raise ValueError("source_effective_kv_len must be positive")

    block_size = kv_cache.shape[2]
    expected_pages = (
        source_effective_kv_len + block_size - 1
    ) // block_size

    if block_ids.numel() != expected_pages:
        raise ValueError(
            "block_ids length must cover the source effective length"
        )
    if block_ids.numel() and (
        torch.any(block_ids < 0)
        or torch.any(block_ids >= kv_cache.shape[0])
    ):
        raise ValueError(
            "block_ids contain an invalid physical block"
        )
    if torch.unique(block_ids).numel() != block_ids.numel():
        raise ValueError("block_ids must be unique")

    if keep_member_indices.numel() == 0:
        raise ValueError("keep_member_indices must be non-empty")
    if torch.any(keep_member_indices < 0) or torch.any(
        keep_member_indices >= source_effective_kv_len
    ):
        raise ValueError(
            "keep_member_indices are outside the source sequence"
        )
    if keep_member_indices.numel() > 1 and torch.any(
        keep_member_indices[1:]
        <= keep_member_indices[:-1]
    ):
        raise ValueError(
            "keep_member_indices must be strictly increasing"
        )

    return block_size


def gather_paged_kv_reference(
    kv_cache: torch.Tensor,
    block_ids: torch.Tensor,
    source_effective_kv_len: int,
    keep_member_indices: torch.Tensor,
) -> torch.Tensor:
    """PyTorch Gather correctness oracle。

    作用：
        将当前 request 的 paged KV 按 request-local member 顺序展开，
        再根据 keep_member_indices 取出需要保留的 KV。

    输入：
        kv_cache:
            全局 KV pool，[B,H,N,C]。

        block_ids:
            当前 request 的 active block row，[P]。

        source_effective_kv_len:
            当前有效 physical KV 长度 E_source。

        keep_member_indices:
            要保留的 current member indices，[K]。

    输出：
        scratch:
            独立 contiguous tensor，[K,H,C]。

    关键点：
        scratch 使用 clone()，不与原 paged KV alias，
        因此后续 writeback 可以安全覆盖原 physical pages。
    """
    _validate_inputs(
        kv_cache,
        block_ids,
        source_effective_kv_len,
        keep_member_indices,
    )

    # 先按 request 的 block row 从全局 KV pool 取出 active pages。
    active_pages = kv_cache.index_select(0, block_ids)

    # [P,H,N,C] -> [P,N,H,C] -> [P*N,H,C]
    # 得到 request-local linear member sequence。
    members = active_pages.permute(0, 2, 1, 3).reshape(
        -1,
        kv_cache.shape[1],
        kv_cache.shape[3],
    )

    # 丢掉最后 source page 中超过 E_source 的无效 slots。
    members = members[:source_effective_kv_len]

    # 根据 keep 选择 retained members，并复制到 independent scratch。
    return members.index_select(
        0,
        keep_member_indices,
    ).clone()


def compact_paged_kv_reference(
    kv_cache: torch.Tensor,
    block_ids: torch.Tensor,
    source_effective_kv_len: int,
    keep_member_indices: torch.Tensor,
) -> tuple[torch.Tensor, int, int]:
    """PyTorch 完整 compaction correctness oracle。

    作用：
        生成 token-level compaction 后整个 global KV pool 的正确结果，
        但不修改输入 kv_cache。

    输入：
        kv_cache:
            全局 paged KV pool，[B,H,N,C]。

        block_ids:
            当前 request 的 source physical block row，[P]。

        source_effective_kv_len:
            当前有效 physical KV 长度 E_source。

        keep_member_indices:
            需要保留的 members，[K]。

    输出：
        expected:
            compaction 后完整 global KV pool。

        new_effective_kv_len:
            新的有效 physical KV 长度，等于 K。

        new_num_blocks:
            compact 后仍然需要的 physical pages 数，即 ceil(K / block_size)。

    关键语义：
        - retained KV 写回 block_ids 的 physical prefix
        - 最后 active page 未使用 tail 自动补 0
        - trailing detached blocks 保持原内容，不在这里 free
    """
    block_size = _validate_inputs(
        kv_cache,
        block_ids,
        source_effective_kv_len,
        keep_member_indices,
    )

    scratch = gather_paged_kv_reference(
        kv_cache,
        block_ids,
        source_effective_kv_len,
        keep_member_indices,
    )

    new_effective_kv_len = scratch.shape[0]
    new_num_blocks = (
        new_effective_kv_len
        + block_size
        - 1
    ) // block_size

    # clone 整个 KV pool，只重写当前 request 的 destination prefix。
    expected = kv_cache.clone()

    # 构造 page-aligned dense destination。
    # zeros 同时定义最后 active page 的 deterministic tail-zero 语义。
    dense = torch.zeros(
        new_num_blocks * block_size,
        kv_cache.shape[1],
        kv_cache.shape[3],
        dtype=kv_cache.dtype,
        device=kv_cache.device,
    )
    dense[:new_effective_kv_len] = scratch

    # [B_new*N,H,C] -> [B_new,H,N,C]
    pages = dense.reshape(
        new_num_blocks,
        block_size,
        kv_cache.shape[1],
        -1,
    ).permute(0, 2, 1, 3)

    # destination 始终是当前 physical block row 的 prefix。
    expected.index_copy_(
        0,
        block_ids[:new_num_blocks],
        pages,
    )

    return expected, new_effective_kv_len, new_num_blocks


@triton.jit
def _gather_paged_kv_nhd_1d_kernel(
    kv_cache,
    block_ids,
    keep_member_indices,
    scratch,
    block_size,
    num_heads,
    content_size,
    stride_b,
    stride_n,
    BLOCK_HC: tl.constexpr,
):
    """NHD-only 1D Gather Kernel。

    作用：
        一个 Triton program 负责一个 retained token，
        从 paged KV 读取完整 [H,C]，写入 scratch[j,:,:]。

    输入：
        kv_cache:
            全局 KV pool。

        block_ids:
            request physical block row。

        keep_member_indices:
            retained source member indices。

        scratch:
            输出 [K,H,C]。

        block_size / num_heads / content_size:
            分别对应 N / H / C。

        stride_b / stride_n:
            NHD 下定位 physical block 与 token offset 的 stride。

    输出：
        通过 side effect 写 scratch，不返回 tensor。

    限制：
        只适用于 physical NHD layout，因为固定 token 后 [H,C] 必须连续。
    """
    j = tl.program_id(0)

    # keep[j] -> source member -> source page / token offset。
    member = tl.load(keep_member_indices + j).to(tl.int64)
    page_idx = member // block_size
    token_offset = member % block_size
    physical_block = tl.load(block_ids + page_idx).to(tl.int64)

    # 一个 program 直接搬完整 H*C。
    hc = num_heads * content_size
    offs = tl.arange(0, BLOCK_HC)
    mask = offs < hc

    src = (
        physical_block * stride_b
        + token_offset * stride_n
        + offs
    )
    dst = j * hc + offs

    tl.store(
        scratch + dst,
        tl.load(kv_cache + src, mask=mask),
        mask=mask,
    )


@triton.jit
def _gather_paged_kv_2d_kernel(
    kv_cache,
    block_ids,
    keep_member_indices,
    scratch,
    block_size,
    num_heads,
    content_size,
    stride_b,
    stride_h,
    stride_n,
    stride_c,
    BLOCK_C: tl.constexpr,
):
    """Stride-safe 2D Gather Kernel。

    作用：
        一个 Triton program 负责一个 retained token 的一个 KV head，
        从 paged KV 读取 [C]，写入 scratch[j,h,:]。

    输入：
        kv_cache:
            全局 KV pool，[B,H,N,C] logical view。

        block_ids:
            当前 request physical block row。

        keep_member_indices:
            retained source member indices。

        scratch:
            输出 contiguous [K,H,C]。

        block_size / num_heads / content_size:
            分别对应 N / H / C。

        stride_b / stride_h / stride_n / stride_c:
            kv_cache 的真实 logical strides。

    输出：
        通过 side effect 写 scratch。

    特点：
        使用完整 stride 计算地址，因此同时支持 HND / NHD。
    """
    j = tl.program_id(0)
    h = tl.program_id(1)

    # keep[j] -> source member -> source page / token offset。
    member = tl.load(keep_member_indices + j).to(tl.int64)
    page_idx = member // block_size
    token_offset = member % block_size
    physical_block = tl.load(block_ids + page_idx).to(tl.int64)

    offs = tl.arange(0, BLOCK_C)
    mask = offs < content_size

    # kv_cache[physical_block, h, token_offset, :]
    src = (
        physical_block * stride_b
        + h * stride_h
        + token_offset * stride_n
        + offs * stride_c
    )

    # scratch[j,h,:]，scratch 为 contiguous。
    dst = (
        (j * num_heads + h)
        * content_size
        + offs
    )

    tl.store(
        scratch + dst,
        tl.load(kv_cache + src, mask=mask),
        mask=mask,
    )


def _validate_triton_inputs(
    kv_cache: torch.Tensor,
    block_ids: torch.Tensor,
    source_effective_kv_len: int,
    keep_member_indices: torch.Tensor,
) -> int:
    """校验 Triton Gather 的输入。

    输入：
        与 _validate_inputs() 相同。

    输出：
        block_size。

    额外约束：
        - kv_cache / block_ids / keep_member_indices 必须都在 CUDA
        - 三者必须位于同一个 CUDA device
    """
    block_size = _validate_inputs(
        kv_cache,
        block_ids,
        source_effective_kv_len,
        keep_member_indices,
    )

    if (
        not kv_cache.is_cuda
        or not block_ids.is_cuda
        or not keep_member_indices.is_cuda
    ):
        raise ValueError(
            "Triton gather requires CUDA tensors"
        )

    if (
        kv_cache.device != block_ids.device
        or kv_cache.device != keep_member_indices.device
    ):
        raise ValueError(
            "Triton gather inputs must share a CUDA device"
        )

    return block_size


def _validate_nhd_layout(
    kv_cache: torch.Tensor,
) -> None:
    """校验 logical [B,H,N,C] 是否真正由连续 NHD [B,N,H,C] storage 支撑。

    输入：
        kv_cache:
            logical shape [B,H,N,C]。

    输出：
        无返回值；不符合 NHD stride contract 时直接 raise。

    NHD logical strides：
        stride_c = 1
        stride_h = C
        stride_n = H*C
        stride_b = N*H*C
    """
    _, num_heads, block_size, content_size = kv_cache.shape

    is_nhd = (
        kv_cache.stride(3) == 1
        and kv_cache.stride(1) == content_size
        and kv_cache.stride(2)
        == num_heads * content_size
        and kv_cache.stride(0)
        == block_size * num_heads * content_size
    )

    if not is_nhd:
        raise ValueError(
            "Variant A requires an NHD-backed [B,H,N,C] tensor"
        )


def gather_paged_kv_triton_nhd_1d(
    kv_cache: torch.Tensor,
    block_ids: torch.Tensor,
    source_effective_kv_len: int,
    keep_member_indices: torch.Tensor,
) -> torch.Tensor:
    """启动 NHD-only 1D Triton Gather。

    输入：
        kv_cache:
            NHD-backed logical [B,H,N,C] KV pool。

        block_ids:
            request active block row，[P]。

        source_effective_kv_len:
            当前 E_source。

        keep_member_indices:
            retained source members，[K]。

    输出：
        scratch:
            contiguous [K,H,C]。

    实现：
        grid = (K,)
        一个 program 直接搬一个完整 token [H,C]。
    """
    _validate_triton_inputs(
        kv_cache,
        block_ids,
        source_effective_kv_len,
        keep_member_indices,
    )
    _validate_nhd_layout(kv_cache)

    _, num_heads, _, content_size = kv_cache.shape

    scratch = torch.empty(
        (
            keep_member_indices.numel(),
            num_heads,
            content_size,
        ),
        device=kv_cache.device,
        dtype=kv_cache.dtype,
    )

    block_hc = triton.next_power_of_2(
        num_heads * content_size
    )

    _gather_paged_kv_nhd_1d_kernel[
        (keep_member_indices.numel(),)
    ](
        kv_cache,
        block_ids,
        keep_member_indices,
        scratch,
        kv_cache.shape[2],
        num_heads,
        content_size,
        kv_cache.stride(0),
        kv_cache.stride(2),
        BLOCK_HC=block_hc,
        num_warps=4,
    )

    return scratch


def gather_paged_kv_triton_2d(
    kv_cache: torch.Tensor,
    block_ids: torch.Tensor,
    source_effective_kv_len: int,
    keep_member_indices: torch.Tensor,
) -> torch.Tensor:
    """启动 stride-safe 2D Triton Gather。

    输入：
        kv_cache:
            logical [B,H,N,C] KV pool，可为 HND 或 NHD。

        block_ids:
            request active block row，[P]。

        source_effective_kv_len:
            当前 E_source。

        keep_member_indices:
            retained source members，[K]。

    输出：
        scratch:
            contiguous [K,H,C]。

    实现：
        grid = (K,H)
        一个 program 搬一个 retained token 的一个 KV head [C]。
    """
    _validate_triton_inputs(
        kv_cache,
        block_ids,
        source_effective_kv_len,
        keep_member_indices,
    )

    _, num_heads, _, content_size = kv_cache.shape

    scratch = torch.empty(
        (
            keep_member_indices.numel(),
            num_heads,
            content_size,
        ),
        device=kv_cache.device,
        dtype=kv_cache.dtype,
    )

    block_c = triton.next_power_of_2(
        content_size
    )

    _gather_paged_kv_2d_kernel[
        (
            keep_member_indices.numel(),
            num_heads,
        )
    ](
        kv_cache,
        block_ids,
        keep_member_indices,
        scratch,
        kv_cache.shape[2],
        num_heads,
        content_size,
        kv_cache.stride(0),
        kv_cache.stride(1),
        kv_cache.stride(2),
        kv_cache.stride(3),
        BLOCK_C=block_c,
        num_warps=4,
    )

    return scratch


@triton.jit
def _writeback_paged_kv_nhd_1d_kernel(
    kv_cache,
    block_ids,
    scratch,
    retained_members,
    block_size,
    num_heads,
    content_size,
    stride_b,
    stride_n,
    BLOCK_HC: tl.constexpr,
):
    """NHD-only 1D Writeback Kernel。

    作用：
        一个 program 负责一个 destination token 的完整 [H,C]，
        将 scratch 写回 paged KV physical prefix。

    输入：
        kv_cache:
            destination KV pool。

        block_ids:
            当前 request physical block row。

        scratch:
            Gather 后的 contiguous [K,H,C]。

        retained_members:
            K，即 scratch 中真正 retained token 数。

        block_size / num_heads / content_size:
            N / H / C。

        stride_b / stride_n:
            NHD destination addressing 所需 stride。

    输出：
        通过 side effect 修改 kv_cache。

    特殊语义：
        当 dst_member >= K 时不读取 scratch，而是写 0，
        用于清理最后 active destination page 的 unused tail。
    """
    dst_member = tl.program_id(0)

    # compact 后 destination member 直接映射到连续 prefix。
    dst_page = dst_member // block_size
    dst_offset = dst_member % block_size
    physical_block = tl.load(
        block_ids + dst_page
    ).to(tl.int64)

    hc = num_heads * content_size
    offs = tl.arange(0, BLOCK_HC)
    valid_hc = offs < hc

    # retained slot 读取 scratch；tail slot 使用 other=0.0 生成 zero。
    is_retained = dst_member < retained_members
    values = tl.load(
        scratch + dst_member * hc + offs,
        mask=is_retained & valid_hc,
        other=0.0,
    )

    # NHD 固定 token 后 [H,C] 连续。
    dst = (
        physical_block * stride_b
        + dst_offset * stride_n
        + offs
    )

    tl.store(
        kv_cache + dst,
        values,
        mask=valid_hc,
    )


@triton.jit
def _writeback_paged_kv_2d_kernel(
    kv_cache,
    block_ids,
    scratch,
    retained_members,
    block_size,
    num_heads,
    content_size,
    stride_b,
    stride_h,
    stride_n,
    stride_c,
    BLOCK_C: tl.constexpr,
):
    """Stride-safe 2D Writeback Kernel。

    作用：
        一个 program 负责一个 destination token 的一个 KV head [C]，
        将 scratch[j,h,:] 写回 paged KV prefix。

    输入：
        kv_cache:
            destination KV pool，[B,H,N,C] logical view。

        block_ids:
            当前 request physical block row。

        scratch:
            contiguous [K,H,C]。

        retained_members:
            K。

        block_size / num_heads / content_size:
            N / H / C。

        stride_b / stride_h / stride_n / stride_c:
            kv_cache 的真实 strides。

    输出：
        通过 side effect 修改 kv_cache。

    特点：
        - HND / NHD 都支持
        - dst_member >= K 时写 0，完成 final-page tail zero
    """
    dst_member = tl.program_id(0)
    h = tl.program_id(1)

    # destination member -> destination page / token offset。
    dst_page = dst_member // block_size
    dst_offset = dst_member % block_size
    physical_block = tl.load(
        block_ids + dst_page
    ).to(tl.int64)

    offs = tl.arange(0, BLOCK_C)
    valid_c = offs < content_size

    # scratch[j,h,:]；tail slot 不读 scratch，直接生成 zero。
    is_retained = dst_member < retained_members
    src = (
        (dst_member * num_heads + h)
        * content_size
        + offs
    )
    values = tl.load(
        scratch + src,
        mask=is_retained & valid_c,
        other=0.0,
    )

    # 使用真实 stride 定位 destination，兼容 HND / NHD。
    dst = (
        physical_block * stride_b
        + h * stride_h
        + dst_offset * stride_n
        + offs * stride_c
    )

    tl.store(
        kv_cache + dst,
        values,
        mask=valid_c,
    )


def _validate_writeback_inputs(
    kv_cache: torch.Tensor,
    block_ids: torch.Tensor,
    scratch: torch.Tensor,
) -> tuple[int, int, int]:
    """校验 Scratch -> Paged KV Writeback 输入。

    输入：
        kv_cache:
            destination KV pool，[B,H,N,C]。

        block_ids:
            当前 request physical block row，[P]。

        scratch:
            Gather 后的 contiguous [K,H,C]。

    输出：
        block_size:
            N。

        retained_members:
            K。

        new_num_blocks:
            compact 后 destination prefix 需要的 block 数。

    约束：
        - scratch H/C 必须与 kv_cache 一致
        - scratch 与 kv_cache dtype/device 必须一致
        - block_ids 至少覆盖 new_num_blocks 个 destination pages
    """
    if kv_cache.ndim != 4:
        raise ValueError(
            "kv_cache must have shape [B, H, N, C]"
        )
    if block_ids.ndim != 1:
        raise ValueError(
            "block_ids must be one-dimensional"
        )
    if scratch.ndim != 3:
        raise ValueError(
            "scratch must have shape [K, H, C]"
        )
    if block_ids.dtype not in _INDEX_DTYPES:
        raise TypeError(
            "block_ids must have an integral dtype"
        )
    if scratch.shape[0] <= 0:
        raise ValueError(
            "scratch must contain at least one retained member"
        )
    if scratch.shape[1:] != (
        kv_cache.shape[1],
        kv_cache.shape[3],
    ):
        raise ValueError(
            "scratch H and C dimensions must match kv_cache"
        )
    if scratch.dtype != kv_cache.dtype:
        raise TypeError(
            "scratch and kv_cache must have the same dtype"
        )
    if not scratch.is_contiguous():
        raise ValueError(
            "scratch must be contiguous"
        )

    if (
        not kv_cache.is_cuda
        or not block_ids.is_cuda
        or not scratch.is_cuda
    ):
        raise ValueError(
            "Triton writeback requires CUDA tensors"
        )

    if (
        kv_cache.device != block_ids.device
        or kv_cache.device != scratch.device
    ):
        raise ValueError(
            "Triton writeback inputs must share a CUDA device"
        )

    if torch.any(block_ids < 0) or torch.any(
        block_ids >= kv_cache.shape[0]
    ):
        raise ValueError(
            "block_ids contain an invalid physical block"
        )
    if torch.unique(block_ids).numel() != block_ids.numel():
        raise ValueError(
            "block_ids must be unique"
        )

    block_size = kv_cache.shape[2]
    retained_members = scratch.shape[0]
    new_num_blocks = (
        retained_members + block_size - 1
    ) // block_size

    if block_ids.numel() < new_num_blocks:
        raise ValueError(
            "block_ids do not cover the destination prefix"
        )

    return (
        block_size,
        retained_members,
        new_num_blocks,
    )


def writeback_paged_kv_triton_nhd_1d(
    kv_cache: torch.Tensor,
    block_ids: torch.Tensor,
    scratch: torch.Tensor,
) -> tuple[int, int]:
    """启动 NHD-only 1D Triton Writeback。

    输入：
        kv_cache:
            NHD-backed destination KV pool。

        block_ids:
            当前 request physical block row。

        scratch:
            contiguous retained KV，[K,H,C]。

    输出：
        retained_members:
            新的 effective KV member 数 K。

        new_num_blocks:
            compact 后需要的 physical block 数。

    实现：
        grid = (dst_capacity,)
        一个 program 写一个完整 destination token [H,C]。
        dst_capacity = new_num_blocks * block_size，
        因此 final-page tail 也会被 launch 并写 0。
    """
    (
        block_size,
        retained_members,
        new_num_blocks,
    ) = _validate_writeback_inputs(
        kv_cache,
        block_ids,
        scratch,
    )

    _validate_nhd_layout(kv_cache)

    num_heads = kv_cache.shape[1]
    content_size = kv_cache.shape[3]

    # launch 到完整 active page capacity，顺便清理最后一页 tail。
    dst_capacity = new_num_blocks * block_size
    block_hc = triton.next_power_of_2(
        num_heads * content_size
    )

    _writeback_paged_kv_nhd_1d_kernel[
        (dst_capacity,)
    ](
        kv_cache,
        block_ids,
        scratch,
        retained_members,
        block_size,
        num_heads,
        content_size,
        kv_cache.stride(0),
        kv_cache.stride(2),
        BLOCK_HC=block_hc,
        num_warps=4,
    )

    return retained_members, new_num_blocks


def writeback_paged_kv_triton_2d(
    kv_cache: torch.Tensor,
    block_ids: torch.Tensor,
    scratch: torch.Tensor,
) -> tuple[int, int]:
    """启动 stride-safe 2D Triton Writeback。

    输入：
        kv_cache:
            destination KV pool，可为 HND 或 NHD。

        block_ids:
            当前 request physical block row。

        scratch:
            contiguous retained KV，[K,H,C]。

    输出：
        retained_members:
            新的 effective KV member 数 K。

        new_num_blocks:
            compact 后需要的 physical block 数。

    实现：
        grid = (dst_capacity,H)
        一个 program 写一个 destination token/head 的 [C]。
        使用真实 stride 完成 HND / NHD destination addressing。
    """
    (
        block_size,
        retained_members,
        new_num_blocks,
    ) = _validate_writeback_inputs(
        kv_cache,
        block_ids,
        scratch,
    )

    num_heads = kv_cache.shape[1]
    content_size = kv_cache.shape[3]

    dst_capacity = new_num_blocks * block_size
    block_c = triton.next_power_of_2(
        content_size
    )

    _writeback_paged_kv_2d_kernel[
        (
            dst_capacity,
            num_heads,
        )
    ](
        kv_cache,
        block_ids,
        scratch,
        retained_members,
        block_size,
        num_heads,
        content_size,
        kv_cache.stride(0),
        kv_cache.stride(1),
        kv_cache.stride(2),
        kv_cache.stride(3),
        BLOCK_C=block_c,
        num_warps=4,
    )

    return retained_members, new_num_blocks


def compact_paged_kv_triton_2d(
    kv_cache: torch.Tensor,
    block_ids: torch.Tensor,
    source_effective_kv_len: int,
    keep_member_indices: torch.Tensor,
) -> tuple[int, int]:
    """执行完整的 stride-safe Triton KV compaction 数据面。

    输入：
        kv_cache:
            当前 layer 的 paged KV pool，[B,H,N,C]。

        block_ids:
            当前 request source physical block row，[P]。

        source_effective_kv_len:
            当前 E_source。

        keep_member_indices:
            需要保留的 current physical members，[K]。

    输出：
        new_effective_kv_len:
            compact 后有效 physical KV 长度 K。

        new_num_blocks:
            compact 后仍需保留的 physical pages 数。

    数据流：
        paged KV
            -> 2D Gather
            -> independent scratch[K,H,C]
            -> 2D Writeback
            -> compacted physical prefix

    注意：
        本函数只修改 KV payload。
        不修改 Worker BlockTable、Scheduler ownership 或 BlockPool。
    """
    scratch = gather_paged_kv_triton_2d(
        kv_cache,
        block_ids,
        source_effective_kv_len,
        keep_member_indices,
    )

    return writeback_paged_kv_triton_2d(
        kv_cache,
        block_ids,
        scratch,
    )
