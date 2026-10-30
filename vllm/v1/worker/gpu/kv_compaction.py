# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stable in-place compaction of packed paged K/V using one CTA per layer/head."""

from math import gcd

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _compact_stable(
    layer_ptrs,
    strides,
    row,
    keep,
    K: tl.constexpr,
    B: tl.constexpr,
    CHANNELS: tl.constexpr,
    TILE: tl.constexpr,
    CHANNEL_TILE: tl.constexpr,
    CHANNEL_STRIDE: tl.constexpr,
    ALIGNMENT: tl.constexpr,
):
    head = tl.program_id(0)
    layer = tl.program_id(1)
    address = tl.multiple_of(tl.load(layer_ptrs + layer), ALIGNMENT * 2)
    cache = address.to(tl.pointer_type(tl.bfloat16))
    sb = tl.load(strides + layer * 4)
    sh = tl.load(strides + layer * 4 + 1)
    st = tl.load(strides + layer * 4 + 2)
    sc = CHANNEL_STRIDE
    if CHANNEL_STRIDE == 0:
        sc = tl.load(strides + layer * 4 + 3)
    channels = tl.arange(0, CHANNEL_TILE)
    for start in range(0, triton.cdiv(K, TILE)):
        j = start * TILE + tl.arange(0, TILE)
        source = tl.load(keep + j, j < K, other=0)
        source_page = tl.load(row + source // B, j < K, other=0).to(tl.int64)
        dest_page = tl.load(row + j // B, j < K, other=0).to(tl.int64)
        source_offset = tl.multiple_of(
            source_page * sb + head * sh + source % B * st, ALIGNMENT
        )
        dest_offset = tl.multiple_of(dest_page * sb + head * sh + j % B * st, ALIGNMENT)
        changed = (j < K) & (source != j)
        mask = changed[:, None] & (channels[None, :] < CHANNELS)
        value = tl.load(
            cache + source_offset[:, None] + channels[None, :] * sc,
            mask,
            other=0,
        )
        # All warps must finish reading overlapping source members before writes.
        tl.debug_barrier()
        tl.store(cache + dest_offset[:, None] + channels[None, :] * sc, value, mask)


class StableKVCompactor:
    def __init__(self, layers: list[torch.Tensor]):
        self.layers = layers
        shape = layers[0].shape
        if any(t.ndim != 4 or t.shape[1:] != shape[1:] for t in layers):
            raise ValueError("Compaction requires equal packed layer shapes")
        if any(t.dtype != torch.bfloat16 or not t.is_cuda for t in layers):
            raise ValueError("Compaction requires CUDA BF16 views")
        self.heads, self.block_size, self.channels = shape[1:]
        channel_strides = {t.stride(3) for t in layers}
        self.channel_stride = channel_strides.pop() if len(channel_strides) == 1 else 0
        alignment = gcd(
            *(s for t in layers for s in (t.data_ptr() // 2, *t.stride()[:3]))
        )
        self.alignment = min(8, alignment & -alignment)
        device = layers[0].device
        self.layer_ptrs = torch.tensor(
            [t.data_ptr() for t in layers], dtype=torch.int64, device=device
        )
        self.strides = torch.tensor(
            [t.stride() for t in layers], dtype=torch.int64, device=device
        )

    def compact(
        self,
        row: torch.Tensor,
        keep: torch.Tensor,
        *,
        tile: int = 16,
        num_warps: int = 4,
    ):
        """Keep must be a strictly increasing int32 current-member plan."""
        return _compact_stable[(self.heads, len(self.layers))](
            self.layer_ptrs,
            self.strides,
            row,
            keep,
            keep.numel(),
            self.block_size,
            self.channels,
            tile,
            triton.next_power_of_2(self.channels),
            self.channel_stride,
            self.alignment,
            num_warps=num_warps,
            num_stages=1,
        )
