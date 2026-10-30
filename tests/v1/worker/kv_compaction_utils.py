# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Small independent CPU scalar oracle; never used by the serving runtime."""

import torch


def scalar_retained_kv(
    layers: list[torch.Tensor],
    source_row: list[int],
    keep: list[int],
    *,
    whole_blocks: bool = False,
) -> tuple[list[torch.Tensor], list[int]]:
    block_size = layers[0].shape[2]
    heads, channels = layers[0].shape[1], layers[0].shape[3]
    retained = []
    for layer in layers:
        assert layer.device.type == "cpu"
        expected = torch.empty((len(keep), heads, channels), dtype=layer.dtype)
        for j, member in enumerate(keep):
            page = source_row[member // block_size]
            offset = member % block_size
            for head in range(heads):
                for channel in range(channels):
                    expected[j, head, channel] = layer[page, head, offset, channel]
        retained.append(expected)
    row = source_row[: (len(keep) + block_size - 1) // block_size]
    if whole_blocks:
        row = []
        for j, member in enumerate(keep):
            if j % block_size == 0:
                row.append(source_row[member // block_size])
            assert member % block_size == j % block_size
            assert source_row[member // block_size] == row[j // block_size]
    return retained, row
