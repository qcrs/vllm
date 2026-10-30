# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate retained bytes, not undefined page tails or FullKV model quality."""

import pytest
import torch

from tests.v1.worker.kv_compaction_utils import scalar_retained_kv


@pytest.mark.parametrize("whole_blocks", [False, True])
def test_scalar_oracle_decodes_nonmonotonic_pages_and_partial_tail(whole_blocks):
    layers = []
    for layer in range(2):
        cache = torch.empty_strided((7, 2, 4, 6), (128, 6, 12, 1))
        for page in range(7):
            for head in range(2):
                for token in range(4):
                    for channel in range(6):
                        cache[page, head, token, channel] = (
                            layer * 10000
                            + page * 1000
                            + token * 100
                            + head * 10
                            + channel
                        )
        layers.append(cache)
    keep = [0, 1, 2, 3, 8, 9]
    values, row = scalar_retained_kv(layers, [6, 1, 4], keep, whole_blocks=whole_blocks)
    assert row == ([6, 4] if whole_blocks else [6, 1])
    for layer in range(2):
        for j, source_page in enumerate([6, 6, 6, 6, 4, 4]):
            assert values[layer][j, 1, 5] == (
                layer * 10000 + source_page * 1000 + j % 4 * 100 + 15
            )
