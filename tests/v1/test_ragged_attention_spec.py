from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from vllm.config import CacheConfig
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVQuantMode,
    RaggedAttentionSpec,
)


def _config(max_model_len: int = 1024):
    return SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=max_model_len),
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
    )


def test_ragged_geometry_and_identity_memory_invariant():
    ragged = RaggedAttentionSpec(
        block_size=16,
        num_kv_heads=8,
        page_group_size=2,
        head_size=128,
        dtype=torch.bfloat16,
    )
    dense = FullAttentionSpec(
        block_size=16,
        num_kv_heads=8,
        head_size=128,
        dtype=torch.bfloat16,
    )

    assert CacheConfig().page_group_size is None
    assert ragged.num_kv_heads == 8
    assert ragged.num_head_groups_per_layer == 4
    assert ragged.real_page_size_bytes == 2 * 16 * 2 * 128 * 2
    assert ragged.num_head_groups_per_layer * ragged.page_size_bytes == (
        dense.page_size_bytes
    )
    assert ragged.max_memory_usage_bytes(_config()) == dense.max_memory_usage_bytes(
        _config()
    )


def test_dataclasses_replace_preserves_ragged_fields_and_type():
    spec = RaggedAttentionSpec(
        block_size=16,
        num_kv_heads=8,
        page_group_size=2,
        head_size=128,
        head_size_v=128,
        dtype=torch.float16,
    )
    replaced = replace(spec, indexes_kv_by_block_stride=True)

    assert isinstance(replaced, RaggedAttentionSpec)
    assert replaced.page_group_size == 2
    assert replaced.head_size_v == 128
    assert replaced.indexes_kv_by_block_stride is True


def test_ragged_rejects_non_divisible_page_width():
    with pytest.raises(ValueError):
        RaggedAttentionSpec(
            block_size=16,
            num_kv_heads=8,
            page_group_size=3,
            head_size=128,
            dtype=torch.bfloat16,
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"head_size_v": 64},
        {"kv_quant_mode": KVQuantMode.FP8_PER_TENSOR},
        {"page_size_padded": 1 << 20},
    ],
)
def test_ragged_rejects_unsupported_core_geometry(kwargs):
    with pytest.raises(ValueError):
        RaggedAttentionSpec(
            block_size=16,
            num_kv_heads=8,
            page_group_size=2,
            head_size=128,
            dtype=torch.bfloat16,
            **kwargs,
        )
