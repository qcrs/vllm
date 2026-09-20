from types import SimpleNamespace

import pytest
import torch

from vllm.config import CacheConfig
from vllm.v1.core.kv_cache_utils import (
    get_kv_cache_config_from_groups,
    get_max_concurrency_for_kv_cache_config,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheGroupSpec,
    RaggedAttentionSpec,
)


def _config(max_model_len: int = 64, *, override: int | None = None):
    return SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=max_model_len),
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
        cache_config=CacheConfig(num_gpu_blocks_override=override),
        kv_transfer_config=None,
    )


def _ragged_spec(*, block_size: int = 16, hp: int = 2):
    return RaggedAttentionSpec(
        block_size=block_size,
        num_kv_heads=8,
        page_group_size=hp,
        head_size=128,
        dtype=torch.bfloat16,
    )


def test_ragged_planner_global_backing_and_identity_memory():
    config = _config()
    layer_names = ["layer.0", "layer.1", "layer.2"]
    spec = _ragged_spec()
    group = KVCacheGroupSpec(layer_names, spec)
    available_memory = spec.page_size_bytes * 37 + spec.page_size_bytes // 2

    kv_config = get_kv_cache_config_from_groups(config, [group], available_memory)

    assert spec.num_kv_heads == 8
    assert spec.num_head_groups_per_layer == 4
    assert kv_config.num_blocks == available_memory // spec.page_size_bytes
    assert len(kv_config.kv_cache_tensors) == 1
    tensor = kv_config.kv_cache_tensors[0]
    assert tensor.size == kv_config.num_blocks * spec.page_size_bytes
    assert tensor.shared_by == layer_names

    dense = FullAttentionSpec(
        block_size=spec.block_size,
        num_kv_heads=spec.num_kv_heads,
        head_size=spec.head_size,
        dtype=spec.dtype,
    )
    depths = 4
    assert (
        len(layer_names)
        * depths
        * spec.num_head_groups_per_layer
        * spec.page_size_bytes
        == len(layer_names) * depths * dense.page_size_bytes
    )
    assert CacheConfig().page_group_size is None


def test_ragged_concurrency_counts_all_layer_physical_pages():
    config = _config(max_model_len=64)
    layer_names = ["layer.0", "layer.1", "layer.2"]
    spec = _ragged_spec()
    group = KVCacheGroupSpec(layer_names, spec)
    depths = 4
    physical_pages_per_request = (
        len(layer_names) * depths * spec.num_head_groups_per_layer
    )
    kv_config = get_kv_cache_config_from_groups(
        config, [group], spec.page_size_bytes * physical_pages_per_request
    )

    assert get_max_concurrency_for_kv_cache_config(config, kv_config) == 1
    assert physical_pages_per_request == len(layer_names) * 4 * depths
    assert physical_pages_per_request != 4 * depths


def test_ragged_planner_rejects_mixed_groups_and_override():
    config = _config()
    ragged_group = KVCacheGroupSpec(["ragged"], _ragged_spec())
    dense_group = KVCacheGroupSpec(
        ["dense"],
        FullAttentionSpec(
            block_size=16,
            num_kv_heads=8,
            head_size=128,
            dtype=torch.bfloat16,
        ),
    )
    with pytest.raises(ValueError, match="exactly one Ragged"):
        get_kv_cache_config_from_groups(
            config, [ragged_group, dense_group], 1 << 30
        )
    with pytest.raises(ValueError, match="num_gpu_blocks_override"):
        get_kv_cache_config_from_groups(
            _config(override=3), [ragged_group], 1 << 30
        )


def test_dense_planner_keeps_general_namespace():
    config = _config()
    groups = [
        KVCacheGroupSpec(
            ["layer.0", "layer.1"],
            FullAttentionSpec(
                block_size=16,
                num_kv_heads=8,
                head_size=128,
                dtype=torch.bfloat16,
            ),
        )
    ]
    page_size = groups[0].kv_cache_spec.page_size_bytes
    kv_config = get_kv_cache_config_from_groups(config, groups, page_size * 11)
    assert kv_config.num_blocks == 5
    assert len(kv_config.kv_cache_tensors) == 2
    assert all(t.size == page_size * 5 for t in kv_config.kv_cache_tensors)
