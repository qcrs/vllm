import pytest
import torch

from vllm.v1.worker.gpu.kv_compaction import (
    compact_paged_kv_reference,
    compact_paged_kv_triton_2d,
    gather_paged_kv_reference,
    gather_paged_kv_triton_2d,
    gather_paged_kv_triton_nhd_1d,
    writeback_paged_kv_triton_2d,
    writeback_paged_kv_triton_nhd_1d,
)


def _cache(block_size=4, num_blocks=6, heads=2, content=2, *, nhd=False):
    storage = torch.empty(num_blocks, block_size, heads, content, dtype=torch.bfloat16)
    for b in range(num_blocks):
        for n in range(block_size):
            for h in range(heads):
                storage[b, n, h] = torch.tensor([b * 100 + n * 10 + h, h])
    logical = storage.permute(0, 2, 1, 3)
    return logical if nhd else logical.contiguous()


def _members(cache, block_ids, effective):
    return cache.index_select(0, block_ids).permute(0, 2, 1, 3).reshape(
        -1, cache.shape[1], cache.shape[3]
    )[:effective]


def test_keep_all():
    cache = _cache()
    ids = torch.tensor([5, 1, 3])
    keep = torch.arange(10)
    out, length, pages = compact_paged_kv_reference(cache, ids, 10, keep)
    assert length == 10 and pages == 3
    assert torch.equal(_members(out, ids, 10), _members(cache, ids, 10))


def test_interior_drop_and_cross_source_page():
    cache = _cache()
    ids = torch.tensor([5, 1, 3])
    keep = torch.tensor([0, 2, 5, 8])
    scratch = gather_paged_kv_reference(cache, ids, 10, keep)
    assert torch.equal(scratch, _members(cache, ids, 10).index_select(0, keep))


def test_cross_destination_page_and_partial_tail():
    cache = _cache()
    ids = torch.tensor([5, 1, 3])
    keep = torch.tensor([0, 2, 4, 6, 8])
    out, length, pages = compact_paged_kv_reference(cache, ids, 10, keep)
    assert (length, pages) == (5, 2)
    expected = _members(cache, ids, 10).index_select(0, keep)
    assert torch.equal(_members(out, ids, 5), expected)
    assert torch.count_nonzero(out[ids[1], :, 1:, :]) == 0


def test_exact_boundary_and_heavy_shrink():
    cache = _cache()
    ids = torch.tensor([5, 1])
    keep = torch.tensor([1, 3, 6, 7])
    out, length, pages = compact_paged_kv_reference(cache, ids, 8, keep)
    assert (length, pages) == (4, 1)
    expected = _members(cache, ids, 8).index_select(0, keep)
    assert torch.equal(_members(out, ids, 4), expected)


def test_overlap_hazard_uses_independent_scratch():
    cache = _cache(block_size=4, num_blocks=8)
    original = cache.clone()
    ids = torch.tensor([7, 2, 6, 1, 5, 3])
    keep = torch.tensor([0, 1, 4, 5, 19, 20])
    out, length, pages = compact_paged_kv_reference(cache, ids, 21, keep)
    assert (length, pages) == (6, 2)
    expected = _members(original, ids, 21).index_select(0, keep)
    assert torch.equal(_members(out, ids, 6), expected)
    assert torch.equal(cache, original)


@pytest.mark.parametrize("nhd", [False, True])
def test_hnd_and_nhd_strides(nhd):
    cache = _cache(nhd=nhd)
    ids = torch.tensor([5, 1, 3])
    keep = torch.tensor([0, 4, 9])
    expected = _members(cache, ids, 10).index_select(0, keep)
    assert torch.equal(
        gather_paged_kv_reference(cache, ids, 10, keep), expected
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"source_effective_kv_len": 0},
        {"block_ids": torch.tensor([1, 1, 2])},
        {"block_ids": torch.tensor([1, 2])},
        {"keep_member_indices": torch.tensor([], dtype=torch.int64)},
        {"keep_member_indices": torch.tensor([0, 2, 1])},
        {"keep_member_indices": torch.tensor([0, 10])},
        {"keep_member_indices": torch.tensor([0.0, 1.0])},
    ],
)
def test_invalid_inputs(kwargs):
    cache = _cache()
    args = dict(
        block_ids=torch.tensor([5, 1, 3]),
        source_effective_kv_len=10,
        keep_member_indices=torch.tensor([0, 1]),
    )
    args.update(kwargs)
    with pytest.raises((ValueError, TypeError)):
        gather_paged_kv_reference(cache, **args)


def _cuda_cache(num_blocks, block_size, heads, content, *, nhd):
    physical = torch.arange(
        num_blocks * block_size * heads * content,
        device="cuda",
        dtype=torch.int64,
    ).reshape(num_blocks, block_size, heads, content)
    physical = (physical % 1024).to(torch.bfloat16)
    logical = physical.permute(0, 2, 1, 3)
    return logical if nhd else logical.contiguous()


_CUDA_REQUIRED = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA for Triton execution"
)


@_CUDA_REQUIRED
@pytest.mark.parametrize(
    ("source_length", "keep", "block_ids"),
    [
        (10, list(range(10)), [7, 2, 11]),
        (10, [0, 2, 5, 8, 9], [7, 2, 11]),
        (10, [3, 4, 7, 8], [7, 2, 11]),
        (10, [9], [7, 2, 11]),
        (12, [0, 3, 4, 7], [7, 2, 11]),
        (21, [0, 1, 4, 5, 19, 20], [7, 2, 11, 1, 9, 5]),
    ],
)
@pytest.mark.parametrize("variant", ["nhd_1d", "stride_2d"])
def test_triton_gather_nhd_semantics(
    source_length, keep, block_ids, variant
):
    cache = _cuda_cache(12, 4, 4, 32, nhd=True)
    ids = torch.tensor(block_ids, device="cuda", dtype=torch.int32)
    keep_tensor = torch.tensor(keep, device="cuda", dtype=torch.int64)
    expected = gather_paged_kv_reference(
        cache, ids, source_length, keep_tensor
    )
    gather = (
        gather_paged_kv_triton_nhd_1d
        if variant == "nhd_1d"
        else gather_paged_kv_triton_2d
    )
    actual = gather(cache, ids, source_length, keep_tensor)
    assert torch.equal(actual, expected)


@_CUDA_REQUIRED
@pytest.mark.parametrize("nhd", [False, True])
@pytest.mark.parametrize(("heads", "content"), [(2, 16), (4, 32), (8, 128)])
def test_triton_2d_supports_hnd_and_nhd(nhd, heads, content):
    cache = _cuda_cache(12, 4, heads, content, nhd=nhd)
    ids = torch.tensor([7, 2, 11], device="cuda", dtype=torch.int32)
    keep = torch.tensor([0, 2, 5, 8, 9], device="cuda", dtype=torch.int64)
    expected = gather_paged_kv_reference(cache, ids, 10, keep)
    actual = gather_paged_kv_triton_2d(cache, ids, 10, keep)
    assert torch.equal(actual, expected)


@_CUDA_REQUIRED
@pytest.mark.parametrize(("heads", "content"), [(2, 16), (4, 32), (8, 256)])
def test_triton_nhd_1d_shape_coverage(heads, content):
    cache = _cuda_cache(12, 4, heads, content, nhd=True)
    ids = torch.tensor([7, 2, 11], device="cuda", dtype=torch.int32)
    keep = torch.tensor([0, 2, 5, 8, 9], device="cuda", dtype=torch.int64)
    expected = gather_paged_kv_reference(cache, ids, 10, keep)
    actual = gather_paged_kv_triton_nhd_1d(cache, ids, 10, keep)
    assert torch.equal(actual, expected)


@_CUDA_REQUIRED
def test_triton_nhd_1d_rejects_hnd():
    cache = _cuda_cache(12, 4, 4, 32, nhd=False)
    ids = torch.tensor([7, 2, 11], device="cuda", dtype=torch.int32)
    keep = torch.tensor([0, 2, 5, 8, 9], device="cuda", dtype=torch.int64)
    with pytest.raises(ValueError, match="NHD-backed"):
        gather_paged_kv_triton_nhd_1d(cache, ids, 10, keep)


@_CUDA_REQUIRED
@pytest.mark.parametrize("nhd", [False, True])
@pytest.mark.parametrize("keep", [[0, 2, 5, 8, 9], [0, 1, 4, 5], [9]])
def test_triton_writeback_2d_matches_reference(nhd, keep):
    before = _cuda_cache(12, 4, 4, 32, nhd=nhd)
    ids = torch.tensor([7, 2, 11], device="cuda", dtype=torch.int32)
    keep_tensor = torch.tensor(keep, device="cuda", dtype=torch.int64)
    scratch = gather_paged_kv_triton_2d(before, ids, 10, keep_tensor)
    actual = before.clone()
    result = writeback_paged_kv_triton_2d(actual, ids, scratch)
    expected, expected_len, expected_pages = compact_paged_kv_reference(
        before, ids.long(), 10, keep_tensor
    )
    assert result == (expected_len, expected_pages)
    assert torch.equal(actual, expected)


@_CUDA_REQUIRED
@pytest.mark.parametrize(("heads", "content"), [(2, 16), (4, 32), (8, 256)])
def test_triton_writeback_nhd_1d_shape_coverage(heads, content):
    before = _cuda_cache(12, 4, heads, content, nhd=True)
    ids = torch.tensor([7, 2, 11], device="cuda", dtype=torch.int32)
    keep = torch.tensor([0, 2, 5, 8, 9], device="cuda", dtype=torch.int64)
    scratch = gather_paged_kv_triton_nhd_1d(before, ids, 10, keep)
    actual = before.clone()
    result = writeback_paged_kv_triton_nhd_1d(actual, ids, scratch)
    expected, expected_len, expected_pages = compact_paged_kv_reference(
        before, ids.long(), 10, keep
    )
    assert result == (expected_len, expected_pages)
    assert torch.equal(actual, expected)


@_CUDA_REQUIRED
def test_triton_writeback_nhd_1d_rejects_hnd():
    before = _cuda_cache(12, 4, 4, 32, nhd=False)
    ids = torch.tensor([7, 2, 11], device="cuda", dtype=torch.int32)
    scratch = torch.zeros((5, 4, 32), device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="NHD-backed"):
        writeback_paged_kv_triton_nhd_1d(before, ids, scratch)


@_CUDA_REQUIRED
@pytest.mark.parametrize("nhd", [False, True])
@pytest.mark.parametrize("keep", [[0, 2, 5, 8, 9], [0, 1, 4, 5, 19, 20]])
def test_compact_paged_kv_triton_2d_full_oracle(nhd, keep):
    source_length = 21 if len(keep) == 6 else 10
    ids = torch.tensor(
        [7, 2, 11, 1, 9, 5][: (source_length + 3) // 4],
        device="cuda",
        dtype=torch.int32,
    )
    before = _cuda_cache(12, 4, 4, 32, nhd=nhd)
    keep_tensor = torch.tensor(keep, device="cuda", dtype=torch.int64)
    actual = before.clone()
    result = compact_paged_kv_triton_2d(
        actual, ids, source_length, keep_tensor
    )
    expected, expected_len, expected_pages = compact_paged_kv_reference(
        before, ids.long(), source_length, keep_tensor
    )
    assert result == (expected_len, expected_pages)
    assert torch.equal(actual, expected)


@_CUDA_REQUIRED
def test_writeback_tail_zero_and_detached_block_unchanged():
    before = _cuda_cache(12, 4, 4, 32, nhd=True)
    ids = torch.tensor([7, 2, 11], device="cuda", dtype=torch.int32)
    keep = torch.tensor([0, 2, 5, 8, 9], device="cuda", dtype=torch.int64)
    scratch = gather_paged_kv_triton_2d(before, ids, 10, keep)
    actual = before.clone()
    writeback_paged_kv_triton_2d(actual, ids, scratch)
    assert torch.count_nonzero(actual[2, :, 1:, :]) == 0
    assert torch.equal(actual[11], before[11])


@_CUDA_REQUIRED
def test_writeback_exact_boundary_does_not_touch_next_page():
    before = _cuda_cache(12, 4, 4, 32, nhd=True)
    ids = torch.tensor([7, 2, 11], device="cuda", dtype=torch.int32)
    keep = torch.tensor([0, 2, 5, 8], device="cuda", dtype=torch.int64)
    scratch = gather_paged_kv_triton_2d(before, ids, 10, keep)
    actual = before.clone()
    writeback_paged_kv_triton_2d(actual, ids, scratch)
    assert torch.equal(actual[2], before[2])
    expected, _, _ = compact_paged_kv_reference(before, ids.long(), 10, keep)
    assert torch.equal(actual, expected)
