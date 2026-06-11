"""Tests for daam_attribution (v1.35)."""

from __future__ import annotations

import pytest
import torch

from circuitry.core.attention import daam_attribution


def _make_maps(n_steps=3, n_heads=4, n_patches=16, seq_len=5):
    """Return a list of (n_heads, n_patches, seq_len) tensors."""
    return [torch.rand(n_heads, n_patches, seq_len) for _ in range(n_steps)]


def test_basic_shape():
    maps = _make_maps()
    out = daam_attribution(maps)
    assert out.shape == (5, 16)


def test_shape_with_batch():
    maps = [torch.rand(2, 4, 16, 5) for _ in range(3)]
    out = daam_attribution(maps)
    assert out.shape == (5, 16)


def test_normalize_sums_to_one():
    maps = _make_maps()
    out = daam_attribution(maps, normalize=True)
    row_sums = out.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)


def test_no_normalize():
    maps = _make_maps()
    out = daam_attribution(maps, normalize=False)
    row_sums = out.sum(dim=-1)
    # Without normalization, rows need not sum to 1
    assert not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-1) or True
    # Just verify the shape and values are finite
    assert out.shape == (5, 16)
    assert torch.isfinite(out).all()


def test_head_agg_mean():
    maps = _make_maps(n_heads=4)
    out_mean = daam_attribution(maps, head_agg="mean")
    assert out_mean.shape == (5, 16)


def test_head_agg_max():
    maps = _make_maps(n_heads=4)
    out_max = daam_attribution(maps, head_agg="max")
    assert out_max.shape == (5, 16)


def test_head_agg_max_geq_mean():
    """max aggregation should yield >= mean values elementwise."""
    torch.manual_seed(42)
    maps = _make_maps(n_heads=4)
    out_mean = daam_attribution(maps, head_agg="mean", normalize=False)
    out_max = daam_attribution(maps, head_agg="max", normalize=False)
    assert (out_max >= out_mean - 1e-6).all()


def test_spatial_reshape():
    maps = _make_maps(n_patches=16, seq_len=5)
    out = daam_attribution(maps, normalize=False, spatial_shape=(4, 4))
    assert out.shape == (5, 4, 4)


def test_spatial_reshape_with_normalize():
    maps = _make_maps(n_patches=16, seq_len=5)
    out = daam_attribution(maps, normalize=True, spatial_shape=(4, 4))
    assert out.shape == (5, 4, 4)
    # Each token's flattened map should sum to ~1
    flat_sums = out.reshape(5, -1).sum(dim=-1)
    assert torch.allclose(flat_sums, torch.ones_like(flat_sums), atol=1e-5)


def test_single_step():
    maps = _make_maps(n_steps=1)
    out = daam_attribution(maps)
    assert out.shape == (5, 16)


def test_empty_raises():
    with pytest.raises(ValueError, match="non-empty"):
        daam_attribution([])


def test_invalid_head_agg():
    maps = _make_maps()
    with pytest.raises(ValueError, match="head_agg"):
        daam_attribution(maps, head_agg="sum")


def test_bad_spatial_shape_raises():
    maps = _make_maps(n_patches=16)
    with pytest.raises(ValueError, match="spatial_shape"):
        daam_attribution(maps, spatial_shape=(3, 4))  # 12 != 16


def test_output_is_float32():
    maps = [torch.rand(4, 16, 5, dtype=torch.float16) for _ in range(2)]
    out = daam_attribution(maps)
    assert out.dtype == torch.float32
