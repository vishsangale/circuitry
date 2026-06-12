"""Tests for circuitry.core.feature_geometry."""

from __future__ import annotations

import math

import torch

from circuitry.core.feature_geometry import (
    feature_coverage,
    feature_interference,
    feature_spread,
)

# ---------------------------------------------------------------------------
# feature_interference tests
# ---------------------------------------------------------------------------

def test_feature_interference_shape():
    """feature_interference returns (n, n) tensor for (n, d) input."""
    dirs = torch.randn(6, 8)
    result = feature_interference(dirs)
    assert result.shape == (6, 6)


def test_feature_interference_diagonal_is_one():
    """Diagonal of feature_interference matrix is all 1.0."""
    dirs = torch.randn(5, 16)
    result = feature_interference(dirs)
    diag = result.diagonal()
    assert torch.allclose(diag, torch.ones(5), atol=1e-5)


def test_feature_interference_orthogonal_off_diagonal_zero():
    """Two orthogonal unit vectors produce off-diagonal ≈ 0.0."""
    # Standard basis vectors are orthogonal
    dirs = torch.zeros(2, 4)
    dirs[0, 0] = 1.0
    dirs[1, 1] = 1.0
    result = feature_interference(dirs)
    assert abs(result[0, 1].item()) < 1e-6
    assert abs(result[1, 0].item()) < 1e-6


def test_feature_interference_identical_vectors_off_diagonal_one():
    """Two identical unit vectors produce off-diagonal ≈ 1.0."""
    v = torch.randn(1, 8)
    v = v / v.norm()
    dirs = v.expand(2, -1)
    result = feature_interference(dirs)
    assert abs(result[0, 1].item() - 1.0) < 1e-5
    assert abs(result[1, 0].item() - 1.0) < 1e-5


def test_feature_interference_normalize_false_gives_raw_dot_products():
    """normalize=False gives raw dot products rather than cosine similarities."""
    # Un-normalized vectors of different magnitudes — raw dot product != cosine sim
    dirs = torch.zeros(2, 3)
    dirs[0] = torch.tensor([2.0, 0.0, 0.0])
    dirs[1] = torch.tensor([3.0, 0.0, 0.0])

    raw = feature_interference(dirs, normalize=False)
    normalized = feature_interference(dirs, normalize=True)

    # Raw: 2*3 + 0 + 0 = 6; normalized cosine: 1.0
    assert abs(raw[0, 1].item() - 6.0) < 1e-5
    assert abs(normalized[0, 1].item() - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# feature_coverage tests
# ---------------------------------------------------------------------------

def test_feature_coverage_returns_float():
    """feature_coverage returns a Python float."""
    dirs = torch.randn(4, 8)
    acts = torch.randn(16, 8)
    result = feature_coverage(dirs, acts)
    assert isinstance(result, float)


def test_feature_coverage_in_range():
    """feature_coverage value lies in [0, 1]."""
    dirs = torch.randn(4, 8)
    acts = torch.randn(16, 8)
    result = feature_coverage(dirs, acts)
    assert 0.0 <= result <= 1.0


def test_feature_coverage_full_span_near_one():
    """Identity-like directions spanning the full space → coverage near 1.0."""
    d = 8
    # Use identity matrix rows as feature directions — they form a complete basis
    dirs = torch.eye(d)
    acts = torch.randn(32, d)
    result = feature_coverage(dirs, acts)
    assert result > 0.95


def test_feature_coverage_orthogonal_direction_near_zero():
    """A direction orthogonal to all activations → coverage ≈ 0.0."""
    # Activations live in the first 3 dims; direction is in dim 4
    acts = torch.zeros(20, 5)
    acts[:, :3] = torch.randn(20, 3)
    # Add tiny noise so variance is non-zero in dims 0-2
    acts[:, :3] += 0.01 * torch.randn(20, 3)

    # Feature direction only in dim 4 (orthogonal to all variance)
    dirs = torch.zeros(1, 5)
    dirs[0, 4] = 1.0

    result = feature_coverage(dirs, acts)
    assert result < 0.05


def test_feature_coverage_k_equals_one():
    """With k=1, only the single best direction is used."""
    dirs = torch.randn(5, 8)
    acts = torch.randn(16, 8)
    result_k1 = feature_coverage(dirs, acts, k=1)
    result_all = feature_coverage(dirs, acts)
    # k=1 coverage <= full coverage
    assert 0.0 <= result_k1 <= 1.0
    assert result_k1 <= result_all + 1e-6


# ---------------------------------------------------------------------------
# feature_spread tests
# ---------------------------------------------------------------------------

def test_feature_spread_returns_float():
    """feature_spread returns a Python float."""
    dirs = torch.randn(4, 8)
    result = feature_spread(dirs)
    assert isinstance(result, float)


def test_feature_spread_orthogonal_directions_near_pi_over_2():
    """Orthogonal unit vectors → spread ≈ π/2."""
    # Standard basis vectors are mutually orthogonal
    d = 4
    dirs = torch.eye(d)  # 4 directions in 4-d space, all orthogonal
    result = feature_spread(dirs)
    assert abs(result - math.pi / 2) < 0.01


def test_feature_spread_identical_directions_near_zero():
    """All identical directions → spread ≈ 0.0."""
    v = torch.randn(1, 8)
    v = v / v.norm()
    # Stack the same direction multiple times
    dirs = v.expand(5, -1)
    result = feature_spread(dirs)
    # The implementation clamps cosine sim to [-1+1e-6, 1-1e-6], so acos(1-1e-6) ≈ 0.00142
    # rather than exactly 0.  A threshold of 0.005 rad covers this numerical floor.
    assert result < 0.005
