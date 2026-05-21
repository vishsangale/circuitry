from __future__ import annotations

import pytest
import torch

from circuitry.core import activation
from circuitry.core.activation import NormStats


def test_dead_fraction_all_zeros_is_one():
    x = torch.zeros(4, 8)
    assert activation.dead_fraction(x) == pytest.approx(1.0)


def test_dead_fraction_none_dead():
    x = torch.ones(4, 8)
    assert activation.dead_fraction(x) == pytest.approx(0.0)


def test_dead_fraction_threshold():
    x = torch.tensor([[0.0, 0.1, 1.0, -0.5]])
    # default threshold=0.0 → "dead" means <= 0
    assert activation.dead_fraction(x) == pytest.approx(0.5)


def test_dead_fraction_returns_float():
    assert isinstance(activation.dead_fraction(torch.zeros(2, 2)), float)


def test_norm_stats_shape_and_fields():
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    s = activation.norm_stats(x)
    assert isinstance(s, NormStats)
    assert s.mean == pytest.approx(2.5)
    assert s.max == pytest.approx(4.0)
    assert s.std > 0
    # frac > k*median: median is 2.5; 4>2.5 → 1/4 = 0.25 with k=1
    assert 0.0 <= s.frac_above_k_median <= 1.0


def test_kurtosis_normal_is_near_zero():
    torch.manual_seed(0)
    x = torch.randn(10_000)
    # Excess kurtosis of N(0,1) ≈ 0 (within sampling noise).
    assert abs(float(activation.kurtosis(x).item())) < 0.3


def test_kurtosis_heavy_tail_is_positive():
    torch.manual_seed(0)
    base = torch.randn(10_000)
    base[:50] *= 20.0  # inject heavy-tail outliers
    assert float(activation.kurtosis(base).item()) > 1.0


def test_kurtosis_along_dim():
    x = torch.randn(8, 100)
    k = activation.kurtosis(x, dim=-1)
    assert k.shape == (8,)


def test_participation_ratio_uniform_is_n():
    # Uniform |x| → PR ≈ n.
    x = torch.ones(16)
    assert activation.participation_ratio(x) == pytest.approx(16.0, rel=1e-5)


def test_participation_ratio_spike_is_one():
    x = torch.zeros(16)
    x[0] = 1.0
    assert activation.participation_ratio(x) == pytest.approx(1.0, rel=1e-5)


def test_participation_ratio_returns_float():
    assert isinstance(activation.participation_ratio(torch.ones(4)), float)


def test_token_similarity_identical_tokens():
    # All tokens identical → cosine similarity = 1.0
    h = torch.ones(1, 5, 8)
    sim = activation.token_similarity(h)
    assert torch.allclose(sim, torch.tensor(1.0), atol=1e-6)


def test_token_similarity_orthogonal_tokens():
    # Standard basis tokens → off-diagonal cosine = 0
    h = torch.eye(4).unsqueeze(0)  # (1, 4, 4)
    sim = activation.token_similarity(h)
    assert torch.allclose(sim, torch.tensor(0.0), atol=1e-6)


def test_token_similarity_handles_batch():
    h = torch.randn(3, 5, 8)
    sim = activation.token_similarity(h)
    assert sim.shape == ()  # scalar, mean across batch
