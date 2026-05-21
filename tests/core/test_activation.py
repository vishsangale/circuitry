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
