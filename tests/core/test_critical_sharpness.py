"""Tests for critical_sharpness and gradient_subspace_saturation."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from circuitry.core.weight import critical_sharpness, gradient_subspace_saturation


class _Quadratic(nn.Module):
    """y = 0.5 * x^T A x, eigenvalues of A are [2, 0.5]."""

    def __init__(self):
        super().__init__()
        self.x = nn.Parameter(torch.tensor([1.0, 0.5]))

    def loss(self):
        A = torch.tensor([[2.0, 0.0], [0.0, 0.5]])
        return 0.5 * (self.x @ A @ self.x)


# ---------------------------------------------------------------------------
# critical_sharpness tests
# ---------------------------------------------------------------------------


def test_critical_sharpness_quadratic_approx():
    """For a quadratic loss with Hessian diag([2, 0.5]), λ_max should be ≈ 2.0."""
    model = _Quadratic()
    result = critical_sharpness(model, model.loss, n_iters=50, tol=1e-6)
    assert result == pytest.approx(2.0, rel=0.05)


def test_critical_sharpness_returns_float():
    """Return type must be a Python float."""
    model = _Quadratic()
    result = critical_sharpness(model, model.loss)
    assert isinstance(result, float)


def test_critical_sharpness_no_params_returns_zero():
    """Model with no parameters must return 0.0."""

    class _Empty(nn.Module):
        def forward(self):
            return torch.tensor(0.0)

    model = _Empty()
    result = critical_sharpness(model, model.forward)
    assert result == 0.0


def test_critical_sharpness_positive():
    """Largest eigenvalue of a convex (PSD Hessian) loss must be non-negative."""
    model = _Quadratic()
    result = critical_sharpness(model, model.loss)
    assert result >= 0.0


def test_critical_sharpness_respects_n_iters():
    """n_iters=1 should not raise and should still return a float."""
    model = _Quadratic()
    result = critical_sharpness(model, model.loss, n_iters=1)
    assert isinstance(result, float)


# ---------------------------------------------------------------------------
# gradient_subspace_saturation tests
# ---------------------------------------------------------------------------


def test_gss_returns_zero_for_short_history():
    """A single-entry history (< 2 entries) returns 0.0."""
    g = torch.tensor([1.0, 0.0])
    result = gradient_subspace_saturation([g])
    assert result == 0.0


def test_gss_saturated_subspace():
    """Current gradient fully in the span of history → saturation ≈ 1.0."""
    direction = torch.tensor([1.0, 0.0])
    # History of 5 scaled copies of the same direction, current = direction.
    grad_history = [direction * float(i) for i in range(1, 6)] + [direction]
    result = gradient_subspace_saturation(grad_history, k=10)
    assert result == pytest.approx(1.0, abs=1e-5)


def test_gss_orthogonal_is_zero():
    """Current gradient orthogonal to all history vectors → saturation ≈ 0.0.

    Use 3-D vectors so the history (e1) spans only 1 of 3 directions.
    The current gradient e3 is orthogonal to e1, so projection onto the
    1-D history subspace (k=1) gives 0.
    """
    e1 = torch.tensor([1.0, 0.0, 0.0])
    e3 = torch.tensor([0.0, 0.0, 1.0])
    history = [e1] * 4
    result = gradient_subspace_saturation(history + [e3], k=1)
    assert result == pytest.approx(0.0, abs=1e-5)


def test_gss_returns_float_in_unit_interval():
    """Result is a Python float in [0, 1]."""
    g1 = torch.randn(8)
    g2 = torch.randn(8)
    g3 = torch.randn(8)
    result = gradient_subspace_saturation([g1, g2, g3], k=2)
    assert isinstance(result, float)
    assert 0.0 <= result <= 1.0 + 1e-6


def test_gss_zero_current_gradient():
    """Zero current gradient returns 0.0 without raising."""
    history = [torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])]
    current = torch.zeros(2)
    result = gradient_subspace_saturation(history + [current], k=5)
    assert result == 0.0


def test_gss_k_capped_at_history_length():
    """k larger than history length should not crash."""
    g1 = torch.randn(16)
    g2 = torch.randn(16)
    # Only 1 history vector; k=100 should be silently capped to 1.
    result = gradient_subspace_saturation([g1, g2], k=100)
    assert isinstance(result, float)
    assert 0.0 <= result <= 1.0 + 1e-6


def test_gss_partial_saturation():
    """Gradient with a component inside and outside the history subspace is in (0, 1)."""
    e1 = torch.tensor([1.0, 0.0, 0.0])
    e2 = torch.tensor([0.0, 1.0, 0.0])
    # Current gradient has equal components along e1 (in history) and e3 (outside).
    current = torch.tensor([1.0, 0.0, 1.0]) / (2.0 ** 0.5)
    result = gradient_subspace_saturation([e1, e2, current], k=2)
    assert 0.0 < result < 1.0
