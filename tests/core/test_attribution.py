"""Tests for circuitry.core.attribution: gradient_input_attribution and integrated_gradients."""

import pytest
import torch
from torch import Tensor

from circuitry.core.attribution import gradient_input_attribution, integrated_gradients


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_3d(batch=2, seq=5, d=8) -> tuple[Tensor, Tensor]:
    """Return (grads, embeds) with shape (batch, seq, d)."""
    torch.manual_seed(0)
    grads = torch.randn(batch, seq, d)
    embeds = torch.randn(batch, seq, d)
    return grads, embeds


def _make_2d(batch=3, d=8) -> tuple[Tensor, Tensor]:
    """Return (grads, embeds) with shape (batch, d)."""
    torch.manual_seed(1)
    grads = torch.randn(batch, d)
    embeds = torch.randn(batch, d)
    return grads, embeds


# ---------------------------------------------------------------------------
# gradient_input_attribution — shape tests
# ---------------------------------------------------------------------------

def test_gia_shape_3d():
    """Output shape is (batch, seq) for 3D input."""
    grads, embeds = _make_3d(batch=2, seq=5, d=8)
    out = gradient_input_attribution(grads, embeds)
    assert out.shape == (2, 5), f"Expected (2, 5), got {out.shape}"


def test_gia_shape_2d():
    """Output shape is (batch,) for 2D input."""
    grads, embeds = _make_2d(batch=3, d=8)
    out = gradient_input_attribution(grads, embeds)
    assert out.shape == (3,), f"Expected (3,), got {out.shape}"


# ---------------------------------------------------------------------------
# gradient_input_attribution — reduction correctness
# ---------------------------------------------------------------------------

def test_gia_l2_non_negative():
    """reduction='l2' produces non-negative values."""
    grads, embeds = _make_3d()
    out = gradient_input_attribution(grads, embeds, reduction="l2")
    assert (out >= 0).all(), "l2 reduction must be non-negative"


def test_gia_dot_can_be_negative():
    """reduction='dot' can produce negative values (signed)."""
    # Construct deliberately anti-aligned grads/embeds so dot product < 0
    torch.manual_seed(42)
    grads = torch.ones(2, 5, 4)
    embeds = -torch.ones(2, 5, 4)
    out = gradient_input_attribution(grads, embeds, reduction="dot")
    assert (out < 0).any(), "dot reduction should be negative for anti-aligned inputs"


def test_gia_abs_non_negative():
    """reduction='abs' produces non-negative values."""
    grads, embeds = _make_3d()
    out = gradient_input_attribution(grads, embeds, reduction="abs")
    assert (out >= 0).all(), "abs reduction must be non-negative"


def test_gia_l1_non_negative():
    """reduction='l1' produces non-negative values."""
    grads, embeds = _make_3d()
    out = gradient_input_attribution(grads, embeds, reduction="l1")
    assert (out >= 0).all(), "l1 reduction must be non-negative"


def test_gia_dot_unit_vectors_equals_one():
    """When grads and embeds are identical unit vectors per token, dot product = 1.0."""
    # shape (1, 3, 4): each token is the same unit vector
    v = torch.tensor([1.0, 0.0, 0.0, 0.0])
    grads = v.unsqueeze(0).unsqueeze(0).expand(1, 3, 4).clone()
    embeds = grads.clone()
    out = gradient_input_attribution(grads, embeds, reduction="dot")
    assert out.shape == (1, 3)
    assert torch.allclose(out, torch.ones(1, 3), atol=1e-5), (
        f"Expected all-ones, got {out}"
    )


def test_gia_invalid_reduction_raises():
    """Invalid reduction string raises ValueError."""
    grads, embeds = _make_3d()
    with pytest.raises(ValueError, match="reduction"):
        gradient_input_attribution(grads, embeds, reduction="invalid_reduction")


# ---------------------------------------------------------------------------
# integrated_gradients — shape
# ---------------------------------------------------------------------------

def test_ig_output_shape():
    """integrated_gradients returns tensor of shape (batch, seq)."""
    torch.manual_seed(0)
    embeds = torch.randn(2, 5, 8)
    model_fn = lambda e: e.sum(dim=(-2, -1))
    out = integrated_gradients(model_fn, embeds)
    assert out.shape == (2, 5), f"Expected (2, 5), got {out.shape}"


# ---------------------------------------------------------------------------
# integrated_gradients — completeness axiom
# ---------------------------------------------------------------------------

def test_ig_completeness_axiom():
    """Completeness: sum of IG over tokens ≈ f(embeds) - f(baseline)."""
    torch.manual_seed(0)
    batch, seq, d = 2, 5, 4
    embeds = torch.ones(batch, seq, d)
    baseline = torch.zeros(batch, seq, d)
    # Linear model: f(e) = sum of all elements → grad is all-ones everywhere
    model_fn = lambda e: e.sum(dim=(-2, -1))
    ig = integrated_gradients(
        model_fn, embeds, baseline=baseline, n_steps=100, reduction="dot"
    )
    # ig shape: (batch, seq); sum over seq should equal f(embeds) - f(baseline)
    f_embeds = model_fn(embeds)   # (batch,) = seq * d = 20 per example
    f_baseline = model_fn(baseline)  # (batch,) = 0
    expected = f_embeds - f_baseline  # (batch,)
    ig_sum = ig.sum(dim=-1)           # (batch,)
    assert torch.allclose(ig_sum, expected, atol=0.5), (
        f"Completeness failed: ig_sum={ig_sum}, expected={expected}"
    )


# ---------------------------------------------------------------------------
# integrated_gradients — custom baseline
# ---------------------------------------------------------------------------

def test_ig_custom_baseline():
    """integrated_gradients accepts a custom baseline tensor."""
    torch.manual_seed(1)
    embeds = torch.randn(1, 4, 6)
    baseline = torch.full_like(embeds, 0.5)
    model_fn = lambda e: e.sum(dim=(-2, -1))
    out = integrated_gradients(model_fn, embeds, baseline=baseline, n_steps=10)
    assert out.shape == (1, 4)


# ---------------------------------------------------------------------------
# integrated_gradients — l2 reduction non-negative
# ---------------------------------------------------------------------------

def test_ig_l2_non_negative():
    """integrated_gradients with reduction='l2' returns non-negative values."""
    torch.manual_seed(2)
    embeds = torch.randn(2, 4, 6)
    model_fn = lambda e: e.sum(dim=(-2, -1))
    out = integrated_gradients(model_fn, embeds, reduction="l2", n_steps=20)
    assert (out >= 0).all(), "l2 IG values must be non-negative"


# ---------------------------------------------------------------------------
# integrated_gradients — n_steps affects results on nonlinear model
# ---------------------------------------------------------------------------

def test_ig_n_steps_affects_result():
    """n_steps=1 and n_steps=20 give different results for a nonlinear model."""
    torch.manual_seed(3)
    embeds = torch.randn(1, 4, 6)
    # Nonlinear model: square then sum
    model_fn = lambda e: (e ** 2).sum(dim=(-2, -1))
    out1 = integrated_gradients(model_fn, embeds, n_steps=1, reduction="dot")
    out20 = integrated_gradients(model_fn, embeds, n_steps=20, reduction="dot")
    assert not torch.allclose(out1, out20, atol=1e-4), (
        "n_steps=1 and n_steps=20 should differ for a nonlinear model"
    )
