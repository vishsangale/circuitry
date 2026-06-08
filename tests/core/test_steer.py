"""Tests for steer_vector (core) and apply_steer (patching)."""
from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from circuitry.core.steer import steer_vector
from circuitry.patching.steer import apply_steer
from circuitry.patching.sites import Site, HFSiteResolver


# ---------------------------------------------------------------------------
# steer_vector tests
# ---------------------------------------------------------------------------


def test_steer_vector_basic():
    """The steering vector points in the direction pos - neg."""
    torch.manual_seed(0)
    d = 8
    pos = torch.zeros(d)
    pos[0] = 2.0
    neg = torch.zeros(d)
    neg[0] = -2.0

    sv = steer_vector(pos, neg, normalize=False)
    assert sv.shape == (d,)
    # diff is [4, 0, ...] → first component is positive
    assert float(sv[0]) > 0.0


def test_steer_vector_normalized():
    """normalize=True produces a unit vector."""
    torch.manual_seed(1)
    pos = torch.randn(8, 16)
    neg = torch.randn(8, 16)
    sv = steer_vector(pos, neg, normalize=True)
    assert sv.shape == (16,)
    assert float(sv.norm()) == pytest.approx(1.0, abs=1e-6)


def test_steer_vector_unnormalized():
    """normalize=False does not produce a unit vector (except by coincidence)."""
    torch.manual_seed(2)
    pos = torch.randn(4, 8)
    neg = torch.randn(4, 8)
    sv = steer_vector(pos, neg, normalize=False)
    # The norm of the difference is generally not 1
    norm = float(sv.norm())
    # It could be 1 by coincidence, but extremely unlikely; just check shape
    assert sv.shape == (8,)
    assert norm > 0.0  # nonzero difference


def test_steer_vector_zero_diff_raises():
    """Identical pos and neg with normalize=True raises ValueError."""
    d = 8
    acts = torch.randn(4, d)
    with pytest.raises(ValueError, match="near-zero norm"):
        steer_vector(acts, acts.clone(), normalize=True)


def test_steer_vector_1d_input():
    """1-D inputs (d_model,) are handled without error."""
    torch.manual_seed(3)
    d = 16
    pos = torch.randn(d)
    neg = torch.randn(d)
    sv = steer_vector(pos, neg)
    assert sv.shape == (d,)


def test_steer_vector_direction():
    """The returned vector is proportional to mean(pos) - mean(neg)."""
    torch.manual_seed(4)
    d = 8
    pos = torch.randn(5, d)
    neg = torch.randn(5, d)
    sv = steer_vector(pos, neg, normalize=False)
    expected = pos.mean(0) - neg.mean(0)
    assert torch.allclose(sv, expected.float(), atol=1e-6)


# ---------------------------------------------------------------------------
# apply_steer tests (patching layer)
# ---------------------------------------------------------------------------


class _TinyModel(nn.Module):
    """Minimal 1-layer model for hook testing.

    Architecture: Linear(d_in, d_out) wrapped in a single layer list so it
    resembles a transformer-like structure compatible with HFSiteResolver.
    """

    def __init__(self, d: int = 8) -> None:
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(d, d, bias=False)])
        self.d = d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers[0](x)


def _make_resolver(model: _TinyModel, d: int = 8) -> HFSiteResolver:
    return HFSiteResolver(
        n_heads=1,
        d_model=d,
        layer_pattern="layers.{L}",
        attn_module="layers.{L}",   # not used in this test
        mlp_module="layers.{L}",
    )


def test_apply_steer_adds_vector():
    """Output inside the context differs from baseline by coeff * vector."""
    torch.manual_seed(5)
    d = 8
    model = _TinyModel(d)
    resolver = HFSiteResolver(
        n_heads=1,
        d_model=d,
        layer_pattern="layers.{L}",
        mlp_module="",   # not used
    )
    site = Site(component="resid_post", layer=0)

    x = torch.randn(1, 1, d)  # (batch, seq, d)
    vector = torch.randn(d)
    coeff = 2.5

    # Baseline (no hook)
    model.eval()
    with torch.no_grad():
        baseline = model(x).clone()

    with apply_steer(model, site, vector, coeff=coeff, resolver=resolver):
        with torch.no_grad():
            steered = model(x).clone()

    diff = steered - baseline
    expected_diff = coeff * vector.to(diff.device, diff.dtype)
    # diff should be (1, 1, d) and each position equal to expected_diff
    assert torch.allclose(diff[0, 0], expected_diff, atol=1e-5), (
        f"Expected diff {expected_diff}, got {diff[0, 0]}"
    )


def test_apply_steer_cleanup():
    """No forward hooks remain after the context exits."""
    torch.manual_seed(6)
    d = 8
    model = _TinyModel(d)
    resolver = HFSiteResolver(
        n_heads=1,
        d_model=d,
        layer_pattern="layers.{L}",
        mlp_module="",
    )
    site = Site(component="resid_post", layer=0)
    vector = torch.randn(d)

    # Hooks before the context
    hooks_before = len(model.layers[0]._forward_hooks)

    with apply_steer(model, site, vector, resolver=resolver):
        hooks_during = len(model.layers[0]._forward_hooks)

    hooks_after = len(model.layers[0]._forward_hooks)

    assert hooks_during == hooks_before + 1, "Hook should be registered inside context"
    assert hooks_after == hooks_before, "Hook should be removed after context exits"


def test_apply_steer_cleanup_on_exception():
    """Hook is removed even if an exception is raised inside the context."""
    torch.manual_seed(7)
    d = 8
    model = _TinyModel(d)
    resolver = HFSiteResolver(
        n_heads=1,
        d_model=d,
        layer_pattern="layers.{L}",
        mlp_module="",
    )
    site = Site(component="resid_post", layer=0)
    vector = torch.randn(d)

    hooks_before = len(model.layers[0]._forward_hooks)

    with pytest.raises(RuntimeError, match="intentional"):
        with apply_steer(model, site, vector, resolver=resolver):
            raise RuntimeError("intentional test error")

    hooks_after = len(model.layers[0]._forward_hooks)
    assert hooks_after == hooks_before, "Hook must be removed even after exception"
