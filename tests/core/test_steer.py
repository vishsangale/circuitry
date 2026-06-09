"""Tests for steer_vector, repe_direction, directional_ablation (core) and apply_steer/apply_ablation (patching)."""
from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from circuitry.core.steer import directional_ablation, repe_direction, steer_vector
from circuitry.patching.steer import apply_ablation, apply_steer
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


# ---------------------------------------------------------------------------
# repe_direction tests
# ---------------------------------------------------------------------------


def test_repe_direction_unit_norm():
    torch.manual_seed(10)
    diffs = torch.randn(8, 16)
    d = repe_direction(diffs)
    assert abs(d.norm().item() - 1.0) < 1e-5


def test_repe_direction_shape():
    torch.manual_seed(11)
    diffs = torch.randn(5, 32)
    d = repe_direction(diffs)
    assert d.shape == (32,)


def test_repe_direction_1d_input():
    """Single difference vector (1-D) should still return a unit vector."""
    torch.manual_seed(12)
    diff = torch.randn(16)
    d = repe_direction(diff)
    assert d.shape == (16,)
    assert abs(d.norm().item() - 1.0) < 1e-5


def test_repe_direction_zero_returns_zero():
    """All-zero diffs should return a zero vector."""
    d = repe_direction(torch.zeros(4, 16))
    assert d.norm().item() < 1e-7


def test_repe_direction_high_signal_first_pc():
    """When one dimension dominates variance, repe_direction should point along it."""
    torch.manual_seed(13)
    n, dim = 20, 16
    diffs = torch.randn(n, dim) * 0.001
    # Add a linearly-growing ramp so dim-0 has high variance even after centering.
    diffs[:, 0] = torch.linspace(-5.0, 5.0, n)
    d = repe_direction(diffs)
    assert abs(d[0].item()) > 0.9, f"Expected direction ≈ e_0, got d[0]={d[0].item():.3f}"


# ---------------------------------------------------------------------------
# directional_ablation tests
# ---------------------------------------------------------------------------


def test_directional_ablation_removes_direction():
    """After ablation, the component along `direction` should be near zero."""
    torch.manual_seed(20)
    d = 16
    acts = torch.randn(10, d)
    direction = torch.randn(d)
    ablated = directional_ablation(acts, direction)
    d_hat = direction / direction.norm()
    residual_proj = (ablated.float() @ d_hat.float()).abs().max().item()
    assert residual_proj < 1e-5, f"Component not removed; max proj = {residual_proj}"


def test_directional_ablation_preserves_orthogonal():
    """Component orthogonal to direction should be unchanged."""
    torch.manual_seed(21)
    d = 8
    direction = torch.zeros(d)
    direction[0] = 1.0  # ablate only dim 0
    acts = torch.randn(5, d)
    ablated = directional_ablation(acts, direction)
    # dims 1..d-1 must be identical
    assert torch.allclose(ablated[:, 1:].float(), acts[:, 1:].float(), atol=1e-6)


def test_directional_ablation_shape_preserved():
    torch.manual_seed(22)
    acts = torch.randn(3, 7, 16)
    direction = torch.randn(16)
    out = directional_ablation(acts, direction)
    assert out.shape == acts.shape


def test_directional_ablation_zero_direction():
    """Near-zero direction should return acts unchanged."""
    torch.manual_seed(23)
    acts = torch.randn(4, 8)
    out = directional_ablation(acts, torch.zeros(8))
    assert torch.allclose(out, acts.float(), atol=1e-6)


# ---------------------------------------------------------------------------
# apply_ablation tests
# ---------------------------------------------------------------------------


def test_apply_ablation_removes_component():
    """Output inside the context should have near-zero projection onto direction."""
    torch.manual_seed(30)
    d = 8
    model = _TinyModel(d)
    resolver = HFSiteResolver(
        n_heads=1, d_model=d, layer_pattern="layers.{L}", mlp_module=""
    )
    site = Site(component="resid_post", layer=0)
    direction = torch.randn(d)

    x = torch.randn(1, 1, d)
    model.eval()
    with apply_ablation(model, site, direction, resolver=resolver):
        with torch.no_grad():
            out = model(x).clone()

    d_hat = direction / direction.norm()
    proj = (out.float().reshape(-1, d) @ d_hat.float()).abs().max().item()
    assert proj < 1e-5, f"Direction not ablated; max proj = {proj}"


def test_apply_ablation_cleanup():
    """No forward hooks remain after the context exits."""
    torch.manual_seed(31)
    d = 8
    model = _TinyModel(d)
    resolver = HFSiteResolver(
        n_heads=1, d_model=d, layer_pattern="layers.{L}", mlp_module=""
    )
    site = Site(component="resid_post", layer=0)
    direction = torch.randn(d)

    hooks_before = len(model.layers[0]._forward_hooks)
    with apply_ablation(model, site, direction, resolver=resolver):
        hooks_during = len(model.layers[0]._forward_hooks)
    hooks_after = len(model.layers[0]._forward_hooks)

    assert hooks_during == hooks_before + 1
    assert hooks_after == hooks_before


def test_apply_ablation_cleanup_on_exception():
    """Hook is removed even if an exception occurs inside the context."""
    torch.manual_seed(32)
    d = 8
    model = _TinyModel(d)
    resolver = HFSiteResolver(
        n_heads=1, d_model=d, layer_pattern="layers.{L}", mlp_module=""
    )
    site = Site(component="resid_post", layer=0)
    direction = torch.randn(d)

    hooks_before = len(model.layers[0]._forward_hooks)
    with pytest.raises(RuntimeError, match="intentional"):
        with apply_ablation(model, site, direction, resolver=resolver):
            raise RuntimeError("intentional test error")
    assert len(model.layers[0]._forward_hooks) == hooks_before
