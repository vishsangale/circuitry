"""Tests for circuitry.core.erase (LEACE concept erasure)."""
from __future__ import annotations

import torch
import pytest

from circuitry.core.erase import EraseProjection, leace_erase


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _linear_accuracy(acts: torch.Tensor, labels: torch.Tensor) -> float:
    """Fit a simple linear classifier (mean-difference + threshold) and return accuracy."""
    classes = torch.unique(labels)
    if classes.shape[0] == 2:
        c0, c1 = int(classes[0].item()), int(classes[1].item())
        mu0 = acts[labels == c0].mean(0)
        mu1 = acts[labels == c1].mean(0)
        direction = mu1 - mu0
        scores = acts @ direction
        threshold = (scores[labels == c0].mean() + scores[labels == c1].mean()) / 2
        preds = (scores > threshold).long()
        return float((preds == (labels == c1).long()).float().mean().item())
    else:
        # Multi-class: nearest-centroid
        centroids = torch.stack([acts[labels == int(c.item())].mean(0) for c in classes])
        dists = torch.cdist(acts.float(), centroids.float())
        pred_idx = dists.argmin(dim=1)
        preds = classes[pred_idx]
        return float((preds == labels).float().mean().item())


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_leace_erase_binary_reduces_accuracy():
    """Probe accuracy before erasure ≈1.0, after ≈0.5 (chance)."""
    torch.manual_seed(0)
    n, d = 200, 16
    # Class 0: centred at +e0, class 1: centred at -e0 — clearly linearly separable
    acts = torch.randn(n, d)
    labels = torch.zeros(n, dtype=torch.long)
    labels[n // 2:] = 1
    acts[:n // 2, 0] += 3.0
    acts[n // 2:, 0] -= 3.0

    acc_before = _linear_accuracy(acts, labels)
    assert acc_before > 0.9, f"pre-erasure accuracy should be high, got {acc_before}"

    proj = leace_erase(acts, labels)
    acts_erased = proj.apply(acts)

    acc_after = _linear_accuracy(acts_erased, labels)
    assert acc_after < 0.6, f"post-erasure accuracy should drop near chance, got {acc_after}"


def test_leace_erase_multiclass_reduces_accuracy():
    """3-class data separated along a SINGLE shared dimension: accuracy drops after erasure.

    All classes are offset along dimension 0 only (means at -4, 0, +4) so the
    first principal direction of the between-class mean matrix captures all the
    discriminative signal.  Erasing it should collapse near-chance accuracy.
    """
    torch.manual_seed(1)
    n_per_class, d = 100, 16
    n = n_per_class * 3

    # All class separation along a single dimension
    acts = torch.randn(n, d) * 0.3
    labels = torch.repeat_interleave(torch.arange(3), n_per_class)
    offsets = [-4.0, 0.0, 4.0]
    for c, offset in enumerate(offsets):
        acts[labels == c, 0] += offset

    acc_before = _linear_accuracy(acts, labels)
    assert acc_before > 0.8, f"pre-erasure accuracy should be high, got {acc_before}"

    proj = leace_erase(acts, labels)
    acts_erased = proj.apply(acts)

    acc_after = _linear_accuracy(acts_erased, labels)
    assert acc_after < acc_before - 0.1, (
        f"post-erasure accuracy {acc_after} should be notably less than {acc_before}"
    )


def test_leace_erase_orthogonal_directions_preserved():
    """Vectors orthogonal to the concept direction are unchanged by the projection."""
    torch.manual_seed(2)
    d = 8
    n = 50
    acts = torch.randn(n, d)
    labels = torch.zeros(n, dtype=torch.long)
    labels[n // 2:] = 1
    acts[:n // 2, 0] += 2.0
    acts[n // 2:, 0] -= 2.0

    proj = leace_erase(acts, labels)
    d_hat = proj.direction  # (d,) unit vector

    # Build a vector orthogonal to d_hat using Gram-Schmidt
    v = torch.randn(d)
    v = v - (v @ d_hat) * d_hat
    v = v / v.norm()

    v_erased = proj.apply(v.unsqueeze(0)).squeeze(0)
    assert torch.allclose(v, v_erased, atol=1e-5), (
        f"orthogonal vector should be preserved; max diff = {(v - v_erased).abs().max()}"
    )


def test_leace_erase_apply_shape():
    """apply() preserves the leading dimensions: (..., d) → (..., d)."""
    torch.manual_seed(3)
    d = 12
    n = 40
    acts = torch.randn(n, d)
    labels = torch.randint(0, 2, (n,))

    proj = leace_erase(acts, labels)

    # 2D
    x2 = torch.randn(10, d)
    assert proj.apply(x2).shape == (10, d)

    # 3D
    x3 = torch.randn(4, 7, d)
    assert proj.apply(x3).shape == (4, 7, d)

    # 1D
    x1 = torch.randn(d)
    assert proj.apply(x1).shape == (d,)


def test_leace_erase_returns_cpu_tensors():
    """P and direction are always on CPU, regardless of input device."""
    torch.manual_seed(4)
    d = 8
    n = 30
    acts = torch.randn(n, d)
    labels = torch.randint(0, 2, (n,))

    proj = leace_erase(acts, labels)

    assert proj.P.device.type == "cpu", f"P should be on CPU, got {proj.P.device}"
    assert proj.direction.device.type == "cpu", (
        f"direction should be on CPU, got {proj.direction.device}"
    )
    assert proj.P.dtype == torch.float32
    assert proj.direction.dtype == torch.float32
