"""Tests for sae_influence_scores (GradSAE). v1.31."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from circuitry.sae.grad import sae_influence_scores


class _LinearSAE:
    """SAE with learnable W_enc / W_dec for gradient tests."""

    def __init__(self, d_model: int, d_sae: int, seed: int = 0) -> None:
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        torch.manual_seed(seed)
        self._W_enc = torch.randn(d_model, d_sae)
        self._W_dec = torch.randn(d_sae, d_model)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x @ self._W_enc)

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        return f @ self._W_dec


class _ZeroSAE:
    """SAE that maps everything to zero while preserving the gradient graph (f * 0)."""

    def __init__(self, d_model: int, d_sae: int) -> None:
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self._d_sae = d_sae
        self._d_model = d_model
        # Fixed projections for graph connectivity; multiplied by zero so output = 0
        torch.manual_seed(99)
        self._W_enc = torch.zeros(d_model, d_sae)
        self._W_dec = torch.zeros(d_sae, d_model)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # graph-connected zero: x @ 0 = 0 but grad flows
        return x.reshape(-1, x.shape[-1]) @ self._W_enc

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        return f @ self._W_dec


def _mse_loss(x_hat: torch.Tensor) -> torch.Tensor:
    target = torch.zeros_like(x_hat)
    return F.mse_loss(x_hat, target)


def test_influence_scores_returns_tensor():
    torch.manual_seed(0)
    sae = _LinearSAE(d_model=8, d_sae=32)
    x = torch.randn(4, 8)
    scores = sae_influence_scores(sae, x, _mse_loss)
    assert isinstance(scores, torch.Tensor)


def test_influence_scores_shape():
    """Output shape must be (d_sae,)."""
    torch.manual_seed(1)
    d_sae = 32
    sae = _LinearSAE(d_model=8, d_sae=d_sae)
    x = torch.randn(4, 8)
    scores = sae_influence_scores(sae, x, _mse_loss)
    assert scores.shape == (d_sae,), f"Expected ({d_sae},), got {scores.shape}"


def test_influence_scores_nonnegative():
    """Scores are |grad| * |feat|, so all values ≥ 0."""
    torch.manual_seed(2)
    sae = _LinearSAE(d_model=8, d_sae=32)
    x = torch.randn(6, 8)
    scores = sae_influence_scores(sae, x, _mse_loss)
    assert (scores >= 0).all(), "Influence scores must be non-negative"


def test_influence_scores_cpu_float32():
    torch.manual_seed(3)
    sae = _LinearSAE(d_model=8, d_sae=32)
    x = torch.randn(3, 8)
    scores = sae_influence_scores(sae, x, _mse_loss)
    assert scores.device.type == "cpu"
    assert scores.dtype == torch.float32


def test_influence_scores_detached():
    """Returned tensor must be detached (no grad_fn)."""
    torch.manual_seed(4)
    sae = _LinearSAE(d_model=8, d_sae=32)
    x = torch.randn(3, 8)
    scores = sae_influence_scores(sae, x, _mse_loss)
    assert scores.grad_fn is None


def test_influence_scores_zero_for_zero_sae():
    """When SAE maps everything to zero, all influence scores are zero."""
    torch.manual_seed(5)
    d_sae = 16
    sae = _ZeroSAE(d_model=8, d_sae=d_sae)
    x = torch.randn(4, 8)
    scores = sae_influence_scores(sae, x, _mse_loss)
    assert torch.allclose(scores, torch.zeros(d_sae), atol=1e-6)


def test_influence_scores_3d_input():
    """3-D input (batch, seq, d_model) should work; scores are (d_sae,)."""
    torch.manual_seed(6)
    d_sae = 32
    sae = _LinearSAE(d_model=8, d_sae=d_sae)
    x = torch.randn(2, 4, 8)  # (batch, seq, d_model)
    scores = sae_influence_scores(sae, x, _mse_loss)
    assert scores.shape == (d_sae,)


def test_influence_scores_larger_loss_gives_larger_scores():
    """Scaling up the MSE target gap should increase total influence."""
    torch.manual_seed(7)
    sae = _LinearSAE(d_model=8, d_sae=32)
    x = torch.randn(4, 8)

    target_small = torch.randn(4, 8) * 0.01
    target_large = torch.randn(4, 8) * 10.0

    scores_small = sae_influence_scores(sae, x, lambda h: F.mse_loss(h, target_small))
    scores_large = sae_influence_scores(sae, x, lambda h: F.mse_loss(h, target_large))

    assert scores_large.sum() > scores_small.sum(), (
        "Larger loss gradient magnitude should produce larger influence scores"
    )
