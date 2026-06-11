"""Tests for fgaa_steering_vector. Spec §4.3 / v1.31."""
from __future__ import annotations

import pytest
import torch

from circuitry.sae.steer import fgaa_steering_vector


class _LinearSAE:
    """SAE with fixed encoder/decoder weight matrices for deterministic tests."""

    def __init__(self, d_model: int, d_sae: int) -> None:
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        torch.manual_seed(42)
        self._W_enc = torch.randn(d_model, d_sae)
        self._W_dec = torch.randn(d_sae, d_model)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x @ self._W_enc)

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        return f @ self._W_dec


class _IdentitySAE:
    """SAE whose encode and decode are identity: steering vector = mean diff."""

    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.dtype = torch.float32

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        return f


def test_fgaa_returns_tensor():
    torch.manual_seed(0)
    sae = _LinearSAE(d_model=8, d_sae=32)
    pos = torch.randn(5, 8)
    neg = torch.randn(5, 8)
    vec = fgaa_steering_vector(sae, pos, neg, n_features=4)
    assert isinstance(vec, torch.Tensor)


def test_fgaa_output_shape():
    """Output shape must be (d_model,)."""
    torch.manual_seed(1)
    d_model = 16
    sae = _LinearSAE(d_model=d_model, d_sae=64)
    pos = torch.randn(4, d_model)
    neg = torch.randn(4, d_model)
    vec = fgaa_steering_vector(sae, pos, neg, n_features=5)
    assert vec.shape == (d_model,), f"Expected ({d_model},), got {vec.shape}"


def test_fgaa_output_is_cpu_float32():
    torch.manual_seed(2)
    sae = _LinearSAE(d_model=8, d_sae=32)
    pos = torch.randn(3, 8)
    neg = torch.randn(3, 8)
    vec = fgaa_steering_vector(sae, pos, neg)
    assert vec.device.type == "cpu"
    assert vec.dtype == torch.float32


def test_fgaa_opposite_inputs_negate():
    """Swapping pos/neg should negate the steering vector."""
    torch.manual_seed(3)
    sae = _LinearSAE(d_model=8, d_sae=32)
    pos = torch.randn(4, 8)
    neg = torch.randn(4, 8)
    vec_pn = fgaa_steering_vector(sae, pos, neg, n_features=8)
    vec_np = fgaa_steering_vector(sae, neg, pos, n_features=8)
    assert torch.allclose(vec_pn, -vec_np, atol=1e-5), (
        "Swapping pos/neg should negate the vector"
    )


def test_fgaa_n_features_clip_to_d_sae():
    """n_features > d_sae should not error — clips to d_sae."""
    torch.manual_seed(4)
    d_sae = 16
    sae = _LinearSAE(d_model=8, d_sae=d_sae)
    pos = torch.randn(2, 8)
    neg = torch.randn(2, 8)
    vec = fgaa_steering_vector(sae, pos, neg, n_features=d_sae * 10)
    assert vec.shape == (8,)


def test_fgaa_1d_input_accepted():
    """Single (d_model,) activations should work without error."""
    torch.manual_seed(5)
    sae = _LinearSAE(d_model=8, d_sae=32)
    pos = torch.randn(8)   # 1-D
    neg = torch.randn(8)
    vec = fgaa_steering_vector(sae, pos, neg, n_features=4)
    assert vec.shape == (8,)


def test_fgaa_zero_diff_gives_zero_vector():
    """If pos == neg, mean diff is zero → all weights zero → zero vector."""
    torch.manual_seed(6)
    sae = _LinearSAE(d_model=8, d_sae=32)
    acts = torch.randn(4, 8)
    vec = fgaa_steering_vector(sae, acts, acts, n_features=8)
    assert torch.allclose(vec, torch.zeros(8), atol=1e-6)


def test_fgaa_n_features_default():
    """Default n_features=10 should work without explicit argument."""
    torch.manual_seed(7)
    sae = _LinearSAE(d_model=8, d_sae=64)
    pos = torch.randn(6, 8)
    neg = torch.randn(6, 8)
    vec = fgaa_steering_vector(sae, pos, neg)
    assert vec.shape == (8,)


def test_fgaa_detached_from_graph():
    """Returned vector must be detached (no grad_fn)."""
    torch.manual_seed(8)
    sae = _LinearSAE(d_model=8, d_sae=32)
    pos = torch.randn(3, 8)
    neg = torch.randn(3, 8)
    vec = fgaa_steering_vector(sae, pos, neg, n_features=4)
    assert vec.grad_fn is None, "Steering vector must be detached"
