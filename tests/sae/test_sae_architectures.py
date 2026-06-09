"""Tests for assert_supported_sae and sae_decompose with new architectures.

Covers: matryoshka, batch_topk (newly added); also verifies that raw
crosscoder objects remain blocked.
"""
from __future__ import annotations

import pytest
import torch

from circuitry.sae.grad import assert_supported_sae, sae_decompose


# ---------------------------------------------------------------------------
# Helpers — minimal synthetic SAE objects
# ---------------------------------------------------------------------------

class _FakeCfg:
    def __init__(self, architecture: str, normalize_activations: str = "none") -> None:
        self.architecture = architecture
        self.normalize_activations = normalize_activations


class _FakeSAE:
    """Minimal SAE: encode is identity projection to d_sae, decode is transpose."""

    def __init__(self, d_model: int, d_sae: int, architecture: str) -> None:
        self.cfg = _FakeCfg(architecture)
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self._W_enc = torch.randn(d_model, d_sae)
        self._W_dec = self._W_enc.T.clone()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # flatten to 2-D, project, relu
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        return torch.relu(flat @ self._W_enc)

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        out = f @ self._W_dec
        # reshape back if needed (encode returns 2-D)
        return out


# ---------------------------------------------------------------------------
# assert_supported_sae — positive cases for new architectures
# ---------------------------------------------------------------------------

def test_matryoshka_sae_assert_supported():
    """assert_supported_sae must not raise for architecture='matryoshka'."""
    sae = _FakeSAE(d_model=8, d_sae=16, architecture="matryoshka")
    # Should complete without any exception
    assert_supported_sae(sae)


def test_batch_topk_sae_assert_supported():
    """assert_supported_sae must not raise for architecture='batch_topk'."""
    sae = _FakeSAE(d_model=8, d_sae=16, architecture="batch_topk")
    assert_supported_sae(sae)


# ---------------------------------------------------------------------------
# sae_decompose — shape + detach correctness for new architectures
# ---------------------------------------------------------------------------

def test_matryoshka_sae_decompose():
    """sae_decompose on a matryoshka SAE returns (f, x_hat, eps) with correct shapes;
    eps is detached (does not require grad) while f and x_hat remain in-graph.
    """
    torch.manual_seed(0)
    d_model, d_sae, batch = 8, 16, 3
    sae = _FakeSAE(d_model=d_model, d_sae=d_sae, architecture="matryoshka")
    x = torch.randn(batch, d_model, requires_grad=True)

    f, x_hat, eps = sae_decompose(sae, x)

    assert f.shape == (batch, d_sae), f"Expected f shape ({batch}, {d_sae}), got {f.shape}"
    assert x_hat.shape == (batch, d_model), f"Expected x_hat shape ({batch}, {d_model}), got {x_hat.shape}"
    assert eps.shape == (batch, d_model), f"Expected eps shape ({batch}, {d_model}), got {eps.shape}"
    assert not eps.requires_grad, "eps must be detached"


def test_batch_topk_sae_decompose():
    """sae_decompose on a batch_topk SAE returns (f, x_hat, eps) with correct shapes."""
    torch.manual_seed(1)
    d_model, d_sae, batch = 8, 16, 4
    sae = _FakeSAE(d_model=d_model, d_sae=d_sae, architecture="batch_topk")
    x = torch.randn(batch, d_model, requires_grad=True)

    f, x_hat, eps = sae_decompose(sae, x)

    assert f.shape == (batch, d_sae)
    assert x_hat.shape == (batch, d_model)
    assert eps.shape == (batch, d_model)
    assert not eps.requires_grad, "eps must be detached"


# ---------------------------------------------------------------------------
# assert_supported_sae — blocked architecture still raises
# ---------------------------------------------------------------------------

def test_crosscoder_raw_still_blocked():
    """A raw SAE with cfg.architecture='crosscoder' must raise NotImplementedError.

    crosscoder requires the CrosscoderWrapper shim; direct use is blocked.
    """
    sae = _FakeSAE(d_model=8, d_sae=16, architecture="crosscoder")
    with pytest.raises(NotImplementedError):
        assert_supported_sae(sae)


# ---------------------------------------------------------------------------
# v1.31 — p_anneal and hierarchical_topk architecture support
# ---------------------------------------------------------------------------

def test_p_anneal_sae_assert_supported():
    """assert_supported_sae must not raise for architecture='p_anneal'."""
    sae = _FakeSAE(d_model=8, d_sae=16, architecture="p_anneal")
    assert_supported_sae(sae)


def test_hierarchical_topk_sae_assert_supported():
    """assert_supported_sae must not raise for architecture='hierarchical_topk'."""
    sae = _FakeSAE(d_model=8, d_sae=16, architecture="hierarchical_topk")
    assert_supported_sae(sae)


def test_p_anneal_in_supported_architectures():
    from circuitry.sae.grad import SUPPORTED_SAE_ARCHITECTURES
    assert "p_anneal" in SUPPORTED_SAE_ARCHITECTURES


def test_hierarchical_topk_in_supported_architectures():
    from circuitry.sae.grad import SUPPORTED_SAE_ARCHITECTURES
    assert "hierarchical_topk" in SUPPORTED_SAE_ARCHITECTURES


def test_p_anneal_sae_decompose():
    """sae_decompose on a p_anneal SAE returns (f, x_hat, eps) correctly."""
    torch.manual_seed(2)
    d_model, d_sae, batch = 8, 16, 3
    sae = _FakeSAE(d_model=d_model, d_sae=d_sae, architecture="p_anneal")
    x = torch.randn(batch, d_model, requires_grad=True)

    f, x_hat, eps = sae_decompose(sae, x)

    assert f.shape == (batch, d_sae)
    assert x_hat.shape == (batch, d_model)
    assert not eps.requires_grad


def test_hierarchical_topk_sae_decompose():
    """sae_decompose on a hierarchical_topk SAE returns correct shapes."""
    torch.manual_seed(3)
    d_model, d_sae, batch = 8, 16, 4
    sae = _FakeSAE(d_model=d_model, d_sae=d_sae, architecture="hierarchical_topk")
    x = torch.randn(batch, d_model, requires_grad=True)

    f, x_hat, eps = sae_decompose(sae, x)

    assert f.shape == (batch, d_sae)
    assert x_hat.shape == (batch, d_model)
    assert not eps.requires_grad
