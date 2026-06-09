"""Tests for sae_reconstruction_error and superposition_index. Spec §4.3."""
from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest
import torch

from circuitry.sae import sae_reconstruction_error
from circuitry.sae.metrics import superposition_index


class _IdentitySAE:
    """SAE whose encode is identity and decode is identity: recon should be exact."""
    def __init__(self, d_model: int) -> None:
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self._d = d_model

    def encode(self, x):
        return x  # features == input

    def decode(self, feats):
        return feats


class _DeadSAE:
    """SAE whose encode returns all zeros: frac_alive must be 0."""
    def __init__(self, d_model: int, n_features: int = 64) -> None:
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self._d = d_model
        self._k = n_features

    def encode(self, x):
        return torch.zeros(x.shape[0], self._k, dtype=x.dtype, device=x.device)

    def decode(self, feats):
        return torch.zeros(feats.shape[0], self._d, dtype=feats.dtype, device=feats.device)


def test_identity_sae_recon_mse_is_zero():
    torch.manual_seed(0)
    x = torch.randn(3, 5, 8)
    out = sae_reconstruction_error(x, _IdentitySAE(d_model=8))
    assert out["recon_mse"] == pytest.approx(0.0, abs=1e-6)


def test_dead_sae_frac_alive_is_zero():
    torch.manual_seed(1)
    x = torch.randn(4, 6, 8)
    out = sae_reconstruction_error(x, _DeadSAE(d_model=8, n_features=32))
    assert out["frac_alive"] == pytest.approx(0.0, abs=1e-6)
    assert out["l0"] == pytest.approx(0.0, abs=1e-6)


def test_returned_dict_has_all_five_keys():
    torch.manual_seed(2)
    x = torch.randn(2, 3, 4)
    out = sae_reconstruction_error(x, _IdentitySAE(d_model=4))
    assert set(out.keys()) == {"recon_mse", "l0", "l1", "frac_alive", "ce_recovered_proxy"}
    for v in out.values():
        assert isinstance(v, float)


def test_ce_recovered_proxy_one_for_identity():
    """1 - recon_mse / var == 1 when recon is exact (mse == 0)."""
    torch.manual_seed(3)
    x = torch.randn(4, 4, 8) * 2.0  # non-trivial variance
    out = sae_reconstruction_error(x, _IdentitySAE(d_model=8))
    assert out["ce_recovered_proxy"] == pytest.approx(1.0, abs=1e-6)


def test_cost_is_real():
    """SAE encode/decode latency must propagate to the metric — sanity-checks
    we're actually running the SAE, not short-circuiting."""
    slow_sae = MagicMock()
    slow_sae.device = torch.device("cpu")
    slow_sae.dtype = torch.float32

    def slow_encode(x):
        time.sleep(0.05)
        return torch.zeros(x.shape[0], 16)

    def slow_decode(feats):
        time.sleep(0.05)
        return torch.zeros(feats.shape[0], 8)

    slow_sae.encode.side_effect = slow_encode
    slow_sae.decode.side_effect = slow_decode

    x = torch.randn(1, 2, 8)
    t0 = time.perf_counter()
    sae_reconstruction_error(x, slow_sae)
    elapsed = time.perf_counter() - t0
    assert elapsed >= 0.09  # both encode + decode ran


# ---------------------------------------------------------------------------
# superposition_index tests
# ---------------------------------------------------------------------------


def test_superposition_index_returns_float():
    torch.manual_seed(0)
    feat = torch.abs(torch.randn(10, 64))
    si = superposition_index(feat)
    assert isinstance(si, float)
    assert si >= 1.0


def test_superposition_index_uniform_is_n_features():
    """Uniform activation magnitudes → exp(H) = n_features."""
    n_features = 64
    feat = torch.ones(1, n_features)  # all equal magnitude
    si = superposition_index(feat)
    assert si == pytest.approx(float(n_features), rel=1e-4)


def test_superposition_index_single_active_is_one():
    """Only one feature active → exp(H) = 1 (no superposition)."""
    n_features = 64
    feat = torch.zeros(1, n_features)
    feat[0, 0] = 1.0
    si = superposition_index(feat)
    assert si == pytest.approx(1.0, abs=1e-5)


def test_superposition_index_all_zero_is_one():
    """All-zero activations → convention returns 1.0."""
    feat = torch.zeros(5, 32)
    si = superposition_index(feat)
    assert si == pytest.approx(1.0, abs=1e-5)


def test_superposition_index_increases_with_spread():
    """More uniformly spread activations should give higher superposition index."""
    torch.manual_seed(1)
    n_features = 128
    # Sparse: only 4 features active
    feat_sparse = torch.zeros(10, n_features)
    feat_sparse[:, :4] = 1.0
    # Dense: all features active uniformly
    feat_dense = torch.ones(10, n_features)
    si_sparse = superposition_index(feat_sparse)
    si_dense = superposition_index(feat_dense)
    assert si_dense > si_sparse, f"Dense ({si_dense:.1f}) should exceed sparse ({si_sparse:.1f})"


def test_superposition_index_nd_input():
    """3-D feature tensor should flatten correctly."""
    torch.manual_seed(2)
    feat = torch.abs(torch.randn(4, 8, 32))
    si = superposition_index(feat)
    assert isinstance(si, float)
    assert si >= 1.0
