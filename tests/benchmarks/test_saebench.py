"""Tests for circuitry.benchmarks.saebench — SAEBench metric runner."""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from circuitry.benchmarks.saebench import (
    SAEBenchResult,
    explained_variance,
    feature_density,
    l0_sparsity,
    reconstruction_mse,
    run_saebench,
    sparse_probing_r2,
)

# ---------------------------------------------------------------------------
# Synthetic SAE fixtures
# ---------------------------------------------------------------------------

D_MODEL  = 8
D_HIDDEN = 16
N_TOKENS = 32


class _LinearSAE(nn.Module):
    """Minimal SAE: ReLU encoder, linear decoder.  No bias."""

    def __init__(self, d_model: int, d_hidden: int, *, seed: int = 0) -> None:
        super().__init__()
        rng = torch.Generator().manual_seed(seed)
        W_enc = torch.randn(d_model, d_hidden, generator=rng)
        W_dec = torch.randn(d_hidden, d_model, generator=rng)
        self.W_enc = nn.Parameter(W_enc)
        self.W_dec = nn.Parameter(W_dec)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x @ self.W_enc)

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        return f @ self.W_dec


class _IdentitySAE(nn.Module):
    """SAE where decode(encode(x)) == x exactly (identity reconstruction).

    encode: ReLU(x @ W)  but W is chosen so that W @ W^T ≈ I for the test
    data.  For an exact identity we use encode(x)=relu(x) with W_enc=W_dec=I
    (d_model == d_hidden, all non-negative inputs).
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        # Identity encoder/decoder via left-inverse trick:
        # Use d_hidden = d_model; W_enc = W_dec = I; but ReLU clips negatives.
        # To guarantee decode(encode(x)) == x we need x >= 0.
        self.d_model = d_model

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x)   # identity when x >= 0

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        return f


def _make_sae(seed: int = 0) -> _LinearSAE:
    return _LinearSAE(D_MODEL, D_HIDDEN, seed=seed)


def _make_acts(seed: int = 1) -> torch.Tensor:
    """Return (N_TOKENS, D_MODEL) float32 activations."""
    rng = torch.Generator().manual_seed(seed)
    return torch.randn(N_TOKENS, D_MODEL, generator=rng)


def _make_nonneg_acts(seed: int = 2) -> torch.Tensor:
    """Return (N_TOKENS, D_MODEL) non-negative activations (for identity SAE)."""
    rng = torch.Generator().manual_seed(seed)
    return torch.abs(torch.randn(N_TOKENS, D_MODEL, generator=rng))


# ---------------------------------------------------------------------------
# l0_sparsity
# ---------------------------------------------------------------------------

def test_l0_sparsity_range():
    sae  = _make_sae()
    acts = _make_acts()
    val  = l0_sparsity(sae, acts)
    assert 0.0 <= val <= D_HIDDEN, f"L0 {val} outside [0, {D_HIDDEN}]"


def test_l0_sparsity_all_zeros():
    """If all features are zero (dead SAE) l0 == 0."""

    class _ZeroSAE:
        def encode(self, x):
            return torch.zeros(x.shape[0], D_HIDDEN)

    val = l0_sparsity(_ZeroSAE(), _make_acts())
    assert val == 0.0


# ---------------------------------------------------------------------------
# explained_variance
# ---------------------------------------------------------------------------

def test_explained_variance_range():
    sae  = _make_sae()
    acts = _make_acts()
    ev   = explained_variance(sae, acts)
    assert 0.0 <= ev <= 1.0, f"EV {ev} outside [0, 1]"


def test_explained_variance_perfect():
    """Identity SAE on non-negative activations should give EV == 1."""
    sae  = _IdentitySAE(D_MODEL)
    acts = _make_nonneg_acts()
    ev   = explained_variance(sae, acts)
    assert abs(ev - 1.0) < 1e-5, f"Expected EV ≈ 1.0 for identity SAE, got {ev}"


# ---------------------------------------------------------------------------
# reconstruction_mse
# ---------------------------------------------------------------------------

def test_reconstruction_mse_perfect():
    """Identity SAE on non-negative activations → MSE ≈ 0."""
    sae  = _IdentitySAE(D_MODEL)
    acts = _make_nonneg_acts()
    mse  = reconstruction_mse(sae, acts)
    assert mse < 1e-10, f"Expected MSE ≈ 0 for identity SAE, got {mse}"


def test_reconstruction_mse_nonneg():
    sae  = _make_sae()
    acts = _make_acts()
    mse  = reconstruction_mse(sae, acts)
    assert mse >= 0.0


# ---------------------------------------------------------------------------
# feature_density
# ---------------------------------------------------------------------------

def test_feature_density_range():
    sae  = _make_sae()
    acts = _make_acts()
    fd   = feature_density(sae, acts)
    assert 0.0 <= fd <= 1.0, f"Feature density {fd} outside [0, 1]"


def test_feature_density_zero_for_dead_sae():
    class _ZeroSAE:
        def encode(self, x):
            return torch.zeros(x.shape[0], D_HIDDEN)

    fd = feature_density(_ZeroSAE(), _make_acts())
    assert fd == 0.0


# ---------------------------------------------------------------------------
# sparse_probing_r2
# ---------------------------------------------------------------------------

def test_sparse_probing_r2_range():
    sae  = _make_sae()
    acts = _make_acts()
    r2   = sparse_probing_r2(sae, acts)
    assert 0.0 <= r2 <= 1.0, f"R² {r2} outside [0, 1]"


def test_sparse_probing_r2_perfect_when_acts_in_feature_span():
    """When acts can be perfectly reconstructed from features, R² should be 1."""
    # Use identity SAE: encode(x) = relu(x) = x for x >= 0, decode = id
    sae  = _IdentitySAE(D_MODEL)
    acts = _make_nonneg_acts()
    r2   = sparse_probing_r2(sae, acts)
    # The feature matrix IS acts for non-negative inputs, so lstsq gives R²=1
    assert r2 > 0.9, f"Expected R² close to 1.0 for identity SAE, got {r2}"


# ---------------------------------------------------------------------------
# run_saebench aggregate runner
# ---------------------------------------------------------------------------

def test_run_saebench_returns_result():
    sae    = _make_sae()
    acts   = _make_acts()
    result = run_saebench(sae, acts)
    assert isinstance(result, SAEBenchResult)


def test_run_saebench_all_fields_populated():
    sae    = _make_sae()
    acts   = _make_acts()
    result = run_saebench(sae, acts)
    assert isinstance(result.l0, float)
    assert isinstance(result.explained_variance, float)
    assert isinstance(result.mse, float)
    assert isinstance(result.feature_density, float)
    assert isinstance(result.sparse_probing_r2, float)
    assert result.ce_loss_score is None   # requires LM; not computed here


def test_run_saebench_perfect_reconstruction():
    """Identity SAE: EV ≈ 1, MSE ≈ 0, L0 == D_MODEL (all features active)."""
    sae    = _IdentitySAE(D_MODEL)
    acts   = _make_nonneg_acts()
    result = run_saebench(sae, acts)
    assert abs(result.explained_variance - 1.0) < 1e-5
    assert result.mse < 1e-10
    # For identity SAE on non-negative data: every feature equals the input dim
    assert result.l0 == float(D_MODEL)


def test_run_saebench_subset_tasks():
    sae    = _make_sae()
    acts   = _make_acts()
    result = run_saebench(sae, acts, tasks=["l0", "mse"])
    # The requested fields should be non-trivially set
    assert result.l0 >= 0.0
    assert result.mse >= 0.0
    # Unrequested fields default to 0.0
    assert result.explained_variance == 0.0
    assert result.feature_density    == 0.0
    assert result.sparse_probing_r2  == 0.0


def test_run_saebench_unknown_task_raises():
    sae  = _make_sae()
    acts = _make_acts()
    with pytest.raises(ValueError, match="Unknown SAEBench task"):
        run_saebench(sae, acts, tasks=["nonexistent_metric"])


# ---------------------------------------------------------------------------
# SAEBenchResult.summary
# ---------------------------------------------------------------------------

def test_saebench_summary_string():
    result = SAEBenchResult(
        l0=4.5,
        explained_variance=0.92,
        mse=0.001,
        feature_density=0.75,
        sparse_probing_r2=0.88,
    )
    s = result.summary()
    assert isinstance(s, str)
    assert len(s) > 0
    assert "L0" in s
    assert "Explained" in s


def test_saebench_summary_includes_ce_when_set():
    result = SAEBenchResult(
        l0=3.0,
        explained_variance=0.95,
        mse=0.0005,
        feature_density=0.6,
        sparse_probing_r2=0.9,
        ce_loss_score=0.85,
    )
    s = result.summary()
    assert "CE" in s
