from __future__ import annotations

import math

import pytest
import torch

from circuitry.core import spectral, weight
from circuitry.core.spectral import spectral_edge_gap


def test_esd_returns_pair_of_tensors():
    W = torch.randn(32, 32)
    edges, counts = spectral.esd(W, bins=20)
    assert edges.shape == (21,)
    assert counts.shape == (20,)
    assert counts.sum().item() > 0


def test_esd_zero_matrix():
    W = torch.zeros(8, 8)
    edges, counts = spectral.esd(W, bins=5)
    # All mass at zero; should not raise.
    assert counts.sum().item() == 8


def test_rank_trajectory_keys_match_state_dict():
    W1 = {"layer.weight": torch.eye(4), "other.weight": torch.zeros(3, 3)}
    W2 = {"layer.weight": torch.eye(4) * 2, "other.weight": torch.eye(3)}
    traj = spectral.rank_trajectory([W1, W2])
    assert set(traj) == {"layer.weight", "other.weight"}
    assert len(traj["layer.weight"]) == 2
    assert traj["layer.weight"][0] == pytest.approx(weight.effective_rank(W1["layer.weight"]), rel=1e-5)


def test_rank_trajectory_skips_non_2d():
    W1 = {"bias": torch.zeros(8), "W": torch.eye(4)}
    W2 = {"bias": torch.ones(8), "W": torch.eye(4) * 0.5}
    traj = spectral.rank_trajectory([W1, W2])
    assert "bias" not in traj
    assert "W" in traj


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_esd_returns_consistent_devices_on_cuda():
    # Regression (v0.9.1): edges (linspace) came back on CPU while counts (histc)
    # were on CUDA — a device-inconsistent pair from a CUDA input.
    W = torch.randn(64, 64, device="cuda")
    edges, counts = spectral.esd(W, bins=20)
    assert edges.device == counts.device


# ---------------------------------------------------------------------------
# spectral_edge_gap tests (v1.30)
# ---------------------------------------------------------------------------



def test_spectral_edge_gap_returns_float():
    torch.manual_seed(10)
    W1 = torch.randn(16, 16)
    W2 = W1 + 0.1 * torch.randn(16, 16)
    gap = spectral_edge_gap(W1, W2, k=3)
    assert isinstance(gap, float)
    assert math.isfinite(gap)
    assert gap >= 0.0


def test_spectral_edge_gap_low_rank_update_large_gap():
    """A rank-1 update should produce a very large gap (singular values drop off sharply)."""
    torch.manual_seed(11)
    W = torch.randn(16, 16)
    # Pure rank-1 update: outer product of two random vectors
    u, v = torch.randn(16), torch.randn(16)
    W2 = W + 3.0 * u.unsqueeze(1) * v.unsqueeze(0)
    gap = spectral_edge_gap(W, W2, k=1)
    assert gap > 2.0, f"Expected large gap for rank-1 update, got {gap:.3f}"


def test_spectral_edge_gap_full_rank_update_near_one():
    """A random full-rank update should produce a gap close to 1 (no special structure)."""
    torch.manual_seed(12)
    W = torch.randn(32, 32)
    dW = torch.randn(32, 32) * 0.1  # random, full-rank
    gap = spectral_edge_gap(W, W + dW, k=5)
    # Gap should not be extreme for a structureless update
    assert 0.5 < gap < 10.0, f"Expected moderate gap for full-rank update, got {gap:.3f}"


def test_spectral_edge_gap_invalid_k_raises():
    W = torch.randn(4, 4)
    with pytest.raises(ValueError):
        spectral_edge_gap(W, W, k=0)
    with pytest.raises(ValueError):
        spectral_edge_gap(W, W, k=4)  # k >= min(m,n)
