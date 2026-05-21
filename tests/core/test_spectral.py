from __future__ import annotations

import pytest
import torch

from circuitry.core import spectral, weight


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
