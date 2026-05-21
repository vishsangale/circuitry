from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from circuitry.core import weight


def test_singular_values_diagonal():
    W = torch.diag(torch.tensor([3.0, 2.0, 1.0]))
    s = weight.singular_values(W)
    assert torch.allclose(s, torch.tensor([3.0, 2.0, 1.0]))


def test_singular_values_k_truncates():
    W = torch.diag(torch.tensor([3.0, 2.0, 1.0, 0.5]))
    s = weight.singular_values(W, k=2)
    assert s.shape == (2,)
    assert torch.allclose(s, torch.tensor([3.0, 2.0]))


def test_singular_values_max_dim_subsamples():
    # Wide matrix; max_dim should cap SVD cost without hanging.
    torch.manual_seed(0)
    W = torch.randn(64, 2048)
    s = weight.singular_values(W, max_dim=256)
    assert s.shape[0] <= 256


def test_singular_values_accepts_numpy():
    W = np.eye(4, dtype=np.float32)
    s = weight.singular_values(W)
    assert torch.allclose(s, torch.ones(4))


def test_effective_rank_identity_is_n():
    n = 5
    assert weight.effective_rank(torch.eye(n)) == pytest.approx(n, rel=1e-6)


def test_effective_rank_rank1_is_one():
    u = torch.randn(8, 1)
    v = torch.randn(1, 8)
    W = u @ v
    assert weight.effective_rank(W) == pytest.approx(1.0, abs=1e-4)


def test_effective_rank_returns_python_float():
    val = weight.effective_rank(torch.eye(3))
    assert isinstance(val, float)


def test_effective_rank_invariant_under_orthogonal():
    torch.manual_seed(0)
    W = torch.randn(16, 16)
    Q, _ = torch.linalg.qr(torch.randn(16, 16))
    assert weight.effective_rank(W) == pytest.approx(
        weight.effective_rank(Q @ W), rel=1e-5
    )


def test_stable_rank_identity():
    n = 5
    assert weight.stable_rank(torch.eye(n)) == pytest.approx(n, rel=1e-6)


def test_stable_rank_rank1_is_one():
    u = torch.randn(8, 1)
    v = torch.randn(1, 8)
    assert weight.stable_rank(u @ v) == pytest.approx(1.0, abs=1e-4)


def test_stable_rank_returns_float():
    assert isinstance(weight.stable_rank(torch.eye(3)), float)


def test_condition_number_orthogonal_is_one():
    Q, _ = torch.linalg.qr(torch.randn(8, 8))
    assert weight.condition_number(Q) == pytest.approx(1.0, abs=1e-4)


def test_condition_number_diag():
    W = torch.diag(torch.tensor([4.0, 1.0]))
    assert weight.condition_number(W) == pytest.approx(4.0, rel=1e-6)


def test_condition_number_returns_float():
    assert isinstance(weight.condition_number(torch.eye(3)), float)


def test_heavy_tail_alpha_random_matrix():
    # Marchenko-Pastur bulk → alpha is finite and positive.
    torch.manual_seed(0)
    W = torch.randn(64, 64)
    alpha = weight.heavy_tail_alpha(W)
    assert isinstance(alpha, float)
    assert math.isfinite(alpha)
    assert alpha > 0


def test_heavy_tail_alpha_low_rank_is_finite():
    # Constructed power-law tail (rank ~10 in 64-dim space).
    torch.manual_seed(0)
    U = torch.randn(64, 10)
    V = torch.randn(10, 64)
    alpha = weight.heavy_tail_alpha(U @ V)
    assert math.isfinite(alpha)


def test_update_delta_zero_when_unchanged():
    sd = {"w": torch.ones(3, 3)}
    out = weight.update_delta(sd, sd)
    assert out["w"] == 0.0


def test_update_delta_l2_when_shifted():
    sd_now = {"w": torch.tensor([[1.0, 0.0], [0.0, 1.0]])}
    sd_prev = {"w": torch.tensor([[0.0, 0.0], [0.0, 0.0]])}
    out = weight.update_delta(sd_now, sd_prev)
    assert abs(out["w"] - (2.0 ** 0.5)) < 1e-6


def test_direction_cosine_collinear_updates():
    # prev_prev -> prev: +I; prev -> now: +2I (same direction)
    sd_pp = {"w": torch.zeros(2, 2)}
    sd_p = {"w": torch.eye(2)}
    sd_n = {"w": torch.eye(2) * 3.0}
    out = weight.direction_cosine(sd_n, sd_p, sd_pp)
    assert abs(out["w"] - 1.0) < 1e-6


def test_direction_cosine_opposite_updates():
    sd_pp = {"w": torch.zeros(2, 2)}
    sd_p = {"w": torch.eye(2)}
    sd_n = {"w": torch.zeros(2, 2)}  # update reverses
    out = weight.direction_cosine(sd_n, sd_p, sd_pp)
    assert abs(out["w"] - (-1.0)) < 1e-6
