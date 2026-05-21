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
