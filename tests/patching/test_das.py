"""Tests for DASRunner — Distributed Alignment Search."""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from circuitry.patching.das import DASResult, DASRunner, _interchange

# ---------------------------------------------------------------------------
# Synthetic "causal model" fixture
# ---------------------------------------------------------------------------

class LinearCausalModel(nn.Module):
    """Two-layer linear model where the hidden state encodes a causal variable.

    The true causal direction is R_true[0] (first row of the ground-truth
    rotation). h = R_true.T @ [causal_var, noise1, noise2, ...]
    output = h @ W_out  (logit for class = causal_var)

    DAS should recover R_true by learning that swapping dimension 0 in the
    rotated space changes the class prediction correctly.
    """

    def __init__(self, d_model: int = 4, n_classes: int = 2, seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        # Fixed random true rotation
        Q, _ = torch.linalg.qr(torch.randn(d_model, d_model))
        self.register_buffer("R_true", Q)

        self.hidden = nn.Linear(d_model, d_model, bias=False)
        self.head = nn.Linear(d_model, n_classes, bias=False)
        # Make output depend on R_true[0] direction (the causal direction)
        W = torch.zeros(n_classes, d_model)
        W[0] = 2.0 * Q[0]    # class 0 ← positive projection onto R_true[0]
        W[1] = -2.0 * Q[0]   # class 1 ← negative projection onto R_true[0]
        self.head.weight.data = W

        nn.init.eye_(self.hidden.weight)  # identity: h passes through unchanged

    def make_inputs(self, causal_values: torch.Tensor) -> torch.Tensor:
        """Build inputs whose first rotated dimension equals causal_values.

        causal_values: (batch,) float tensor, typically ±1
        Returns: (batch, d_model) tensor in the unrotated space
        """
        batch = causal_values.shape[0]
        d = self.R_true.shape[0]
        z = torch.zeros(batch, d)
        z[:, 0] = causal_values  # first rotated dim = causal var
        return z @ self.R_true   # rotate to model input space

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, d_model)
        h = self.hidden(x)       # identity — just passes x through
        return self.head(h)      # (batch, n_classes)


@pytest.fixture
def causal_model():
    return LinearCausalModel(d_model=4, n_classes=2, seed=42)


# ---------------------------------------------------------------------------
# _interchange unit tests
# ---------------------------------------------------------------------------

def test_interchange_swaps_subspace():
    """Swapping subspace_dim=2 of a 4D vector gives source[:2] + base[2:]."""
    R = torch.eye(4)
    h_base   = torch.tensor([[1., 2., 3., 4.]])
    h_source = torch.tensor([[5., 6., 7., 8.]])
    h_int = _interchange(R, h_base, h_source, subspace_dim=2)
    # With identity R: z = h, so z_int = [source0, source1, base2, base3]
    expected = torch.tensor([[5., 6., 3., 4.]])
    assert torch.allclose(h_int, expected, atol=1e-5)


def test_interchange_subspace_dim_0_is_identity():
    """subspace_dim=0: no swap → h_int == h_base."""
    R = torch.eye(4)
    h_base   = torch.randn(2, 4)
    h_source = torch.randn(2, 4)
    h_int = _interchange(R, h_base, h_source, subspace_dim=0)
    assert torch.allclose(h_int, h_base, atol=1e-5)


def test_interchange_full_swap():
    """subspace_dim=d_model: full swap → h_int == h_source."""
    R = torch.eye(4)
    h_base   = torch.randn(2, 4)
    h_source = torch.randn(2, 4)
    h_int = _interchange(R, h_base, h_source, subspace_dim=4)
    assert torch.allclose(h_int, h_source, atol=1e-5)


def test_interchange_non_identity_rotation():
    """Interchange with a non-identity R should be invertible (rotates back)."""
    torch.manual_seed(0)
    Q, _ = torch.linalg.qr(torch.randn(4, 4))
    h_base   = torch.randn(3, 4)
    h_source = torch.randn(3, 4)
    # With subspace_dim=0: h_int == h_base regardless of R
    h_int = _interchange(Q, h_base, h_source, subspace_dim=0)
    assert torch.allclose(h_int, h_base, atol=1e-5)


# ---------------------------------------------------------------------------
# DASRunner API tests
# ---------------------------------------------------------------------------

def test_das_returns_result(causal_model):
    torch.manual_seed(0)
    base   = causal_model.make_inputs(torch.ones(8))
    source = causal_model.make_inputs(-torch.ones(8))
    labels = torch.zeros(8, dtype=torch.long)  # after swap: should predict class 0-like
    runner = DASRunner(causal_model)
    result = runner.run(
        base, source, labels,
        module=causal_model.hidden,
        subspace_dim=1,
        n_steps=10,
        lr=0.05,
    )
    assert isinstance(result, DASResult)
    assert result.rotation.shape == (4, 4)
    assert result.subspace_dim == 1
    assert len(result.losses) == 10


def test_das_rotation_is_orthogonal(causal_model):
    """After training, R @ R.T should be close to identity."""
    torch.manual_seed(1)
    base   = causal_model.make_inputs(torch.ones(8))
    source = causal_model.make_inputs(-torch.ones(8))
    labels = torch.zeros(8, dtype=torch.long)
    runner = DASRunner(causal_model)
    result = runner.run(
        base, source, labels,
        module=causal_model.hidden,
        subspace_dim=1,
        n_steps=50,
        lr=0.05,
    )
    RRt = result.rotation @ result.rotation.T
    assert torch.allclose(RRt, torch.eye(4), atol=1e-4)


def test_das_recovers_causal_direction(causal_model):
    """DAS should align its first row with the true causal direction.

    The synthetic model encodes the causal variable in R_true[0].
    After training, |cos(R_learned[0], R_true[0])| should be close to 1.
    """
    torch.manual_seed(2)
    n = 32
    # base: causal_var = +1; source: causal_var = −1
    base   = causal_model.make_inputs(torch.ones(n))
    source = causal_model.make_inputs(-torch.ones(n))
    # After swapping causal subspace from source → predict class 1 (causal=-1)
    labels = torch.ones(n, dtype=torch.long)

    runner = DASRunner(causal_model)
    result = runner.run(
        base, source, labels,
        module=causal_model.hidden,
        subspace_dim=1,
        n_steps=400,
        lr=0.05,
    )

    r_true = causal_model.R_true[0]  # true causal direction
    r_learned = result.rotation[0]    # learned first row
    # abs: DAS recovers direction up to sign (both ±R_true[0] are valid)
    cosine = abs(torch.dot(r_true, r_learned).item())
    assert cosine > 0.90, f"cosine to ground-truth direction = {cosine:.4f} (expected > 0.90)"


def test_das_iia_score_range(causal_model):
    """IIA score must be in [0, 1]."""
    torch.manual_seed(3)
    base   = causal_model.make_inputs(torch.ones(8))
    source = causal_model.make_inputs(-torch.ones(8))
    labels = torch.ones(8, dtype=torch.long)
    runner = DASRunner(causal_model)
    result = runner.run(
        base, source, labels,
        module=causal_model.hidden,
        subspace_dim=1,
        n_steps=20,
    )
    assert 0.0 <= result.iia_score <= 1.0


def test_das_subspace_directions_shape(causal_model):
    """subspace_directions() should return (subspace_dim, d_model)."""
    base   = causal_model.make_inputs(torch.ones(4))
    source = causal_model.make_inputs(-torch.ones(4))
    labels = torch.zeros(4, dtype=torch.long)
    runner = DASRunner(causal_model)
    result = runner.run(
        base, source, labels,
        module=causal_model.hidden,
        subspace_dim=2,
        n_steps=5,
    )
    dirs = result.subspace_directions()
    assert dirs.shape == (2, 4)
