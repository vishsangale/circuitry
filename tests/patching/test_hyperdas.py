"""Tests for HyperDASRunner — input-conditioned alignment search."""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from circuitry.patching.hyperdas import (
    HyperDASNet,
    HyperDASResult,
    HyperDASRunner,
    _gram_schmidt,
)


# ---------------------------------------------------------------------------
# Toy model
# ---------------------------------------------------------------------------

class _TinyMLP(nn.Module):
    class _Block(nn.Module):
        def __init__(self, d: int) -> None:
            super().__init__()
            self.mlp = nn.ModuleDict({
                "up": nn.Linear(d, d * 2, bias=False),
                "down": nn.Linear(d * 2, d, bias=False),
            })

        def forward(self, x: Tensor) -> Tensor:
            return x + self.mlp["down"](torch.relu(self.mlp["up"](x)))

    def __init__(self, d: int = 8, vocab: int = 32, n_layers: int = 3) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, d)
        self.layers = nn.ModuleList([self._Block(d) for _ in range(n_layers)])
        self.lm_head = nn.Linear(d, vocab, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        h = self.embed_tokens(x)
        for layer in self.layers:
            h = layer(h)
        return self.lm_head(h)


D, VOCAB = 8, 32


def _make_runner(subspace_dim: int = 1) -> tuple[_TinyMLP, nn.Module, HyperDASRunner]:
    torch.manual_seed(0)
    model = _TinyMLP(d=D, vocab=VOCAB)
    module = model.layers[1]
    return model, module, HyperDASRunner(
        model, module, d_model=D, subspace_dim=subspace_dim
    )


def _make_inputs(batch: int = 4, seq: int = 5):
    base = torch.randint(0, VOCAB, (batch, seq))
    src = torch.randint(0, VOCAB, (batch, seq))
    labels = torch.randint(0, VOCAB, (batch,))
    return base, src, labels


# ---------------------------------------------------------------------------
# HyperDASNet unit tests
# ---------------------------------------------------------------------------

def test_hypernet_output_shape() -> None:
    """HyperDASNet(8, 2, 32)(randn(3, 8)) should produce shape (3, 2, 8)."""
    net = HyperDASNet(d_model=8, subspace_dim=2, hidden_dim=32)
    h = torch.randn(3, 8)
    out = net(h)
    assert out.shape == (3, 2, 8), f"expected (3, 2, 8), got {tuple(out.shape)}"


def test_hypernet_rows_orthonormal() -> None:
    """Output rows should have unit norm and be mutually orthogonal."""
    torch.manual_seed(42)
    net = HyperDASNet(d_model=8, subspace_dim=3, hidden_dim=32)
    h = torch.randn(5, 8)
    out = net(h)  # (5, 3, 8)

    # Row norms ≈ 1
    norms = out.norm(dim=-1)  # (5, 3)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), (
        f"row norms not unit: {norms}"
    )

    # Pairwise dots ≈ 0
    # out @ out.T over last dim: (5, 3, 3)
    dots = torch.bmm(out, out.transpose(1, 2))  # (5, 3, 3)
    eye = torch.eye(3).unsqueeze(0).expand(5, -1, -1)
    assert torch.allclose(dots, eye, atol=1e-5), (
        f"rows not mutually orthogonal: max off-diag = {(dots - eye).abs().max().item():.2e}"
    )


# ---------------------------------------------------------------------------
# _gram_schmidt unit tests
# ---------------------------------------------------------------------------

def test_gram_schmidt_unit_norms() -> None:
    """After Gram-Schmidt, each row should have norm ≈ 1."""
    torch.manual_seed(7)
    V = torch.randn(2, 3, 8)
    Q = _gram_schmidt(V)
    norms = Q.norm(dim=-1)  # (2, 3)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6), (
        f"norms not unit: {norms}"
    )


def test_gram_schmidt_orthogonal() -> None:
    """After Gram-Schmidt, rows within each batch element should be orthogonal."""
    torch.manual_seed(13)
    V = torch.randn(2, 3, 8)
    Q = _gram_schmidt(V)  # (2, 3, 8)
    dots = torch.bmm(Q, Q.transpose(1, 2))  # (2, 3, 3)
    eye = torch.eye(3).unsqueeze(0).expand(2, -1, -1)
    assert torch.allclose(dots, eye, atol=1e-5), (
        f"rows not orthogonal: max off-diag = {(dots - eye).abs().max().item():.2e}"
    )


# ---------------------------------------------------------------------------
# HyperDASRunner end-to-end tests
# ---------------------------------------------------------------------------

def test_hyperdas_result_type() -> None:
    """run() should return a HyperDASResult."""
    _, _, runner = _make_runner()
    base, src, labels = _make_inputs()
    result = runner.run(base, src, labels, n_steps=5)
    assert isinstance(result, HyperDASResult)


def test_hyperdas_result_has_network() -> None:
    """result.network should be a HyperDASNet instance."""
    _, _, runner = _make_runner()
    base, src, labels = _make_inputs()
    result = runner.run(base, src, labels, n_steps=5)
    assert isinstance(result.network, HyperDASNet)


def test_hyperdas_losses_list() -> None:
    """result.losses should be a list with one entry per training step."""
    _, _, runner = _make_runner()
    base, src, labels = _make_inputs()
    n_steps = 5
    result = runner.run(base, src, labels, n_steps=n_steps)
    assert isinstance(result.losses, list)
    assert len(result.losses) == n_steps, (
        f"expected {n_steps} losses, got {len(result.losses)}"
    )


def test_hyperdas_iia_in_unit_interval() -> None:
    """IIA score must be in [0, 1]."""
    _, _, runner = _make_runner()
    base, src, labels = _make_inputs()
    result = runner.run(base, src, labels, n_steps=5)
    assert 0.0 <= result.iia_score <= 1.0, (
        f"iia_score out of range: {result.iia_score}"
    )


def test_hyperdas_model_back_to_eval() -> None:
    """model.training should remain False after run() completes."""
    model, _, runner = _make_runner()
    base, src, labels = _make_inputs()
    runner.run(base, src, labels, n_steps=5)
    assert not model.training, "model should be in eval mode after run()"


def test_hyperdas_n_steps_respected() -> None:
    """n_steps=5 should give exactly 5 loss entries."""
    _, _, runner = _make_runner()
    base, src, labels = _make_inputs()
    result = runner.run(base, src, labels, n_steps=5)
    assert len(result.losses) == 5


def test_hyperdas_network_on_same_device() -> None:
    """Trained hypernetwork parameters should be on the same device as the model."""
    model, _, runner = _make_runner()
    base, src, labels = _make_inputs()
    result = runner.run(base, src, labels, n_steps=5)

    model_device = next(model.parameters()).device
    for name, p in result.network.named_parameters():
        assert p.device == model_device, (
            f"param {name} on {p.device}, expected {model_device}"
        )


def test_hyperdas_subspace_dim_2() -> None:
    """subspace_dim=2 should complete without error and return valid result."""
    _, _, runner = _make_runner(subspace_dim=2)
    base, src, labels = _make_inputs()
    result = runner.run(base, src, labels, n_steps=5)
    assert isinstance(result, HyperDASResult)
    assert result.network.subspace_dim == 2
    assert len(result.losses) == 5
    assert 0.0 <= result.iia_score <= 1.0
