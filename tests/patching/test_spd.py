"""Tests for patching/spd.py — Stochastic Parameter Decomposition (v1.48)."""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from circuitry.patching.spd import SPDResult, SPDRunner

D_IN, D_OUT, C = 6, 4, 12


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.target = nn.Linear(D_IN, D_OUT)
        self.head = nn.Linear(D_OUT, 3)

    def forward(self, x):
        return self.head(torch.tanh(self.target(x)))


@pytest.fixture
def model() -> Toy:
    torch.manual_seed(0)
    return Toy()


@pytest.fixture
def batches() -> list[torch.Tensor]:
    torch.manual_seed(1)
    return [torch.randn(16, D_IN) for _ in range(4)]


def _quick_run(model, batches, **kwargs) -> SPDResult:
    runner = SPDRunner(model, model.target, n_components=C,
                       importance_hidden=16, seed=0)
    defaults = dict(n_steps=20, lr=1e-2)
    defaults.update(kwargs)
    return runner.run(batches, **defaults)


# ---------------------------------------------------------------------------
# Construction and validation
# ---------------------------------------------------------------------------


def test_non_linear_module_raises(model):
    with pytest.raises(TypeError, match="nn.Linear"):
        SPDRunner(model, nn.ReLU(), n_components=4)


def test_empty_batches_raises(model):
    runner = SPDRunner(model, model.target, n_components=4)
    with pytest.raises(ValueError, match="empty"):
        runner.run([])


def test_bad_output_loss_raises(model, batches):
    runner = SPDRunner(model, model.target, n_components=4)
    with pytest.raises(ValueError, match="output_loss"):
        runner.run(batches, output_loss="nope")


# ---------------------------------------------------------------------------
# Result shape and bookkeeping
# ---------------------------------------------------------------------------


def test_result_shapes_and_losses(model, batches):
    result = _quick_run(model, batches)
    assert result.U.shape == (C, D_OUT)
    assert result.V.shape == (D_IN, C)
    assert result.n_components == C
    assert all(len(v) == 20 for v in result.losses.values())
    assert set(result.losses) == {"faith", "stoch", "imp"}


def test_importance_in_unit_interval(model, batches):
    result = _quick_run(model, batches)
    ci = result.importance(batches[0])
    assert ci.shape == (16, C)
    assert float(ci.min()) > 0.0 and float(ci.max()) < 1.0


def test_active_components_subset(model, batches):
    result = _quick_run(model, batches)
    active = result.active_components(batches[0], threshold=0.0)
    assert active == list(range(C))  # everything beats threshold 0
    assert result.active_components(batches[0], threshold=1.0) == []


def test_component_weight_shape_and_sum(model, batches):
    result = _quick_run(model, batches)
    wc = result.component_weight(0)
    assert wc.shape == (D_OUT, D_IN)
    total = sum(result.component_weight(c) for c in range(C))
    torch.testing.assert_close(total, result.reconstructed_weight(),
                               atol=1e-5, rtol=1e-4)


def test_to_markdown(model, batches):
    result = _quick_run(model, batches)
    md = result.to_markdown(x=batches[0], top_k=5)
    assert "## SPD Decomposition" in md
    assert "mean importance" in md


# ---------------------------------------------------------------------------
# Training behaviour
# ---------------------------------------------------------------------------


def test_faithfulness_converges(model, batches):
    result = _quick_run(model, batches, n_steps=600, coeff_faith=10.0,
                        coeff_stoch=1.0, coeff_imp=1e-4)
    assert result.faithfulness_error < 0.05
    torch.testing.assert_close(
        result.reconstructed_weight(), model.target.weight.detach(),
        atol=0.05, rtol=0.5,
    )


def test_faithfulness_loss_decreases(model, batches):
    result = _quick_run(model, batches, n_steps=300, coeff_faith=10.0)
    faith = result.losses["faith"]
    assert faith[-1] < faith[0] / 10


def test_kl_output_loss_runs(model, batches):
    result = _quick_run(model, batches, output_loss="kl")
    assert all(v >= 0 for v in result.losses["stoch"])


def test_forward_fn_used(model, batches):
    calls = {"n": 0}

    def fwd(m, batch):
        calls["n"] += 1
        return m(batch)

    _quick_run(model, batches, n_steps=5, forward_fn=fwd)
    assert calls["n"] == 10  # target pass + masked pass per step


def test_deterministic_given_seed(model, batches):
    r1 = SPDRunner(model, model.target, n_components=C, seed=7).run(
        batches, n_steps=10)
    r2 = SPDRunner(model, model.target, n_components=C, seed=7).run(
        batches, n_steps=10)
    torch.testing.assert_close(r1.U, r2.U)
    torch.testing.assert_close(r1.V, r2.V)


# ---------------------------------------------------------------------------
# Model hygiene
# ---------------------------------------------------------------------------


def test_model_untouched_after_run(model, batches):
    model.train()
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    _quick_run(model, batches)
    assert model.training  # mode restored
    for n, p in model.named_parameters():
        assert p.requires_grad, f"{n} left frozen"
        torch.testing.assert_close(p.detach(), before[n])  # weights unchanged
    assert len(model.target._forward_hooks) == 0  # hooks removed


def test_model_output_unchanged_after_run(model, batches):
    with torch.no_grad():
        before = model(batches[0]).clone()
    _quick_run(model, batches)
    with torch.no_grad():
        after = model(batches[0])
    torch.testing.assert_close(before, after)


def test_exports():
    import circuitry
    from circuitry import patching

    assert circuitry.SPDRunner is SPDRunner
    assert patching.SPDResult is SPDResult
    assert "SPDRunner" in circuitry.__all__
