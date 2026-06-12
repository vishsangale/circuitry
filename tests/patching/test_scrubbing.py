"""Tests for CausalScrubRunner — causal scrubbing faithfulness scores."""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from circuitry.patching.scrubbing import (
    CausalScrubResult,
    CausalScrubRunner,
    CircuitHypothesis,
)

# ---------------------------------------------------------------------------
# Toy model with a known "correct" circuit
# ---------------------------------------------------------------------------

class TwoPathModel(nn.Module):
    """Model with two additive paths: signal + noise.

    output = W_signal(x) + W_noise(noise_input)

    The "circuit" is just the signal path (signal_layer).
    The noise path should be ablated.

    For clean run: noise_input = 0 → output = W_signal(x)
    For corrupted run: noise_input = 1 → output = W_signal(x) + W_noise(1)

    If we scrub out noise_layer (replace with corrupted acts), the output
    should change. If we keep the circuit (signal_layer), it should not.
    """

    def __init__(self, d: int = 4, n_classes: int = 2, seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        self.signal_layer = nn.Linear(d, d, bias=False)
        self.noise_layer  = nn.Linear(d, d, bias=False)
        self.head = nn.Linear(d, n_classes, bias=False)
        nn.init.eye_(self.signal_layer.weight)
        nn.init.eye_(self.head.weight[:, :n_classes] if d >= n_classes else self.head.weight)
        nn.init.zeros_(self.noise_layer.weight)  # noise path starts at 0 weight

    def forward(self, signal: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        if noise is None:
            noise = torch.zeros_like(signal)
        h = self.signal_layer(signal) + self.noise_layer(noise)
        return self.head(h)


class SimpleLinear(nn.Module):
    """Dead-simple 2-layer linear model for minimal scrubbing tests."""

    def __init__(self, d: int = 4, vocab: int = 4):
        super().__init__()
        self.layer0 = nn.Linear(d, d, bias=False)
        self.layer1 = nn.Linear(d, vocab, bias=False)
        nn.init.eye_(self.layer0.weight)
        nn.init.eye_(self.layer1.weight[:d, :] if vocab >= d else self.layer1.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer1(self.layer0(x))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_model():
    return SimpleLinear(d=4, vocab=4)


@pytest.fixture
def two_path_model():
    return TwoPathModel(d=4, n_classes=2, seed=0)


def _metric(logits: torch.Tensor) -> torch.Tensor:
    """Differentiable scalar: sum of max logits."""
    return logits.max(dim=-1).values.sum()


# ---------------------------------------------------------------------------
# CircuitHypothesis API
# ---------------------------------------------------------------------------

def test_circuit_hypothesis_empty():
    """Empty circuit is valid."""
    hyp = CircuitHypothesis(circuit_modules=[])
    assert hyp.circuit_modules == []


def test_circuit_hypothesis_stores_labels(simple_model):
    hyp = CircuitHypothesis(
        circuit_modules=[simple_model.layer0],
        node_labels={simple_model.layer0: "layer0"},
    )
    assert hyp.node_labels[simple_model.layer0] == "layer0"


# ---------------------------------------------------------------------------
# CausalScrubRunner API / return-type tests
# ---------------------------------------------------------------------------

def test_scrub_returns_result(simple_model):
    clean = torch.randn(4, 4)
    corrupted = torch.randn(4, 4)
    hyp = CircuitHypothesis(circuit_modules=[simple_model.layer0])
    runner = CausalScrubRunner(simple_model)
    result = runner.run(clean, corrupted, _metric, hyp)
    assert isinstance(result, CausalScrubResult)


def test_scrub_faithfulness_is_float(simple_model):
    clean = torch.randn(4, 4)
    corrupted = torch.randn(4, 4)
    hyp = CircuitHypothesis(circuit_modules=[simple_model.layer0])
    runner = CausalScrubRunner(simple_model)
    result = runner.run(clean, corrupted, _metric, hyp)
    assert isinstance(result.faithfulness, float)


def test_scrub_metrics_are_finite(simple_model):
    clean = torch.randn(4, 4)
    corrupted = torch.randn(4, 4)
    hyp = CircuitHypothesis(circuit_modules=[simple_model.layer0])
    runner = CausalScrubRunner(simple_model)
    result = runner.run(clean, corrupted, _metric, hyp)
    import math
    assert math.isfinite(result.clean_metric)
    assert math.isfinite(result.corrupted_metric)
    assert math.isfinite(result.scrubbed_metric)


# ---------------------------------------------------------------------------
# Correctness tests
# ---------------------------------------------------------------------------

def test_correct_hypothesis_high_faithfulness():
    """When ALL modules are included in the circuit, faithfulness ≈ 1.0.

    Keeping the full model clean → scrubbed == clean → faithfulness = 1.
    """
    torch.manual_seed(0)
    model = SimpleLinear(d=4, vocab=4)
    clean = torch.randn(8, 4)
    corrupted = torch.randn(8, 4)
    # Full circuit: both layers included
    hyp = CircuitHypothesis(circuit_modules=list(model.modules()))
    runner = CausalScrubRunner(model)
    result = runner.run(clean, corrupted, _metric, hyp, compute_per_module=False)
    assert result.faithfulness > 0.9, f"full-circuit faithfulness = {result.faithfulness:.4f}"


def test_empty_circuit_low_faithfulness():
    """With an empty circuit, all activations are replaced by corrupted ones.

    scrubbed ≈ corrupted → faithfulness ≈ 0.
    """
    torch.manual_seed(1)
    model = SimpleLinear(d=4, vocab=4)
    clean = torch.randn(8, 4)
    corrupted = torch.randn(8, 4)
    hyp = CircuitHypothesis(circuit_modules=[])
    runner = CausalScrubRunner(model)
    result = runner.run(clean, corrupted, _metric, hyp, compute_per_module=False)
    # scrubbed should be equal to or near corrupted run
    assert result.faithfulness < 0.5, f"empty-circuit faithfulness = {result.faithfulness:.4f}"


def test_constant_model_faithfulness_is_one():
    """If clean == corrupted, faithfulness is always 1 (or undefined → 1)."""
    torch.manual_seed(2)
    model = SimpleLinear(d=4, vocab=4)
    inputs = torch.randn(4, 4)
    hyp = CircuitHypothesis(circuit_modules=[model.layer0])
    runner = CausalScrubRunner(model)
    result = runner.run(inputs, inputs, _metric, hyp, compute_per_module=False)
    # clean == corrupted → denom == 0 → faithfulness defaults to 1.0
    assert result.faithfulness == 1.0


def test_per_module_delta_computed(simple_model):
    """per_module_delta should have one entry per circuit module."""
    torch.manual_seed(3)
    clean = torch.randn(4, 4)
    corrupted = torch.randn(4, 4)
    hyp = CircuitHypothesis(
        circuit_modules=[simple_model.layer0],
        node_labels={simple_model.layer0: "layer0"},
    )
    runner = CausalScrubRunner(simple_model)
    result = runner.run(clean, corrupted, _metric, hyp, compute_per_module=True)
    assert "layer0" in result.per_module_delta


def test_per_module_delta_disabled(simple_model):
    """compute_per_module=False → per_module_delta is empty."""
    clean = torch.randn(4, 4)
    corrupted = torch.randn(4, 4)
    hyp = CircuitHypothesis(circuit_modules=[simple_model.layer0])
    runner = CausalScrubRunner(simple_model)
    result = runner.run(clean, corrupted, _metric, hyp, compute_per_module=False)
    assert result.per_module_delta == {}


def test_scrub_does_not_mutate_model_params(simple_model):
    """Running scrubbing should not change any model parameters."""
    clean = torch.randn(4, 4)
    corrupted = torch.randn(4, 4)
    hyp = CircuitHypothesis(circuit_modules=[simple_model.layer0])
    params_before = {n: p.clone() for n, p in simple_model.named_parameters()}
    runner = CausalScrubRunner(simple_model)
    runner.run(clean, corrupted, _metric, hyp)
    for name, p in simple_model.named_parameters():
        assert torch.allclose(p, params_before[name]), f"param {name} was mutated"
