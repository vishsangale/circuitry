"""Tests for circuitry.patching.causal_trace.CausalTraceResult and CausalTraceRunner."""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from circuitry.patching.causal_trace import CausalTraceResult, CausalTraceRunner


# ---------------------------------------------------------------------------
# Toy model
# ---------------------------------------------------------------------------

class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(4, 8, bias=False)
        self.l2 = nn.Linear(8, 4, bias=False)

    def forward(self, x):
        return self.l2(torch.relu(self.l1(x)))


def _metric(out: torch.Tensor) -> float:
    return out.mean().item()


def _make_runner(**kwargs):
    model = TinyMLP()
    torch.manual_seed(42)
    nn.init.normal_(model.l1.weight)
    nn.init.normal_(model.l2.weight)
    return model, CausalTraceRunner(model, **kwargs)


def _clean_corrupted():
    torch.manual_seed(0)
    clean = torch.randn(1, 4)
    corrupted = torch.randn(1, 4)
    return clean, corrupted


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_result_is_causal_trace_result():
    model = TinyMLP()
    runner = CausalTraceRunner(
        model,
        modules=[model.l1, model.l2],
        module_names=["layer1", "layer2"],
    )
    clean, corrupted = _clean_corrupted()
    result = runner.run(clean, corrupted, _metric)
    assert isinstance(result, CausalTraceResult)


def test_recovery_shape_matches_modules():
    model = TinyMLP()
    runner = CausalTraceRunner(
        model,
        modules=[model.l1, model.l2],
        module_names=["layer1", "layer2"],
    )
    clean, corrupted = _clean_corrupted()
    result = runner.run(clean, corrupted, _metric)
    assert result.recovery.shape == (2,)


def test_layer_names_length_matches_modules():
    model = TinyMLP()
    runner = CausalTraceRunner(
        model,
        modules=[model.l1, model.l2],
        module_names=["layer1", "layer2"],
    )
    clean, corrupted = _clean_corrupted()
    result = runner.run(clean, corrupted, _metric)
    assert len(result.layer_names) == 2


def test_clean_and_corrupted_scores_are_floats():
    model = TinyMLP()
    runner = CausalTraceRunner(
        model,
        modules=[model.l1, model.l2],
        module_names=["layer1", "layer2"],
    )
    clean, corrupted = _clean_corrupted()
    result = runner.run(clean, corrupted, _metric)
    assert isinstance(result.clean_score, float)
    assert isinstance(result.corrupted_score, float)


def test_recovery_values_are_finite():
    model = TinyMLP()
    runner = CausalTraceRunner(
        model,
        modules=[model.l1, model.l2],
        module_names=["layer1", "layer2"],
    )
    clean, corrupted = _clean_corrupted()
    result = runner.run(clean, corrupted, _metric)
    assert torch.all(torch.isfinite(result.recovery))


def test_top_layers_sorted_descending():
    model = TinyMLP()
    runner = CausalTraceRunner(
        model,
        modules=[model.l1, model.l2],
        module_names=["layer1", "layer2"],
    )
    clean, corrupted = _clean_corrupted()
    result = runner.run(clean, corrupted, _metric)
    top = result.top_layers(k=2)
    assert isinstance(top, list)
    assert all(isinstance(name, str) and isinstance(val, float) for name, val in top)
    scores = [v for _, v in top]
    assert scores == sorted(scores, reverse=True)


def test_top_layers_k1_returns_one_element():
    model = TinyMLP()
    runner = CausalTraceRunner(
        model,
        modules=[model.l1, model.l2],
        module_names=["layer1", "layer2"],
    )
    clean, corrupted = _clean_corrupted()
    result = runner.run(clean, corrupted, _metric)
    top = result.top_layers(k=1)
    assert len(top) == 1


def test_to_markdown_contains_header():
    model = TinyMLP()
    runner = CausalTraceRunner(
        model,
        modules=[model.l1, model.l2],
        module_names=["layer1", "layer2"],
    )
    clean, corrupted = _clean_corrupted()
    result = runner.run(clean, corrupted, _metric)
    md = result.to_markdown()
    assert isinstance(md, str)
    assert "Causal Trace" in md


def test_raises_if_neither_modules_nor_pattern():
    model = TinyMLP()
    with pytest.raises(ValueError):
        CausalTraceRunner(model)


def test_raises_if_both_modules_and_pattern():
    model = TinyMLP()
    with pytest.raises(ValueError):
        CausalTraceRunner(model, modules=[model.l1], module_pattern="l[12]")


def test_module_pattern_matches_both_layers():
    model = TinyMLP()
    runner = CausalTraceRunner(model, module_pattern="l[12]")
    clean, corrupted = _clean_corrupted()
    result = runner.run(clean, corrupted, _metric)
    # Both l1 and l2 should have been matched
    assert result.recovery.shape[0] == 2
    assert len(result.layer_names) == 2


def test_recovery_near_one_when_clean_equals_corrupted():
    """When clean == corrupted the denominator (clean - corrupted score) is ~0.

    The convention is that recovery = 1.0 in this degenerate case, since
    patching the clean activation into a run that is already clean changes nothing.
    """
    model = TinyMLP()
    runner = CausalTraceRunner(
        model,
        modules=[model.l1, model.l2],
        module_names=["layer1", "layer2"],
    )
    torch.manual_seed(10)
    x = torch.randn(1, 4)
    result = runner.run(x, x.clone(), _metric)
    # All recovery values should be close to 1.0 (or at least finite and non-negative)
    assert torch.all(torch.isfinite(result.recovery))
    assert torch.allclose(result.recovery, torch.ones_like(result.recovery), atol=1e-4)
