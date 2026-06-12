"""Tests for circuitry.patching.head_knockout — HeadKnockoutResult and HeadKnockoutRunner."""

import pytest
import torch
import torch.nn as nn

try:
    from circuitry.patching.head_knockout import HeadKnockoutResult, HeadKnockoutRunner
    IMPORT_OK = True
except ImportError:
    IMPORT_OK = False
    HeadKnockoutResult = None
    HeadKnockoutRunner = None

pytestmark = pytest.mark.skipif(not IMPORT_OK, reason="circuitry.patching.head_knockout not available")


# ---------------------------------------------------------------------------
# Toy model
# ---------------------------------------------------------------------------

class TinyPerHeadModel(nn.Module):
    """Two-layer, two-head-per-layer model where each head is a separate Linear."""

    def __init__(self):
        super().__init__()
        self.l0h0 = nn.Linear(4, 4, bias=False)
        self.l0h1 = nn.Linear(4, 4, bias=False)
        self.l1h0 = nn.Linear(4, 4, bias=False)
        self.l1h1 = nn.Linear(4, 4, bias=False)
        self.out = nn.Linear(4, 2, bias=False)

    def forward(self, x):
        x = torch.relu(self.l0h0(x) + self.l0h1(x))
        x = torch.relu(self.l1h0(x) + self.l1h1(x))
        return self.out(x)


@pytest.fixture
def model_and_runner():
    torch.manual_seed(0)
    model = TinyPerHeadModel()
    head_modules = [[model.l0h0, model.l0h1], [model.l1h0, model.l1h1]]
    runner = HeadKnockoutRunner(model, head_modules)
    inputs = torch.randn(3, 4)
    def metric(out):
        return out.mean().item()
    result = runner.run(inputs, metric)
    return model, runner, head_modules, inputs, metric, result


# ---------------------------------------------------------------------------
# Test 1: result type is HeadKnockoutResult
# ---------------------------------------------------------------------------

def test_result_type(model_and_runner):
    *_, result = model_and_runner
    assert isinstance(result, HeadKnockoutResult)


# ---------------------------------------------------------------------------
# Test 2: importance shape is (n_layers, n_heads)
# ---------------------------------------------------------------------------

def test_importance_shape(model_and_runner):
    *_, result = model_and_runner
    # 2 layers, 2 heads each
    assert result.importance.shape == (2, 2)


# ---------------------------------------------------------------------------
# Test 3: layer_names length matches len(head_modules)
# ---------------------------------------------------------------------------

def test_layer_names_length(model_and_runner):
    _, _, head_modules, _, _, result = model_and_runner
    assert len(result.layer_names) == len(head_modules)


# ---------------------------------------------------------------------------
# Test 4: clean_score is a float
# ---------------------------------------------------------------------------

def test_clean_score_is_float(model_and_runner):
    *_, result = model_and_runner
    assert isinstance(result.clean_score, float)


# ---------------------------------------------------------------------------
# Test 5: knockout_scores shape is (n_layers, n_heads)
# ---------------------------------------------------------------------------

def test_knockout_scores_shape(model_and_runner):
    *_, result = model_and_runner
    assert result.knockout_scores.shape == (2, 2)


# ---------------------------------------------------------------------------
# Test 6: importance values are finite
# ---------------------------------------------------------------------------

def test_importance_values_finite(model_and_runner):
    *_, result = model_and_runner
    assert torch.isfinite(result.importance).all()


# ---------------------------------------------------------------------------
# Test 7: top_heads() returns list of (str, int, float) triples
# ---------------------------------------------------------------------------

def test_top_heads_returns_triples(model_and_runner):
    *_, result = model_and_runner
    tops = result.top_heads()
    assert isinstance(tops, list)
    for item in tops:
        assert len(item) == 3
        layer_name, head_idx, importance = item
        assert isinstance(layer_name, str)
        assert isinstance(head_idx, int)
        assert isinstance(importance, float)


# ---------------------------------------------------------------------------
# Test 8: top_heads(k=1) returns exactly 1 element
# ---------------------------------------------------------------------------

def test_top_heads_k1_returns_one(model_and_runner):
    *_, result = model_and_runner
    tops = result.top_heads(k=1)
    assert len(tops) == 1


# ---------------------------------------------------------------------------
# Test 9: top_heads() sorted by importance descending
# ---------------------------------------------------------------------------

def test_top_heads_sorted_descending(model_and_runner):
    *_, result = model_and_runner
    tops = result.top_heads()
    importances = [t[2] for t in tops]
    assert importances == sorted(importances, reverse=True)


# ---------------------------------------------------------------------------
# Test 10: to_markdown() returns string containing "Head Knockout"
# ---------------------------------------------------------------------------

def test_to_markdown_contains_header(model_and_runner):
    *_, result = model_and_runner
    md = result.to_markdown()
    assert isinstance(md, str)
    assert "Head Knockout" in md


# ---------------------------------------------------------------------------
# Test 11: importance = clean_score - knockout_scores elementwise
# ---------------------------------------------------------------------------

def test_importance_equals_clean_minus_knockout(model_and_runner):
    *_, result = model_and_runner
    expected = result.clean_score - result.knockout_scores
    assert torch.allclose(result.importance, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# Test 12: custom layer_names are stored correctly
# ---------------------------------------------------------------------------

def test_custom_layer_names_stored():
    torch.manual_seed(1)
    model = TinyPerHeadModel()
    head_modules = [[model.l0h0, model.l0h1], [model.l1h0, model.l1h1]]
    custom_names = ["layer_zero", "layer_one"]
    runner = HeadKnockoutRunner(model, head_modules, layer_names=custom_names)
    inputs = torch.randn(3, 4)
    def metric(out):
        return out.mean().item()
    result = runner.run(inputs, metric)
    assert result.layer_names == custom_names
