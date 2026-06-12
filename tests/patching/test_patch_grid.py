"""Tests for circuitry.patching.patch_grid: PatchGridResult and PatchGridRunner."""

import pytest
import torch
import torch.nn as nn

from circuitry.patching.patch_grid import PatchGridResult, PatchGridRunner

# ---------------------------------------------------------------------------
# Toy model
# ---------------------------------------------------------------------------

class TinySeqModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(4, 4, bias=False)
        self.l2 = nn.Linear(4, 4, bias=False)

    def forward(self, x):  # x: (batch, seq, 4)
        x = torch.relu(self.l1(x))
        return self.l2(x)  # (batch, seq, 4)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def model():
    torch.manual_seed(0)
    m = TinySeqModel()
    m.eval()
    return m


@pytest.fixture
def inputs(model):
    torch.manual_seed(1)
    clean = torch.randn(2, 5, 4)
    corrupted = torch.randn(2, 5, 4)
    return clean, corrupted


def metric(out):
    return out.mean().item()


# ---------------------------------------------------------------------------
# Helper: build a result using explicit modules list
# ---------------------------------------------------------------------------

def _run(model, inputs):
    clean, corrupted = inputs
    runner = PatchGridRunner(model, modules=[model.l1, model.l2])
    return runner.run(clean, corrupted, metric)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_result_type(model, inputs):
    """run() returns a PatchGridResult instance."""
    result = _run(model, inputs)
    assert isinstance(result, PatchGridResult)


def test_recovery_shape(model, inputs):
    """recovery tensor has shape (n_layers, seq_len) — 2 modules, seq_len=5."""
    result = _run(model, inputs)
    assert result.recovery.shape == (2, 5), (
        f"Expected (2, 5), got {result.recovery.shape}"
    )


def test_layer_names_length(model, inputs):
    """layer_names length matches number of modules."""
    result = _run(model, inputs)
    assert len(result.layer_names) == 2


def test_scores_are_floats(model, inputs):
    """clean_score and corrupted_score are Python floats."""
    result = _run(model, inputs)
    assert isinstance(result.clean_score, float)
    assert isinstance(result.corrupted_score, float)


def test_recovery_finite(model, inputs):
    """All recovery values are finite."""
    result = _run(model, inputs)
    assert torch.isfinite(result.recovery).all(), "recovery contains non-finite values"


def test_top_sites_returns_triples(model, inputs):
    """top_sites() returns a list of (str, int, float) triples."""
    result = _run(model, inputs)
    sites = result.top_sites()
    assert isinstance(sites, list)
    for item in sites:
        assert len(item) == 3, f"Expected 3-tuple, got {item}"
        layer_name, pos, score = item
        assert isinstance(layer_name, str), f"layer_name must be str, got {type(layer_name)}"
        assert isinstance(pos, int), f"pos must be int, got {type(pos)}"
        assert isinstance(score, float), f"score must be float, got {type(score)}"


def test_top_sites_k1(model, inputs):
    """top_sites(k=1) returns exactly 1 element."""
    result = _run(model, inputs)
    sites = result.top_sites(k=1)
    assert len(sites) == 1


def test_top_sites_sorted_descending(model, inputs):
    """top_sites() is sorted by score descending."""
    result = _run(model, inputs)
    sites = result.top_sites(k=5)
    scores = [s for _, _, s in sites]
    assert scores == sorted(scores, reverse=True), (
        f"top_sites not sorted descending: {scores}"
    )


def test_to_markdown_contains_header(model, inputs):
    """to_markdown() returns a string containing 'Patch Grid'."""
    result = _run(model, inputs)
    md = result.to_markdown()
    assert isinstance(md, str)
    assert "Patch Grid" in md, f"'Patch Grid' not found in markdown output:\n{md[:300]}"


def test_raises_without_modules_or_pattern(model):
    """Raises ValueError if neither modules nor module_pattern is provided."""
    with pytest.raises(ValueError):
        PatchGridRunner(model)


def test_raises_with_both_modules_and_pattern(model):
    """Raises ValueError if both modules and module_pattern are provided."""
    with pytest.raises(ValueError):
        PatchGridRunner(model, modules=[model.l1], module_pattern=r"l[12]")


def test_module_pattern_matches_layers(model, inputs):
    """module_pattern='l[12]' matches both l1 and l2 layers."""
    clean, corrupted = inputs
    runner = PatchGridRunner(model, module_pattern=r"l[12]")
    result = runner.run(clean, corrupted, metric)
    assert isinstance(result, PatchGridResult)
    # Should have matched 2 layers
    assert result.recovery.shape[0] == 2, (
        f"Expected 2 matched layers, got {result.recovery.shape[0]}"
    )
