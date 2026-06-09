"""Tests for circuitry.core.neuron — NeuronStats and neuron_stats."""

import pytest
import torch

try:
    from circuitry.core.neuron import NeuronStats, neuron_stats
    IMPORT_OK = True
except ImportError:
    IMPORT_OK = False
    NeuronStats = None
    neuron_stats = None

pytestmark = pytest.mark.skipif(not IMPORT_OK, reason="circuitry.core.neuron not available")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def acts_3d():
    """Shape (batch=4, seq=6, d=8) — a typical 3-D activation tensor."""
    torch.manual_seed(42)
    return torch.randn(4, 6, 8)


@pytest.fixture
def acts_2d():
    """Shape (batch=4, d=8) — a 2-D activation tensor without a seq dimension."""
    torch.manual_seed(42)
    return torch.randn(4, 8)


# ---------------------------------------------------------------------------
# Test 1: result type is NeuronStats
# ---------------------------------------------------------------------------

def test_result_type(acts_3d):
    result = neuron_stats(acts_3d)
    assert isinstance(result, NeuronStats)


# ---------------------------------------------------------------------------
# Test 2: mean shape is (d,) for (batch, seq, d) input
# ---------------------------------------------------------------------------

def test_mean_shape_3d(acts_3d):
    result = neuron_stats(acts_3d)
    d = acts_3d.shape[-1]
    assert result.mean.shape == (d,)


# ---------------------------------------------------------------------------
# Test 3: std shape is (d,)
# ---------------------------------------------------------------------------

def test_std_shape_3d(acts_3d):
    result = neuron_stats(acts_3d)
    d = acts_3d.shape[-1]
    assert result.std.shape == (d,)


# ---------------------------------------------------------------------------
# Test 4: max shape is (d,)
# ---------------------------------------------------------------------------

def test_max_shape_3d(acts_3d):
    result = neuron_stats(acts_3d)
    d = acts_3d.shape[-1]
    assert result.max.shape == (d,)


# ---------------------------------------------------------------------------
# Test 5: dead_fraction is a float in [0, 1]
# ---------------------------------------------------------------------------

def test_dead_fraction_is_float_in_range(acts_3d):
    result = neuron_stats(acts_3d)
    assert isinstance(result.dead_fraction, float)
    assert 0.0 <= result.dead_fraction <= 1.0


# ---------------------------------------------------------------------------
# Test 6: kurtosis shape is (d,)
# ---------------------------------------------------------------------------

def test_kurtosis_shape_3d(acts_3d):
    result = neuron_stats(acts_3d)
    d = acts_3d.shape[-1]
    assert result.kurtosis.shape == (d,)


# ---------------------------------------------------------------------------
# Test 7: dead_fraction = 1.0 when all activations are negative (threshold=0.0)
# ---------------------------------------------------------------------------

def test_dead_fraction_all_negative():
    acts = -torch.ones(4, 6, 8)  # all negative
    result = neuron_stats(acts, threshold=0.0)
    assert result.dead_fraction == 1.0


# ---------------------------------------------------------------------------
# Test 8: dead_fraction = 0.0 when all activations are positive
# ---------------------------------------------------------------------------

def test_dead_fraction_all_positive():
    acts = torch.ones(4, 6, 8)  # all positive
    result = neuron_stats(acts, threshold=0.0)
    assert result.dead_fraction == 0.0


# ---------------------------------------------------------------------------
# Test 9: mean values correct for constant activations
# ---------------------------------------------------------------------------

def test_mean_correct_for_constant():
    acts = torch.full((4, 6, 8), 3.0)  # constant 3.0
    result = neuron_stats(acts)
    expected = torch.full((8,), 3.0)
    assert torch.allclose(result.mean, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# Test 10: std = 0 for constant activations
# ---------------------------------------------------------------------------

def test_std_zero_for_constant():
    acts = torch.full((4, 6, 8), 3.0)
    result = neuron_stats(acts)
    assert torch.allclose(result.std, torch.zeros(8), atol=1e-6)


# ---------------------------------------------------------------------------
# Test 11: max is correct — equals max of acts along non-feature dims
# ---------------------------------------------------------------------------

def test_max_correct(acts_3d):
    result = neuron_stats(acts_3d)
    # Flatten leading dims, take max over them
    flat = acts_3d.reshape(-1, acts_3d.shape[-1])
    expected_max = flat.max(dim=0).values
    assert torch.allclose(result.max, expected_max, atol=1e-6)


# ---------------------------------------------------------------------------
# Test 12: works with 2D input (batch, d) — no seq dimension
# ---------------------------------------------------------------------------

def test_works_with_2d_input(acts_2d):
    result = neuron_stats(acts_2d)
    d = acts_2d.shape[-1]
    assert result.mean.shape == (d,)
    assert result.std.shape == (d,)
    assert result.max.shape == (d,)
    assert result.kurtosis.shape == (d,)
    assert isinstance(result.dead_fraction, float)
