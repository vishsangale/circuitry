"""Tests for circuitry.patching.iti (Inference-Time Intervention)."""
from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from circuitry.patching.iti import ITIConfig, apply_iti, fit_iti


# ---------------------------------------------------------------------------
# Toy model helpers
# ---------------------------------------------------------------------------

class _AttentionLike(nn.Module):
    """Simulates a single-layer attention output: Linear(d_in -> n_heads * d_head)."""

    def __init__(self, d_in: int, n_heads: int, d_head: int):
        super().__init__()
        self.proj = nn.Linear(d_in, n_heads * d_head, bias=False)
        self.n_heads = n_heads
        self.d_head = d_head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class _SingleLayerModel(nn.Module):
    """Wraps one _AttentionLike as self.layers[0] for easy hook attachment."""

    def __init__(self, d_in: int, n_heads: int, d_head: int):
        super().__init__()
        self.layers = nn.ModuleList([_AttentionLike(d_in, n_heads, d_head)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers[0](x)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def sample_head_acts():
    """Two heads (layer 0 head 0, layer 0 head 1), 20 samples, d_head=4."""
    torch.manual_seed(0)
    n, d_head = 20, 4
    labels = torch.cat([torch.zeros(10, dtype=torch.long), torch.ones(10, dtype=torch.long)])
    acts = {
        (0, 0): torch.randn(n, d_head),
        (0, 1): torch.randn(n, d_head),
    }
    return acts, labels, d_head


@pytest.fixture()
def toy_model():
    torch.manual_seed(1)
    return _SingleLayerModel(d_in=8, n_heads=2, d_head=4)


# ---------------------------------------------------------------------------
# fit_iti tests
# ---------------------------------------------------------------------------

def test_fit_iti_returns_iti_config(sample_head_acts):
    acts, labels, _ = sample_head_acts
    result = fit_iti(acts, labels)
    assert isinstance(result, ITIConfig)


def test_fit_iti_directions_are_unit_vectors(sample_head_acts):
    acts, labels, _ = sample_head_acts
    config = fit_iti(acts, labels)
    for key, direction in config.head_directions.items():
        norm = direction.norm().item()
        assert abs(norm - 1.0) < 1e-5, f"Direction for {key} has norm {norm}, expected ≈ 1"


def test_fit_iti_d_head_inferred(sample_head_acts):
    acts, labels, d_head = sample_head_acts
    config = fit_iti(acts, labels)
    assert config.d_head == d_head


def test_fit_iti_empty_raises():
    labels = torch.zeros(5, dtype=torch.long)
    with pytest.raises(ValueError, match="head_acts must not be empty"):
        fit_iti({}, labels)


def test_fit_iti_stores_all_heads(sample_head_acts):
    acts, labels, _ = sample_head_acts
    config = fit_iti(acts, labels)
    assert set(config.head_directions.keys()) == {(0, 0), (0, 1)}


def test_fit_iti_custom_coeff(sample_head_acts):
    acts, labels, _ = sample_head_acts
    config = fit_iti(acts, labels, coeff=7.5)
    assert config.coeff == 7.5


# ---------------------------------------------------------------------------
# apply_iti tests
# ---------------------------------------------------------------------------

def test_apply_iti_steers_output(toy_model):
    """Output slice for head 0 changes by exactly coeff * direction."""
    torch.manual_seed(42)
    d_in, d_head, n_heads = 8, 4, 2
    x = torch.randn(1, d_in)

    # Build a known unit direction
    direction = torch.zeros(d_head)
    direction[0] = 1.0  # already unit norm

    coeff = 15.0
    config = ITIConfig(
        head_directions={(0, 0): direction},
        d_head=d_head,
        coeff=coeff,
    )

    attn_modules = {0: toy_model.layers[0]}

    # Baseline output without steering
    with torch.no_grad():
        baseline = toy_model(x).clone()

    # Steered output
    with apply_iti(toy_model, config, attn_modules=attn_modules):
        with torch.no_grad():
            steered = toy_model(x).clone()

    # Head 0 slice: positions 0..d_head
    delta = steered[..., :d_head] - baseline[..., :d_head]
    expected = coeff * direction.to(dtype=delta.dtype)
    assert torch.allclose(delta, expected.expand_as(delta), atol=1e-5), (
        f"Expected delta ≈ {expected}, got {delta}"
    )

    # Head 1 slice should be unchanged
    delta_head1 = steered[..., d_head:] - baseline[..., d_head:]
    assert torch.allclose(delta_head1, torch.zeros_like(delta_head1), atol=1e-5)


def test_apply_iti_hook_removed_after(toy_model):
    """After the context manager exits, output is identical to pre-hook output."""
    torch.manual_seed(0)
    d_head = 4
    direction = torch.randn(d_head)
    direction = direction / direction.norm()

    config = ITIConfig(
        head_directions={(0, 0): direction},
        d_head=d_head,
        coeff=15.0,
    )

    attn_modules = {0: toy_model.layers[0]}
    x = torch.randn(1, 8)

    with torch.no_grad():
        before = toy_model(x).clone()

    with apply_iti(toy_model, config, attn_modules=attn_modules):
        pass  # enter and immediately exit

    with torch.no_grad():
        after = toy_model(x).clone()

    assert torch.allclose(before, after, atol=1e-6), "Hook still active after context exit"


def test_apply_iti_model_back_to_eval(toy_model):
    """model.training is False after apply_iti when model was already in eval mode."""
    toy_model.eval()
    d_head = 4
    direction = torch.zeros(d_head)
    direction[0] = 1.0

    config = ITIConfig(head_directions={(0, 0): direction}, d_head=d_head, coeff=1.0)
    attn_modules = {0: toy_model.layers[0]}

    with apply_iti(toy_model, config, attn_modules=attn_modules):
        pass

    assert not toy_model.training, "model.training should be False after apply_iti"


def test_apply_iti_model_restores_train_mode(toy_model):
    """model.training is restored to True when model was in train mode before apply_iti."""
    toy_model.train()
    d_head = 4
    direction = torch.zeros(d_head)
    direction[0] = 1.0

    config = ITIConfig(head_directions={(0, 0): direction}, d_head=d_head, coeff=1.0)
    attn_modules = {0: toy_model.layers[0]}

    with apply_iti(toy_model, config, attn_modules=attn_modules):
        assert not toy_model.training, "should be eval inside context"

    assert toy_model.training, "model.training should be restored to True after apply_iti"


def test_apply_iti_no_attn_modules_no_config_raises():
    """apply_iti with no attn_modules= and no model.config raises ValueError."""
    model = nn.Linear(4, 4)  # plain module, no .config attribute
    direction = torch.zeros(4)
    direction[0] = 1.0
    config = ITIConfig(head_directions={(0, 0): direction}, d_head=4, coeff=1.0)

    with pytest.raises(ValueError, match="apply_iti"):
        with apply_iti(model, config):
            pass


def test_iti_coeff_zero_no_change(toy_model):
    """coeff=0 gives identical output to baseline."""
    torch.manual_seed(5)
    d_head = 4
    direction = torch.randn(d_head)
    direction = direction / direction.norm()

    config = ITIConfig(
        head_directions={(0, 0): direction},
        d_head=d_head,
        coeff=0.0,
    )
    attn_modules = {0: toy_model.layers[0]}
    x = torch.randn(1, 8)

    with torch.no_grad():
        baseline = toy_model(x).clone()

    with apply_iti(toy_model, config, attn_modules=attn_modules):
        with torch.no_grad():
            steered = toy_model(x).clone()

    assert torch.allclose(baseline, steered, atol=1e-6), (
        "coeff=0 should give zero delta"
    )
