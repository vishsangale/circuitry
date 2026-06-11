"""Recorder wiring tests for the opt-in ``moe_routing`` diagnostic (v1.44)."""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from circuitry.recipes import Recipe, _clear_registry_for_tests, register_recipe
from circuitry.recorder.hooks import HookPoint, TensorSource
from circuitry.recorder.live import Recorder
from circuitry.writers.base import RecordingWriter

N_EXPERTS = 4
D = 8


@pytest.fixture(autouse=True)
def _clean():
    _clear_registry_for_tests()
    yield
    _clear_registry_for_tests()


class _MLP(nn.Module):
    """MoE-shaped MLP: ``.gate`` emits router logits (..., n_experts)."""

    def __init__(self):
        super().__init__()
        self.gate = nn.Linear(D, N_EXPERTS)
        self.proj = nn.Linear(D, D)

    def forward(self, x):
        _router_logits = self.gate(x)
        return self.proj(x)


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = _MLP()

    def forward(self, x):
        return self.mlp(x)


class TinyMoE(nn.Module):
    def __init__(self, n_layers: int = 2):
        super().__init__()
        self.layers = nn.ModuleList(_Block() for _ in range(n_layers))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def _recipe(name: str = "moe-test", **overrides) -> Recipe:
    return Recipe(
        name=name,
        hook_points=[
            HookPoint(source=TensorSource.OUTPUT, pattern=r".*\.mlp\.gate$"),
            HookPoint(source=TensorSource.OUTPUT, pattern=r".*\.mlp\.proj$"),
        ],
        activation_diagnostics=["moe_routing"],
        **overrides,
    )


def _run(recipe: Recipe, tmp_path) -> RecordingWriter:
    register_recipe(recipe)
    torch.manual_seed(0)
    model = TinyMoE()
    writer = RecordingWriter()
    rec = Recorder(model, run_dir=tmp_path, recipe=recipe.name,
                   writer=writer, every_n_steps=1)
    rec.attach()
    _ = model(torch.randn(3, 5, D))
    rec.step(0)
    rec.detach()
    return writer


def test_emits_per_router_entropy_and_balance(tmp_path):
    writer = _run(_recipe(), tmp_path)
    tags = {t for t, _, _ in writer.scalars}
    for layer in (0, 1):
        assert f"moe/routing_entropy/layers.{layer}.mlp.gate" in tags
        assert f"moe/expert_load_balance/layers.{layer}.mlp.gate" in tags


def test_emits_pathway_complexity_across_layers(tmp_path):
    writer = _run(_recipe(), tmp_path)
    vals = [v for t, v, _ in writer.scalars if t == "moe/pathway_complexity"]
    assert len(vals) == 1
    assert vals[0] >= 1.0


def test_values_in_valid_ranges(tmp_path):
    import math

    writer = _run(_recipe(), tmp_path)
    for t, v, _ in writer.scalars:
        if t.startswith("moe/routing_entropy/"):
            assert 0.0 <= v <= math.log(N_EXPERTS) + 1e-6
        if t.startswith("moe/expert_load_balance/"):
            assert 0.0 < v <= 1.0


def test_non_router_activations_ignored(tmp_path):
    # .mlp.proj outputs are captured by a hook but must not produce moe/ tags
    # (they don't match moe_router_pattern).
    writer = _run(_recipe(), tmp_path)
    assert not any("proj" in t for t, _, _ in writer.scalars if t.startswith("moe/"))


def test_disabled_emits_nothing(tmp_path):
    writer = _run(_recipe(enabled={"moe_routing": False}), tmp_path)
    assert not any(t.startswith("moe/") for t, _, _ in writer.scalars)


def test_custom_router_pattern(tmp_path):
    # Point the pattern at proj instead — proj outputs (B, T, D=8) become the
    # "router"; gate is ignored.
    writer = _run(_recipe(moe_router_pattern=r".*\.mlp\.proj$"), tmp_path)
    tags = {t for t, _, _ in writer.scalars if t.startswith("moe/")}
    assert any("proj" in t for t in tags)
    assert not any("gate" in t for t in tags)


def test_llm_recipe_has_moe_routing_disabled_by_default():
    from circuitry.recipes.llm import RECIPE

    assert "moe_routing" in RECIPE.activation_diagnostics
    assert RECIPE.enabled.get("moe_routing") is False
    assert any(
        hp.pattern == r".*\.mlp\.gate$" and hp.source == TensorSource.OUTPUT
        for hp in RECIPE.hook_points
    )
