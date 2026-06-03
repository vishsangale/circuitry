"""Tests for Recipe.disable() and Recipe.only() helpers."""
from __future__ import annotations

import pytest

from circuitry.recipes import _clear_registry_for_tests, get_recipe
from circuitry.recipes.llm import register


@pytest.fixture(autouse=True)
def _clean():
    _clear_registry_for_tests()
    register()
    yield
    _clear_registry_for_tests()


def test_disable_sets_enabled_false():
    r = get_recipe("llm")
    r2 = r.disable(["effective_rank"])
    assert r2.enabled.get("effective_rank") is False
    # Other names should not appear as disabled (absent = default True via _enabled).
    assert r2.enabled.get("stable_rank", True) is True


def test_disable_empty_is_noop():
    r = get_recipe("llm")
    r2 = r.disable([])
    assert r2.enabled == r.enabled


def test_disable_unknown_raises():
    r = get_recipe("llm")
    with pytest.raises(ValueError, match="nonexistent_diag"):
        r.disable(["nonexistent_diag"])


def test_only_disables_complement():
    r = get_recipe("llm")
    r2 = r.only(["effective_rank"])
    assert r2.enabled.get("effective_rank") is True
    # Every other diagnostic in the three lists must be False.
    _all = set(r.weight_diagnostics + r.activation_diagnostics + r.gradient_diagnostics)
    for name in _all:
        if name != "effective_rank":
            assert r2.enabled.get(name) is False, f"{name} should be disabled"


def test_only_unknown_raises():
    r = get_recipe("llm")
    with pytest.raises(ValueError, match="no_such"):
        r.only(["no_such"])


def test_effective_diagnostics_reflects_disable():
    """effective_diagnostics() drops disabled names; the raw lists are unchanged
    (the whole point — inspecting the lists after .disable() is misleading)."""
    r = get_recipe("llm")
    r2 = r.disable(["effective_rank"])
    eff = r2.effective_diagnostics()
    assert "effective_rank" not in eff["weight"]
    assert "stable_rank" in eff["weight"]  # still active
    # Raw list is untouched.
    assert "effective_rank" in r2.weight_diagnostics


def test_effective_diagnostics_default_all_enabled():
    """With no enabled overrides, effective == declared (per family).

    (Uses a clean Recipe — the stock llm recipe disables drift_probe by default,
    so effective != declared there.)"""
    from circuitry.recipes import Recipe

    r = Recipe(
        name="t", hook_points=[],
        weight_diagnostics=["a", "b"],
        activation_diagnostics=["c"],
        gradient_diagnostics=["d"],
    )
    assert r.effective_diagnostics() == {
        "weight": ["a", "b"], "activation": ["c"], "gradient": ["d"],
    }
    assert r.active_diagnostics == ["a", "b", "c", "d"]


def test_active_diagnostics_flat_and_only():
    r = get_recipe("llm")
    r2 = r.only(["effective_rank", "norms_per_param"])
    active = r2.active_diagnostics
    assert set(active) == {"effective_rank", "norms_per_param"}
    # Flat list spans families (weight + gradient here).
    assert "effective_rank" in active and "norms_per_param" in active


def test_only_empty_disables_all():
    r = get_recipe("llm")
    r2 = r.only([])
    _all = set(r.weight_diagnostics + r.activation_diagnostics + r.gradient_diagnostics)
    for name in _all:
        assert r2.enabled.get(name) is False


def test_disable_then_only_composes():
    """Latest-wins: disable heavy_tail_alpha first, then only(["effective_rank"])
    should result in effective_rank=True and everything else False."""
    r = get_recipe("llm")
    r2 = r.disable(["heavy_tail_alpha"]).only(["effective_rank"])
    assert r2.enabled.get("effective_rank") is True
    _all = set(r.weight_diagnostics + r.activation_diagnostics + r.gradient_diagnostics)
    for name in _all:
        if name != "effective_rank":
            assert r2.enabled.get(name) is False


def test_custom_diagnostics_unaffected_by_disable(tmp_path):
    """Custom DiagnosticFn still fires even after disable() on all named diagnostics."""
    import torch
    import torch.nn as nn

    from circuitry import Recorder
    from circuitry.recipes import Recipe
    from circuitry.recorder.hooks import HookPoint, TensorSource
    from circuitry.writers.null import NullWriter

    fired = []

    def _custom(ctx):
        fired.append(ctx.step)
        return {"custom_val": 1.0}

    recipe = Recipe(
        name="__test_custom__",
        hook_points=[HookPoint(pattern=r".*", source=TensorSource.WEIGHT)],
        weight_diagnostics=["effective_rank"],
        custom=[_custom],
    ).disable(["effective_rank"])

    model = nn.Linear(4, 4)
    rec = Recorder(model, run_dir=tmp_path, recipe=recipe, writer=NullWriter(), every_n_steps=1)
    rec.attach()
    with torch.no_grad():
        _ = model(torch.randn(2, 4))
    rec.step(0)
    rec.detach()
    assert fired, "custom DiagnosticFn must fire even after disable()"
