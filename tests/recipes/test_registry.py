from __future__ import annotations

import pytest

from circuitry.recipes import Recipe, get_recipe, list_recipes, register_recipe, _clear_registry_for_tests
from circuitry.recorder.hooks import HookPoint, TensorSource


@pytest.fixture(autouse=True)
def _clean_registry():
    _clear_registry_for_tests()
    yield
    _clear_registry_for_tests()


def _make(name: str = "demo") -> Recipe:
    return Recipe(
        name=name,
        hook_points=[HookPoint(source=TensorSource.WEIGHT, pattern=r".*")],
        weight_diagnostics=["effective_rank"],
        activation_diagnostics=[],
        gradient_diagnostics=[],
    )


def test_register_and_get_round_trip():
    register_recipe(_make("custom-a"))
    r = get_recipe("custom-a")
    assert r.name == "custom-a"
    assert "custom-a" in list_recipes()


def test_register_duplicate_raises():
    register_recipe(_make("custom-b"))
    with pytest.raises(ValueError):
        register_recipe(_make("custom-b"))


def test_get_unknown_raises():
    with pytest.raises(KeyError):
        get_recipe("nonexistent-recipe-xyz")


def test_recipe_diagnostic_toggle_enabled_default_true():
    r = _make("custom-c")
    # Per-diagnostic toggle (design §10) — represented as a dict on the Recipe.
    assert r.enabled.get("effective_rank", True) is True
