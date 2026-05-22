# tests/recipes/test_recipe_with_prefix.py
"""Tests for Recipe.with_prefix() — v0.7.0 H2 feature."""
from __future__ import annotations

import pytest

from circuitry.recipes import Recipe, _clear_registry_for_tests
from circuitry.recorder.hooks import HookPoint, TensorSource


@pytest.fixture(autouse=True)
def _clean():
    _clear_registry_for_tests()
    yield
    _clear_registry_for_tests()


def _base_recipe(name: str = "base") -> Recipe:
    return Recipe(
        name=name,
        hook_points=[HookPoint(source=TensorSource.WEIGHT, pattern=r".*")],
        weight_diagnostics=["effective_rank"],
    )


def test_with_prefix_returns_new_instance():
    """with_prefix must not mutate self."""
    r = _base_recipe()
    r2 = r.with_prefix("model.language_model")
    assert r is not r2
    assert r.module_prefix is None  # original unchanged
    assert r2.module_prefix == "model.language_model"


def test_with_prefix_renames_name():
    """New recipe name is `<original>@<prefix>`."""
    r = _base_recipe("llm")
    r2 = r.with_prefix("model.language_model")
    assert r2.name == "llm@model.language_model"
    assert r.name == "llm"  # original unchanged


def test_with_prefix_sets_module_prefix():
    r = _base_recipe()
    r2 = r.with_prefix("model.text")
    assert r2.module_prefix == "model.text"


def test_with_prefix_latest_wins():
    """Calling with_prefix twice: last prefix wins, not concatenated."""
    r = _base_recipe("base")
    r2 = r.with_prefix("a").with_prefix("b")
    assert r2.module_prefix == "b"


def test_with_prefix_preserves_hook_points_and_diagnostics():
    r = _base_recipe()
    r2 = r.with_prefix("foo")
    assert r2.hook_points == r.hook_points
    assert r2.weight_diagnostics == r.weight_diagnostics


def test_module_prefix_default_is_none():
    r = _base_recipe()
    assert r.module_prefix is None


def test_with_prefix_idempotency_name_accumulates():
    """Name accumulates @prefixes on double call — this is the documented behaviour."""
    r = _base_recipe("base")
    r2 = r.with_prefix("a")
    r3 = r2.with_prefix("b")
    # module_prefix is the last one
    assert r3.module_prefix == "b"
    # name has both — this is by design (latest-wins on prefix, name reflects history)
    assert r3.name == "base@a@b"
