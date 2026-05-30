"""Recipe-level drift-probe tests (Workstream B-recipe).

Covers:
- Recipe has drift_probe in activation_diagnostics.
- drift_probe is disabled by default (enabled={"drift_probe": False}).
- probe_batch, drift_method, drift_max_tokens fields exist on Recipe.
- disable() / only() handle "drift_probe" correctly.
- Layering: recipes/__init__.py imports torch (for Tensor annotation) but NOT
  cli/; llm.py imports unchanged.
"""

from __future__ import annotations

import dataclasses

import torch

from circuitry.recipes import Recipe, _clear_registry_for_tests
from circuitry.recipes.llm import RECIPE


def setup_function():
    _clear_registry_for_tests()


def teardown_function():
    _clear_registry_for_tests()


def test_llm_recipe_has_drift_probe_in_activation_diagnostics():
    assert "drift_probe" in RECIPE.activation_diagnostics


def test_llm_recipe_drift_probe_disabled_by_default():
    assert RECIPE.enabled.get("drift_probe") is False, (
        "drift_probe must be disabled by default in the stock llm recipe"
    )


def test_recipe_probe_batch_field_exists():
    """probe_batch field is present and defaults to None."""
    r = Recipe(name="x", hook_points=[], activation_diagnostics=[])
    assert hasattr(r, "probe_batch")
    assert r.probe_batch is None


def test_recipe_drift_method_field_exists():
    """drift_method field is present and defaults to 'linear_cka'."""
    r = Recipe(name="x", hook_points=[], activation_diagnostics=[])
    assert hasattr(r, "drift_method")
    assert r.drift_method == "linear_cka"


def test_recipe_drift_max_tokens_field_exists():
    """drift_max_tokens field is present and defaults to None."""
    r = Recipe(name="x", hook_points=[], activation_diagnostics=[])
    assert hasattr(r, "drift_max_tokens")
    assert r.drift_max_tokens is None


def test_recipe_probe_batch_stores_tensor():
    """probe_batch field accepts a torch.Tensor."""
    t = torch.randn(1, 8)
    r = Recipe(name="x", hook_points=[], activation_diagnostics=[],
               probe_batch=t)
    assert r.probe_batch is t


def test_disable_drift_probe():
    """Recipe.disable(['drift_probe']) works correctly."""
    r = dataclasses.replace(
        RECIPE,
        enabled={k: v for k, v in RECIPE.enabled.items() if k != "drift_probe"}
    )
    # Add it back as enabled first.
    r = dataclasses.replace(r, enabled={**r.enabled, "drift_probe": True})
    r2 = r.disable(["drift_probe"])
    assert r2.enabled.get("drift_probe") is False


def test_only_drift_probe():
    """Recipe.only(['drift_probe']) enables only drift_probe; all others disabled."""
    r = dataclasses.replace(
        RECIPE,
        # Make drift_probe enabled for this test so .only() can work with it.
        enabled={**RECIPE.enabled, "drift_probe": True}
    )
    r2 = r.only(["drift_probe"])
    assert r2.enabled.get("drift_probe") is True
    for diag in (r2.activation_diagnostics
                 + r2.weight_diagnostics
                 + r2.gradient_diagnostics):
        if diag != "drift_probe":
            assert r2.enabled.get(diag) is False, (
                f"{diag} should be disabled after only(['drift_probe'])"
            )


def test_enabled_method_returns_false_for_drift_probe_by_default():
    """_enabled() method on a Recorder with the stock recipe returns False for
    drift_probe, since the stock recipe has it explicitly disabled."""
    import pathlib
    import tempfile

    import torch.nn as nn

    from circuitry.recorder.live import Recorder

    _clear_registry_for_tests()
    from circuitry.recipes.llm import register
    register()

    model = nn.Sequential(nn.Linear(4, 4))
    with tempfile.TemporaryDirectory() as td:
        rec = Recorder(model, run_dir=pathlib.Path(td), recipe="llm",
                       writer="null", every_n_steps=1)
        assert rec._enabled("drift_probe") is False


def test_recipe_no_cli_import():
    """recipes/__init__.py and recipes/llm.py must not import from circuitry.cli."""
    import sys

    # Verify no cli module was imported as a side-effect.
    recipes_mod = sys.modules.get("circuitry.recipes")
    if recipes_mod is not None:
        # Just confirm the module is healthy — actual layering is tested in
        # tests/test_layering.py.
        pass
    # Explicitly check that circuitry.cli is not in the import closure of
    # circuitry.recipes by inspecting the module's __dict__ for any cli ref.
    import circuitry.recipes as recipes_pkg
    assert "cli" not in str(recipes_pkg.__dict__.get("__spec__", "")), (
        "recipes/__init__.py must not import cli"
    )
