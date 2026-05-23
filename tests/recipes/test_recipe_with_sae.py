"""Tests for Recipe.with_sae and the new sae_checkpoints /
induction_probe_seq_len fields. Spec §4.4."""
from __future__ import annotations

from circuitry.recipes import Recipe
from circuitry.recorder.hooks import HookPoint, TensorSource


def _bare_recipe(name: str = "test") -> Recipe:
    return Recipe(
        name=name,
        hook_points=[HookPoint(source=TensorSource.OUTPUT, pattern=r".*\.mlp$")],
        activation_diagnostics=["dead_fraction"],
    )


def test_default_sae_checkpoints_is_none():
    r = _bare_recipe()
    assert r.sae_checkpoints is None


def test_default_induction_probe_seq_len_is_25():
    r = _bare_recipe()
    assert r.induction_probe_seq_len == 25


def test_with_sae_sets_checkpoints_and_renames():
    r = _bare_recipe("llm")
    mapping = {r".*\.layers\.8$": ("release-x", "id-y")}
    r2 = r.with_sae(mapping)
    assert r2.sae_checkpoints == mapping
    # Original recipe unchanged (dataclass replace returns a copy).
    assert r.sae_checkpoints is None


def test_with_sae_does_not_modify_activation_diagnostics():
    """Spec §4.4: with_sae loads checkpoints but does NOT auto-append
    'sae_reconstruction'. User must opt in explicitly."""
    r = _bare_recipe()
    r2 = r.with_sae({r".*\.mlp$": ("r", "i")})
    assert "sae_reconstruction" not in r2.activation_diagnostics


def test_with_sae_latest_wins_idempotent():
    r = _bare_recipe()
    r1 = r.with_sae({r"a$": ("r1", "i1")})
    r2 = r1.with_sae({r"b$": ("r2", "i2")})
    assert r2.sae_checkpoints == {r"b$": ("r2", "i2")}  # not merged


def test_with_sae_composes_with_with_prefix():
    """Spec §4.4: prefix first, then SAE patterns match against prefixed names."""
    r = _bare_recipe("llm")
    r2 = r.with_prefix("model.language_model").with_sae(
        {r".*\.layers\.8$": ("rel", "id")}
    )
    assert r2.module_prefix == "model.language_model"
    assert r2.sae_checkpoints == {r".*\.layers\.8$": ("rel", "id")}
    # name carries the prefix annotation
    assert "@model.language_model" in r2.name
