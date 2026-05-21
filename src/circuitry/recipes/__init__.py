"""Recipe dataclass + global registry. See docs/design.md §4.4."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from circuitry.recorder.hooks import HookPoint, StepContext

DiagnosticFn = Callable[[StepContext], dict[str, float]]


@dataclass
class Recipe:
    name: str
    hook_points: list[HookPoint]
    weight_diagnostics: list[str] = field(default_factory=list)
    activation_diagnostics: list[str] = field(default_factory=list)
    gradient_diagnostics: list[str] = field(default_factory=list)
    custom: list[DiagnosticFn] = field(default_factory=list)
    expected_min_matches: dict[str, int] = field(default_factory=dict)
    enabled: dict[str, bool] = field(default_factory=dict)


_REGISTRY: dict[str, Recipe] = {}


def register_recipe(recipe: Recipe) -> None:
    if recipe.name in _REGISTRY:
        raise ValueError(f"recipe {recipe.name!r} already registered")
    _REGISTRY[recipe.name] = recipe


def get_recipe(name: str) -> Recipe:
    if name not in _REGISTRY:
        raise KeyError(f"unknown recipe {name!r}; known: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def list_recipes() -> list[str]:
    return sorted(_REGISTRY)


def _clear_registry_for_tests() -> None:
    """Test-only escape hatch. Not part of the public API."""
    _REGISTRY.clear()
