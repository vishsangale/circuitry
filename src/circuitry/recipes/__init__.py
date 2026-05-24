"""Recipe dataclass + global registry. See docs/design.md §4.4."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from dataclasses import dataclass, field

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
    module_prefix: str | None = None
    sae_checkpoints: dict[str, tuple[str, str]] | None = None
    induction_probe_seq_len: int = 25
    lens_max_tokens: int | None = None

    def with_prefix(self, prefix: str) -> Recipe:
        """Return a new Recipe scoped to ``prefix``.

        Matched modules will be restricted to those whose dotted name equals
        ``prefix`` or starts with ``prefix + "."``.

        **Latest-wins**: calling ``r.with_prefix("a").with_prefix("b")`` yields
        a recipe with ``module_prefix="b"`` — prefixes are NOT concatenated.

        .. note::
            If you set ``expected_min_matches``, you'll likely want to lower it
            after scoping — thresholds calibrated to whole-model counts won't
            hold after a prefix filter.
        """
        return dataclasses.replace(
            self,
            module_prefix=prefix,
            name=f"{self.name}@{prefix}",
        )

    def with_sae(
        self,
        mapping: dict[str, tuple[str, str]],
    ) -> Recipe:
        """Return a new Recipe with sae_checkpoints replaced (latest-wins).

        Mapping keys are regex patterns matched against resolved
        fully-qualified module names (e.g. r".*\\.layers\\.8$"). Values are
        (sae_lens_release, sae_id) pairs forwarded to load_sae(...).

        **Does NOT modify activation_diagnostics.** Loading SAEs is cheap;
        running them every step is not (an SAE encode+decode is two large
        matmuls). Users must explicitly add "sae_reconstruction" to
        ``activation_diagnostics`` to actually pay that cost.

        **Interaction with `.with_prefix()`:** patterns in `mapping` are
        matched against fully-qualified module names *after* any prefix
        from `.with_prefix()` has been applied. Always call
        `.with_prefix()` *before* `.with_sae()` so patterns are written
        against the final module paths.
        """
        return dataclasses.replace(self, sae_checkpoints=mapping)


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


def _register_stock_recipes() -> None:
    from circuitry.recipes import llm, two_tower, vision
    for mod in (llm, vision, two_tower):
        try:
            mod.register()
        except ValueError:
            pass  # already registered


_register_stock_recipes()
