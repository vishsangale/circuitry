"""Recipe dataclass + global registry. See docs/design.md §4.4."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from dataclasses import dataclass, field

import torch

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
    # Explicit attention-head metadata for attention_head_rank, used when the
    # model exposes no resolvable ``config`` (custom non-HF models). Keys:
    # ``n_heads`` (required), ``n_kv_heads`` (optional, defaults to n_heads),
    # ``head_dim`` (optional — derived from ``hidden_size`` if both that and
    # n_heads are supplied). When set this overrides config-based resolution.
    # See recorder._resolve_attn_meta for the full search order.
    attn_head_meta: dict[str, int] | None = None
    # Custom forward entry point for the recorder's internal probe passes
    # (induction_score, drift_probe). HF-style diagnostics call
    # ``model(probe, output_attentions=True)``; non-HF models whose forward
    # entry point differs (e.g. SASRec.predict_scores) ``TypeError`` or no-op.
    # Set ``forward_fn(model, batch) -> output`` to run the right forward; the
    # recorder's capture hooks still fire during that call. When ``None`` the
    # recorder uses the HF-style call with a wrapper-safe fallback.
    forward_fn: Callable[..., object] | None = None
    # v1.4 drift-probe config fields (Workstream B).
    # probe_batch: fixed input tensor for the second forward pass that captures
    # reference activations and computes representational drift.  None = no
    # probe pass (drift_probe is also disabled by default via ``enabled``).
    # Storage warning: reference activations are stored as float32 CPU copies,
    # up to n_layers × max_samples × d_model × 4 bytes (~270 MB for a large LLM
    # at max_samples=256, d_model=4096, 32 layers).  Use drift_max_tokens to cap
    # the token dimension and reduce storage.
    #
    # Shape requirement for CKA methods (linear_cka, rbf_cka): the captured
    # activation must yield >= 2 rows after flattening all but the last dim.
    # A single-token probe (batch=1, seq=1) will raise ValueError for CKA;
    # use drift_method="cosine" for single-row probes, or pass >= 2 tokens.
    # Integer (token-ID) probe_batch is supported; the recorder casts to device
    # only (not dtype) so embedding lookups receive the correct integer type.
    probe_batch: torch.Tensor | None = None
    drift_method: str = "linear_cka"
    drift_max_tokens: int | None = None

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

    def disable(self, names: list[str]) -> Recipe:
        """Return a new Recipe with each name in *names* disabled.

        *names* must be a subset of the recipe's own
        ``weight_diagnostics + activation_diagnostics + gradient_diagnostics``.
        Raises ``ValueError`` for any name not present in those lists.

        Custom ``DiagnosticFn`` callables in ``self.custom`` are not
        name-addressable and are unaffected by this helper.
        """
        _all = set(
            self.weight_diagnostics + self.activation_diagnostics + self.gradient_diagnostics
        )
        unknown = set(names) - _all
        if unknown:
            raise ValueError(
                f"Recipe {self.name!r}: unknown diagnostic name(s) {sorted(unknown)}. "
                f"Available: {sorted(_all)}"
            )
        new_enabled = {**self.enabled, **{n: False for n in names}}
        return dataclasses.replace(self, enabled=new_enabled)

    def only(self, names: list[str]) -> Recipe:
        """Return a new Recipe running *only* the diagnostics in *names*.

        The complement (everything in
        ``weight_diagnostics + activation_diagnostics + gradient_diagnostics``
        not in *names*) is disabled. Raises ``ValueError`` for any name
        not present in those lists.

        Custom ``DiagnosticFn`` callables are unaffected.
        """
        _all = set(
            self.weight_diagnostics + self.activation_diagnostics + self.gradient_diagnostics
        )
        unknown = set(names) - _all
        if unknown:
            raise ValueError(
                f"Recipe {self.name!r}: unknown diagnostic name(s) {sorted(unknown)}. "
                f"Available: {sorted(_all)}"
            )
        new_enabled = {**self.enabled, **{n: (n in set(names)) for n in _all}}
        return dataclasses.replace(self, enabled=new_enabled)


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
    from circuitry.recipes import llm, recsys, two_tower, vision
    for mod in (llm, vision, two_tower, recsys):
        try:
            mod.register()
        except ValueError:
            pass  # already registered


_register_stock_recipes()
