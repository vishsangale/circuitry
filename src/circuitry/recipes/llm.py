"""Stock LLM recipe. See docs/design.md §5."""

from __future__ import annotations

from circuitry.recipes import Recipe, register_recipe
from circuitry.recorder.hooks import HookPoint, TensorSource

RECIPE = Recipe(
    name="llm",
    hook_points=[
        HookPoint(source=TensorSource.WEIGHT,
                  pattern=r".*\.(q|k|v|o)_proj$"),
        HookPoint(source=TensorSource.WEIGHT,
                  pattern=r".*\.(w1|w2|w3|gate_proj|up_proj|down_proj)$"),
        HookPoint(source=TensorSource.OUTPUT, pattern=r".*\.attn$"),
        HookPoint(source=TensorSource.OUTPUT, pattern=r".*\.mlp$"),
        HookPoint(source=TensorSource.OUTPUT, pattern=r".*\.ln_[12]$"),
        HookPoint(source=TensorSource.WEIGHT, pattern=r"embed.*"),
        HookPoint(source=TensorSource.WEIGHT, pattern=r"lm_head$"),
    ],
    weight_diagnostics=["effective_rank", "stable_rank", "heavy_tail_alpha"],
    activation_diagnostics=["dead_fraction", "kurtosis", "participation_ratio"],
    gradient_diagnostics=["layer_norm"],
)


def register() -> None:
    """Register the LLM recipe. Idempotent under test fixtures via
    ``_clear_registry_for_tests``."""
    register_recipe(RECIPE)
