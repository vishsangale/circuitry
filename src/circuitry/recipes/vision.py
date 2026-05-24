"""Stock vision recipe — covers conv-based and ViT-style backbones."""

from __future__ import annotations

from circuitry.recipes import Recipe, register_recipe
from circuitry.recorder.hooks import HookPoint, TensorSource

RECIPE = Recipe(
    name="vision",
    hook_points=[
        HookPoint(source=TensorSource.WEIGHT,
                  pattern=r"(conv\d+|fc\d+|patch_embed|blocks\.\d+\.(attn|mlp))(\.weight)?$"),
        HookPoint(source=TensorSource.OUTPUT,
                  pattern=r"(conv\d+|fc\d+|blocks\.\d+\.(attn|mlp))$"),
    ],
    weight_diagnostics=["effective_rank", "stable_rank"],
    activation_diagnostics=["dead_fraction", "participation_ratio"],
    gradient_diagnostics=["grad_norm_per_module"],
)


def register() -> None:
    """Register the vision recipe. Idempotent under test fixtures via
    ``_clear_registry_for_tests``."""
    register_recipe(RECIPE)
