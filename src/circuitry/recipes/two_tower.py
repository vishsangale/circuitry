"""Two-tower recipe (recsys). Adds an embedding-alignment custom diagnostic:
cosine similarity between the mean query-tower output and the mean item-tower
output captured this step.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from circuitry.recipes import Recipe, register_recipe
from circuitry.recorder.hooks import HookPoint, StepContext, TensorSource

_QUERY_KEYS = ("query_tower", "user_tower", "left_tower")
_ITEM_KEYS = ("item_tower", "right_tower")


def _mean_output(ctx: StepContext, prefix: tuple[str, ...]) -> torch.Tensor | None:
    for name, t in ctx.activations.items():
        if any(name == p or name.startswith(p + ".") for p in prefix):
            return t.flatten(0, -2).mean(dim=0)
    return None


def embedding_alignment(ctx: StepContext) -> dict[str, float]:
    q = _mean_output(ctx, _QUERY_KEYS)
    i = _mean_output(ctx, _ITEM_KEYS)
    if q is None or i is None or q.shape != i.shape:
        return {}
    cos = F.cosine_similarity(q.unsqueeze(0), i.unsqueeze(0)).item()
    return {"embedding_alignment": float(cos)}


RECIPE = Recipe(
    name="two_tower",
    hook_points=[
        HookPoint(source=TensorSource.WEIGHT,
                  pattern=r"(query_tower|item_tower|interaction).*"),
        HookPoint(source=TensorSource.OUTPUT,
                  pattern=r"(query_tower|item_tower)$"),
    ],
    weight_diagnostics=["effective_rank", "stable_rank"],
    activation_diagnostics=["dead_fraction", "participation_ratio"],
    gradient_diagnostics=["layer_norm"],
    custom=[embedding_alignment],
)


def register() -> None:
    register_recipe(RECIPE)
