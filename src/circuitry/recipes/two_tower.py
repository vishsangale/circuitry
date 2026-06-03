"""Two-tower recipe (recsys). Adds an embedding-alignment custom diagnostic:
cosine similarity between the mean query-tower output and the mean item-tower
output captured this step.

Also covers standard DLRM module names (``embed_tables``, ``bottom_mlp``,
``top_mlp``) in addition to the two-tower convention
(``query_tower``, ``item_tower``, ``interaction``).  Per-table embedding
children (``item_tower.0``, ``item_tower.1``, …) are hooked individually so
that activation diagnostics fire on each embedding table.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from circuitry.recipes import Recipe, register_recipe
from circuitry.recorder.hooks import HookPoint, StepContext, TensorSource

_QUERY_KEYS = ("query_tower", "user_tower", "left_tower")
_ITEM_KEYS = ("item_tower", "right_tower")


def _mean_output(ctx: StepContext, prefix: tuple[str, ...]) -> torch.Tensor | None:
    """Return the mean activation for the first matching tower.

    When the tower is an ``nn.ModuleList`` its forward hook never fires, but
    the individual children (``item_tower.0``, ``item_tower.1``, …) *do* fire
    because the output pattern now includes them.  Aggregate those children
    by averaging their mean activations so that ``embedding_alignment`` still
    works even when the top-level tower container is a ModuleList.
    """
    # First try an exact match on the top-level tower name (Sequential case).
    for name, t in ctx.activations.items():
        if name in prefix:
            return t.flatten(0, -2).mean(dim=0)

    # Fallback: gather per-child activations for each prefix (ModuleList case).
    child_means: list[torch.Tensor] = []
    for name, t in ctx.activations.items():
        if any(name.startswith(p + ".") for p in prefix):
            child_means.append(t.flatten(0, -2).mean(dim=0))
    if child_means:
        stacked = torch.stack(child_means, dim=0)  # (N_children, dim)
        return stacked.mean(dim=0)

    return None


def embedding_alignment(ctx: StepContext) -> dict[str, float]:
    q = _mean_output(ctx, _QUERY_KEYS)
    i = _mean_output(ctx, _ITEM_KEYS)
    if q is None or i is None or q.shape != i.shape:
        return {}
    cos = F.cosine_similarity(q.unsqueeze(0), i.unsqueeze(0)).item()
    return {"embedding_alignment": float(cos)}


# Weight pattern — covers both two-tower names and standard DLRM names.
#   Two-tower: query_tower, item_tower, interaction
#   DLRM:      embed_tables, bottom_mlp, top_mlp
_WEIGHT_PATTERN = (
    r"(query_tower|item_tower|interaction"
    r"|embed_tables|bottom_mlp|top_mlp).*"
)

# Output pattern — matches top-level towers AND per-child embedding tables
# (item_tower.0, item_tower.1, …) so that:
#   (a) per-child activation diagnostics fire (F34), and
#   (b) _mean_output can aggregate children for embedding_alignment (F33).
_OUTPUT_PATTERN = r"(query_tower|item_tower)(\.\d+)?$"

RECIPE = Recipe(
    name="two_tower",
    hook_points=[
        HookPoint(source=TensorSource.WEIGHT, pattern=_WEIGHT_PATTERN),
        # GRAD mirrors WEIGHT so grad_norm_per_module has gradients to read —
        # ctx.gradients is populated only from GRAD hook points.
        HookPoint(source=TensorSource.GRAD, pattern=_WEIGHT_PATTERN),
        HookPoint(source=TensorSource.OUTPUT, pattern=_OUTPUT_PATTERN),
    ],
    weight_diagnostics=["effective_rank", "stable_rank"],
    activation_diagnostics=["dead_fraction", "participation_ratio"],
    gradient_diagnostics=["grad_norm_per_module"],
    custom=[embedding_alignment],
)


def register() -> None:
    register_recipe(RECIPE)
