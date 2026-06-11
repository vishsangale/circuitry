"""Hook-point data classes and module-matching logic.

See docs/design.md §4.4. Recorder uses ``match_modules`` to resolve a
HookPoint against ``model.named_modules()`` and to enforce
``expected_min_matches`` invariants at attach time.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import torch
import torch.nn as nn


class TensorSource(str, Enum):
    WEIGHT = "weight"
    INPUT = "input"
    OUTPUT = "output"
    GRAD = "grad"
    # NAMED_PARAM matches its ``pattern`` against parameter names
    # (``model.named_parameters()``) rather than module names, and feeds the
    # matched ≥2-D parameter to the weight diagnostics keyed by its parameter
    # name. Use it to reach a fused parameter that the WEIGHT source can't —
    # e.g. ``nn.MultiheadAttention.in_proj_weight`` (a direct Parameter on the
    # module, not a child Linear's ``.weight``). Pattern only; ≥2-D params only.
    NAMED_PARAM = "named_param"
    # Multi-process variants (design §11, v1.46). Identical to WEIGHT / OUTPUT
    # in a single-process run; in a torch.distributed run the recorder gathers
    # the full tensor before any primitive sees it: WEIGHT_FULL materializes
    # DTensor-sharded params via full_tensor(), ACTIVATION_FULL all-gathers the
    # captured activation across ranks (concat on the batch dim). Non-zero
    # ranks participate in the collectives but emit nothing (rank-0 writes).
    WEIGHT_FULL = "weight_full"
    ACTIVATION_FULL = "activation_full"


@dataclass
class HookPoint:
    source: TensorSource
    pattern: str | None = None
    modules: list[nn.Module] | None = None
    selector: Callable[[nn.Module], list[str]] | None = None
    optional: bool = False
    """If True, a 0-match is a soft skip even under ``strict=True``.

    Use for patterns that are *structurally absent* on some architectures the
    recipe also targets — e.g. MoE router/expert patterns on a dense model, or
    DLRM/GRU patterns on a transformer-recsys model. Genuinely-required patterns
    must stay ``optional=False`` so strict mode still catches a misconfigured
    recipe (a pattern that should have matched but didn't).
    """

    def __post_init__(self) -> None:
        targets = sum(x is not None for x in (self.pattern, self.modules, self.selector))
        if targets != 1:
            raise ValueError(
                "HookPoint requires exactly one of {pattern, modules, selector}; "
                f"got {targets}"
            )


@dataclass
class StepContext:
    """Snapshot passed to every diagnostic on an emit step.

    Fields are dicts keyed by hooked-module name (the dotted name from
    ``model.named_modules()``). Built-in diagnostics ignore the ``user`` dict;
    custom diagnostics can use it to thread arbitrary state through
    ``Recorder.step(**kwargs)``.
    """

    step: int
    model: nn.Module
    activations: dict[str, torch.Tensor] = field(default_factory=dict)
    gradients: dict[str, torch.Tensor] = field(default_factory=dict)
    weights: dict[str, torch.Tensor] = field(default_factory=dict)
    loss: float | None = None
    user: dict[str, Any] = field(default_factory=dict)


def match_modules(model: nn.Module, hp: HookPoint) -> list[str]:
    """Return the dotted module names matched by ``hp`` against ``model``.

    Resolution rules:
      - ``pattern``  : regex against ``dict(model.named_modules()).keys()``
      - ``modules``  : reverse-lookup each instance to its name
      - ``selector`` : delegate; selector must return module names
    """
    if hp.source is TensorSource.NAMED_PARAM and hp.pattern is not None:
        # Match against parameter names, not module names (e.g. to reach a fused
        # nn.MultiheadAttention.in_proj_weight that has no owning child module).
        rx = re.compile(hp.pattern)
        return [n for n, _ in model.named_parameters() if rx.search(n)]
    name_to_mod = dict(model.named_modules())
    if hp.pattern is not None:
        rx = re.compile(hp.pattern)
        return [n for n in name_to_mod if rx.search(n)]
    if hp.modules is not None:
        mod_to_name = {id(m): n for n, m in name_to_mod.items()}
        return [mod_to_name[id(m)] for m in hp.modules if id(m) in mod_to_name]
    assert hp.selector is not None
    return list(hp.selector(model))


def filtered_matches(model: nn.Module, hp: HookPoint, recipe: Any) -> list[str]:
    """Like ``match_modules``, but apply ``recipe.module_prefix`` if set.

    Keeps only module names that equal the prefix or start with
    ``prefix + "."``.  When ``recipe.module_prefix`` is ``None`` the result is
    identical to ``match_modules(model, hp)``.

    Parameters
    ----------
    model:
        The model to inspect.
    hp:
        The HookPoint whose matching rules are applied.
    recipe:
        The Recipe object. Must have a ``module_prefix`` attribute
        (``str | None``).
    """
    names = match_modules(model, hp)
    prefix: str | None = getattr(recipe, "module_prefix", None)
    if prefix is None:
        return names
    return [n for n in names if n == prefix or n.startswith(prefix + ".")]
