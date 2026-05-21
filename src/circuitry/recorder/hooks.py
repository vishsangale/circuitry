"""Hook-point data classes and module-matching logic.

See docs/design.md §4.4. Recorder uses ``match_modules`` to resolve a
HookPoint against ``model.named_modules()`` and to enforce
``expected_min_matches`` invariants at attach time.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

import torch
import torch.nn as nn


class TensorSource(str, Enum):
    WEIGHT = "weight"
    INPUT = "input"
    OUTPUT = "output"
    GRAD = "grad"


@dataclass
class HookPoint:
    source: TensorSource
    pattern: str | None = None
    modules: list[nn.Module] | None = None
    selector: Callable[[nn.Module], list[str]] | None = None

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
    name_to_mod = dict(model.named_modules())
    if hp.pattern is not None:
        rx = re.compile(hp.pattern)
        return [n for n in name_to_mod if rx.search(n)]
    if hp.modules is not None:
        mod_to_name = {id(m): n for n, m in name_to_mod.items()}
        return [mod_to_name[id(m)] for m in hp.modules if id(m) in mod_to_name]
    assert hp.selector is not None
    return list(hp.selector(model))
