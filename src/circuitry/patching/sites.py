"""Site dataclass + resolution for activation patching. Design spec §3."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch.nn as nn
from torch import Tensor

VALID_COMPONENTS = frozenset({
    "resid_pre",
    "resid_post",
    "attn_head_out",
    "mlp_out",
    "mlp_neuron",
})


@dataclass(frozen=True)
class Site:
    """A named intervention point in the model's computation graph."""

    component: str
    layer: int
    head: int | None = None
    neuron: int | None = None
    position: int | slice | None = None

    def __post_init__(self) -> None:
        if self.component not in VALID_COMPONENTS:
            raise ValueError(
                f"Unknown component {self.component!r}; "
                f"valid: {sorted(VALID_COMPONENTS)}"
            )
        if self.component == "attn_head_out" and self.head is None:
            raise ValueError("attn_head_out requires head index")
        if self.component == "mlp_neuron" and self.neuron is None:
            raise ValueError("mlp_neuron requires neuron index")


@dataclass
class ResolvedSite:
    """A Site resolved to a concrete module + hook functions."""

    module: nn.Module
    is_input_hook: bool
    extract: Callable[[Tensor], Tensor]
    inject: Callable[[Tensor, Tensor], Tensor]
