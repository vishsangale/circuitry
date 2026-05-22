"""Parameter-centric model inventory.

Built once at ``Recorder.attach()`` time. Replaces the old
``getattr(module, "weight", None)`` heuristic for resolving weight HookPoints,
which silently dropped weights hidden inside wrapper Linear classes (e.g.
HuggingFace ``Gemma4ClippableLinear``) whose ``.weight`` lives on a child
module rather than on themselves.

Layering: this module is in ``core/`` — pure data + ``torch``, no I/O. Callers
in ``recorder/`` are responsible for persisting :meth:`ModelInventory.to_json`
to disk.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class ParameterRecord:
    """One Parameter in a model, with everything needed to route to it.

    Fields:
        name: Full dotted name from ``model.named_parameters()``, e.g.
            ``"model.layers.0.self_attn.q_proj.weight"``.
        shape: Parameter shape as a tuple.
        dtype: Parameter dtype.
        numel: Total element count.
        requires_grad: Whether the Parameter participates in autograd.
        owning_module_name: Dotted name of the module that owns this
            Parameter (the segment before the final ``.<leaf>``). ``""`` if
            the Parameter is registered directly on the root model.
        owning_module_class: ``type(owner).__name__`` — used for diagnostic
            messages and to recognize wrapper classes.
        leaf_attr: The final name segment (e.g. ``"weight"``, ``"bias"``).
    """

    name: str
    shape: tuple[int, ...]
    dtype: torch.dtype
    numel: int
    requires_grad: bool
    owning_module_name: str
    owning_module_class: str
    leaf_attr: str


@dataclass(frozen=True)
class ModelInventory:
    """Frozen snapshot of every named Parameter in a model.

    Built once at attach time. The recorder uses this as the source of truth
    for ``WEIGHT`` / ``GRAD`` HookPoint resolution instead of attribute-guessing
    on matched modules.

    Construction is cheap — one pass over ``model.named_parameters()`` plus a
    dict build of ``model.named_modules()`` for owning-class lookup.
    """

    parameters: tuple[ParameterRecord, ...]

    # ------------------------------------------------------------------ build

    @classmethod
    def build(cls, model: nn.Module) -> ModelInventory:
        """Walk every named Parameter once and capture its metadata.

        Uses ``remove_duplicate=False`` so tied weights (e.g.
        ``embed_tokens.weight`` and ``lm_head.weight`` when
        ``tie_word_embeddings=True``) are listed under both names — recipes
        that hook either side will resolve correctly.
        """
        name_to_mod = dict(model.named_modules())
        records: list[ParameterRecord] = []
        for full_name, p in model.named_parameters(remove_duplicate=False):
            owner_name, _, leaf = full_name.rpartition(".")
            owner = name_to_mod.get(owner_name, model)
            records.append(
                ParameterRecord(
                    name=full_name,
                    shape=tuple(p.shape),
                    dtype=p.dtype,
                    numel=p.numel(),
                    requires_grad=p.requires_grad,
                    owning_module_name=owner_name,
                    owning_module_class=type(owner).__name__,
                    leaf_attr=leaf,
                )
            )
        return cls(parameters=tuple(records))

    # ------------------------------------------------------------------ query

    def with_prefix(self, prefix: str) -> ModelInventory:
        """Return a new inventory containing only parameters whose name starts
        with ``prefix``. Used for modality scoping on multimodal models — e.g.
        ``inv.with_prefix("model.language_model")``.
        """
        return ModelInventory(
            parameters=tuple(r for r in self.parameters if r.name.startswith(prefix))
        )

    def match_pattern(self, pattern: str) -> tuple[ParameterRecord, ...]:
        """Return parameters whose full name matches ``pattern`` (regex search)."""
        rx = re.compile(pattern)
        return tuple(r for r in self.parameters if rx.search(r.name))

    def find_primary_weight(self, module_name: str) -> ParameterRecord | None:
        """Resolve a matched module name to its primary 2-D+ weight Parameter.

        Resolution order:

        1. Exact ``<module>.weight`` (direct attribute) — preferred when it
           exists and is at least 2-D.
        2. Otherwise: count all 2-D+ Parameters anywhere in the module's
           subtree (the module itself plus all descendants). If exactly one,
           return it. If zero or multiple, return ``None`` — the caller can
           decide to warn and skip.

        Returning ``None`` is the loud failure mode; the silent
        ``getattr(module, "weight", None)`` path is gone.
        """
        # Direct .weight wins if present and 2-D+.
        direct = next(
            (
                r
                for r in self.parameters
                if r.owning_module_name == module_name
                and r.leaf_attr == "weight"
                and len(r.shape) >= 2
            ),
            None,
        )
        if direct is not None:
            return direct

        # Otherwise, scan the subtree for any 2-D+ Parameter.
        prefix = module_name + "." if module_name else ""
        subtree = [
            r
            for r in self.parameters
            if len(r.shape) >= 2
            and (
                r.owning_module_name == module_name
                or (prefix and r.owning_module_name.startswith(prefix))
            )
        ]
        return subtree[0] if len(subtree) == 1 else None

    # ----------------------------------------------------------- persistence

    def to_json(self) -> str:
        """Serialize the inventory to a stable JSON string.

        Caller writes it to disk (typically ``<run_dir>/circuitry/inventory.json``).
        Useful for "why didn't my regex match?" debugging and as an auditable
        record of what circuitry can see in a model.
        """
        return json.dumps(
            [
                {
                    "name": r.name,
                    "shape": list(r.shape),
                    "dtype": str(r.dtype),
                    "numel": r.numel,
                    "requires_grad": r.requires_grad,
                    "owning_module_name": r.owning_module_name,
                    "owning_module_class": r.owning_module_class,
                    "leaf_attr": r.leaf_attr,
                }
                for r in self.parameters
            ],
            indent=2,
        )
