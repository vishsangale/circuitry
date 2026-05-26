"""Shared HF model-layout helpers for the patching backend (EAP/AtP*).

The HF-eager patching backend targets Llama-family layouts
(``model.model.layers`` + ``self_attn.{q,k,v,o}_proj``). Non-Llama models
(GPT-2, etc.) should be routed through the TransformerLens backend via
``circuitry.patching.to_hooked_transformer``.
"""

from __future__ import annotations

import torch.nn as nn

_UNSUPPORTED_MSG = (
    "circuitry's HF patching backend supports Llama-family layouts "
    "(model.model.layers + self_attn.{{q,k,v,o}}_proj). This model ({cls}) is not "
    "supported directly. For GPT-2 and other architectures, convert it with "
    "circuitry.patching.to_hooked_transformer(model, \"<name>\") and use the "
    "TransformerLens backend (TLSiteResolver). See docs/design.md §4.6."
)


def locate_layers(model: nn.Module) -> nn.ModuleList:
    """Return the transformer layers list (model.model.layers or model.layers)."""
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "layers"):
        return inner.layers  # type: ignore[return-value]
    if hasattr(model, "layers"):
        return model.layers  # type: ignore[return-value]
    raise ValueError(_UNSUPPORTED_MSG.format(cls=type(model).__name__))


def locate_embed(model: nn.Module) -> nn.Module:
    """Return the token-embedding module (model.model.embed_tokens or model.embed_tokens)."""
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "embed_tokens"):
        return inner.embed_tokens
    if hasattr(model, "embed_tokens"):
        return model.embed_tokens  # type: ignore[return-value]
    raise ValueError(_UNSUPPORTED_MSG.format(cls=type(model).__name__))
