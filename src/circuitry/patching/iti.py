"""Inference-Time Intervention (ITI). Li et al. 2023, arXiv:2306.03341.

Trains per-(layer, head) mass-mean probes on labelled activation data, then
at inference time adds coeff * direction to each head's output slice.
"""
from __future__ import annotations

import contextlib
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from circuitry.core.probe import mass_mean_probe

__all__ = ["ITIConfig", "fit_iti", "apply_iti"]


@dataclass
class ITIConfig:
    """Steering config from fit_iti.

    head_directions: {(layer, head): (d_head,) unit direction}
    d_head: size of each head's output slice
    coeff: additive steering magnitude (Li et al. use 15.0)
    """
    head_directions: dict[tuple[int, int], Tensor]
    d_head: int
    coeff: float = 15.0


def fit_iti(
    head_acts: dict[tuple[int, int], Tensor],
    labels: Tensor,
    *,
    coeff: float = 15.0,
) -> ITIConfig:
    """Train per-head mass-mean probes and return ITIConfig.

    Args:
        head_acts: {(layer, head): (n, d_head) activation tensor}
        labels: (n,) binary labels — 0=negative, 1=positive
        coeff: steering magnitude at inference (default 15.0)
    """
    if not head_acts:
        raise ValueError("fit_iti: head_acts must not be empty")
    d_head = next(iter(head_acts.values())).shape[-1]
    directions: dict[tuple[int, int], Tensor] = {}
    for key, acts in head_acts.items():
        probe = mass_mean_probe(acts, labels)
        directions[key] = probe.direction
    return ITIConfig(head_directions=directions, d_head=d_head, coeff=coeff)


@contextlib.contextmanager
def apply_iti(
    model: nn.Module,
    config: ITIConfig,
    *,
    attn_modules: dict[int, nn.Module] | None = None,
    resolver=None,
) -> Iterator[None]:
    """Context manager that injects ITI steering into attention head outputs.

    For each (layer, head) in config, registers a forward hook on
    attn_modules[layer] that adds coeff * direction to
    output[..., head*d_head:(head+1)*d_head].

    Args:
        model: PyTorch model.
        config: ITIConfig from fit_iti.
        attn_modules: {layer_idx: nn.Module} — attention output module per layer.
            When None, uses resolver to look up Site(attn_out, layer).
        resolver: SiteResolver. Defaults to HFSiteResolver.from_config when
            attn_modules=None and model has .config.
    """
    if attn_modules is None:
        from circuitry.patching.sites import HFSiteResolver, Site
        if resolver is None:
            cfg = getattr(model, "config", None)
            if cfg is None:
                raise ValueError(
                    "apply_iti: pass attn_modules= or resolver= (or model.config)."
                )
            resolver = HFSiteResolver.from_config(cfg)
        layers_needed = {layer for (layer, _) in config.head_directions}
        attn_modules = {
            layer: resolver.resolve(model, Site(component="attn_out", layer=layer)).module
            for layer in layers_needed
        }

    layer_to_heads: dict[int, dict[int, Tensor]] = defaultdict(dict)
    for (layer, head), direction in config.head_directions.items():
        layer_to_heads[layer][head] = direction

    handles = []
    was_training = model.training
    try:
        model.eval()
        for layer_idx, head_dirs in layer_to_heads.items():
            module = attn_modules[layer_idx]
            d_head = config.d_head
            coeff = config.coeff

            def _make_hook(hd: dict[int, Tensor], d: int, c: float):
                def _hook(mod, inp, output):  # noqa: ARG001
                    out = output[0] if isinstance(output, tuple) else output
                    out = out.clone()
                    for head_idx, direction in hd.items():
                        v = direction.to(device=out.device, dtype=out.dtype)
                        start, end = head_idx * d, (head_idx + 1) * d
                        out[..., start:end] = out[..., start:end] + c * v
                    return (out,) + output[1:] if isinstance(output, tuple) else out
                return _hook

            handles.append(module.register_forward_hook(_make_hook(head_dirs, d_head, coeff)))
        yield
    finally:
        for h in handles:
            h.remove()
        if was_training:
            model.train()
        else:
            model.eval()
