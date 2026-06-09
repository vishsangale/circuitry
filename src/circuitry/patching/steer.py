"""Activation steering and ablation context managers. See docs/design.md §4.

Registers forward hooks that additively inject or subtract a concept direction
at the resolved site.  Hooks are always removed on exit.

Re-exports ``steer_vector``, ``repe_direction``, and ``directional_ablation``
from ``circuitry.core.steer`` for convenience.
"""
from __future__ import annotations

import contextlib
from collections.abc import Iterator

import torch
import torch.nn as nn
from torch import Tensor

from circuitry.core.steer import directional_ablation, repe_direction, steer_vector  # re-exports
from circuitry.patching.sites import Site, SiteResolver, HFSiteResolver

__all__ = ["apply_ablation", "apply_steer", "directional_ablation", "repe_direction", "steer_vector"]


def _resolve_module(model: nn.Module, site: Site, resolver: SiteResolver) -> nn.Module:
    """Return the nn.Module at *site* using *resolver*."""
    resolved = resolver.resolve(model, site)
    return resolved.module


@contextlib.contextmanager
def apply_steer(
    model: nn.Module,
    site: Site,
    vector: Tensor,
    *,
    coeff: float = 1.0,
    resolver: SiteResolver | None = None,
) -> Iterator[None]:
    """Context manager that adds ``coeff * vector`` to activations at *site*.

    Registers a forward hook on the module resolved from *site*.  The hook
    adds ``coeff * vector`` (broadcast over batch and seq dims) to the module
    output tensor.  The hook is always removed on exit, even if an exception
    occurs inside the ``with`` block.

    Args:
        model: The PyTorch model.
        site: The :class:`~circuitry.patching.sites.Site` to hook
            (``component``/``layer`` must resolve to an ``nn.Module``).
        vector: ``(d_model,)`` steering direction, e.g. from
            :func:`~circuitry.core.steer.steer_vector`.
        coeff: Scale factor; positive values steer toward the vector
            direction, negative values steer away.
        resolver: :class:`~circuitry.patching.sites.SiteResolver` to use.
            Defaults to ``HFSiteResolver.from_config(model.config)`` when
            *model* has a ``.config`` attribute, otherwise raises
            ``ValueError``.
    """
    if resolver is None:
        config = getattr(model, "config", None)
        if config is None:
            raise ValueError(
                "apply_steer: resolver=None requires model.config to exist. "
                "Pass an explicit SiteResolver via the resolver= argument."
            )
        resolver = HFSiteResolver.from_config(config)

    module = _resolve_module(model, site, resolver)
    hook_handle = None
    was_training = model.training

    try:
        model.eval()

        def _steer_hook(
            mod: nn.Module,  # noqa: ARG001
            inputs: tuple,   # noqa: ARG001
            output: object,
        ) -> object:
            # Handle tuple outputs (some HF modules return (tensor, ...))
            if isinstance(output, tuple):
                first = output[0]
                steered = _add_vector(first, vector, coeff)
                return (steered,) + output[1:]
            return _add_vector(output, vector, coeff)

        hook_handle = module.register_forward_hook(_steer_hook)
        yield

    finally:
        if hook_handle is not None:
            hook_handle.remove()
        # Restore training mode
        if was_training:
            model.train()
        else:
            model.eval()


@contextlib.contextmanager
def apply_ablation(
    model: nn.Module,
    site: Site,
    direction: Tensor,
    *,
    resolver: SiteResolver | None = None,
) -> Iterator[None]:
    """Context manager that removes a concept direction from activations at *site*.

    Registers a forward hook that projects out ``direction`` (calls
    :func:`~circuitry.core.steer.directional_ablation`) from the module's
    output tensor.  The hook is always removed on exit.

    Args:
        model:     The PyTorch model.
        site:      The :class:`~circuitry.patching.sites.Site` to hook.
        direction: ``(d_model,)`` concept direction to ablate (e.g. the refusal
            direction from :func:`~circuitry.core.steer.steer_vector` or
            :func:`~circuitry.core.steer.repe_direction`).
        resolver:  :class:`~circuitry.patching.sites.SiteResolver` to use.
            Defaults to ``HFSiteResolver.from_config(model.config)`` when
            *model* has a ``.config`` attribute, otherwise raises ``ValueError``.

    Reference: Arditi et al. NeurIPS 2024 "Refusal in LMs" arXiv:2406.11717
    """
    if resolver is None:
        config = getattr(model, "config", None)
        if config is None:
            raise ValueError(
                "apply_ablation: resolver=None requires model.config to exist. "
                "Pass an explicit SiteResolver via the resolver= argument."
            )
        resolver = HFSiteResolver.from_config(config)

    module = _resolve_module(model, site, resolver)
    hook_handle = None
    was_training = model.training

    try:
        model.eval()

        def _ablation_hook(
            mod: nn.Module,   # noqa: ARG001
            inputs: tuple,    # noqa: ARG001
            output: object,
        ) -> object:
            if isinstance(output, tuple):
                first = output[0]
                return (directional_ablation(first, direction),) + output[1:]
            return directional_ablation(output, direction)

        hook_handle = module.register_forward_hook(_ablation_hook)
        yield

    finally:
        if hook_handle is not None:
            hook_handle.remove()
        if was_training:
            model.train()
        else:
            model.eval()


def _add_vector(output: Tensor, vector: Tensor, coeff: float) -> Tensor:
    """Add ``coeff * vector`` to *output*, broadcasting over batch/seq dims."""
    v = vector.to(device=output.device, dtype=output.dtype)
    # Broadcast: v is (d_model,); output may be (d_model,), (batch, d_model),
    # or (batch, seq, d_model) — standard broadcasting handles all cases.
    return output + coeff * v
