"""Activation patching context manager. Design spec §4.

Guarantees: hook removed on exit, eval mode restored, param requires_grad
restored, even on exception. Mutation-last discipline.
"""
from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field

import torch.nn as nn
from torch import Tensor

from circuitry.patching.sites import ResolvedSite, Site


@dataclass
class PatchHandle:
    """Handle returned by patch_site context manager."""

    activation_grad: Tensor | None = field(default=None, init=False)
    _grad_tensor: Tensor | None = field(default=None, init=False, repr=False)


@contextmanager
def patch_site(
    model: nn.Module,
    site: Site,
    value: Tensor,
    resolver: object,
    *,
    enable_activation_grad: bool = False,
) -> Generator[PatchHandle, None, None]:
    """Patch a site's activation with ``value`` for the duration of the context.

    Guarantees:
    - Hook is removed on exit (including on exception).
    - Model eval mode is set on entry, restored on exit.
    - All param requires_grad are set to False, restored on exit.
    - Param values are never modified.
    """
    resolved: ResolvedSite = resolver.resolve(model, site)  # type: ignore[attr-defined]
    handle = PatchHandle()
    hook_handle = None
    was_training = model.training
    original_requires_grad: dict[str, bool] = {}

    for name, p in model.named_parameters():
        original_requires_grad[name] = p.requires_grad
        p.requires_grad_(False)

    try:
        model.eval()

        def _attach_grad_hook(modified: Tensor) -> Tensor:
            """Detach, mark requires_grad, and register a hook to capture grad."""
            activated = modified.detach().requires_grad_(True)

            def _capture_grad(grad: Tensor) -> None:
                handle.activation_grad = grad.clone()

            activated.register_hook(_capture_grad)
            handle._grad_tensor = activated
            return activated

        if resolved.is_input_hook:
            def pre_hook(module: nn.Module, args: tuple) -> tuple:
                x = args[0]
                modified = resolved.inject(x, value)
                if enable_activation_grad:
                    modified = _attach_grad_hook(modified)
                return (modified,) + args[1:]

            hook_handle = resolved.module.register_forward_pre_hook(pre_hook)
        else:
            def post_hook(module: nn.Module, inputs: tuple, output: object) -> object:
                if isinstance(output, tuple):
                    first = output[0]
                    modified = resolved.inject(first, value)
                    if enable_activation_grad:
                        modified = _attach_grad_hook(modified)
                    return (modified,) + output[1:]
                modified = resolved.inject(output, value)
                if enable_activation_grad:
                    modified = _attach_grad_hook(modified)
                return modified

            hook_handle = resolved.module.register_forward_hook(post_hook)

        yield handle

    finally:
        if hook_handle is not None:
            hook_handle.remove()
        if was_training:
            model.train()
        else:
            model.eval()
        for name, p in model.named_parameters():
            if name in original_requires_grad:
                p.requires_grad_(original_requires_grad[name])
