"""Mean ablation — replace a module's output with dataset-mean activations.

Zero ablation (zeroing a module's output) is the simplest intervention but
sends activations out of distribution, potentially inflating circuit importance
estimates.  Mean ablation patches with the empirical mean activation computed
over a reference dataset, providing a more realistic null hypothesis.

Reference:
    Wang et al. 2022, "Interpretability in the Wild: a Circuit for Indirect
    Object Identification in GPT-2 small" (uses both zero and mean ablation).
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

__all__ = ["mean_ablation", "compute_mean_activation"]


def compute_mean_activation(
    model: nn.Module,
    module: nn.Module,
    dataset_inputs: list[Any],
) -> Tensor:
    """Compute the mean output activation of *module* over a list of inputs.

    Args:
        model:          PyTorch model (run in eval mode, no grad).
        module:         Target module whose output is captured.
        dataset_inputs: List of inputs (each a Tensor or dict).  The module
                        output is captured for each and averaged.

    Returns:
        Mean activation tensor, same shape as a single module output.
    """
    model.eval()
    accum: Tensor | None = None
    count = 0

    def _capture(mod: nn.Module, inp: tuple, out: Any) -> None:  # noqa: ARG001
        nonlocal accum, count
        val = out[0] if isinstance(out, tuple) else out
        v = val.detach().to(torch.float32)
        if accum is None:
            accum = v.mean(dim=0)          # mean over batch
        else:
            accum = accum + v.mean(dim=0)
        count += 1

    handle = module.register_forward_hook(_capture)
    with torch.no_grad():
        for inp in dataset_inputs:
            if isinstance(inp, Tensor):
                model(inp)
            elif isinstance(inp, dict):
                model(**inp)
            else:
                model(inp)
    handle.remove()

    if accum is None or count == 0:
        raise ValueError("compute_mean_activation: no inputs were processed")
    return accum / count


@contextlib.contextmanager
def mean_ablation(
    model: nn.Module,
    module: nn.Module,
    mean_act: Tensor,
) -> Iterator[None]:
    """Context manager: replace *module*'s output with a pre-computed mean activation.

    On each forward pass inside the context, the module's output is replaced by
    *mean_act* broadcast to match the batch/sequence shape of the live output.

    Args:
        model:    PyTorch model (not modified; hook is removed on exit).
        module:   Module to ablate.
        mean_act: Pre-computed mean activation from :func:`compute_mean_activation`
                  or computed manually; shape should be broadcastable to the
                  module's output (e.g. ``(d_model,)`` or ``(seq, d_model)``).

    Example::

        mean = compute_mean_activation(model, model.layers[3], ref_inputs)
        with mean_ablation(model, model.layers[3], mean):
            out = model(x)  # layers[3]'s output is replaced by mean
    """
    mean_t = mean_act.detach().to(torch.float32)

    def _hook(mod: nn.Module, inp: tuple, out: Any) -> Any:  # noqa: ARG001
        val = out[0] if isinstance(out, tuple) else out
        replacement = mean_t.expand_as(val).to(val.dtype)
        if isinstance(out, tuple):
            return (replacement,) + out[1:]
        return replacement

    handle = module.register_forward_hook(_hook)
    try:
        yield
    finally:
        handle.remove()
