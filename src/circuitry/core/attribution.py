"""Gradient-based input token attribution primitives.

gradient_input_attribution — gradient × input (Simonyan et al. 2013 / Shrikumar et al. 2016):
    pointwise product of gradient w.r.t. embedding and embedding value, reduced
    over the feature dimension to a per-token scalar.  One backward pass.

integrated_gradients — Sundararajan et al. 2017 (arXiv:1703.01365):
    accumulates gradients along a straight-line path from a baseline (zero
    embeddings) to the actual embeddings, then multiplies by (embed − baseline).
    Satisfies completeness: sum of per-token scores ≈ f(embed) − f(baseline)
    when reduction="dot".
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor

_REDUCTIONS = frozenset({"l2", "dot", "abs", "l1"})


def _reduce(product: Tensor, reduction: str) -> Tensor:
    """Reduce (batch, …, d_model) elementwise product → (batch, …)."""
    if reduction == "dot":
        return product.sum(dim=-1)
    if reduction == "l2":
        return product.norm(dim=-1)
    if reduction == "abs":
        return product.abs().sum(dim=-1)
    if reduction == "l1":
        return product.abs().sum(dim=-1)
    raise ValueError(f"reduction must be one of {sorted(_REDUCTIONS)}, got {reduction!r}")


def gradient_input_attribution(
    grads: Any,
    embeds: Any,
    *,
    reduction: str = "l2",
) -> Tensor:
    """Per-token attribution via gradient × input.

    Computes ``grads ⊙ embeds`` elementwise and reduces over the feature
    dimension, yielding a scalar importance score per token.

    Args:
        grads:     Gradient of a scalar target w.r.t. the embedding layer output,
                   shape ``(batch, seq, d_model)`` or ``(batch, d_model)``.
        embeds:    Embedding vectors, same shape as *grads*.
        reduction: How to reduce the ``d_model`` axis:
                   ``"l2"`` (default) — L2 norm of the product (always ≥ 0);
                   ``"dot"`` — dot product / sum (signed, fastest);
                   ``"abs"`` — L1 norm of the product (always ≥ 0);
                   ``"l1"`` — alias for ``"abs"``.

    Returns:
        ``(batch, seq)`` or ``(batch,)`` float tensor of per-token scores.

    Reference:
        Simonyan et al. 2013 "Deep Inside Convolutional Networks"
        (gradient saliency); Shrikumar et al. 2016 "Not Just a Black Box"
        (gradient × input formulation).
    """
    if reduction not in _REDUCTIONS:
        raise ValueError(f"reduction must be one of {sorted(_REDUCTIONS)}, got {reduction!r}")
    g = torch.as_tensor(grads).detach().to(torch.float32)
    e = torch.as_tensor(embeds).detach().to(torch.float32)
    return _reduce(g * e, reduction)


def integrated_gradients(
    model_fn: Callable[[Tensor], Tensor],
    embeds: Tensor,
    *,
    baseline: Tensor | None = None,
    n_steps: int = 50,
    reduction: str = "dot",
) -> Tensor:
    """Per-token attribution via Integrated Gradients (IG).

    Integrates the gradient of *model_fn* along a straight-line path from
    *baseline* to *embeds*, then multiplies by ``embeds − baseline``.  With
    ``reduction="dot"`` this satisfies the **completeness axiom**:
    ``attribution.sum() ≈ model_fn(embeds) − model_fn(baseline)``.

    Args:
        model_fn:  Callable that takes ``(batch, seq, d_model)`` embeddings and
                   returns a ``(batch,)`` scalar per example.  The model should
                   not perform embedding lookup — it receives raw float tensors.
        embeds:    ``(batch, seq, d_model)`` actual embedding vectors.
        baseline:  Reference point, same shape as *embeds*.  Defaults to
                   ``torch.zeros_like(embeds)`` (zero-embedding baseline).
        n_steps:   Number of interpolation steps (default 50).  Higher = more
                   accurate integral approximation.
        reduction: Reduction over ``d_model`` after multiplying by
                   ``(embeds − baseline)``.  Same options as
                   :func:`gradient_input_attribution`.  ``"dot"`` (default)
                   preserves the completeness property.

    Returns:
        ``(batch, seq)`` float tensor of per-token attribution scores.

    Reference:
        Sundararajan et al. 2017, ICML "Axiomatic Attribution for Deep Networks".
        https://arxiv.org/abs/1703.01365
    """
    if reduction not in _REDUCTIONS:
        raise ValueError(f"reduction must be one of {sorted(_REDUCTIONS)}, got {reduction!r}")

    e = embeds.detach().to(torch.float32)
    b = (baseline.detach().to(torch.float32) if baseline is not None
         else torch.zeros_like(e))
    delta = e - b   # (batch, seq, d_model)

    grad_accum = torch.zeros_like(e)
    alphas = torch.linspace(0.0, 1.0, n_steps, device=e.device)

    for alpha in alphas:
        interp = (b + alpha * delta).requires_grad_(True)
        out = model_fn(interp)            # (batch,)
        out.sum().backward()
        grad_accum = grad_accum + interp.grad.detach()

    avg_grads = grad_accum / n_steps      # trapezoidal approx of ∫ grad dα
    product = delta * avg_grads           # (batch, seq, d_model)
    return _reduce(product, reduction)
