"""Activation steering vector primitive. See docs/design.md §4.

Pure function: no I/O, no .cuda(), no nn.Module state.

Reference: Rimsky et al. 2024 "Steering Llama 2 via Contrastive Activation
Addition" (CAA) — https://arxiv.org/abs/2312.06681
"""
from __future__ import annotations

import torch
from torch import Tensor


def steer_vector(
    positive_acts: Tensor,
    negative_acts: Tensor,
    *,
    normalize: bool = True,
) -> Tensor:
    """Compute a contrastive steering vector (Rimsky et al. 2024 CAA).

    Averages activations across the batch dimension for each polarity,
    returns ``mean_positive - mean_negative``. If ``normalize=True``,
    divides by the L2 norm so the returned vector is unit length.

    Args:
        positive_acts: ``(batch, d_model)`` or ``(d_model,)`` — activations
            for the "positive" / steered-toward direction.
        negative_acts: same shape — activations for the opposite direction.
        normalize: if ``True``, return a unit vector.

    Returns:
        Tensor of shape ``(d_model,)``.

    Raises:
        ValueError: if ``normalize=True`` and the difference vector has
            near-zero norm.
    """
    pos = torch.as_tensor(positive_acts).detach().to(torch.float32)
    neg = torch.as_tensor(negative_acts).detach().to(torch.float32)

    # Mean-reduce batch dim if present
    if pos.ndim == 2:
        pos = pos.mean(0)
    if neg.ndim == 2:
        neg = neg.mean(0)

    diff = pos - neg

    if normalize:
        norm = diff.norm()
        if norm < 1e-8:
            raise ValueError(
                "steer_vector: difference vector has near-zero norm; check inputs"
            )
        diff = diff / norm

    return diff
