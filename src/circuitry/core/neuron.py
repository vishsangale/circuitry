"""Per-neuron activation statistics.

neuron_stats — fast vectorised per-neuron diagnostics over a batch of activations:
    dead fraction, mean, std, max, and excess kurtosis.

Useful for identifying dead neurons, diagnosing activation collapse, detecting
polysemanticity signals (high kurtosis), and comparing pre- and post-training
activation distributions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor


@dataclass
class NeuronStats:
    """Per-neuron statistics computed over a batch of activations.

    All tensor attributes have shape ``(d,)`` where ``d`` is the feature
    dimension (last axis of the input).

    Attributes:
        mean:          Per-neuron mean activation.
        std:           Per-neuron standard deviation.
        max:           Per-neuron maximum activation across all examples.
        dead_fraction: Fraction of neurons whose maximum activation is below
                       the dead threshold.  A neuron that never fires above
                       the threshold is considered dead.
        kurtosis:      Per-neuron excess kurtosis ``E[(x−μ)⁴]/σ⁴ − 3``.
                       High kurtosis ≈ sparse / peaky distribution; near 0 ≈
                       Gaussian; negative ≈ flat / bimodal.
    """

    mean: Tensor
    std: Tensor
    max: Tensor
    dead_fraction: float
    kurtosis: Tensor


def neuron_stats(acts: Any, *, threshold: float = 0.0) -> NeuronStats:
    """Compute per-neuron statistics over a batch of activations.

    Args:
        acts:      Activation tensor of shape ``(..., d)``.  Any leading
                   dimensions (batch, sequence, etc.) are flattened before
                   computing statistics, so the result always has shape ``(d,)``.
        threshold: A neuron is counted as *dead* if its maximum activation
                   across all examples is strictly below this value (default 0).

    Returns:
        :class:`NeuronStats` with per-neuron ``mean``, ``std``, ``max``,
        ``dead_fraction``, and ``kurtosis`` tensors of shape ``(d,)``.
    """
    t = torch.as_tensor(acts).detach().to(torch.float32)
    d = t.shape[-1]
    flat = t.reshape(-1, d)             # (n, d)

    mean = flat.mean(dim=0)             # (d,)
    std = flat.std(dim=0, unbiased=True)
    max_val = flat.max(dim=0).values    # (d,)

    dead_fraction = (max_val < threshold).float().mean().item()

    # Excess kurtosis: E[(x − μ)⁴] / σ⁴ − 3
    centered = flat - mean.unsqueeze(0)
    fourth_moment = (centered ** 4).mean(dim=0)
    var = std ** 2
    kurtosis = fourth_moment / (var ** 2).clamp_min(1e-8) - 3.0

    return NeuronStats(
        mean=mean,
        std=std,
        max=max_val,
        dead_fraction=dead_fraction,
        kurtosis=kurtosis,
    )
