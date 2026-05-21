"""Activation-space diagnostics. Pure; CPU-deterministic.

See docs/design.md §4.1 for the contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import numpy as np
import torch

ArrayLike = Union[torch.Tensor, np.ndarray]


@dataclass(frozen=True)
class NormStats:
    mean: float
    std: float
    max: float
    frac_above_k_median: float


def _as_tensor(x: ArrayLike) -> torch.Tensor:
    return torch.as_tensor(x).to(dtype=torch.float32)


def dead_fraction(x: ArrayLike, threshold: float = 0.0) -> float:
    """Fraction of activations at or below ``threshold``."""
    t = _as_tensor(x)
    if t.numel() == 0:
        return 0.0
    return float((t <= threshold).float().mean().item())


def norm_stats(x: ArrayLike, k: float = 3.0) -> NormStats:
    """Per-element norm statistics. ``frac_above_k_median`` is the fraction of
    elements whose absolute value exceeds ``k * median(|x|)`` — a cheap
    heavy-tail indicator.
    """
    t = _as_tensor(x).flatten()
    if t.numel() == 0:
        return NormStats(0.0, 0.0, 0.0, 0.0)
    abs_t = t.abs()
    med = abs_t.median().item()
    return NormStats(
        mean=float(t.mean().item()),
        std=float(t.std(unbiased=False).item()),
        max=float(abs_t.max().item()),
        frac_above_k_median=float((abs_t > k * med).float().mean().item()) if med > 0 else 0.0,
    )
