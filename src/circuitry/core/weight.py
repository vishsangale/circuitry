"""Weight-space diagnostics. Pure functions; CPU-deterministic; no I/O.

See docs/design.md §4.1 for the contract.
"""

from __future__ import annotations

import math
from typing import Union

import numpy as np
import torch

ArrayLike = Union[torch.Tensor, np.ndarray]


def _as_2d(W: ArrayLike) -> torch.Tensor:
    t = torch.as_tensor(W)
    if t.ndim == 1:
        t = t.unsqueeze(0)
    elif t.ndim > 2:
        t = t.reshape(t.shape[0], -1)
    return t.to(dtype=torch.float32 if t.dtype not in (torch.float32, torch.float64) else t.dtype)


def singular_values(
    W: ArrayLike,
    k: int | None = None,
    max_dim: int | None = 512,
) -> torch.Tensor:
    """Singular values of ``W`` in descending order.

    ``max_dim`` caps the SVD cost on wide matrices by truncating to a
    ``max_dim``-column random subsample before the decomposition. Pass
    ``max_dim=None`` to disable. ``k`` truncates the returned vector.
    """
    M = _as_2d(W)
    if max_dim is not None and min(M.shape) > max_dim:
        # Sample columns from the longer axis to keep SVD bounded.
        axis = 1 if M.shape[1] > M.shape[0] else 0
        n = M.shape[axis]
        idx = torch.randperm(n)[:max_dim]
        M = M.index_select(axis, idx)
    s = torch.linalg.svdvals(M)
    s, _ = torch.sort(s, descending=True)
    if k is not None:
        s = s[:k]
    return s


def effective_rank(W: ArrayLike, eps: float = 1e-12) -> float:
    """Roy & Vetterli (2007) effective rank: ``exp(H(p))`` where ``p`` is the
    normalized singular-value distribution.
    """
    s = singular_values(W)
    s = s[s > eps]
    if s.numel() == 0:
        return 0.0
    p = s / s.sum()
    H = -(p * torch.log(p)).sum().item()
    return float(math.exp(H))


def stable_rank(W: ArrayLike) -> float:
    """``||W||_F^2 / ||W||_2^2``. Lower-bounds the algebraic rank and is
    numerically robust on near-singular matrices.
    """
    s = singular_values(W)
    if s.numel() == 0:
        return 0.0
    return float((s.pow(2).sum() / (s[0].pow(2))).item())


def condition_number(W: ArrayLike, eps: float = 1e-12) -> float:
    """``sigma_max / sigma_min``. Returns ``+inf`` if the smallest singular
    value is below ``eps``.
    """
    s = singular_values(W)
    if s.numel() == 0 or s[-1].item() < eps:
        return float("inf")
    return float((s[0] / s[-1]).item())
