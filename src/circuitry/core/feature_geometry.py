"""SAE / probe feature direction geometry primitives.

feature_interference — pairwise cosine-similarity matrix between feature directions.
feature_coverage     — fraction of activation variance explained by a set of directions.
feature_spread       — mean pairwise angular distance (diversity of the direction set).

All pure functions; no forward passes.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor


def feature_interference(
    feature_dirs: Any,
    *,
    normalize: bool = True,
) -> Tensor:
    """Pairwise cosine-similarity matrix between feature directions.

    High off-diagonal values indicate that two features point in similar
    directions, meaning interventions on one may bleed into the other.
    Useful for diagnosing redundant or competing features in an SAE or probe
    bank.

    Args:
        feature_dirs: ``(n_features, d_model)`` matrix of feature directions.
        normalize:    If True (default), L2-normalize each direction before
                      computing cosine similarity.  Set to False when the
                      directions are already normalised or when the raw dot
                      product is desired.

    Returns:
        ``(n_features, n_features)`` float tensor of pairwise cosine similarities.
        Diagonal is 1.0; off-diagonal values are in ``[−1, 1]``.
    """
    F = torch.as_tensor(feature_dirs).detach().to(torch.float32)  # (n, d)
    if normalize:
        F = F / F.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return F @ F.T   # (n, n)


def feature_coverage(
    feature_dirs: Any,
    acts: Any,
    *,
    k: int | None = None,
) -> float:
    """Fraction of activation variance explained by a set of feature directions.

    Projects the activation batch onto the feature directions and measures how
    much variance is captured.  With ``k=None`` all directions are used; with
    ``k`` set only the top-k directions (by explained variance) are included.

    Args:
        feature_dirs: ``(n_features, d_model)`` matrix of feature directions.
        acts:         ``(batch, d_model)`` or ``(..., d_model)`` activation tensor.
                      Leading dimensions are flattened.
        k:            Use only the top-k most explanatory directions.
                      ``None`` (default) uses all directions.

    Returns:
        Scalar float in ``[0, 1]``.  1.0 means the features span the full
        activation variance; 0.0 means they are orthogonal to all variance.
    """
    F = torch.as_tensor(feature_dirs).detach().to(torch.float32)   # (n, d)
    A = torch.as_tensor(acts).detach().to(torch.float32)
    d = A.shape[-1]
    flat = A.reshape(-1, d)      # (N, d)

    # Centre activations
    flat = flat - flat.mean(dim=0, keepdim=True)
    total_var = (flat ** 2).sum().item()
    if total_var < 1e-12:
        return 1.0               # degenerate: zero variance, trivially covered

    # Normalise directions
    F_norm = F / F.norm(dim=-1, keepdim=True).clamp_min(1e-8)   # (n, d)

    # Projection variance for each direction: Var(flat @ f_i)
    proj = flat @ F_norm.T       # (N, n)
    proj_var = (proj ** 2).sum(dim=0)   # (n,)

    if k is not None:
        proj_var = proj_var.topk(min(k, proj_var.shape[0])).values

    return (proj_var.sum() / total_var).clamp(max=1.0).item()


def feature_spread(
    feature_dirs: Any,
    *,
    normalize: bool = True,
) -> float:
    """Mean pairwise angular distance between feature directions (diversity score).

    A spread near ``π/2`` (≈ 1.57 radians) means directions are nearly
    orthogonal on average — high diversity.  Near 0 means all directions point
    in the same direction — low diversity / redundant feature set.

    Args:
        feature_dirs: ``(n_features, d_model)`` matrix of feature directions.
        normalize:    L2-normalise before computing angles (default True).

    Returns:
        Mean pairwise angular distance in radians.
    """
    F = torch.as_tensor(feature_dirs).detach().to(torch.float32)  # (n, d)
    if normalize:
        F = F / F.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    sim = (F @ F.T).clamp(-1.0 + 1e-6, 1.0 - 1e-6)   # (n, n) cosine sim
    angles = torch.acos(sim)                             # (n, n) in [0, π]
    # Exclude diagonal (self-similarity = 0)
    n = F.shape[0]
    if n < 2:
        return 0.0
    mask = ~torch.eye(n, dtype=torch.bool)
    return angles[mask].mean().item()
