"""Spectral diagnostics across snapshots. Pure; CPU-deterministic.

See docs/design.md §4.1 for the contract.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import torch

from circuitry.core import weight

ArrayLike = torch.Tensor | np.ndarray


def esd(W: ArrayLike, bins: int = 100) -> tuple[torch.Tensor, torch.Tensor]:
    """Empirical spectral density: a histogram of singular values.

    Returns ``(bin_edges, counts)`` so the result is drop-in for
    ``torch.utils.tensorboard.SummaryWriter.add_histogram`` after a small
    reshape, and also human-plottable.
    """
    return _esd_from_sv(weight.singular_values(W), bins)


def _esd_from_sv(s: torch.Tensor, bins: int = 100) -> tuple[torch.Tensor, torch.Tensor]:
    if s.numel() == 0:
        edges = torch.linspace(0.0, 1.0, bins + 1, device=s.device)
        counts = torch.zeros(bins, device=s.device)
        return edges, counts
    counts = torch.histc(s, bins=bins, min=float(s.min().item()), max=float(s.max().item()))
    edges = torch.linspace(float(s.min().item()), float(s.max().item()), bins + 1,
                           device=s.device)
    if counts.sum() == 0:  # degenerate (all values identical)
        counts[0] = s.numel()
    return edges, counts


def _flatten_to_2d(t: torch.Tensor) -> torch.Tensor:
    """Fold a ≥2-D parameter to 2-D as ``[shape[0], -1]`` for the rank
    primitive. Correct for conv weights (``[out, in, kh, kw] -> [out, in*kh*kw]``);
    ``effective_rank`` itself rejects >2-D so the fold must happen here."""
    t = torch.as_tensor(t)
    return t if t.ndim <= 2 else t.reshape(t.shape[0], -1)


def rank_trajectory(
    state_dicts: Sequence[Mapping[str, torch.Tensor]],
) -> dict[str, list[float]]:
    """Effective rank per 2D+ parameter across an ordered sequence of state dicts.

    1-D tensors (biases, layer norms) are skipped. Conv and other ≥2-D weights
    are folded to 2-D as ``[out, -1]`` before the rank is computed.
    """
    if not state_dicts:
        return {}
    keys = [k for k, v in state_dicts[0].items() if torch.as_tensor(v).ndim >= 2]
    out: dict[str, list[float]] = {k: [] for k in keys}
    for sd in state_dicts:
        for k in keys:
            if k not in sd:
                out[k].append(float("nan"))
            else:
                out[k].append(weight.effective_rank(_flatten_to_2d(sd[k])))
    return out


def spectral_edge_gap(
    W_prev: ArrayLike,
    W_curr: ArrayLike,
    *,
    k: int = 5,
) -> float:
    """Spectral gap in the top-k singular values of the weight update ΔW.

    Computes ``s[k-1] / s[k]`` (1-indexed; 0-indexed: ``s[k-1] / s[k]``) of
    the sorted singular values of ``W_curr − W_prev``.  A growing gap between
    the ``k``-th and ``(k+1)``-th singular value fingerprints circuit formation
    during grokking: structured computation crystallises into a low-rank update.

    Complements :func:`~circuitry.core.dynamics.grokking_step` (which detects
    *when* a phase transition occurs) by characterising *what* the update looks
    like spectrally (sharp low-rank update = circuit formation).

    Args:
        W_prev: (m, n) weight matrix before the update.
        W_curr: (m, n) weight matrix after the update.
        k:      rank boundary; must satisfy ``1 <= k < min(m, n)``.

    Returns:
        ``s[k-1] / s[k]`` where singular values are sorted descending.
        Returns ``1.0`` if all singular values beyond index k−1 are < 1e-12
        (degenerate / rank-deficient update).

    Reference: arXiv:2604.06256 "Spectral Signatures of Circuit Formation".
    """
    import torch as _torch

    a = _torch.as_tensor(W_prev).to(_torch.float64)
    b = _torch.as_tensor(W_curr).to(_torch.float64)
    if a.ndim > 2:
        a = a.reshape(a.shape[0], -1)
    if b.ndim > 2:
        b = b.reshape(b.shape[0], -1)
    delta = b - a
    sv = _torch.linalg.svdvals(delta)  # descending order
    if k < 1:
        raise ValueError(f"spectral_edge_gap: k must be >= 1, got {k}")
    if k >= sv.shape[0]:
        raise ValueError(
            f"spectral_edge_gap: k={k} >= min(m,n)={sv.shape[0]}; "
            "reduce k or use a larger weight matrix"
        )
    s_k = float(sv[k - 1].item())
    s_k1 = float(sv[k].item())
    if s_k1 < 1e-12:
        return float(s_k / 1e-12) if s_k > 1e-12 else 1.0
    return s_k / s_k1
