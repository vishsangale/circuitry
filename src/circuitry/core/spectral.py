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
