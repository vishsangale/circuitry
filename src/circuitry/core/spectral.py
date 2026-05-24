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
    s = weight.singular_values(W)
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


def rank_trajectory(
    state_dicts: Sequence[Mapping[str, torch.Tensor]],
) -> dict[str, list[float]]:
    """Effective rank per 2D parameter across an ordered sequence of state dicts.

    Non-2D tensors (biases, layer norms) are skipped.
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
                out[k].append(weight.effective_rank(sd[k]))
    return out
