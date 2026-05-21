"""Gradient-space diagnostics. Pure; CPU-deterministic.

See docs/design.md §4.1 for the contract.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch


def layer_norm(grads: Mapping[str, torch.Tensor]) -> dict[str, float]:
    """Frobenius norm per layer. ``None`` values are skipped."""
    out: dict[str, float] = {}
    for name, g in grads.items():
        if g is None:
            continue
        out[name] = float(torch.linalg.vector_norm(g).item())
    return out


def signal_propagation_depth(
    grads_by_depth: Sequence[torch.Tensor],
    eps_ratio: float = 1e-3,
) -> int:
    """Deepest layer whose gradient norm exceeds ``eps_ratio * norm(layer_0)``.

    Returns 0 if the first layer is itself zero, ``len(grads_by_depth)`` if all
    layers are alive.
    """
    if not grads_by_depth:
        return 0
    norms = [float(torch.linalg.vector_norm(g).item()) for g in grads_by_depth]
    if norms[0] == 0.0:
        return 0
    threshold = eps_ratio * norms[0]
    depth = 0
    for n in norms:
        if n > threshold:
            depth += 1
        else:
            break
    return depth
