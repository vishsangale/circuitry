"""Gradient-space diagnostics. Pure; CPU-deterministic.

See docs/design.md §4.1 for the contract.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch


def grad_norm_per_module(grads: Mapping[str, torch.Tensor]) -> dict[str, float]:
    """Frobenius (L2) norm of each module's gradient tensor, computed in
    float32 for precision on low-precision (bf16/fp16) gradients. ``None``
    values are skipped."""
    out: dict[str, float] = {}
    for name, g in grads.items():
        if g is None:
            continue
        # Sparse gradients (e.g. nn.Embedding(sparse=True), standard in recsys)
        # have no vector_norm kernel — densify first (F31).
        if g.is_sparse:
            g = g.to_dense()
        out[name] = float(torch.linalg.vector_norm(g.to(torch.float32)).item())
    return out


def total_grad_norm(per_module_norms: Mapping[str, float]) -> float:
    """Global gradient L2 norm: ``sqrt(sum(n**2))`` over the per-module norms
    produced by :func:`grad_norm_per_module`. Returns ``0.0`` for an empty
    mapping."""
    return float(sum(n * n for n in per_module_norms.values()) ** 0.5)


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
