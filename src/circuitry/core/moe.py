"""MoE routing diagnostics — pure functions over router outputs.

Mixture-of-experts models route each token through a small subset of experts;
the router distribution is a first-class training-health signal (collapsed
routing, dead experts, degenerate pathways).  Reference: MoE pathway
complexity, arXiv:2506.21551; load-balancing losses, Shazeer et al. 2017.

  routing_entropy     — mean per-token entropy of the router distribution.
  expert_load_balance — effective fraction of experts receiving traffic
                        (1 = perfectly uniform).
  pathway_complexity  — effective number of distinct cross-layer expert paths.

All functions are pure: tensors in, floats out; no hooks, no I/O.
"""
from __future__ import annotations

from collections import Counter
from typing import Any

import torch
from torch import Tensor


def _as_float(x: Any) -> Tensor:
    return torch.as_tensor(x).detach().to(torch.float32)


def routing_entropy(gate_scores: Any, *, from_logits: bool = True) -> float:
    """Mean per-token entropy (nats) of the router distribution.

    Low entropy = confident routing (each token strongly prefers few
    experts); entropy near ``log(n_experts)`` = near-uniform (undertrained or
    collapsed-to-uniform router).

    Args:
        gate_scores: ``(..., n_experts)`` router outputs; leading dims are
            flattened to tokens.
        from_logits: when True (default) applies softmax over the last dim;
            set False if *gate_scores* already holds probabilities.

    Returns:
        Mean token-level Shannon entropy in nats.
    """
    t = _as_float(gate_scores)
    if t.ndim < 1 or t.shape[-1] < 2:
        raise ValueError(
            f"gate_scores must have a final n_experts dim >= 2, got shape {tuple(t.shape)}"
        )
    flat = t.reshape(-1, t.shape[-1])
    if from_logits:
        logp = torch.log_softmax(flat, dim=-1)
        p = logp.exp()
    else:
        p = flat / flat.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        logp = p.clamp_min(1e-12).log()
    return float(-(p * logp).sum(dim=-1).mean().item())


def expert_load_balance(expert_ids: Any, n_experts: int) -> float:
    """Effective fraction of experts receiving traffic, in ``(0, 1]``.

    Computes the empirical load distribution over experts from the routed
    assignments and returns ``exp(H(load)) / n_experts`` — the exponentiated
    entropy ("effective number of experts in use") normalised by the expert
    count.  1.0 = perfectly uniform load; → ``1/n_experts`` as all traffic
    collapses onto a single expert.

    Args:
        expert_ids: integer tensor of routed expert indices, any shape
            (e.g. ``(tokens,)`` top-1 assignments or ``(tokens, k)`` top-k).
        n_experts: total number of experts (defines the support; experts
            receiving zero traffic lower the score).

    Returns:
        Scalar in ``(0, 1]``.
    """
    ids = torch.as_tensor(expert_ids).detach().reshape(-1).to(torch.int64)
    if ids.numel() == 0:
        raise ValueError("expert_ids is empty")
    if int(ids.max()) >= n_experts or int(ids.min()) < 0:
        raise ValueError(
            f"expert id out of range [0, {n_experts}): "
            f"min={int(ids.min())}, max={int(ids.max())}"
        )
    load = torch.bincount(ids, minlength=n_experts).float()
    p = load / load.sum()
    nz = p[p > 0]
    entropy = float(-(nz * nz.log()).sum().item())
    return float(torch.exp(torch.tensor(entropy)).item()) / n_experts


def pathway_complexity(expert_ids_per_layer: list[Any]) -> float:
    """Effective number of distinct cross-layer expert paths.

    A sample's *path* is the tuple of experts it is routed through across
    layers (top-k sets are order-normalised within a layer).  Returns
    ``exp(H)`` of the empirical path distribution — the effective path count:
    1.0 when every sample takes the same path, up to ``n_samples`` when all
    paths are distinct.  Low values on a deep MoE indicate pathway collapse.

    Args:
        expert_ids_per_layer: one integer tensor per layer, each ``(n,)``
            (top-1) or ``(n, k)`` (top-k) with the SAME ``n`` samples in the
            same order across layers.

    Returns:
        Effective number of distinct paths (≥ 1.0).

    Reference: arXiv:2506.21551.
    """
    if not expert_ids_per_layer:
        raise ValueError("expert_ids_per_layer is empty")
    per_layer: list[list[tuple[int, ...]]] = []
    n = None
    for li, ids in enumerate(expert_ids_per_layer):
        t = torch.as_tensor(ids).detach().to(torch.int64)
        if t.ndim == 1:
            t = t.unsqueeze(-1)
        if t.ndim != 2:
            raise ValueError(
                f"layer {li}: expected (n,) or (n, k) expert ids, got shape {tuple(t.shape)}"
            )
        if n is None:
            n = t.shape[0]
        elif t.shape[0] != n:
            raise ValueError(
                f"layer {li}: {t.shape[0]} samples but layer 0 has {n} — "
                "all layers must route the same samples"
            )
        per_layer.append([tuple(sorted(row.tolist())) for row in t])

    paths = Counter(
        tuple(layer[s] for layer in per_layer) for s in range(n or 0)
    )
    total = sum(paths.values())
    entropy = 0.0
    for count in paths.values():
        p = count / total
        entropy -= p * torch.log(torch.tensor(p)).item()
    return float(torch.exp(torch.tensor(entropy)).item())
