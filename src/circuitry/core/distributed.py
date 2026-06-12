"""Distributed reduce helpers — design §11, first increment (v1.45).

Pure tensor-in / tensor-out helpers the Recorder calls *before* a primitive
so that `core/` primitives never know about ranks (design §11 contract).
Every helper degrades to a no-op passthrough when ``torch.distributed`` is
unavailable or uninitialized, so single-process behaviour is unchanged.

This module is the foundation increment of the multi-process milestone; the
``TensorSource.WEIGHT_FULL`` / ``ACTIVATION_FULL`` recorder integration that
drives these helpers shipped in v1.46 (see design §11 and §4.4).
"""
from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

__all__ = [
    "is_distributed",
    "world_size",
    "is_main_process",
    "all_gather_concat",
    "all_gather_named",
    "full_tensor",
]


def is_distributed() -> bool:
    """True when ``torch.distributed`` is available *and* initialized."""
    import torch.distributed as dist

    return dist.is_available() and dist.is_initialized()


def world_size() -> int:
    """Process-group world size; 1 when not distributed."""
    if not is_distributed():
        return 1
    import torch.distributed as dist

    return dist.get_world_size()


def is_main_process() -> bool:
    """True on rank 0 (and always true when not distributed).

    The writer-side guard: diagnostics are emitted from the main process
    only (design §11 rank-0 semantics).
    """
    if not is_distributed():
        return True
    import torch.distributed as dist

    return dist.get_rank() == 0


def all_gather_concat(t: Tensor, *, dim: int = 0, group: Any = None) -> Tensor:
    """All-gather *t* across ranks and concatenate along *dim*.

    The activation-side reduce: per-rank activation batches become the full
    cross-rank batch before a primitive sees them.  Requires the same tensor
    shape on every rank (the recorder guarantees this for activation hooks
    capturing same-shaped micro-batches; pad or crop upstream otherwise).
    Passthrough (returns *t* unchanged) when not distributed or
    ``world_size == 1``.

    Args:
        t: local tensor shard.
        dim: concatenation dimension (default 0, the batch dim).
        group: optional process group (default: the global group).

    Returns:
        ``(world_size * t.shape[dim], ...)`` concatenated tensor on every
        rank (all-gather is symmetric).
    """
    ws = world_size()
    if ws == 1:
        return t
    import torch.distributed as dist

    local = t.contiguous()
    gathered = [torch.empty_like(local) for _ in range(ws)]
    dist.all_gather(gathered, local, group=group)
    return torch.cat(gathered, dim=dim)


def all_gather_named(
    named: dict[str, Tensor], *, dim: int = 0, group: Any = None
) -> dict[str, Tensor]:
    """All-gather a ``{name: tensor}`` dict across ranks; concat per name.

    The recorder-facing activation reduce (``ACTIVATION_FULL``, design §11).
    Uses ``all_gather_object`` — one collective per call regardless of how
    many names are present — so it tolerates per-rank shape differences
    (e.g. uneven batch shards) and names captured on only a subset of ranks
    (those are concatenated over the ranks that have them). Tensors travel as
    detached CPU copies; this runs at emit cadence, not in the hot path.

    Every rank must call this the same number of times per emit step (it is
    a collective); the result is symmetric — all ranks receive the full
    concatenation. Passthrough when not distributed or ``world_size == 1``.

    Args:
        named: this rank's local ``{name: tensor}`` contribution (may be
            empty — the rank still participates).
        dim: concatenation dimension (default 0, the batch dim).
        group: optional process group.

    Returns:
        ``{name: concatenated tensor}`` over rank order.
    """
    if world_size() == 1:
        return named
    import torch.distributed as dist

    local = {
        name: t.detach().to("cpu", copy=True) for name, t in named.items()
    }
    gathered: list[dict[str, Tensor] | None] = [None] * world_size()
    dist.all_gather_object(gathered, local, group=group)
    out: dict[str, list[Tensor]] = {}
    for rank_dict in gathered:
        for name, t in (rank_dict or {}).items():
            out.setdefault(name, []).append(t)
    return {name: torch.cat(parts, dim=dim) for name, parts in out.items()}


def full_tensor(param: Any) -> Tensor:
    """Materialize the full, unsharded value of a (possibly sharded) parameter.

    The weight-side reduce. Handles:

    - ``DTensor`` (FSDP2 / tensor-parallel sharded params): gathers via
      ``DTensor.full_tensor()``.
    - plain ``Tensor`` / ``nn.Parameter``: returned as-is (already full).

    FSDP1 flat-param sharding cannot be resolved from a single parameter
    object — the recorder-side integration will use
    ``summon_full_params``-style gathering for that case (follow-up release;
    see design §11). Until then, passing an FSDP1-sharded param returns the
    shard unchanged, which is exactly the wrong-numbers hazard §11 warns
    about — the recorder, not this helper, is responsible for refusing that
    configuration.
    """
    if hasattr(param, "full_tensor"):  # torch.distributed.tensor.DTensor
        return param.full_tensor()
    return param
