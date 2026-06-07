"""Lens KL diagnostics. See docs/design.md §4 and the v0.9 / v1.10 specs §4.1.

Per-layer KL between a lens projection of the residual stream and the model's
final logits.

- ``logit_lens_kl`` — the parameter-free logit lens (Nostalgebraist 2020,
  https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru): project the residual
  straight through the unembedding.
- ``tuned_lens_kl`` — the tuned lens (Belrose et al. 2023,
  https://arxiv.org/abs/2303.08112): apply a learned per-layer affine
  translator ``h ↦ A h + b`` before projecting, so the lens distribution is no
  longer confounded by the early/mid-layer basis mismatch. With ``A = I`` and
  ``b = 0`` it reduces exactly to the logit lens.

Both are pure functions: tensors in, a float out. float32 upcast, token-axis
chunking, device-deterministic (no ``.cuda()``).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor

logger = logging.getLogger("circuitry.core.lens")


def _as_tensor(x: Any) -> Tensor:
    if isinstance(x, Tensor):
        return x
    return torch.as_tensor(x)


def _resolve_unembed(W: Tensor, d_model: int, *, who: str) -> Tensor:
    """Return the unembed oriented as (d_model, vocab).

    Orientation auto-detect + the d_model == vocab square edge case, shared by
    the logit and tuned lenses. ``who`` names the caller for error/warn text.
    """
    if W.ndim != 2:
        raise ValueError(f"{who}: unembed must be 2-D, got shape {tuple(W.shape)}")
    if W.shape[0] == d_model and W.shape[1] == d_model:
        logger.warning(
            "%s: unembed shape %s is square (d_model == vocab); "
            "orientation cannot be inferred, assuming (d_model, vocab) layout.",
            who, tuple(W.shape),
        )
        return W
    if W.shape[0] == d_model:
        return W
    if W.shape[1] == d_model:
        return W.t()
    raise ValueError(
        f"{who}: unembed shape {tuple(W.shape)} has no dim matching d_model={d_model}"
    )


def _lens_kl_from_residual(
    res_f32: Tensor,
    proj_W_f32: Tensor,
    fl_f32: Tensor,
    chunk_size: int,
) -> float:
    """Mean KL(softmax(res @ proj_W) || softmax(final)) over the token axis.

    ``res_f32`` is the (already float32, already lens-transformed + normalized)
    residual; ``proj_W_f32`` is oriented (d_model, vocab). Chunk over the
    flattened token axis so the (tokens, vocab) lens-logits transient never
    materializes for the whole batch at once. Exact up to float accumulation
    order. Stays on the input's device (no .cuda()).
    """
    res_flat = res_f32.reshape(-1, res_f32.shape[-1])  # (N, d_model)
    fl_flat = fl_f32.reshape(-1, fl_f32.shape[-1])      # (N, vocab)
    n = res_flat.shape[0]
    if n == 0:
        return 0.0
    kl_sum = res_flat.new_zeros(())
    for start in range(0, n, max(1, chunk_size)):
        r = res_flat[start:start + chunk_size]
        f = fl_flat[start:start + chunk_size]
        lens_logits = r @ proj_W_f32
        log_q = torch.log_softmax(lens_logits, dim=-1)  # lens distribution
        log_p = torch.log_softmax(f, dim=-1)             # final distribution
        q = log_q.exp()
        kl_sum = kl_sum + (q * (log_q - log_p)).sum(dim=-1).sum()
    return float((kl_sum / n).item())


def logit_lens_kl(
    residual: Any,
    unembed: Any,
    final_logits: Any,
    *,
    layer_norm: Callable[[Tensor], Tensor] | None = None,
    chunk_size: int = 256,
) -> float:
    """KL(softmax(layer_norm(residual) @ unembed) || softmax(final_logits)),
    mean over leading (batch, seq) dims.

    See docstring in spec §4.1 for the orientation auto-detect rule and the
    d_model == vocab edge case.
    """
    res = _as_tensor(residual)
    W = _as_tensor(unembed)
    fl = _as_tensor(final_logits)

    d_model = int(res.shape[-1])
    proj_W = _resolve_unembed(W, d_model, who="logit_lens_kl")

    res_f32 = res.detach().to(torch.float32)
    if layer_norm is not None:
        res_f32 = layer_norm(res_f32)
    W_f32 = proj_W.detach().to(torch.float32)
    fl_f32 = fl.detach().to(torch.float32)
    return _lens_kl_from_residual(res_f32, W_f32, fl_f32, chunk_size)


def tuned_lens_kl(
    residual: Any,
    translator: tuple[Any, Any],
    unembed: Any,
    final_logits: Any,
    *,
    layer_norm: Callable[[Tensor], Tensor] | None = None,
    chunk_size: int = 256,
) -> float:
    """KL(softmax(layer_norm(A @ residual + b) @ unembed) || softmax(final)),
    mean over leading (batch, seq) dims.

    ``translator`` is the per-layer affine ``(A, b)``: ``A`` is ``(d_model,
    d_model)`` and ``b`` is ``(d_model,)`` (broadcastable). The residual is
    mapped ``h ↦ A h + b`` (applied as ``residual @ Aᵀ + b`` for a trailing
    d_model axis) *before* the final layer-norm and unembed projection. With
    ``A = I`` and ``b = 0`` this reduces exactly to ``logit_lens_kl``.

    Pure: float32 upcast, orientation auto-detect + d_model == vocab edge case
    identical to ``logit_lens_kl``, token-axis chunking, device-deterministic
    (no .cuda()). ``A`` / ``b`` are moved to the residual's device.
    """
    res = _as_tensor(residual)
    A = _as_tensor(translator[0])
    b = _as_tensor(translator[1])
    W = _as_tensor(unembed)
    fl = _as_tensor(final_logits)

    d_model = int(res.shape[-1])
    if A.ndim != 2 or A.shape[0] != d_model or A.shape[1] != d_model:
        raise ValueError(
            f"tuned_lens_kl: translator A must be ({d_model}, {d_model}), "
            f"got shape {tuple(A.shape)}"
        )
    if b.shape[-1] != d_model:
        raise ValueError(
            f"tuned_lens_kl: translator b must have last dim {d_model}, "
            f"got shape {tuple(b.shape)}"
        )
    proj_W = _resolve_unembed(W, d_model, who="tuned_lens_kl")

    res_f32 = res.detach().to(torch.float32)
    A_f32 = A.detach().to(torch.float32).to(res_f32.device)
    b_f32 = b.detach().to(torch.float32).to(res_f32.device)
    # h ↦ A h + b, applied across the trailing d_model axis.
    res_f32 = res_f32 @ A_f32.t() + b_f32
    if layer_norm is not None:
        res_f32 = layer_norm(res_f32)
    W_f32 = proj_W.detach().to(torch.float32)
    fl_f32 = fl.detach().to(torch.float32)
    return _lens_kl_from_residual(res_f32, W_f32, fl_f32, chunk_size)
