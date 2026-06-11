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
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

logger = logging.getLogger("circuitry.core.lens")


@dataclass
class LayerPrediction:
    """Top-k token predictions from the logit lens at a single layer."""

    layer_idx: int          # which layer (0-based)
    token_ids: torch.Tensor  # shape (top_k,) — top-k token indices
    probs: torch.Tensor      # shape (top_k,) — corresponding softmax probabilities


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


def logit_lens_distributions(
    residuals: dict[int, torch.Tensor] | list[torch.Tensor],
    unembed: Any,
    *,
    layer_norm: torch.nn.LayerNorm | None = None,
    top_k: int = 5,
) -> list[LayerPrediction]:
    """Project residual-stream hidden states through the unembedding and return
    top-k token predictions for each layer.

    Args:
        residuals: Either a ``{layer_idx: tensor}`` dict or a list of tensors
            (list index = layer index).  Each tensor may be ``(d_model,)``,
            ``(seq, d_model)``, or ``(batch, seq, d_model)``.  The function
            collapses all leading dimensions to a single ``(d_model,)`` vector
            via ``reshape(-1, d_model).mean(0)``.
        unembed: The unembedding weight matrix; orientation auto-detected
            (same rules as ``logit_lens_kl``).
        layer_norm: Optional ``nn.LayerNorm`` applied to the collapsed residual
            before projection.
        top_k: Number of top tokens to return per layer.

    Returns:
        List of :class:`LayerPrediction`, one per layer, sorted by
        ``layer_idx`` ascending.
    """
    if isinstance(residuals, list):
        items: list[tuple[int, torch.Tensor]] = list(enumerate(residuals))
    else:
        items = list(residuals.items())

    if not items:
        return []

    # Determine d_model from first tensor
    first_tensor = _as_tensor(items[0][1])
    d_model = int(first_tensor.shape[-1])

    W = _as_tensor(unembed)
    proj_W = _resolve_unembed(W, d_model, who="logit_lens_distributions")
    W_f32 = proj_W.detach().to(torch.float32)

    results: list[LayerPrediction] = []
    for layer_idx, res in items:
        res_t = _as_tensor(res).detach().to(torch.float32)
        # Collapse all leading dims to a single (d_model,) vector
        vec = res_t.reshape(-1, d_model).mean(0)  # (d_model,)
        if layer_norm is not None:
            # LayerNorm expects at least 1-D input matching normalized_shape
            vec = layer_norm(vec)
        logits = vec @ W_f32  # (vocab,)
        probs_full = torch.softmax(logits, dim=-1)
        actual_k = min(top_k, probs_full.shape[-1])
        top_probs, top_ids = torch.topk(probs_full, actual_k)
        results.append(LayerPrediction(
            layer_idx=int(layer_idx),
            token_ids=top_ids,
            probs=top_probs,
        ))

    results.sort(key=lambda lp: lp.layer_idx)
    return results


def future_lens_kl(
    residual: Any,
    unembed: Any,
    target_logits: Any,
    *,
    horizon: int = 1,
    layer_norm: torch.nn.LayerNorm | None = None,
    chunk_size: int = 256,
) -> float:
    """KL divergence between residual's logit-lens projection and future-token targets.

    Extends logit_lens_kl to look ahead by ``horizon`` positions: the residual at
    position t is projected through the unembed and compared against
    target_logits[t + horizon].  Measures how much information about future tokens
    is already encoded at this layer.

    At horizon=0 this reduces to logit_lens_kl (same-position comparison).

    Args:
        residual:      (seq, d_model) or (batch, seq, d_model) hidden state.
        unembed:       (d_model, vocab) or (vocab, d_model) unembedding matrix.
        target_logits: (seq, vocab) or (batch, seq, vocab) — the reference logit
                       distribution.  Typically the model's final-layer logits.
                       Positions 0..horizon-1 of target_logits are NOT used as targets
                       (they have no residual to predict them from).
        horizon:       how many positions ahead to look (default 1).
        layer_norm:    optional LayerNorm applied before projecting.
        chunk_size:    token-chunking bound for (tokens, vocab) transient.

    Returns:
        Mean KL divergence (float) over the valid (seq - horizon) positions.
        Returns 0.0 if horizon >= seq (no valid positions).
    """
    res = _as_tensor(residual)
    W = _as_tensor(unembed)
    tgt = _as_tensor(target_logits)

    # --- normalise to (seq, d_model) / (seq, vocab) by mean-reducing batch ---
    if res.ndim == 3:
        res = res.float().mean(0)    # (seq, d_model)
    if tgt.ndim == 3:
        tgt = tgt.float().mean(0)    # (seq, vocab)

    seq = int(res.shape[0])
    if horizon >= seq:
        return 0.0

    d_model = int(res.shape[-1])
    proj_W = _resolve_unembed(W, d_model, who="future_lens_kl")

    res_f32 = res.detach().to(torch.float32)
    W_f32 = proj_W.detach().to(torch.float32)
    tgt_f32 = tgt.detach().to(torch.float32)

    # Slice: source positions 0..seq-horizon-1, targets horizon..seq-1
    if horizon == 0:
        res_src = res_f32          # (seq, d_model)
        tgt_slice = tgt_f32        # (seq, vocab)
    else:
        res_src = res_f32[:-horizon]       # (seq-horizon, d_model)
        tgt_slice = tgt_f32[horizon:]      # (seq-horizon, vocab)

    if layer_norm is not None:
        res_src = layer_norm(res_src)

    return _lens_kl_from_residual(res_src, W_f32, tgt_slice, chunk_size)


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
