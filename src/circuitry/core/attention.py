"""Per-head attention-pattern diagnostics. See docs/design.md §4 and v0.9 spec §4.2.

induction_score — Olsson et al. 2022 prefix-matching probability on a
repeated-random-token probe (https://arxiv.org/abs/2209.11895).

attention_pattern_entropy — per-head Shannon entropy (nats) of the
attention distribution over keys.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor


def _as_tensor(x: Any) -> Tensor:
    if isinstance(x, Tensor):
        return x
    return torch.as_tensor(x)


def _ensure_4d(t: Tensor) -> Tensor:
    if t.ndim == 3:
        return t.unsqueeze(0)
    if t.ndim == 4:
        return t
    raise ValueError(
        f"expected 3-D or 4-D attention pattern, got shape {tuple(t.shape)}"
    )


def induction_score(attn_pattern: Any, *, seq_len_repeat: int) -> list[float]:
    """Per-head prefix-matching probability on a repeated-random-token probe.
    See spec §4.2 docstring for the full contract.
    """
    t = _ensure_4d(_as_tensor(attn_pattern)).detach().to(torch.float32)
    _batch, n_heads, seq, seq2 = t.shape
    if seq != seq2:
        raise ValueError(
            f"induction_score: expected square attn pattern, got {seq}x{seq2}"
        )
    if seq < 2 * seq_len_repeat:
        raise ValueError(
            f"induction_score: seq={seq} must be >= 2 * seq_len_repeat="
            f"{2 * seq_len_repeat}"
        )
    ts = torch.arange(seq_len_repeat - 1, device=t.device)
    query_idx = ts + seq_len_repeat
    key_idx = ts + 1
    selected = t[:, :, query_idx, key_idx]  # (batch, n_heads, len(ts))
    return selected.mean(dim=(0, 2)).tolist()


def attention_pattern_entropy(
    attn_pattern: Any, *, valid_mask: Any | None = None
) -> list[float]:
    """Per-head mean Shannon entropy (nats). See spec §4.2 docstring.

    ``valid_mask`` (optional) restricts the per-head mean to *valid query
    positions*. Left-padded models (recsys / decoder inference) feed PAD query
    rows an all-``-inf`` key set; the resulting softmax row is all-``NaN`` and
    poisons a naive mean. Pass a boolean mask — ``True`` marks a valid query
    row — broadcastable to the per-head entropy shape ``(B, H, T_query)``.
    Common forms: ``(B, T)``, ``(B, 1, T)`` or ``(B, H, T)``; a 2-D ``(B, T)``
    mask is auto-expanded across the head axis.

    Even **without** a mask the mean is NaN-aware: query rows whose entropy is
    ``NaN`` (a fully-``-inf``-masked softmax) are dropped from the per-head
    average. For an unpadded pattern (no NaN rows) this is identical to the
    plain mean, so the change is backward-compatible.
    """
    t = _ensure_4d(_as_tensor(attn_pattern)).detach().to(torch.float32)
    # Normalize each query row to a probability distribution before computing
    # entropy. Softmax rows already sum to 1 (no-op within fp tolerance); for
    # sigmoid / linear attention whose weights don't sum to 1 this makes the
    # entropy normalization-invariant (a pure concentration measure) so it is
    # comparable across attention variants. A fully-masked row (sums to 0) stays
    # all-zero after the eps-clamped divide, and xlogy(0, 0) = 0 -> entropy 0.
    row_sum = t.sum(dim=-1, keepdim=True)
    p = t / row_sum.clamp_min(torch.finfo(t.dtype).eps)
    # xlogy(p, p) = p * log(p), and xlogy(0, 0) = 0 (no NaN).
    plogp = torch.special.xlogy(p, p)
    entropy = -plogp.sum(dim=-1)  # (batch, n_heads, seq)
    if valid_mask is not None:
        mask = _as_tensor(valid_mask).to(torch.bool)
        if mask.ndim == entropy.ndim - 1:
            # (B, T) -> (B, 1, T): insert the head axis for the common
            # per-(batch, position) padding mask.
            mask = mask.unsqueeze(1)
        mask = torch.broadcast_to(mask, entropy.shape)
        # Mark invalid query rows NaN so nanmean ignores them. NaN already
        # present at invalid rows (fully-masked softmax) is dropped the same way.
        entropy = entropy.masked_fill(~mask, float("nan"))
    # NaN-aware per-head mean over (batch, query) — drops fully-masked PAD rows.
    return torch.nanmean(entropy, dim=(0, 2)).tolist()
