"""Per-head attention-pattern diagnostics. See docs/design.md §4 and v0.9 spec §4.2.

induction_score — Olsson et al. 2022 prefix-matching probability on a
repeated-random-token probe (https://arxiv.org/abs/2209.11895).

copy_suppression_score — McDougall et al. 2023 same-token attention on the
same repeated-random-token probe: at position T+i, how much does each head
attend back to position i (the prior occurrence of the same token)?

attention_sink_score — Xiao et al. 2023 per-head mean attention weight on a
designated sink position (default 0, the initial / BOS token). Unlike the two
probe-based diagnostics above, this operates on the live training-forward
attention pattern so it reflects the actual training distribution.

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


def copy_suppression_score(attn_pattern: Any, *, seq_len_repeat: int) -> list[float]:
    """Per-head same-token attention on a repeated-random-token probe.

    Uses the same probe structure as :func:`induction_score` — a sequence of
    the form ``[t_0 … t_{T-1} | t_0 … t_{T-1}]`` — but selects the attention
    weight at query position ``T+i`` to key position ``i`` (the *prior*
    occurrence of the same token) rather than ``i+1`` (the induction offset).

    High score → the head strongly attends to earlier positions carrying the
    same token, the characteristic attention pattern of copy-suppression heads
    (McDougall et al. 2023). The score is complementary to
    :func:`induction_score`: induction heads peak on offset +1; copy-suppression
    heads peak on offset 0.

    Returns a ``list[float]`` of per-head means over the batch and sequence
    axes, identical in shape to :func:`induction_score`.
    """
    t = _ensure_4d(_as_tensor(attn_pattern)).detach().to(torch.float32)
    _batch, n_heads, seq, seq2 = t.shape
    if seq != seq2:
        raise ValueError(
            f"copy_suppression_score: expected square attn pattern, got {seq}x{seq2}"
        )
    if seq < 2 * seq_len_repeat:
        raise ValueError(
            f"copy_suppression_score: seq={seq} must be >= 2 * seq_len_repeat="
            f"{2 * seq_len_repeat}"
        )
    ts = torch.arange(seq_len_repeat, device=t.device)
    query_idx = ts + seq_len_repeat  # second-half positions: T, T+1, ..., 2T-1
    key_idx = ts                     # first-half positions:  0,   1, ...,  T-1
    selected = t[:, :, query_idx, key_idx]  # (batch, n_heads, seq_len_repeat)
    return selected.mean(dim=(0, 2)).tolist()


def attention_sink_score(
    attn_pattern: Any, *, sink_pos: int = 0
) -> list[float]:
    """Per-head mean attention weight on a designated sink position.

    Attention sinks are positions that accumulate disproportionately large
    attention weights from all subsequent query positions regardless of
    semantic relevance — the initial token (position 0, often BOS) is the
    canonical sink in decoder-only LLMs (Xiao et al. 2023,
    https://arxiv.org/abs/2309.17453). A head with ``attention_sink_score``
    near 1 directs almost all its attention to the sink position; near 0 it
    ignores it.

    Unlike :func:`induction_score` and :func:`copy_suppression_score` this
    primitive operates on the **live training-forward** attention pattern, so
    it measures the real training distribution rather than a synthetic probe.

    Args:
        attn_pattern: ``(batch, n_heads, seq, seq)`` or ``(n_heads, seq, seq)``
            attention weights (not required to be normalized).
        sink_pos: Key position to treat as the sink.  ``0`` (default) is the
            initial / BOS token; ``-1`` selects the last position.

    Returns:
        ``list[float]`` of per-head means over ``(batch, query)`` dimensions.
        Length equals ``n_heads``.
    """
    t = _ensure_4d(_as_tensor(attn_pattern)).detach().to(torch.float32)
    _batch, n_heads, seq, _seq2 = t.shape
    pos = sink_pos % seq  # normalize negative indices
    # t[:, :, :, pos] has shape (batch, n_heads, seq_q); mean over batch+query.
    return t[:, :, :, pos].mean(dim=(0, 2)).tolist()


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
