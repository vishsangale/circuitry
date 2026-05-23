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
    ts = torch.arange(seq_len_repeat - 1)
    query_idx = ts + seq_len_repeat
    key_idx = ts + 1
    selected = t[:, :, query_idx, key_idx]  # (batch, n_heads, len(ts))
    return selected.mean(dim=(0, 2)).tolist()


def attention_pattern_entropy(attn_pattern: Any) -> list[float]:
    """Per-head mean Shannon entropy (nats). See spec §4.2 docstring."""
    t = _ensure_4d(_as_tensor(attn_pattern)).detach().to(torch.float32)
    # xlogy(p, p) = p * log(p), and xlogy(0, 0) = 0 (no NaN).
    plogp = torch.special.xlogy(t, t)
    entropy = -plogp.sum(dim=-1)  # (batch, n_heads, seq)
    return entropy.mean(dim=(0, 2)).tolist()
