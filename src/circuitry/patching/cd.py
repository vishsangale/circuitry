"""Contextual Decomposition for Transformers (CD-T).

Propagates per-source-token contribution scores through the attention stack.
Jain et al., ICLR 2025, arXiv:2407.00886.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

__all__ = ["CDResult", "cd_token_contributions"]


@dataclass
class CDResult:
    """Per-token contribution scores after CD-T propagation.

    contributions: (seq_len, seq_len) float tensor.
        contributions[q, s] = fraction of position q's representation
        attributable to source token s. Each row sums to 1.0.
    """
    contributions: Tensor


def cd_token_contributions(
    attn_weights: list[Tensor],
    *,
    head_agg: str = "mean",
    add_residual: bool = True,
) -> CDResult:
    """Propagate per-token contributions through the attention stack.

    Args:
        attn_weights: Per-layer attention weight tensors. Each has shape
            (n_heads, seq_len, seq_len) or (batch, n_heads, seq_len, seq_len).
            Rows must sum to 1 (softmax weights). Batch dim is averaged out.
        head_agg: "mean" (default) or "max" — how to aggregate heads.
        add_residual: When True (default) each layer blends C 50/50 with
            the attention-redistributed C_new, approximating the residual stream.

    Returns:
        CDResult with contributions[q, s] = fraction of q's representation
        from source s. Rows sum to 1.0.

    Reference: Jain et al. "CD-T: Contextual Decomposition for Transformers",
    ICLR 2025, arXiv:2407.00886.
    """
    if not attn_weights:
        raise ValueError("cd_token_contributions: attn_weights must be non-empty")

    def _normalise(A: Tensor) -> Tensor:
        if A.ndim == 4:
            A = A.float().mean(dim=0)  # average over batch
        elif A.ndim == 3:
            A = A.float()
        else:
            raise ValueError(f"Expected 3-D or 4-D attn_weights, got shape {tuple(A.shape)}")
        return A

    A0 = _normalise(attn_weights[0])
    seq_len = A0.shape[1]
    device = A0.device

    # C[q, s] = fraction of position q attributable to source s
    C = torch.eye(seq_len, dtype=torch.float32, device=device)

    for A_layer in attn_weights:
        A = _normalise(A_layer)
        if A.shape[1] != seq_len or A.shape[2] != seq_len:
            raise ValueError(
                f"cd_token_contributions: all layers must have seq_len={seq_len}, "
                f"got {tuple(A.shape)}"
            )

        if head_agg == "mean":
            A_agg = A.mean(dim=0)  # (seq, seq)
        elif head_agg == "max":
            A_agg = A.max(dim=0).values
        else:
            raise ValueError(f"head_agg must be 'mean' or 'max', got {head_agg!r}")

        # Redistribute: C_new[q, s] = sum_j A_agg[q, j] * C[j, s]
        C_new = A_agg @ C  # (seq, seq)

        if add_residual:
            C = 0.5 * C + 0.5 * C_new
        else:
            C = C_new

        # Re-normalise rows to prevent drift
        row_sums = C.sum(dim=1, keepdim=True).clamp_min(1e-12)
        C = C / row_sums

    return CDResult(contributions=C)
