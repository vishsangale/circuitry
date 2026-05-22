"""Activation-space diagnostics. Pure; CPU-deterministic.

See docs/design.md §4.1 for the contract.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

ArrayLike = torch.Tensor | np.ndarray


@dataclass(frozen=True)
class NormStats:
    mean: float
    std: float
    max: float
    frac_above_k_median: float


def _as_tensor(x: ArrayLike) -> torch.Tensor:
    return torch.as_tensor(x).to(dtype=torch.float32)


def dead_fraction(x: ArrayLike, threshold: float = 0.0) -> float:
    """Fraction of activations at or below ``threshold``."""
    t = _as_tensor(x)
    if t.numel() == 0:
        return 0.0
    return float((t <= threshold).float().mean().item())


def norm_stats(x: ArrayLike, k: float = 3.0) -> NormStats:
    """Per-element norm statistics. ``frac_above_k_median`` is the fraction of
    elements whose absolute value exceeds ``k * median(|x|)`` — a cheap
    heavy-tail indicator.
    """
    t = _as_tensor(x).flatten()
    if t.numel() == 0:
        return NormStats(0.0, 0.0, 0.0, 0.0)
    abs_t = t.abs()
    med = abs_t.median().item()
    return NormStats(
        mean=float(t.mean().item()),
        std=float(t.std(unbiased=False).item()),
        max=float(abs_t.max().item()),
        frac_above_k_median=float((abs_t > k * med).float().mean().item()) if med > 0 else 0.0,
    )


def kurtosis(x: ArrayLike, dim: int | tuple[int, ...] = -1) -> torch.Tensor:
    """Excess kurtosis along ``dim``. Returns a tensor (not a Python float)
    because callers commonly want per-channel kurtosis."""
    t = _as_tensor(x)
    mean = t.mean(dim=dim, keepdim=True)
    centered = t - mean
    var = centered.pow(2).mean(dim=dim)
    m4 = centered.pow(4).mean(dim=dim)
    # Avoid div-by-zero
    out = m4 / var.clamp_min(1e-30).pow(2) - 3.0
    out = torch.where(var > 0, out, torch.zeros_like(out))
    return out


def participation_ratio(x: ArrayLike) -> float:
    """``(sum |x|)^2 / sum(x^2)`` — soft count of "active" units.

    Equals ``n`` when ``|x|`` is uniform, equals 1 when ``x`` is a one-hot.
    """
    t = _as_tensor(x).flatten()
    if t.numel() == 0:
        return 0.0
    num = t.abs().sum().pow(2)
    den = t.pow(2).sum().clamp_min(1e-30)
    return float((num / den).item())


def token_similarity(h: torch.Tensor) -> torch.Tensor:
    """Mean off-diagonal cosine similarity between token hidden states.

    Args:
        h: ``(batch, seq, dim)`` or ``(seq, dim)`` hidden states.

    Returns:
        Scalar mean off-diagonal cosine similarity (averaged over batch).
    """
    if h.dim() == 2:
        h = h.unsqueeze(0)
    normalized = torch.nn.functional.normalize(h, dim=-1)
    gram = torch.matmul(normalized, normalized.transpose(-2, -1))  # (B, S, S)
    seq = gram.shape[-1]
    if seq < 2:
        return torch.tensor(0.0, dtype=h.dtype, device=h.device)
    off_diag_mask = ~torch.eye(seq, dtype=torch.bool, device=h.device)
    off_diag = gram[..., off_diag_mask].view(gram.shape[0], -1)
    return off_diag.mean()


def gate_stats(x: ArrayLike, eps: float = 1e-6) -> dict[str, float]:
    """Statistics on a post-gate MLP activation tensor.

    For Llama-/Gemma-style gated MLPs the input to ``down_proj`` is
    ``act(gate_proj(x)) * up_proj(x)`` — i.e. the activations the MLP
    actually integrates. ``dead_fraction`` on the MLP output can't tell
    you whether a channel was gated off versus computed-and-discarded;
    this primitive reads the gated tensor directly.

    Returns a dict so the recorder can fan out ``gate_stats/<sub>``
    scalars without the primitive needing to know about tag naming.

    Keys:
        ``frac_active`` — fraction of entries with ``|x| > eps``.
        ``mean_abs`` — mean of ``|x|`` over all entries.
        ``std`` — standard deviation of ``x`` (unbiased=False).
    """
    t = _as_tensor(x).detach().to(torch.float32)
    abs_t = t.abs()
    return {
        "frac_active": float((abs_t > eps).float().mean().item()),
        "mean_abs": float(abs_t.mean().item()),
        "std": float(t.std(unbiased=False).item()),
    }
