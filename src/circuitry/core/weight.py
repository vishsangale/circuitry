"""Weight-space diagnostics. Pure functions; CPU-deterministic; no I/O.

See docs/design.md §4.1 for the contract.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
import torch

ArrayLike = torch.Tensor | np.ndarray


def _as_2d(W: ArrayLike) -> torch.Tensor:
    t = torch.as_tensor(W)
    if t.ndim == 1:
        t = t.unsqueeze(0)
    elif t.ndim > 2:
        t = t.reshape(t.shape[0], -1)
    return t.to(dtype=torch.float32 if t.dtype not in (torch.float32, torch.float64) else t.dtype)


def _require_at_most_2d(W: ArrayLike, fn: str) -> None:
    """Reject >2-D inputs to the scalar rank diagnostics (F38).

    For a 3-D *batched* tensor — e.g. an MoE expert stack
    ``[n_experts, d_in, d_out]`` — silently reshaping the leading axis into
    rows yields a semantically wrong rank (≈ ``n_experts`` instead of the
    per-expert rank). The shape alone cannot distinguish that case from a
    legitimate conv weight, so these primitives require ≤2-D and the caller
    must decide how to fold extra axes (e.g. ``rank_trajectory`` flattens conv
    weights to ``[out, in*kh*kw]``; a MoE-aware caller iterates per expert).
    """
    ndim = torch.as_tensor(W).ndim
    if ndim > 2:
        raise ValueError(
            f"{fn} requires a 1-D or 2-D tensor, got ndim={ndim} "
            f"(shape {tuple(torch.as_tensor(W).shape)}). Reshape/fold the extra "
            "axes explicitly — a batched (e.g. MoE-expert) tensor must be "
            "handled per-slice, not flattened into rows."
        )


def singular_values(
    W: ArrayLike,
    k: int | None = None,
    max_dim: int | None = None,
    *,
    seed: int | None = None,
    use_gram: bool | str = "auto",
) -> torch.Tensor:
    """Singular values of ``W`` in descending order.

    By default (``max_dim=None``) the full, deterministic SVD is computed, so
    every SVD-derived diagnostic is accurate and reproducible.  Setting
    ``max_dim`` is an opt-in *performance* hatch that caps SVD cost on wide
    matrices by truncating to a ``max_dim``-column random subsample before the
    decomposition — this biases σ_min and the spectral tail and (unless
    ``seed`` is set) is non-deterministic, so use it only when wide-matrix SVD
    cost is the bottleneck.  ``k`` truncates the returned vector.

    Args:
        seed: When not ``None``, the subsample draw (only relevant when
            ``max_dim`` triggers subsampling) uses a seeded
            ``torch.Generator`` so results are reproducible across calls on
            the same matrix.  Default ``None`` leaves the subsample draw
            unseeded (non-deterministic).
        use_gram: Control the ``eigvalsh(Gram)`` fast path for strongly
            rectangular matrices.

            * ``"auto"`` (default) — engage the Gram path when the aspect
              ratio is large enough to win (larger dim ≥ 3·smaller dim)
              *and* no subsampling was required (both dimensions ≤
              ``max_dim``).  Falls back to ``torch.linalg.svdvals`` when the
              matrix is near-square or when subsampling was applied (the
              post-sample matrix is already bounded and svdvals is fine).
            * ``True`` — always use the Gram path on the (possibly
              subsampled) matrix.
            * ``False`` — always use ``torch.linalg.svdvals`` (today's
              behaviour).

            **Numerical note:** the Gram path promotes to float64 internally
            to recover precision, but small singular values (below
            ``~sqrt(eps) · sigma_max``) are still unreliable.  Prefer
            ``use_gram=False`` (or the hard override in
            :func:`condition_number`) whenever accurate ``sigma_min`` is
            required.
    """
    M = _as_2d(W)
    subsampled = False
    if max_dim is not None and min(M.shape) > max_dim:
        # Sample columns from the longer axis to keep SVD bounded.
        axis = 1 if M.shape[1] > M.shape[0] else 0
        n = M.shape[axis]
        if seed is not None:
            g = torch.Generator(device=M.device).manual_seed(int(seed))
            idx = torch.randperm(n, device=M.device, generator=g)[:max_dim]
        else:
            idx = torch.randperm(n, device=M.device)[:max_dim]
        M = M.index_select(axis, idx)
        subsampled = True

    # Gram fast path: cheaper than full SVD for strongly rectangular matrices.
    # sigma_i(M) = sqrt(lambda_i(M M^T))  (or M^T M for the smaller Gram).
    # Promote to float64 to partially recover the precision lost by squaring.
    # Auto policy: only engage when both dims are within max_dim (no subsampling
    # already occurred) AND the aspect ratio is >= 3:1.
    use_gram_bool: bool
    if use_gram is True:
        use_gram_bool = True
    elif use_gram is False:
        use_gram_bool = False
    else:  # "auto"
        if subsampled:
            # Post-sample matrix is already bounded; svdvals is fine.
            use_gram_bool = False
        else:
            a, b = M.shape[0], M.shape[1]
            smaller, larger = (a, b) if a <= b else (b, a)
            use_gram_bool = larger >= 3 * smaller

    if use_gram_bool:
        orig_dtype = M.dtype
        Mf = M.to(torch.float64)
        a, b = Mf.shape
        if a <= b:
            G = Mf @ Mf.T  # (a, a)
        else:
            G = Mf.T @ Mf  # (b, b)
        eig = torch.linalg.eigvalsh(G).clamp_min(0.0)
        s_f64 = eig.sqrt()
        s = s_f64.to(orig_dtype)
    else:
        s = torch.linalg.svdvals(M)

    s, _ = torch.sort(s, descending=True)
    if k is not None:
        s = s[:k]
    return s


def effective_rank(
    W: ArrayLike, eps: float = 1e-12, *, max_dim: int | None = None, seed: int | None = None
) -> float:
    """Roy & Vetterli (2007) effective rank: ``exp(H(p))`` where ``p`` is the
    normalized singular-value distribution.

    Requires a ≤2-D input (see :func:`_require_at_most_2d`). ``max_dim``/``seed``
    are the opt-in perf hatch documented on :func:`singular_values`.
    """
    _require_at_most_2d(W, "effective_rank")
    return _effective_rank_from_sv(singular_values(W, max_dim=max_dim, seed=seed), eps)


def _effective_rank_from_sv(s: torch.Tensor, eps: float = 1e-12) -> float:
    s = s[s > eps]
    if s.numel() == 0:
        return 0.0
    p = s / s.sum()
    H = -(p * torch.log(p)).sum().item()
    return float(math.exp(H))


def stable_rank(
    W: ArrayLike, *, max_dim: int | None = None, seed: int | None = None
) -> float:
    """``||W||_F^2 / ||W||_2^2``. Lower-bounds the algebraic rank and is
    numerically robust on near-singular matrices.

    Requires a ≤2-D input (see :func:`_require_at_most_2d`).
    """
    _require_at_most_2d(W, "stable_rank")
    return _stable_rank_from_sv(singular_values(W, max_dim=max_dim, seed=seed))


def _stable_rank_from_sv(s: torch.Tensor) -> float:
    if s.numel() == 0:
        return 0.0
    return float((s.pow(2).sum() / (s[0].pow(2))).item())


def condition_number(
    W: ArrayLike, eps: float = 1e-12, *, max_dim: int | None = None
) -> float:
    """``sigma_max / sigma_min``. Returns ``+inf`` if the smallest singular
    value is below ``eps``.

    Always uses the full SVD path (``use_gram=False``) because accurate
    ``sigma_min`` is the numerically sensitive quantity and the Gram path
    squares the condition number, destroying precision for small singular
    values. ``max_dim`` is the opt-in perf hatch from :func:`singular_values`
    (default ``None`` → full, accurate SVD); note that subsampling biases
    ``sigma_min`` so the result is only a lower bound on the true condition
    number. Requires a ≤2-D input.
    """
    _require_at_most_2d(W, "condition_number")
    return _condition_number_from_sv(singular_values(W, max_dim=max_dim, use_gram=False), eps)


def _condition_number_from_sv(s: torch.Tensor, eps: float = 1e-12) -> float:
    if s.numel() == 0 or s[-1].item() < eps:
        return float("inf")
    return float((s[0] / s[-1]).item())


def heavy_tail_alpha(
    W: ArrayLike, top_frac: float = 0.5, *, max_dim: int | None = None, seed: int | None = None
) -> float:
    """Hill estimator of the tail index of the squared-singular-value
    distribution. Computed on the top ``top_frac`` (default half) of squared
    singular values — the empirically robust default.

    Returns ``+inf`` on degenerate inputs. Requires a ≤2-D input.
    """
    _require_at_most_2d(W, "heavy_tail_alpha")
    return _heavy_tail_alpha_from_sv(singular_values(W, max_dim=max_dim, seed=seed), top_frac)


def _heavy_tail_alpha_from_sv(s: torch.Tensor, top_frac: float = 0.5) -> float:
    if s.numel() < 4:
        return float("inf")
    s2 = s.pow(2).sort(descending=True).values
    k = max(2, int(s2.numel() * top_frac))
    top = s2[:k]
    smin = top[-1].clamp_min(1e-30)
    # Hill: alpha_hat = k / sum(log(s_i / s_k))
    logs = torch.log(top / smin)
    denom = logs.sum().item()
    if denom <= 0:
        return float("inf")
    return float(k / denom)


def update_delta(
    sd_now: Mapping[str, torch.Tensor],
    sd_prev: Mapping[str, torch.Tensor],
) -> dict[str, float]:
    """L2 norm of the delta between two state-dict snapshots, per parameter.

    Returns ``{name: ||sd_now[name] - sd_prev[name]||_2}`` for every name
    present in both. Names missing from either side are skipped.
    """
    out: dict[str, float] = {}
    for name in sd_now:
        if name not in sd_prev:
            continue
        # Align devices: in the live Recorder, sd_now holds GPU weights while the
        # prior snapshot (sd_prev) is a CPU copy — subtracting across devices
        # raises. Move sd_prev onto sd_now's device.
        a = sd_now[name].to(torch.float32)
        b = sd_prev[name].to(device=a.device, dtype=torch.float32)
        out[name] = float((a - b).norm().item())
    return out


def direction_cosine(
    sd_now: Mapping[str, torch.Tensor],
    sd_prev: Mapping[str, torch.Tensor],
    sd_prev_prev: Mapping[str, torch.Tensor],
) -> dict[str, float]:
    """Cosine similarity between two consecutive parameter updates.

    Update_1 = sd_prev - sd_prev_prev
    Update_2 = sd_now  - sd_prev

    Returns ``{name: cos(Update_1, Update_2)}``. Zero-norm updates return 0.0.
    """
    out: dict[str, float] = {}
    for name in sd_now:
        if name not in sd_prev or name not in sd_prev_prev:
            continue
        # Align devices onto sd_now's (the live Recorder keeps prior snapshots on
        # CPU while current weights are on GPU — cross-device math raises).
        now = sd_now[name].to(torch.float32)
        prev = sd_prev[name].to(device=now.device, dtype=torch.float32)
        prev_prev = sd_prev_prev[name].to(device=now.device, dtype=torch.float32)
        u2 = (now - prev).flatten()
        u1 = (prev - prev_prev).flatten()
        n2 = float(u2.norm().item())
        n1 = float(u1.norm().item())
        if n1 == 0.0 or n2 == 0.0:
            out[name] = 0.0
        else:
            out[name] = float((u1 @ u2).item()) / (n1 * n2)
    return out


def attention_head_rank(
    W: ArrayLike,
    n_heads: int,
    head_dim: int,
    axis: int,
) -> list[float]:
    """Per-head ``effective_rank`` of an attention projection weight.

    Args:
        W: A 2-D weight matrix. For q/k/v_proj the head dimension lives
            in the OUTPUT axis (``axis=0``); for o_proj it lives in the
            INPUT axis (``axis=1``).
        n_heads: Number of heads in this projection. For GQA models the
            caller passes ``num_key_value_heads`` for k/v_proj and
            ``num_attention_heads`` for q/o_proj — the primitive does
            not infer this.
        head_dim: Per-head dimension. The product ``n_heads * head_dim``
            must equal ``W.shape[axis]``.
        axis: 0 for "output-head" projections (q/k/v), 1 for
            "input-head" projections (o).

    Returns:
        ``n_heads`` floats, the ``effective_rank`` of each per-head
        slice. Order matches head index.

    Raises:
        ValueError: if ``W.shape[axis] != n_heads * head_dim`` or if
            ``W`` is not 2-D.
    """
    t = _as_2d(W)
    if t.ndim != 2:
        raise ValueError(f"attention_head_rank expects 2-D, got shape {tuple(t.shape)}")
    if t.shape[axis] != n_heads * head_dim:
        raise ValueError(
            f"attention_head_rank: W.shape[{axis}] == {t.shape[axis]} but "
            f"n_heads * head_dim == {n_heads * head_dim}"
        )
    ranks: list[float] = []
    for i in range(n_heads):
        if axis == 0:
            slice_i = t[i * head_dim : (i + 1) * head_dim]
        else:
            slice_i = t[:, i * head_dim : (i + 1) * head_dim]
        ranks.append(effective_rank(slice_i))
    return ranks
