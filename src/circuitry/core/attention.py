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

head_specialization — classify each head into a behavioral type
(induction / copy_suppression / sink / uniform) from the three scores above.

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


def head_specialization(
    induction: list[float],
    copy_suppression: list[float],
    sink: list[float],
    *,
    induction_threshold: float = 0.4,
    copy_suppression_threshold: float = 0.3,
    sink_threshold: float = 0.5,
) -> list[str]:
    """Classify each attention head into a behavioral type.

    For each head, checks which scores exceed their respective thresholds.
    When none exceed: ``"uniform"``. When exactly one exceeds: that type.
    When multiple exceed: the type with the highest score-to-threshold ratio
    wins (the "most strongly typed" signal). Thresholds match the report
    FLAG_RULES for consistency.

    Args:
        induction: per-head :func:`induction_score` values.
        copy_suppression: per-head :func:`copy_suppression_score` values.
        sink: per-head :func:`attention_sink_score` values.
        induction_threshold: min score to qualify as induction (default 0.4).
        copy_suppression_threshold: min score for copy-suppression (default 0.3).
        sink_threshold: min score for sink (default 0.5).

    Returns:
        ``list[str]`` of per-head labels — one of ``"induction"``,
        ``"copy_suppression"``, ``"sink"``, or ``"uniform"``.
    """
    if not (len(induction) == len(copy_suppression) == len(sink)):
        raise ValueError(
            "head_specialization: induction, copy_suppression, and sink must have "
            f"the same length, got {len(induction)}, {len(copy_suppression)}, {len(sink)}"
        )
    labels: list[str] = []
    for ind, css, snk in zip(induction, copy_suppression, sink, strict=True):
        candidates: dict[str, float] = {}
        if ind >= induction_threshold:
            candidates["induction"] = ind / induction_threshold
        if css >= copy_suppression_threshold:
            candidates["copy_suppression"] = css / copy_suppression_threshold
        if snk >= sink_threshold:
            candidates["sink"] = snk / sink_threshold
        labels.append("uniform" if not candidates else max(candidates, key=candidates.__getitem__))
    return labels


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


def attention_rollout(
    attn_weights: list[Any],
    *,
    grads: list[Any] | None = None,
    discard_ratio: float = 0.0,
) -> torch.Tensor:
    """Recursive attention rollout — patch-level saliency map for ViTs.

    Accumulates attention across layers by multiplying attention matrices with
    a residual-stream identity term at each layer.  The optional ``grads``
    parameter enables Gradient-weighted Multi-head Attention Rollout (GMAR):
    each head's contribution is weighted by the magnitude of the gradient of
    the loss with respect to that head's attention weights before averaging.

    **Uniform rollout** (Abnar & Zuidema 2020): at each layer compute
    ``A_hat = A + I`` (add identity for the skip connection), row-normalise,
    and multiply through layers: ``R = A_hat_L @ … @ A_hat_1``.  The first
    row of ``R`` gives the attention map from the CLS token to all patches.

    **GMAR** (gradient-weighted): for each head, weight its attention matrix by
    ``mean(|grad|)`` before averaging over heads.

    Args:
        attn_weights: list of per-layer attention weight tensors, each of shape
            ``(B, H, T, T)`` (batch, heads, queries, keys).  Layers are given
            in bottom-to-top order (first element = first layer).
        grads:        optional list of gradient tensors, same shapes as
            ``attn_weights``.  When provided, each head's matrix is weighted by
            ``mean(|grad|)`` over the spatial dimensions before averaging.
        discard_ratio: fraction of the lowest attention values to zero out at
            each layer before rollout (noise suppression; default 0 = off).

    Returns:
        ``(B, T)`` float tensor — saliency of each token/patch for each sample
        in the batch.  ``T`` includes the CLS token at index 0 (for standard
        ViT tokenisation); the patch saliency is ``result[:, 1:]``.

    Reference:
        Abnar & Zuidema 2020, ACL "Quantifying Attention Flow in Transformers".
        GMAR: arXiv:2504.19414 "Gradient-weighted Multi-head Attention Rollout".
    """
    if not attn_weights:
        raise ValueError("attention_rollout: attn_weights must be non-empty")

    def _prep(a: Any) -> torch.Tensor:
        t = torch.as_tensor(a).detach().to(torch.float32)
        if t.ndim == 3:  # (H, T, T) — add batch dim
            t = t.unsqueeze(0)
        return t  # (B, H, T, T)

    layers_a = [_prep(a) for a in attn_weights]
    B, H, T, _ = layers_a[0].shape

    layers_g: list[torch.Tensor] | None = None
    if grads is not None:
        layers_g = [_prep(g) for g in grads]

    result = torch.eye(T, dtype=torch.float32).unsqueeze(0).expand(B, -1, -1).clone()  # (B, T, T)

    for idx, A in enumerate(layers_a):
        # --- per-head weighting (GMAR) ---
        if layers_g is not None:
            g = layers_g[idx]  # (B, H, T, T)
            head_weight = g.abs().mean(dim=(-2, -1), keepdim=True)  # (B, H, 1, 1)
            head_weight = head_weight / (head_weight.sum(dim=1, keepdim=True) + 1e-12)
            A_eff = (A * head_weight).sum(dim=1)  # (B, T, T)
        else:
            A_eff = A.mean(dim=1)  # (B, T, T)

        # Discard lowest-attention values
        if discard_ratio > 0.0:
            flat = A_eff.flatten(start_dim=1)  # (B, T*T)
            n_discard = int(flat.shape[1] * discard_ratio)
            if n_discard > 0:
                threshold, _ = flat.kthvalue(n_discard + 1, dim=1)  # (B,)
                mask = A_eff < threshold.unsqueeze(-1).unsqueeze(-1)
                A_eff = A_eff.masked_fill(mask, 0.0)

        # Add residual (identity) and row-normalise
        A_hat = A_eff + torch.eye(T, dtype=A_eff.dtype).unsqueeze(0)
        row_sum = A_hat.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        A_hat = A_hat / row_sum

        result = A_hat @ result

    # Return saliency: first row = CLS→all tokens
    return result[:, 0, :]  # (B, T)


def daam_attribution(
    attn_maps: list[Any],
    *,
    head_agg: str = "mean",
    normalize: bool = True,
    spatial_shape: tuple[int, int] | None = None,
) -> torch.Tensor:
    """DAAM: aggregate cross-attention maps across denoising steps.

    Computes per-token attribution heatmaps by averaging cross-attention maps
    over all denoising steps, then optionally normalising each token's map to
    sum to 1.  Suitable for diffusion-model interpretability where each step
    produces a set of cross-attention maps.

    Args:
        attn_maps:     List of per-step cross-attention tensors, each of shape
                       ``(n_heads, n_patches, seq_len)`` or
                       ``(batch, n_heads, n_patches, seq_len)``.  Batch
                       dimensions are averaged before step aggregation.
        head_agg:      How to aggregate over the head axis — ``"mean"``
                       (default) or ``"max"``.
        normalize:     If True, L1-normalise each token's attribution map so
                       that its values sum to 1 (makes tokens comparable
                       independent of raw attention scale).
        spatial_shape: Optional ``(H, W)`` to reshape the patch axis into a 2-D
                       spatial grid.  Must satisfy ``H * W == n_patches``.

    Returns:
        ``(seq_len, n_patches)`` float tensor (or ``(seq_len, H, W)`` when
        ``spatial_shape`` is supplied).

    Reference:
        Tang et al. 2023, ICCV "What the DAAM: Interpreting Stable Diffusion
        Using Cross Attention".  https://arxiv.org/abs/2210.04885
    """
    if not attn_maps:
        raise ValueError("daam_attribution: attn_maps must be non-empty")
    if head_agg not in ("mean", "max"):
        raise ValueError(f"daam_attribution: head_agg must be 'mean' or 'max', got {head_agg!r}")

    aggregated: list[torch.Tensor] = []
    for step_map in attn_maps:
        t = torch.as_tensor(step_map).detach().to(torch.float32)
        if t.ndim == 3:
            t = t.unsqueeze(0)  # (1, n_heads, n_patches, seq_len)
        elif t.ndim != 4:
            raise ValueError(
                f"daam_attribution: each attn_map must be 3-D or 4-D, got {t.ndim}-D"
            )
        # Average over batch
        t = t.mean(dim=0)  # (n_heads, n_patches, seq_len)
        # Aggregate heads
        if head_agg == "mean":
            t = t.mean(dim=0)  # (n_patches, seq_len)
        else:
            t = t.max(dim=0).values  # (n_patches, seq_len)
        aggregated.append(t)

    # Average over steps → (n_patches, seq_len)
    result = torch.stack(aggregated, dim=0).mean(dim=0)

    # Transpose → (seq_len, n_patches)
    result = result.transpose(0, 1)

    if normalize:
        row_sum = result.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        result = result / row_sum

    if spatial_shape is not None:
        H, W = spatial_shape
        seq_len, n_patches = result.shape
        if H * W != n_patches:
            raise ValueError(
                f"daam_attribution: spatial_shape {spatial_shape} requires "
                f"H*W={H*W} but got {n_patches} patches"
            )
        result = result.reshape(seq_len, H, W)

    return result
