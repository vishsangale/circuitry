"""Differentiable SAE helpers for feature attribution. v1.5.0.

This module wraps sae.encode / sae.decode under NORMAL autograd (no
inference_mode, no_grad, or detach on the live path) so gradients flow
through the feature tensor.  Device/dtype alignment mirrors metrics.py:26-28.

Exported from sae/__init__.py.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor

# Architectures that are supported for gradient-based attribution.
# BatchTopK and Matryoshka may load as "jumprelu" in some sae_lens versions,
# but are also registered under their own names in newer releases.
SUPPORTED_SAE_ARCHITECTURES = frozenset({
    "standard",
    "topk",
    "jumprelu",
    "gated",
    "matryoshka",       # Bussmann et al. 2025, arxiv:2503.17547
    "batch_topk",       # batch-level TopK; gradient path identical to topk
    "p_anneal",         # p-annealing ReLU; inference pass identical to standard
    "hierarchical_topk", # HierarchicalTopK; inference pass identical to topk
})

_BLOCKED_ARCHITECTURES = frozenset({
    "transcoder",
    "skip_transcoder",
    "jumprelu_transcoder",
    "matching_pursuit",
    "temporal",
    # Raw crosscoder SAEs must be wrapped in CrosscoderWrapper before
    # passing to gradient-based attribution helpers.
    "crosscoder",
})

_SUPPORTED_NORM_MODES = frozenset({"none", "layer_norm", "constant_norm_rescale"})


def assert_supported_sae(sae: Any) -> None:
    """Raise NotImplementedError for unsupported SAE architectures or normalisation modes.

    Blocked: transcoder, skip_transcoder, jumprelu_transcoder,
    matching_pursuit, temporal.
    Normalisation: only {none, layer_norm, constant_norm_rescale}.
    """
    cfg = getattr(sae, "cfg", None)
    if cfg is None:
        return  # can't check — let it through; errors will surface at encode/decode

    # Architecture check
    arch = cfg.architecture() if callable(getattr(cfg, "architecture", None)) else getattr(cfg, "architecture", None)
    if arch is not None:
        arch_str = str(arch).lower()
        if arch_str in _BLOCKED_ARCHITECTURES:
            raise NotImplementedError(
                f"SAE architecture {arch_str!r} is not supported for gradient-based "
                f"feature attribution in v1.5.0 (deferred). "
                f"Supported: {sorted(SUPPORTED_SAE_ARCHITECTURES)}."
            )

    # Normalisation mode check
    norm = getattr(cfg, "normalize_activations", None)
    if norm is not None:
        norm_str = str(norm).lower()
        if norm_str not in _SUPPORTED_NORM_MODES:
            raise NotImplementedError(
                f"SAE normalize_activations={norm_str!r} is not supported for "
                f"gradient-based feature attribution in v1.5.0. "
                f"Supported: {sorted(_SUPPORTED_NORM_MODES)}."
            )


def _device_dtype_align(x: Tensor, sae: Any) -> Tensor:
    """Move x to the SAE's device and dtype, mirroring metrics.py:26-28."""
    sae_device = getattr(sae, "device", x.device)
    sae_dtype = getattr(sae, "dtype", torch.float32)
    return x.to(sae_device, sae_dtype)


def encode_features(sae: Any, x: Tensor) -> Tensor:
    """Differentiable encode: returns feature activations f (b, s, d_sae) or (n, d_sae).

    Calls sae.encode under normal autograd — NO inference_mode / no_grad / detach.
    Device/dtype-aligns x to the SAE before encoding.
    """
    x_aligned = _device_dtype_align(x, sae)
    return sae.encode(x_aligned)


def decode_features(sae: Any, f: Tensor) -> Tensor:
    """Differentiable decode: returns reconstruction x_hat in x-space.

    Calls sae.decode under normal autograd — NO inference_mode / no_grad / detach.
    """
    return sae.decode(f)


def sae_influence_scores(
    sae: Any,
    x: Tensor,
    loss_fn: Callable[[Tensor], Tensor],
) -> Tensor:
    """GradSAE per-feature influence scores: ``|∂loss/∂f_i| · |f_i|``.

    Weights feature activation magnitude by the magnitude of the output-side
    gradient.  This filters "active but irrelevant" features (high ``|f_i|``
    but near-zero gradient, e.g. punctuation or positional features) and
    surfaces features that causally drive the loss objective.

    Args:
        sae:     SAE object with ``.encode`` / ``.decode``.
        x:       ``(n, d_model)`` or ``(b, s, d_model)`` activation tensor.
                 Must require grad or be leaf; the function enables grad internally.
        loss_fn: callable ``(x_hat: Tensor) → scalar Tensor``.  The loss is
                 evaluated on the SAE reconstruction of ``x``.  Example:
                 ``lambda x_hat: F.cross_entropy(model(x_hat), labels)``.

    Returns:
        ``(n_features,)`` float tensor of influence scores (mean over batch /
        sequence positions).  Detached CPU float32.

    Reference: arXiv:2505.08080 "GradSAE: Gradient-Weighted SAE Features".
    """

    x_in = _device_dtype_align(x, sae)
    if not x_in.requires_grad:
        x_in = x_in.detach().requires_grad_(True)

    f = sae.encode(x_in)           # (... d_sae)
    x_hat = sae.decode(f)          # (... d_model)
    loss = loss_fn(x_hat)
    loss.backward()

    # ∂loss/∂f via chain rule: ∂loss/∂x_hat is x_in.grad (since x_hat = decode(f)
    # and f = encode(x_in)). But we need the gradient w.r.t. f directly.
    # Rerun with f requiring grad.
    x_in2 = x_in.detach()
    f2 = sae.encode(x_in2)
    f2 = f2.detach().requires_grad_(True)
    x_hat2 = sae.decode(f2)
    loss2 = loss_fn(x_hat2)
    loss2.backward()

    grad_f = f2.grad  # (... d_sae)
    f_detached = f2.detach()

    # influence = |∂loss/∂f_i| * |f_i|, mean over all positions
    scores = (grad_f.abs() * f_detached.abs())
    flat = scores.reshape(-1, scores.shape[-1])
    return flat.mean(0).detach().cpu().float()


def sae_decompose(sae: Any, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Single PAIRED encode→decode.

    Required for stateful normalisation modes (layer_norm, constant_norm_rescale)
    that cache statistics set during encode and used during decode — NEVER cache
    f and decode later; always decode in the same call.

    Returns:
        f     : feature activations, (b, s, d_sae) or (n, d_sae).  In the live
                autograd graph (no detach).
        x_hat : reconstruction = sae.decode(f), in x-space.  In the live graph.
        eps   : (x - x_hat).detach() — frozen clean reconstruction error.
    """
    x_aligned = _device_dtype_align(x, sae)
    f = sae.encode(x_aligned)
    x_hat = sae.decode(f)
    eps = (x_aligned - x_hat).detach()
    return f, x_hat, eps
