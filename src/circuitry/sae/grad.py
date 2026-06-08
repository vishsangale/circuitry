"""Differentiable SAE helpers for feature attribution. v1.5.0.

This module wraps sae.encode / sae.decode under NORMAL autograd (no
inference_mode, no_grad, or detach on the live path) so gradients flow
through the feature tensor.  Device/dtype alignment mirrors metrics.py:26-28.

Exported from sae/__init__.py.
"""
from __future__ import annotations

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
    "matryoshka",   # Bussmann et al. 2025, arxiv:2503.17547
    "batch_topk",   # batch-level TopK; gradient path identical to topk
})

_BLOCKED_ARCHITECTURES = frozenset({
    "transcoder",
    "skip_transcoder",
    "jumprelu_transcoder",
    "matching_pursuit",
    "temporal",
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
