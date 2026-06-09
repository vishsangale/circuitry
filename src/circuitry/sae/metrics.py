"""SAE reconstruction metrics. See v0.9 spec §4.3."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor


def _as_tensor(x: Any) -> Tensor:
    if isinstance(x, Tensor):
        return x
    return torch.as_tensor(x)


def sae_reconstruction_error(x: Any, sae: Any) -> dict[str, float]:
    """Apply sae.encode then sae.decode to x; return per-batch metrics.

    Keys: recon_mse, l0, l1, frac_alive, ce_recovered_proxy.
    See spec §4.3 docstring for semantics.
    """
    t = _as_tensor(x).detach().to(torch.float32)
    flat = t.reshape(-1, t.shape[-1])  # (n_tokens, d_model)

    sae_device = getattr(sae, "device", flat.device)
    sae_dtype = getattr(sae, "dtype", torch.float32)
    flat_for_sae = flat.to(sae_device).to(sae_dtype)

    with torch.inference_mode():
        feats = sae.encode(flat_for_sae)
        recon = sae.decode(feats)

    recon_f32 = recon.detach().to(torch.float32).to(flat.device)
    feats_f32 = feats.detach().to(torch.float32)

    recon_mse = float(((flat - recon_f32) ** 2).mean().item())
    nonzero = feats_f32.abs() > 0
    l0 = float(nonzero.float().sum(dim=-1).mean().item())
    l1 = float(feats_f32.abs().sum(dim=-1).mean().item())
    frac_alive = float(nonzero.any(dim=0).float().mean().item())
    var = float(flat.var(unbiased=False).item())
    ce_recovered_proxy = float(1.0 - (recon_mse / var)) if var > 0 else 0.0

    return {
        "recon_mse": recon_mse,
        "l0": l0,
        "l1": l1,
        "frac_alive": frac_alive,
        "ce_recovered_proxy": ce_recovered_proxy,
    }


def superposition_index(feature_acts: Any) -> float:
    """Effective number of SAE features (superposition measure).

    Returns ``exp(H)`` where ``H`` is the Shannon entropy of the activation
    magnitude distribution across features.  When ``exp(H) >> n_neurons``,
    the layer is operating under superposition (more effective features than
    embedding dimensions).

    Args:
        feature_acts: (..., n_features) SAE activation tensor (output of
            ``sae.encode(x)``).

    Returns:
        Effective feature count (float >= 1.0).

    Reference: arXiv:2512.13568 "Superposition as Lossy Compression".
    """
    t = _as_tensor(feature_acts).detach().to(torch.float32)
    mags = t.abs().flatten()
    total = mags.sum()
    if total.item() < 1e-10:
        return 1.0
    probs = mags / total
    log_probs = torch.where(probs > 0, probs.log(), torch.zeros_like(probs))
    entropy = float(-(probs * log_probs).sum().item())
    return float(torch.exp(torch.tensor(entropy)).item())
