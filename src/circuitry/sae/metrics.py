"""SAE reconstruction metrics. See v0.9 spec §4.3."""

from __future__ import annotations

import warnings
from typing import Any

import torch
from torch import Tensor

# Metrics known to be unreliable across seeds and architectures.
# Emits UserWarning if requested; does not block execution.
# Reference: arXiv:2605.18229 "SAE Metric Reliability".
UNRELIABLE_METRICS: frozenset[str] = frozenset({"tpp", "scr"})


def warn_if_unreliable(metric_name: str) -> None:
    """Emit a UserWarning if ``metric_name`` is in ``UNRELIABLE_METRICS``."""
    if metric_name in UNRELIABLE_METRICS:
        warnings.warn(
            f"SAE metric {metric_name!r} is known to be unreliable across seeds "
            f"and architectures (arXiv:2605.18229). Interpret results with caution.",
            UserWarning,
            stacklevel=3,
        )


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


def sae_downstream_loss(
    sae: Any,
    model: Any,
    tokens: Any,
    *,
    site: Any,
    resolver: Any = None,
) -> dict[str, float]:
    """Gold-standard SAE faithfulness: KL divergence between model outputs with/without SAE.

    Runs two forward passes:
    1. **Clean**: ``model(tokens)`` — no SAE intervention.
    2. **SAE substitution**: ``model(tokens)`` with a forward hook that replaces
       the residual stream at ``site`` with the SAE reconstruction.

    Returns the KL divergence between clean and substituted logit distributions,
    the approximate CE delta, and the SAE's L0 sparsity.

    Unlike ``ce_recovered_proxy`` (which is purely variance-based and can be
    computed without a model), this metric measures actual downstream task
    impact — the gold-standard SAE faithfulness metric used in SAEBench.

    Args:
        sae:      SAE object with ``.encode`` / ``.decode``.
        model:    PyTorch model; must produce logits as final output or as the
                  first element of a tuple output.
        tokens:   Input token IDs (any shape accepted by ``model``).
        site:     :class:`~circuitry.patching.sites.Site` identifying where to
                  substitute the reconstruction.
        resolver: :class:`~circuitry.patching.sites.SiteResolver`; defaults to
                  ``HFSiteResolver.from_config(model.config)`` when the model
                  has a ``.config`` attribute.

    Returns:
        Dict with keys:
        - ``"kl_divergence"``: mean token-level KL(clean ‖ sae_subst) in nats.
        - ``"ce_delta"``: approximate change in cross-entropy; positive = worse.
        - ``"l0"``: mean number of non-zero SAE features per token.

    Reference: arXiv:2406.04093 "Scaling and evaluating SAEs"; SAEBench
               arXiv:2503.09532.
    """
    import torch.nn as nn
    import torch.nn.functional as F

    from circuitry.patching.sites import HFSiteResolver

    if resolver is None:
        config = getattr(model, "config", None)
        if config is None:
            raise ValueError(
                "sae_downstream_loss: resolver=None requires model.config. "
                "Pass an explicit SiteResolver via resolver=."
            )
        resolver = HFSiteResolver.from_config(config)

    resolved = resolver.resolve(model, site)
    module = resolved.module

    sae_device = getattr(sae, "device", next(model.parameters()).device)
    sae_dtype = getattr(sae, "dtype", torch.float32)

    l0_accumulator: list[float] = []

    def _sae_hook(
        mod: nn.Module,  # noqa: ARG001
        inputs: tuple,   # noqa: ARG001
        output: object,
    ) -> object:
        tensor = output[0] if isinstance(output, tuple) else output
        flat = tensor.detach().reshape(-1, tensor.shape[-1])
        flat_sae = flat.to(sae_device, sae_dtype)
        with torch.inference_mode():
            feats = sae.encode(flat_sae)
            recon = sae.decode(feats)
        # Track L0
        nonzero = (feats.detach().abs() > 0).float().sum(dim=-1).mean().item()
        l0_accumulator.append(float(nonzero))
        recon_f32 = recon.detach().to(dtype=tensor.dtype, device=tensor.device)
        recon_shaped = recon_f32.reshape_as(tensor)
        if isinstance(output, tuple):
            return (recon_shaped,) + output[1:]
        return recon_shaped

    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            clean_out = model(tokens)
        clean_logits = clean_out[0] if isinstance(clean_out, tuple) else clean_out
        clean_logits = clean_logits.detach().float()

        handle = module.register_forward_hook(_sae_hook)
        try:
            with torch.inference_mode():
                sae_out = model(tokens)
        finally:
            handle.remove()
        sae_logits = sae_out[0] if isinstance(sae_out, tuple) else sae_out
        sae_logits = sae_logits.detach().float()
    finally:
        model.train() if was_training else model.eval()

    # KL(clean || sae_subst) — mean over all token positions
    clean_log_probs = F.log_softmax(clean_logits, dim=-1)
    sae_log_probs = F.log_softmax(sae_logits, dim=-1)
    # KL = sum_v clean_p * (clean_log_p - sae_log_p)
    clean_probs = clean_log_probs.exp()
    kl = (clean_probs * (clean_log_probs - sae_log_probs)).sum(dim=-1)
    kl_mean = float(kl.mean().item())

    # CE delta ≈ KL (upper bound; exact when distributions are close)
    ce_delta = kl_mean

    l0 = float(sum(l0_accumulator) / max(len(l0_accumulator), 1))

    return {
        "kl_divergence": kl_mean,
        "ce_delta": ce_delta,
        "l0": l0,
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
