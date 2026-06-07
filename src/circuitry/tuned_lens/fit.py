"""``fit_tuned_lens`` — post-hoc trainer for a :class:`TunedLens`.

This is the *only* optimizer loop in the library, and it lives strictly in the
workflow layer (it may import ``core/``; it never imports ``recorder/`` /
``recipes/`` / ``cli/``). Fitting is post-hoc — it runs many forward passes and
must NOT be wired into the live training loop (it would blow the §10 budget).
The recorder only ever *applies* a frozen lens (forward-only).

The trainer mirrors the recorder's lens dispatch exactly so a fitted lens lines
up with how it is later applied:

- residual sources are the per-block outputs (``...layers.N``; ``out[0]`` for
  tuple-returning HF blocks);
- the unembed is ``model.get_output_embeddings().weight`` and the final
  layer-norm is resolved from the usual HF locations;
- the regression *target* is the model's final distribution, reconstructed as
  ``softmax(LN_f(last_block_residual) @ W_U)`` — the same reconstruction the
  recorder uses — so the last-layer translator (A=I, b=0) is a fixed point.
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections.abc import Callable, Iterable
from typing import Any

import torch
from torch import Tensor, nn

from circuitry.core.lens import _resolve_unembed
from circuitry.tuned_lens.container import TunedLens

logger = logging.getLogger("circuitry.tuned_lens")

_BLOCK_RE = re.compile(r"(?:^|\.)layers\.(\d+)$")


def model_fingerprint(model: nn.Module) -> str:
    """A stable hash of the model's parameter names + shapes.

    Architecture-sensitive (catches a different model / width / depth) but
    weight-value-insensitive (a fine-tuned copy of the same architecture keeps
    the same fingerprint). Used to guard tuned-lens application.
    """
    h = hashlib.sha256()
    for name, p in sorted(model.named_parameters(), key=lambda kv: kv[0]):
        h.update(name.encode("utf-8"))
        h.update(repr(tuple(p.shape)).encode("utf-8"))
    return h.hexdigest()[:16]


def _resolve_final_layer_norm(model: nn.Module) -> nn.Module | None:
    return (
        getattr(getattr(model, "model", None), "norm", None)
        or getattr(getattr(model, "transformer", None), "ln_f", None)
        or getattr(model, "norm", None)
        or getattr(model, "ln_f", None)
    )


def _resolve_unembed_weight(model: nn.Module) -> Tensor | None:
    try:
        emb = model.get_output_embeddings()
    except (AttributeError, NotImplementedError):
        emb = None
    w = getattr(emb, "weight", None) if emb is not None else None
    return w if isinstance(w, Tensor) else None


def _block_modules(model: nn.Module) -> list[tuple[int, str, nn.Module]]:
    """Return ``(layer_idx, name, module)`` for every residual-block module."""
    out: list[tuple[int, str, nn.Module]] = []
    for name, mod in model.named_modules():
        m = _BLOCK_RE.search(name)
        if m is not None:
            out.append((int(m.group(1)), name, mod))
    out.sort(key=lambda t: t[0])
    return out


def _run_forward(model: nn.Module, batch: Any,
                 forward_fn: Callable[..., object] | None) -> None:
    """Run a forward pass (output discarded — residuals are captured by hooks).

    Mirrors the recorder's ``forward_fn`` convention, with fallbacks for a bare
    input tensor or an HF-style mapping batch.
    """
    if forward_fn is not None:
        forward_fn(model, batch)
        return
    if isinstance(batch, dict):
        model(**batch)
        return
    model(batch)


def fit_tuned_lens(
    model: nn.Module,
    batches: Iterable[Any],
    *,
    layers: list[int] | None = None,
    steps: int = 250,
    lr: float = 1e-3,
    weight_decay: float = 1e-3,
    device: str | torch.device | None = None,
    forward_fn: Callable[..., object] | None = None,
) -> TunedLens:
    """Fit per-layer affine translators for a tuned lens.

    Args:
        model: the (frozen) transformer to fit a lens for. Must expose an output
            embedding (``get_output_embeddings``) and residual blocks named
            ``...layers.N``.
        batches: an iterable of model inputs (tensors / HF-style mappings /
            anything ``forward_fn`` accepts). Consumed once and cached; the
            caller controls how much data the fit sees.
        layers: block indices to fit. Default: every resolvable block except the
            last (the last block IS the target frame, so its translator is the
            identity by construction and is not trained).
        steps: optimizer iterations over the cached activations.
        lr / weight_decay: AdamW hyperparameters for the translator params.
        device: device to run the fit on (default: the model's device).
        forward_fn: ``forward_fn(model, batch) -> output``; reuses the recipe
            convention for non-standard forward signatures.

    Returns:
        A :class:`TunedLens` with one ``(A, b)`` per fitted layer.
    """
    dev = torch.device(device) if device is not None else next(model.parameters()).device

    unembed = _resolve_unembed_weight(model)
    if unembed is None:
        raise ValueError(
            "fit_tuned_lens: model has no resolvable output embedding "
            "(get_output_embeddings().weight) — cannot fit a tuned lens."
        )
    blocks = _block_modules(model)
    if not blocks:
        raise ValueError(
            "fit_tuned_lens: no residual blocks matched '...layers.N' — "
            "cannot fit a tuned lens."
        )
    final_ln = _resolve_final_layer_norm(model)

    # --- capture residuals + reconstruct the target final distribution -------
    captured: dict[int, Tensor] = {}

    def _mk_hook(idx: int):
        def _hook(_m, _inp, out):
            captured[idx] = out[0] if isinstance(out, tuple) else out
        return _hook

    handles = [mod.register_forward_hook(_mk_hook(idx)) for idx, _name, mod in blocks]

    all_idx = [idx for idx, _n, _m in blocks]
    last_idx = all_idx[-1]
    fit_layers = layers if layers is not None else all_idx[:-1]
    fit_layers = [layer for layer in fit_layers if layer in all_idx]
    if not fit_layers:
        raise ValueError(
            "fit_tuned_lens: no layers to fit (the only block is the final "
            "frame, or `layers` matched nothing)."
        )

    d_model = int(unembed.shape[-1] if unembed.shape[0] >= unembed.shape[-1]
                  else unembed.shape[0])
    proj_W = _resolve_unembed(unembed.detach().to(torch.float32), d_model,
                              who="fit_tuned_lens").to(dev)
    ln = final_ln

    # Per-layer cache: list of residual chunks (float32, on `dev`); and the
    # matching final log-probs chunks (the regression target).
    res_cache: dict[int, list[Tensor]] = {layer: [] for layer in fit_layers}
    target_logprob_cache: list[Tensor] = []

    was_training = model.training
    model.eval()
    n_batches = 0
    with torch.no_grad():
        for batch in batches:
            captured.clear()
            _run_forward(model, batch, forward_fn)
            if last_idx not in captured:
                continue
            last_res = captured[last_idx].detach().to(torch.float32).to(dev)
            last_res = last_res.reshape(-1, last_res.shape[-1])
            last_normed = ln(last_res) if ln is not None else last_res
            target_logprob_cache.append(
                torch.log_softmax(last_normed @ proj_W, dim=-1)
            )
            for layer in fit_layers:
                r = captured[layer].detach().to(torch.float32).to(dev)
                res_cache[layer].append(r.reshape(-1, r.shape[-1]))
            n_batches += 1
    for h in handles:
        h.remove()
    if was_training:
        model.train()

    if n_batches == 0:
        raise ValueError(
            "fit_tuned_lens: no usable batches (the final block produced no "
            "residual on any batch)."
        )

    # --- train the affine translators ----------------------------------------
    res_cat = {layer: torch.cat(res_cache[layer], dim=0) for layer in fit_layers}
    target = torch.cat(target_logprob_cache, dim=0)  # (N, vocab) log-probs

    params: list[nn.Parameter] = []
    A: dict[int, nn.Parameter] = {}
    b: dict[int, nn.Parameter] = {}
    for layer in fit_layers:
        A[layer] = nn.Parameter(torch.eye(d_model, device=dev))
        b[layer] = nn.Parameter(torch.zeros(d_model, device=dev))
        params += [A[layer], b[layer]]

    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    for _step in range(steps):
        opt.zero_grad()
        loss = target.new_zeros(())
        for layer in fit_layers:
            r = res_cat[layer]
            lens_logits = (r @ A[layer].t() + b[layer])
            lens_logits = ln(lens_logits) @ proj_W if ln is not None else lens_logits @ proj_W
            log_q = torch.log_softmax(lens_logits, dim=-1)
            # KL(q || p) = Σ q (log q - log p), mean over tokens.
            loss = loss + (log_q.exp() * (log_q - target)).sum(dim=-1).mean()
        loss = loss / len(fit_layers)
        loss.backward()
        opt.step()

    translators = [
        (A[layer].detach().cpu(), b[layer].detach().cpu()) for layer in fit_layers
    ]
    return TunedLens(
        translators=translators,
        layers=list(fit_layers),
        d_model=d_model,
        model_fingerprint=model_fingerprint(model),
    )
