"""Logit-lens KL diagnostic. See docs/design.md §4 and the v0.9 spec §4.1.

Per-layer KL between the logit-lens projection of the residual stream and
the model's final logits. Nostalgebraist 2020 logit lens
(https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor

logger = logging.getLogger("circuitry.core.lens")


def _as_tensor(x: Any) -> Tensor:
    if isinstance(x, Tensor):
        return x
    return torch.as_tensor(x)


def logit_lens_kl(
    residual: Any,
    unembed: Any,
    final_logits: Any,
    *,
    layer_norm: Callable[[Tensor], Tensor] | None = None,
) -> float:
    """KL(softmax(layer_norm(residual) @ unembed) || softmax(final_logits)),
    mean over leading (batch, seq) dims.

    See docstring in spec §4.1 for the orientation auto-detect rule and the
    d_model == vocab edge case.
    """
    res = _as_tensor(residual)
    W = _as_tensor(unembed)
    fl = _as_tensor(final_logits)

    if W.ndim != 2:
        raise ValueError(
            f"logit_lens_kl: unembed must be 2-D, got shape {tuple(W.shape)}"
        )
    d_model = int(res.shape[-1])

    if W.shape[0] == d_model and W.shape[1] == d_model:
        logger.warning(
            "logit_lens_kl: unembed shape %s is square (d_model == vocab); "
            "orientation cannot be inferred, assuming (d_model, vocab) layout.",
            tuple(W.shape),
        )
        proj_W = W
    elif W.shape[0] == d_model:
        proj_W = W
    elif W.shape[1] == d_model:
        proj_W = W.t()
    else:
        raise ValueError(
            f"logit_lens_kl: unembed shape {tuple(W.shape)} has no dim "
            f"matching d_model={d_model}"
        )

    res_f32 = res.detach().to(torch.float32)
    if layer_norm is not None:
        res_f32 = layer_norm(res_f32)
    W_f32 = proj_W.detach().to(torch.float32)
    fl_f32 = fl.detach().to(torch.float32)

    lens_logits = res_f32 @ W_f32
    log_q = torch.log_softmax(lens_logits, dim=-1)  # lens distribution
    log_p = torch.log_softmax(fl_f32, dim=-1)        # final distribution
    q = log_q.exp()
    # KL(Q || P) = sum q * (log_q - log_p)
    kl_per_token = (q * (log_q - log_p)).sum(dim=-1)
    return float(kl_per_token.mean().item())
