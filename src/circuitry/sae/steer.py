"""SAE-based steering vectors. v1.31.

fgaa_steering_vector — Feature-Guided Activation Addition (FGAA).
Encodes positive and negative example activations through the SAE,
selects the most discriminative features, and returns a weighted sum
of their decoder columns as the steering vector.

Outperforms raw CAA (mean-difference on activations) and naive
single-feature decoder steering on AxBench (arXiv:2501.09929).
"""
from __future__ import annotations

from typing import Any

import torch
from torch import Tensor


def fgaa_steering_vector(
    sae: Any,
    positive_acts: Tensor,
    negative_acts: Tensor,
    *,
    n_features: int = 10,
) -> Tensor:
    """Feature-Guided Activation Addition (FGAA) steering vector.

    Encodes positive and negative example activations through the SAE,
    identifies the ``n_features`` most discriminative SAE features by
    ``|mean_pos - mean_neg|``, and returns a weighted sum of their decoder
    columns (scaled by the signed activation difference).

    The result is a ``(d_model,)`` vector that can be passed directly to
    :func:`~circuitry.patching.steer.apply_steer`.

    Args:
        sae:           SAE object with ``.encode`` / ``.decode``.
        positive_acts: ``(n_pos, d_model)`` activations for the "positive"
                       / steered-toward concept.
        negative_acts: ``(n_neg, d_model)`` activations for the opposite.
        n_features:    Number of top discriminative SAE features to include.

    Returns:
        ``(d_model,)`` steering vector (float32, CPU).

    Reference: arXiv:2501.09929 "Feature-Guided Activation Addition".
    """
    sae_device = getattr(sae, "device", positive_acts.device)
    sae_dtype = getattr(sae, "dtype", torch.float32)

    pos = torch.as_tensor(positive_acts).to(sae_device, sae_dtype)
    neg = torch.as_tensor(negative_acts).to(sae_device, sae_dtype)
    if pos.ndim == 1:
        pos = pos.unsqueeze(0)
    if neg.ndim == 1:
        neg = neg.unsqueeze(0)

    with torch.inference_mode():
        f_pos = sae.encode(pos)   # (n_pos, d_sae)
        f_neg = sae.encode(neg)   # (n_neg, d_sae)

    mean_pos = f_pos.detach().float().mean(0)  # (d_sae,)
    mean_neg = f_neg.detach().float().mean(0)  # (d_sae,)
    diff = mean_pos - mean_neg                 # (d_sae,) signed

    # Select top-n_features by |diff|
    k = min(n_features, diff.shape[0])
    _, top_idx = diff.abs().topk(k)

    # Reconstruct the decoder weight matrix: decode(e_i) - decode(0) for each feature i
    # For a standard linear decoder W_dec this equals W_dec[:, i].
    # We approximate by decoding one-hot feature vectors (works for all SAE types).
    d_sae = diff.shape[0]
    d_model: int | None = None
    decoder_cols: list[Tensor] = []
    for i in top_idx:
        one_hot = torch.zeros(1, d_sae, dtype=sae_dtype, device=sae_device)
        one_hot[0, i] = 1.0
        with torch.inference_mode():
            col = sae.decode(one_hot).squeeze(0).float().cpu()
        decoder_cols.append(col)
        if d_model is None:
            d_model = col.shape[0]

    # Weight each decoder column by the signed activation difference at that feature
    weights = diff[top_idx].cpu()  # (k,)
    vector = sum(w * col for w, col in zip(weights, decoder_cols))  # (d_model,)
    return vector.detach().cpu().float()
