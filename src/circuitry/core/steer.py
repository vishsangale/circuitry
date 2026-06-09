"""Activation steering vector primitive. See docs/design.md §4.

Pure function: no I/O, no .cuda(), no nn.Module state.

Reference: Rimsky et al. 2024 "Steering Llama 2 via Contrastive Activation
Addition" (CAA) — https://arxiv.org/abs/2312.06681
"""
from __future__ import annotations

import torch
from torch import Tensor


def steer_vector(
    positive_acts: Tensor,
    negative_acts: Tensor,
    *,
    normalize: bool = True,
) -> Tensor:
    """Compute a contrastive steering vector (Rimsky et al. 2024 CAA).

    Averages activations across the batch dimension for each polarity,
    returns ``mean_positive - mean_negative``. If ``normalize=True``,
    divides by the L2 norm so the returned vector is unit length.

    Args:
        positive_acts: ``(batch, d_model)`` or ``(d_model,)`` — activations
            for the "positive" / steered-toward direction.
        negative_acts: same shape — activations for the opposite direction.
        normalize: if ``True``, return a unit vector.

    Returns:
        Tensor of shape ``(d_model,)``.

    Raises:
        ValueError: if ``normalize=True`` and the difference vector has
            near-zero norm.
    """
    pos = torch.as_tensor(positive_acts).detach().to(torch.float32)
    neg = torch.as_tensor(negative_acts).detach().to(torch.float32)

    # Mean-reduce batch dim if present
    if pos.ndim == 2:
        pos = pos.mean(0)
    if neg.ndim == 2:
        neg = neg.mean(0)

    diff = pos - neg

    if normalize:
        norm = diff.norm()
        if norm < 1e-8:
            raise ValueError(
                "steer_vector: difference vector has near-zero norm; check inputs"
            )
        diff = diff / norm

    return diff


def repe_direction(diffs: Tensor) -> Tensor:
    """First-PC concept direction from activation differences (Zou et al. 2023 RepE).

    Given contrastive activation differences (positive_acts − negative_acts) for
    a set of stimuli pairs, returns the first principal component as the concept
    direction.  Unlike ``steer_vector`` (mean-difference), PCA gives orthogonality
    guarantees when multiple concepts are extracted from the same data.

    Args:
        diffs: (n_pairs, d_model) pre-computed activation differences.

    Returns:
        (d_model,) unit vector — first PC of the centred difference matrix.

    Reference: Zou et al. 2023 "Representation Engineering" arXiv:2310.01405
    """
    d = torch.as_tensor(diffs).detach().to(torch.float32)
    if d.ndim == 1:
        d = d.unsqueeze(0)
    d_c = d - d.mean(0, keepdim=True)
    if d_c.shape[0] < 2:
        # Single sample: centering collapses to zero; use the uncentered vector.
        v = d[0]
    elif d_c.norm() < 1e-10:
        return torch.zeros(d.shape[1])
    else:
        _, _, V = torch.pca_lowrank(d_c, q=1, center=False)
        v = V[:, 0]
    norm = v.norm()
    if norm < 1e-8:
        return torch.zeros_like(v)
    return v / norm


def directional_ablation(acts: Tensor, direction: Tensor) -> Tensor:
    """Remove a concept direction from activations (orthogonal projection).

    Returns ``acts − (acts · d̂) d̂`` where ``d̂`` is the unit-normalised
    direction.  Equivalent to projecting onto the orthogonal complement.

    Args:
        acts:      (..., d_model) activation tensor.
        direction: (d_model,) concept direction (need not be unit-length).

    Returns:
        Tensor of same shape as ``acts`` with the direction component removed.

    Reference: Arditi et al. NeurIPS 2024 "Refusal in LMs" arXiv:2406.11717
    """
    acts_f = acts.to(torch.float32)
    d = direction.detach().to(torch.float32)
    norm = d.norm()
    if norm < 1e-8:
        return acts_f
    d_hat = (d / norm).to(acts_f.device)
    proj = (acts_f @ d_hat).unsqueeze(-1)  # (..., 1)
    return acts_f - proj * d_hat
