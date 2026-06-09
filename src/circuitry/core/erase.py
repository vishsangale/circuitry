"""Concept erasure via mean-direction orthogonal projection (LEACE).

Park et al. 2023 "LEACE: Perfect linear concept erasure in closed form"
https://arxiv.org/abs/2306.03819
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class EraseProjection:
    P: Tensor          # (d_model, d_model) orthogonal projection onto concept complement
    direction: Tensor  # (d_model,) unit concept direction that was erased

    def apply(self, acts: Tensor) -> Tensor:
        """Project activations onto the concept-erased subspace.

        Acts: (..., d_model) → (..., d_model).
        """
        return acts.float() @ self.P.T


def leace_erase(
    acts: Tensor,
    labels: Tensor,
) -> EraseProjection:
    """Compute an orthogonal projection that erases the concept encoded in labels.

    Finds the mean-difference direction between class-conditional means
    (LEACE closed-form for binary concepts) and returns the projection
    onto its orthogonal complement.

    For multi-class labels, erases the first principal direction of the
    between-class mean matrix (the direction of maximum class-conditional
    mean variance).

    Args:
        acts:   (n, d_model) float tensor of activations.
        labels: (n,) integer class labels. Supports binary (2 classes) and
                multi-class (≥3 classes).

    Returns:
        EraseProjection with P and direction as detached CPU tensors.

    Example::

        proj = leace_erase(acts_train, labels_train)
        acts_erased = proj.apply(acts_test)
    """
    acts_f32 = acts.detach().to(torch.float32).cpu()   # (n, d_model)
    labels_cpu = labels.detach().cpu()

    classes = torch.unique(labels_cpu)
    n_classes = int(classes.shape[0])
    d_model = acts_f32.shape[-1]

    if n_classes < 2:
        raise ValueError(
            f"leace_erase: labels must contain at least 2 distinct classes, "
            f"got {n_classes}"
        )

    if n_classes == 2:
        # Binary LEACE: concept direction = difference of class-conditional means.
        c0, c1 = int(classes[0].item()), int(classes[1].item())
        mu0 = acts_f32[labels_cpu == c0].mean(0)  # (d_model,)
        mu1 = acts_f32[labels_cpu == c1].mean(0)  # (d_model,)
        d = mu1 - mu0                              # (d_model,)
    else:
        # Multi-class: stack per-class means; first right singular vector of
        # mean-centred matrix = direction of maximum between-class variance.
        class_means = torch.stack(
            [acts_f32[labels_cpu == int(c.item())].mean(0) for c in classes],
            dim=0,
        )  # (n_classes, d_model)
        M_centred = class_means - class_means.mean(0)  # (n_classes, d_model)
        # torch.svd returns (U, S, V) where M = U @ diag(S) @ V.T
        _U, _S, V = torch.svd(M_centred)
        d = V[:, 0]  # (d_model,) — first right singular vector

    norm = d.norm()
    if norm < 1e-12:
        # Degenerate case: all class means identical; return identity projection.
        d_hat = torch.zeros(d_model, dtype=torch.float32)
        P = torch.eye(d_model, dtype=torch.float32)
    else:
        d_hat = d / norm                                              # (d_model,)
        P = torch.eye(d_model, dtype=torch.float32) - d_hat.unsqueeze(1) @ d_hat.unsqueeze(0)

    return EraseProjection(
        P=P.detach(),
        direction=d_hat.detach(),
    )


def rlace_erase(
    acts: Tensor,
    labels: Tensor,
    *,
    rank: int = 1,
) -> EraseProjection:
    """Rank-k adversarial concept erasure (RLACE).

    Finds the rank-``rank`` orthogonal projection P = I − U Uᵀ (where U is a
    ``(d_model, rank)`` column-orthonormal matrix) that maximally removes a
    concept from a linear classifier.  The adversarially optimal subspace to
    erase is spanned by the top-``rank`` eigenvectors of the between-class
    scatter matrix B = M_c^T M_c (where M_c is the matrix of centred class
    means).

    For ``rank=1`` this recovers the LEACE direction (same subspace; different
    derivation from the adversarial perspective).

    For multi-class labels with ``rank > 1``, RLACE finds the subspace that
    most aggressively removes all concept information detectable by a linear
    probe — orthogonalising out the top-``rank`` directions of inter-class
    variance simultaneously.

    Args:
        acts:   ``(n, d_model)`` float activation tensor.
        labels: ``(n,)`` integer class labels (≥ 2 classes required).
        rank:   Number of directions to erase.  Must satisfy
                ``1 ≤ rank < d_model``.

    Returns:
        :class:`EraseProjection` where ``.direction`` is the first erased
        direction (the leading eigenvector) and ``.P`` is the full rank-``rank``
        orthogonal projection ``I − U Uᵀ``.

    Reference: Ravfogel et al. ICML 2022 "Linear Adversarial Concept Erasure"
               arXiv:2201.12091.
    """
    acts_f32 = acts.detach().to(torch.float32).cpu()
    labels_cpu = labels.detach().cpu()

    classes = torch.unique(labels_cpu)
    n_classes = int(classes.shape[0])
    d_model = acts_f32.shape[-1]

    if n_classes < 2:
        raise ValueError(
            f"rlace_erase: labels must have >= 2 distinct classes, got {n_classes}"
        )
    if rank < 1 or rank >= d_model:
        raise ValueError(
            f"rlace_erase: rank must satisfy 1 <= rank < d_model={d_model}, got rank={rank}"
        )
    rank = min(rank, n_classes - 1)  # can't erase more than n_classes-1 directions

    # Between-class scatter: B = M_c^T M_c where M_c = centred class means
    class_means = torch.stack(
        [acts_f32[labels_cpu == int(c.item())].mean(0) for c in classes],
        dim=0,
    )  # (C, d_model)
    M_c = class_means - class_means.mean(0)  # centred class means (C, d_model)
    B = M_c.T @ M_c  # (d_model, d_model) between-class scatter

    # Top-rank eigenvectors of B = the adversarially optimal subspace to erase
    # torch.linalg.eigh returns eigenvalues in ascending order → last `rank` are top
    eigvals, eigvecs = torch.linalg.eigh(B)  # eigvecs: (d_model, d_model)
    U = eigvecs[:, -rank:]  # (d_model, rank) — top eigenvectors

    # Orthogonalise U via QR to ensure exact orthonormality
    U, _ = torch.linalg.qr(U)

    P = torch.eye(d_model, dtype=torch.float32) - U @ U.T

    return EraseProjection(
        P=P.detach(),
        direction=U[:, 0].detach(),  # first erased direction
    )
