"""Linear probing API.

Trains a single linear layer (logistic regression) on frozen activations
to measure how well a concept is linearly decodable.  Pure functions;
no hooks, no I/O.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor


@dataclass(frozen=True)
class MDLResult:
    """Result of MDL (Minimum Description Length) probing (Voita & Titov 2020)."""
    code_length: float   # total online-coding code length in nats
    data_entropy: float  # H(Y) = -Σ p_c log p_c (nats)
    mdl_ratio: float     # code_length / (n_samples * data_entropy); < 1 = genuine encoding


@dataclass
class MassMeanProbe:
    """Mass-mean probe (Marks & Tegmark, COLM 2024, arXiv:2310.06824).

    Concept direction = normalised(μ₁ − μ₀).  Causally verified to match or
    beat logistic regression under intervention in most settings.  Binary only.
    """
    direction: Tensor   # (d_model,) unit vector
    threshold: float    # projection midpoint: (proj(μ₀) + proj(μ₁)) / 2
    classes: list       # [class_negative, class_positive]

    def predict(self, acts: Tensor) -> Tensor:
        """(n,) integer predictions via projection threshold."""
        proj = acts.to(torch.float32) @ self.direction.to(acts.device)
        return torch.where(
            proj > self.threshold,
            torch.tensor(self.classes[1], dtype=torch.long),
            torch.tensor(self.classes[0], dtype=torch.long),
        )

    def accuracy(self, acts: Tensor, labels: Tensor) -> float:
        """Fraction correctly classified."""
        preds = self.predict(acts)
        return float((preds == labels.long()).float().mean().item())


@dataclass
class LinearProbe:
    weight: Tensor  # (n_classes, d_model) — probe weight matrix; (1, d_model) for binary
    bias: Tensor    # (n_classes,) or (1,) for binary
    classes: list   # sorted list of unique label values seen during training

    def predict(self, acts: Tensor) -> Tensor:
        """Return integer class predictions.  acts: (..., d_model)."""
        proba = self.predict_proba(acts)
        indices = proba.argmax(dim=-1)
        class_tensor = torch.tensor(self.classes, dtype=torch.long)
        return class_tensor[indices]

    def predict_proba(self, acts: Tensor) -> Tensor:
        """Return softmax probabilities.  Shape (..., n_classes)."""
        acts_f = acts.to(torch.float32)
        logits = acts_f @ self.weight.T + self.bias
        return torch.softmax(logits, dim=-1)

    def accuracy(self, acts: Tensor, labels: Tensor) -> float:
        """Fraction of correctly classified samples.  acts: (n, d_model), labels: (n,)."""
        preds = self.predict(acts)
        return (preds == labels.long()).float().mean().item()

    def direction(self) -> Tensor:
        """(d_model,) unit vector — the concept direction.

        For binary probes: normalized weight[0].
        For multi-class: first principal component of the weight matrix.
        """
        if len(self.classes) == 2:
            v = self.weight[0]
            norm = v.norm()
            if norm < 1e-8:
                return v
            return v / norm
        else:
            # First left singular vector via pca_lowrank (U, S, V) — spec: [0][:, 0]
            U, _, _ = torch.pca_lowrank(self.weight, q=1)
            v = U[:, 0]
            norm = v.norm()
            if norm < 1e-8:
                return v
            return v / norm


def train_linear_probe(
    acts: Tensor,
    labels: Tensor,
    *,
    max_iter: int = 1000,
    C: float = 1.0,
    tol: float = 1e-4,
    device: str | torch.device | None = None,
) -> LinearProbe:
    """Train a logistic-regression linear probe on frozen activations.

    Uses a simple gradient-descent loop (Adam + L2 regularisation) — no
    scikit-learn dependency.

    Args:
        acts:     (n, d_model) float tensor of activations.
        labels:   (n,) integer or long tensor of class labels.
        max_iter: maximum number of optimiser steps.
        C:        inverse regularisation strength (higher = less regularisation).
        tol:      early-stop tolerance on loss change between consecutive steps.
        device:   device for the optimisation loop; defaults to acts.device.

    Returns:
        Fitted LinearProbe.
    """
    if device is None:
        device = acts.device

    acts_f = acts.detach().to(torch.float32).to(device)

    # Map labels to contiguous 0-based indices
    classes = sorted(labels.unique().tolist())
    label_to_idx = {c: i for i, c in enumerate(classes)}
    idx_labels = torch.tensor(
        [label_to_idx[v.item()] for v in labels], dtype=torch.long, device=device
    )

    n, d_model = acts_f.shape
    n_classes = len(classes)

    linear = nn.Linear(d_model, n_classes, bias=True).to(device)
    nn.init.zeros_(linear.weight)
    nn.init.zeros_(linear.bias)

    weight_decay = 1.0 / (C * n) if n > 0 else 0.0
    optimizer = torch.optim.Adam(linear.parameters(), lr=0.01, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    prev_loss = float("inf")
    consecutive_tol = 0

    for _ in range(max_iter):
        optimizer.zero_grad()
        logits = linear(acts_f)
        loss = loss_fn(logits, idx_labels)
        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        if abs(prev_loss - loss_val) < tol:
            consecutive_tol += 1
            if consecutive_tol >= 3:
                break
        else:
            consecutive_tol = 0
        prev_loss = loss_val

    weight = linear.weight.detach().cpu()
    bias = linear.bias.detach().cpu()

    return LinearProbe(weight=weight, bias=bias, classes=classes)


def mdl_probe(
    acts: Tensor,
    labels: Tensor,
    *,
    n_chunks: int = 8,
) -> MDLResult:
    """MDL probe via online coding (Voita & Titov 2020, arXiv:2003.12298).

    Measures how well a linear probe compresses the concept labels via
    sequential online coding: for each chunk i, trains a probe on chunks
    1..(i-1) and evaluates negative log-likelihood on chunk i.  The first
    chunk uses the uniform prior (log n_classes nats/sample).

    MDLResult.mdl_ratio < 1 means the layer compresses the labels better
    than the uniform code — evidence of genuine concept encoding.

    Args:
        acts:     (n, d_model) activation tensor.
        labels:   (n,) integer class labels.
        n_chunks: number of sequential chunks (8 = Voita & Titov default).

    Returns:
        MDLResult with code_length, data_entropy, mdl_ratio.
    """
    n = acts.shape[0]
    if n < n_chunks:
        raise ValueError(f"mdl_probe: n={n} must be >= n_chunks={n_chunks}")

    classes = sorted(labels.unique().tolist())
    n_classes = len(classes)

    # Data entropy H(Y) in nats
    counts = torch.zeros(n_classes, dtype=torch.float32)
    for i, c in enumerate(classes):
        counts[i] = float((labels == c).sum().item())
    probs = counts / counts.sum()
    data_entropy = float(
        -(probs * torch.where(probs > 0, probs.log(), torch.zeros_like(probs))).sum().item()
    )

    chunk_size = n // n_chunks
    total_code_length = 0.0

    for i in range(n_chunks):
        eval_start = i * chunk_size
        eval_end = (i + 1) * chunk_size if i < n_chunks - 1 else n
        eval_acts = acts[eval_start:eval_end].to(torch.float32)
        eval_labels = labels[eval_start:eval_end]
        n_eval = eval_end - eval_start

        if i == 0:
            # No training data: uniform prior → log(n_classes) nats per sample
            total_code_length += math.log(max(n_classes, 1)) * n_eval
        else:
            train_acts = acts[:eval_start]
            train_labels = labels[:eval_start]
            probe = train_linear_probe(train_acts, train_labels, max_iter=300, C=1.0)
            proba = probe.predict_proba(eval_acts).to(torch.float32)  # (n_eval, n_classes)
            probe_label_to_idx = {c: j for j, c in enumerate(probe.classes)}
            idx = torch.tensor(
                [probe_label_to_idx.get(int(v.item()), 0) for v in eval_labels],
                dtype=torch.long,
            )
            selected = proba[torch.arange(len(idx)), idx].clamp(min=1e-10)
            total_code_length += float(-selected.log().sum().item())

    uniform_code = n * data_entropy
    mdl_ratio = total_code_length / uniform_code if uniform_code > 1e-10 else 1.0

    return MDLResult(
        code_length=total_code_length,
        data_entropy=data_entropy,
        mdl_ratio=mdl_ratio,
    )


def mass_mean_probe(acts: Tensor, labels: Tensor) -> MassMeanProbe:
    """Fit a mass-mean probe (binary; mean-difference direction).

    The concept direction is μ₁ − μ₀ (normalised).  Unlike a trained linear
    probe, no optimisation is performed — the direction is purely geometric.
    This is causally verified to match or beat logistic regression under
    intervention (Marks & Tegmark COLM 2024, arXiv:2310.06824).

    Args:
        acts:   (n, d_model) activation tensor.
        labels: (n,) binary integer labels (exactly two unique values).

    Returns:
        MassMeanProbe with .direction, .threshold, .classes, .predict(), .accuracy().

    Raises:
        ValueError: if labels has more than two unique values.
    """
    acts_f = acts.detach().to(torch.float32)
    classes = sorted(labels.unique().tolist())
    if len(classes) != 2:
        raise ValueError(
            f"mass_mean_probe: requires exactly 2 classes; got {len(classes)}: {classes}"
        )
    c0, c1 = classes
    mu0 = acts_f[labels == c0].mean(0)
    mu1 = acts_f[labels == c1].mean(0)

    diff = mu1 - mu0
    norm = diff.norm()
    if norm < 1e-8:
        raise ValueError("mass_mean_probe: class means are identical; direction undefined")
    direction = (diff / norm).cpu()

    proj0 = float((mu0 @ direction.to(acts_f.device)).item())
    proj1 = float((mu1 @ direction.to(acts_f.device)).item())
    threshold = (proj0 + proj1) / 2.0

    return MassMeanProbe(direction=direction, threshold=threshold, classes=classes)


def verify_linear_representation(
    probe: LinearProbe,
    steer_vec: Tensor,
) -> float:
    """Cosine similarity between linear probe direction and a steering vector.

    Tests the linear representation hypothesis: if a concept is linearly
    represented, the discriminative direction (probe) and the causal direction
    (steering vector from ``steer_vector`` or ``repe_direction``) should agree.

    Args:
        probe:     A fitted LinearProbe or MassMeanProbe with a .direction() method.
        steer_vec: (d_model,) steering vector.

    Returns:
        float in [-1, 1]; ≈ ±1 = hypothesis holds; ≈ 0 = orthogonal / unrelated.

    Reference: Park et al. arXiv:2311.03658; Choe et al. arXiv:2502.16385.
    """
    raw = probe.direction() if callable(probe.direction) else probe.direction
    d = raw.to(torch.float32).flatten()
    v = steer_vec.detach().to(torch.float32).flatten()

    d_norm = d.norm()
    v_norm = v.norm()
    if d_norm < 1e-8 or v_norm < 1e-8:
        return 0.0

    min_len = min(d.shape[0], v.shape[0])
    return float(((d[:min_len] / d_norm) @ (v[:min_len] / v_norm)).item())
