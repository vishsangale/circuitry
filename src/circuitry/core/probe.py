"""Linear probing API.

Trains a single linear layer (logistic regression) on frozen activations
to measure how well a concept is linearly decodable.  Pure functions;
no hooks, no I/O.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor


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
