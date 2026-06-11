"""DAS — Distributed Alignment Search.

Geiger et al., "Finding Alignments Between Interpretable Causal Variables and
Distributed Neural Representations", NeurIPS 2023.
https://arxiv.org/abs/2303.02536

Learns an orthogonal rotation R such that the first ``subspace_dim`` columns
of R·h align with a specified causal variable via interchange-intervention
training.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

_Inputs = Tensor | dict[str, Any]


@dataclass
class DASResult:
    """Result of a DAS run."""

    rotation: Tensor          # (d_model, d_model) orthogonal matrix R
    subspace_dim: int         # dimension of the causal subspace
    iia_score: float          # interchange intervention accuracy on training data
    losses: list[float] = field(default_factory=list)  # per-step training losses

    def subspace_directions(self) -> Tensor:
        """Return the first ``subspace_dim`` rows of R — the causal directions."""
        return self.rotation[:self.subspace_dim]  # (subspace_dim, d_model)


def _run_model(model: nn.Module, inputs: _Inputs) -> Tensor:
    if isinstance(inputs, Tensor):
        return model(inputs)
    return model(**inputs)


def _capture_hook(store: dict, key: str) -> Callable:
    def hook(mod: nn.Module, inp: tuple, output: object) -> None:  # noqa: ARG001
        out = output[0] if isinstance(output, tuple) else output
        store[key] = out.detach()
    return hook


def _inject_hook(store: dict, key: str) -> Callable:
    def hook(mod: nn.Module, inp: tuple, output: object) -> object:  # noqa: ARG001
        val = store[key]
        if isinstance(output, tuple):
            return (val,) + output[1:]
        return val
    return hook


def _get_d_model(tensor: Tensor) -> int:
    """Last dim of the captured activation."""
    return tensor.shape[-1]


class DASRunner:
    """Learns a rotation R such that R·h aligns with causal variables.

    Usage::

        runner = DASRunner(model)
        result = runner.run(
            base_inputs, source_inputs, labels,
            module=model.layers[2],
            subspace_dim=1,
            n_steps=300,
        )
        # result.rotation — the learned orthogonal R
    """

    def __init__(self, model: nn.Module) -> None:
        self._model = model

    def run(
        self,
        base_inputs: _Inputs,
        source_inputs: _Inputs,
        labels: Tensor,
        *,
        module: nn.Module,
        subspace_dim: int = 1,
        n_steps: int = 500,
        lr: float = 0.01,
        loss_fn: Callable[[Tensor, Tensor], Tensor] | None = None,
    ) -> DASResult:
        """Run DAS training.

        Args:
            base_inputs: Inputs for the "base" distribution.
            source_inputs: Inputs for the "source" distribution (different
                causal variable value).
            labels: Target class labels (``torch.long``) that the model
                SHOULD produce after the interchange intervention.  Shape
                must be compatible with ``F.cross_entropy(output, labels)``.
            module: The ``nn.Module`` whose output activations to align.
            subspace_dim: Dimension of the causal subspace (number of
                rotated dimensions to swap).
            n_steps: Gradient-descent steps.
            lr: Adam learning rate for R.
            loss_fn: Optional ``(logits, labels) -> scalar`` loss.
                Defaults to cross-entropy on the last token/position
                (logits reshaped to ``(batch * seq, vocab)``).

        Returns:
            :class:`DASResult` with the converged rotation and IIA score.
        """
        model = self._model
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

        # ── Step 1: capture d_model from a sample forward pass ───────────────
        store: dict[str, Tensor] = {}
        h_cap = module.register_forward_hook(_capture_hook(store, "sample"))
        with torch.no_grad():
            _run_model(model, base_inputs)
        h_cap.remove()
        d_model = _get_d_model(store["sample"])

        # ── Step 2: capture base and source activations (no grad needed) ──────
        h_base = _capture_activations(model, base_inputs, module)
        h_source = _capture_activations(model, source_inputs, module)

        # ── Step 3: initialise rotation R as identity ─────────────────────────
        R = torch.eye(d_model, dtype=h_base.dtype, device=h_base.device,
                      requires_grad=True)
        # Use a plain list so we can swap .data in-place after each projection
        optimizer = torch.optim.Adam([R], lr=lr)

        _loss_fn = loss_fn or _default_loss_fn
        losses: list[float] = []

        # ── Step 4: interchange-intervention training loop ────────────────────
        for _ in range(n_steps):
            # Build h_int via differentiable interchange
            h_int = _interchange(R, h_base.detach(), h_source.detach(), subspace_dim)

            # Inject h_int at the target module and run forward
            inj_store: dict[str, Tensor] = {"inject": h_int}
            inj_handle = module.register_forward_hook(_inject_hook(inj_store, "inject"))
            logits = _run_model(model, base_inputs)
            inj_handle.remove()

            loss = _loss_fn(logits, labels)
            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Project R back to nearest orthogonal matrix (Stiefel retraction)
            with torch.no_grad():
                U, _, Vh = torch.linalg.svd(R.data, full_matrices=True)
                R.data = U @ Vh

        # ── Step 5: compute IIA on training data ──────────────────────────────
        iia = _compute_iia(model, base_inputs, source_inputs, labels,
                           module, R.detach(), subspace_dim, _loss_fn)

        return DASResult(
            rotation=R.detach().clone(),
            subspace_dim=subspace_dim,
            iia_score=iia,
            losses=losses,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _capture_activations(model: nn.Module, inputs: _Inputs,
                         module: nn.Module) -> Tensor:
    """Run model and return the detached activation at *module* output."""
    store: dict[str, Tensor] = {}
    handle = module.register_forward_hook(_capture_hook(store, "h"))
    with torch.no_grad():
        _run_model(model, inputs)
    handle.remove()
    return store["h"]


def _interchange(R: Tensor, h_base: Tensor, h_source: Tensor,
                 subspace_dim: int) -> Tensor:
    """Differentiable interchange intervention.

    Rotates both activations into R-space, replaces the first
    ``subspace_dim`` dimensions with those from h_source, rotates back.

    R : (d_model, d_model)
    h_base, h_source : (..., d_model)
    """
    # (..., d_model) @ (d_model, d_model).T = (..., d_model)
    z_base = h_base @ R.T
    z_source = h_source @ R.T

    d_model = R.shape[0]
    # Mask: 1 for source dims, 0 for base dims — avoids in-place ops
    mask = torch.zeros(d_model, dtype=R.dtype, device=R.device)
    mask[:subspace_dim] = 1.0
    z_int = z_base * (1.0 - mask) + z_source * mask

    return z_int @ R  # rotate back


def _default_loss_fn(logits: Tensor, labels: Tensor) -> Tensor:
    """CE on last token position, or flat CE for non-seq outputs."""
    if logits.dim() == 3:  # (batch, seq, vocab) — use last position
        logits = logits[:, -1, :]
    elif logits.dim() > 2:
        logits = logits.view(-1, logits.shape[-1])
    if labels.dim() > 1:
        labels = labels.view(-1)
    return F.cross_entropy(logits, labels)


def _compute_iia(
    model: nn.Module,
    base_inputs: _Inputs,
    source_inputs: _Inputs,
    labels: Tensor,
    module: nn.Module,
    R: Tensor,
    subspace_dim: int,
    loss_fn: Callable,
) -> float:
    """Interchange intervention accuracy — fraction of correct predictions."""
    h_base = _capture_activations(model, base_inputs, module)
    h_source = _capture_activations(model, source_inputs, module)

    with torch.no_grad():
        h_int = _interchange(R, h_base, h_source, subspace_dim)

    inj_store: dict[str, Tensor] = {"inject": h_int}
    inj_handle = module.register_forward_hook(_inject_hook(inj_store, "inject"))
    with torch.no_grad():
        logits = _run_model(model, base_inputs)
    inj_handle.remove()

    if logits.dim() == 3:
        logits = logits[:, -1, :]

    lbl = labels.view(-1) if labels.dim() > 1 else labels
    preds = logits.argmax(dim=-1)
    return (preds == lbl).float().mean().item()
