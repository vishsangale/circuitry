"""HyperDAS — input-conditioned alignment search via hypernetworks.

arXiv:2503.10894

Extends DAS by replacing the global fixed rotation R with a hypernetwork
H: activation → orthonormal subspace basis.  This makes the causal subspace
data-dependent, improving generalisation over DAS when the alignment varies
across inputs.

Training objective: same interchange-intervention cross-entropy as DAS, but
optimises H's parameters rather than R directly.  Differentiable Gram-Schmidt
orthonormalization preserves the orthogonality constraint throughout training.
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

__all__ = ["HyperDASNet", "HyperDASResult", "HyperDASRunner"]


# ---------------------------------------------------------------------------
# Differentiable Gram-Schmidt
# ---------------------------------------------------------------------------

def _gram_schmidt(V: Tensor) -> Tensor:
    """Differentiable Gram-Schmidt orthonormalization on rows of V.

    Args:
        V: (batch, k, d) tensor of k row vectors per batch element.

    Returns:
        (batch, k, d) tensor with orthonormal rows.
    """
    k = V.shape[1]
    result: list[Tensor] = []
    for i in range(k):
        v = V[:, i, :]  # (batch, d)
        for prev in result:
            v = v - (v * prev).sum(dim=-1, keepdim=True) * prev
        v = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        result.append(v)
    return torch.stack(result, dim=1)  # (batch, k, d)


# ---------------------------------------------------------------------------
# Hypernetwork
# ---------------------------------------------------------------------------

class HyperDASNet(nn.Module):
    """Hypernetwork: activation → per-example orthonormal subspace basis.

    Maps (batch, d_model) → (batch, subspace_dim, d_model) via MLP +
    differentiable Gram-Schmidt orthonormalization.

    Args:
        d_model:      Hidden dimension of the target module's output.
        subspace_dim: Number of basis vectors in the causal subspace.
        hidden_dim:   Width of the single hidden layer (default 64).
    """

    def __init__(self, d_model: int, subspace_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.d_model = d_model
        self.subspace_dim = subspace_dim
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, subspace_dim * d_model),
        )

    def forward(self, h: Tensor) -> Tensor:
        """Map activations to an orthonormal subspace basis.

        Args:
            h: (batch, d_model) pooled activation tensor.

        Returns:
            (batch, subspace_dim, d_model) tensor with orthonormal rows.
        """
        batch = h.shape[0]
        raw = self.net(h.to(torch.float32)).reshape(
            batch, self.subspace_dim, self.d_model
        )
        return _gram_schmidt(raw)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class HyperDASResult:
    """Result of a HyperDAS run.

    Attributes:
        network:    Trained hypernetwork mapping activations → subspace basis.
        iia_score:  Interchange intervention accuracy on training data.
        losses:     Per-step training loss values.
    """

    network: HyperDASNet
    iia_score: float
    losses: list[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class HyperDASRunner:
    """Trains HyperDASNet via interchange interventions.

    Args:
        model:        PyTorch model (frozen during training).
        module:       nn.Module whose output activations to align.
        d_model:      Hidden dimension at *module*'s output.
        subspace_dim: Causal subspace dimension (default 1).
        hidden_dim:   MLP hidden units in the hypernetwork (default 64).
    """

    def __init__(
        self,
        model: nn.Module,
        module: nn.Module,
        *,
        d_model: int,
        subspace_dim: int = 1,
        hidden_dim: int = 64,
    ) -> None:
        self._model = model
        self._module = module
        self._d_model = d_model
        self._subspace_dim = subspace_dim
        self._hidden_dim = hidden_dim

    def run(
        self,
        base_inputs: _Inputs,
        source_inputs: _Inputs,
        labels: Tensor,
        *,
        n_steps: int = 200,
        lr: float = 1e-3,
        loss_fn: Callable[[Tensor, Tensor], Tensor] | None = None,
    ) -> HyperDASResult:
        """Train the hypernetwork via interchange-intervention cross-entropy.

        Args:
            base_inputs:   Inputs for the "base" distribution.
            source_inputs: Inputs for the "source" distribution.
            labels:        Target class labels (torch.long) that the model
                           SHOULD produce after the interchange intervention.
            n_steps:       Gradient-descent steps (default 200).
            lr:            Adam learning rate (default 1e-3).
            loss_fn:       Optional ``(logits, labels) -> scalar`` loss.
                           Defaults to cross-entropy on the last token position.

        Returns:
            :class:`HyperDASResult` with the trained network and IIA score.
        """
        model = self._model
        module = self._module
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

        # ── Step 1: capture base and source activations ───────────────────────
        h_base = _capture_activations(model, base_inputs, module)    # (B, ..., d)
        h_source = _capture_activations(model, source_inputs, module)

        # ── Step 2: pool over non-batch, non-feature dims → (B, d_model) ──────
        h_base_pooled = _pool_activations(h_base)    # (B, d_model)
        device = h_base_pooled.device

        # ── Step 3: build hypernetwork ────────────────────────────────────────
        hyper_net = HyperDASNet(
            self._d_model, self._subspace_dim, self._hidden_dim
        ).to(device)

        optimizer = torch.optim.Adam(hyper_net.parameters(), lr=lr)
        _loss_fn = loss_fn or _default_loss_fn
        losses: list[float] = []

        # ── Step 4: interchange-intervention training loop ────────────────────
        for _ in range(n_steps):
            basis = hyper_net(h_base_pooled)  # (B, k, d) — gradients flow here
            h_int = _hyperdas_interchange(basis, h_base, h_source)

            # Inject h_int at the target module and run forward
            inj_store: dict[str, Tensor] = {"inject": h_int}
            inj_handle = module.register_forward_hook(
                _inject_hook(inj_store, "inject")
            )
            logits = _run_model(model, base_inputs)
            inj_handle.remove()

            loss = _loss_fn(logits, labels)
            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # ── Step 5: compute IIA on training data ──────────────────────────────
        iia = _compute_iia(
            model, base_inputs, source_inputs, labels,
            module, hyper_net, h_base_pooled, _loss_fn,
        )

        return HyperDASResult(
            network=hyper_net,
            iia_score=iia,
            losses=losses,
        )


# ---------------------------------------------------------------------------
# Interchange helper
# ---------------------------------------------------------------------------

def _hyperdas_interchange(
    basis: Tensor, h_base: Tensor, h_source: Tensor
) -> Tensor:
    """Replace h_base's subspace component with h_source's using the given basis.

    Gradients flow through *basis* but not through h_base / h_source
    (they are frozen model outputs).

    Args:
        basis:    (B, k, d) orthonormal basis rows from the hypernetwork.
        h_base:   (B, ..., d) base activations (detached inside).
        h_source: (B, ..., d) source activations (detached inside).

    Returns:
        (B, ..., d) — h_base with the causal subspace swapped from h_source.
    """
    orig_shape = h_base.shape
    d = orig_shape[-1]
    h_b = h_base.detach().reshape(orig_shape[0], -1, d)   # (B, P, d)
    h_s = h_source.detach().reshape(orig_shape[0], -1, d)  # (B, P, d)

    BT = basis.transpose(1, 2)              # (B, d, k)
    proj_b = torch.bmm(h_b, BT)            # (B, P, k)
    proj_s = torch.bmm(h_s, BT)            # (B, P, k)
    h_int = h_b - torch.bmm(proj_b, basis) + torch.bmm(proj_s, basis)
    return h_int.reshape(orig_shape)


# ---------------------------------------------------------------------------
# Private helpers (shared with das.py pattern)
# ---------------------------------------------------------------------------

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


def _capture_activations(
    model: nn.Module, inputs: _Inputs, module: nn.Module
) -> Tensor:
    """Run model and return the detached activation at *module* output."""
    store: dict[str, Tensor] = {}
    handle = module.register_forward_hook(_capture_hook(store, "h"))
    with torch.no_grad():
        _run_model(model, inputs)
    handle.remove()
    return store["h"]


def _pool_activations(h: Tensor) -> Tensor:
    """Mean-pool over all non-batch, non-feature dims → (B, d_model)."""
    if h.ndim == 2:
        return h  # already (B, d_model)
    # Flatten middle dims and mean-pool: (B, P, d) → (B, d)
    B, *mid, d = h.shape
    return h.reshape(B, -1, d).mean(dim=1)


def _default_loss_fn(logits: Tensor, labels: Tensor) -> Tensor:
    """CE on last token position, or flat CE for non-seq outputs."""
    if logits.dim() == 3:
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
    hyper_net: HyperDASNet,
    h_base_pooled: Tensor,
    loss_fn: Callable,
) -> float:
    """Compute interchange intervention accuracy with the trained hypernetwork."""
    h_base = _capture_activations(model, base_inputs, module)
    h_source = _capture_activations(model, source_inputs, module)

    with torch.no_grad():
        basis = hyper_net(h_base_pooled)
        h_int = _hyperdas_interchange(basis, h_base, h_source)

    inj_store: dict[str, Tensor] = {"inject": h_int}
    inj_handle = module.register_forward_hook(_inject_hook(inj_store, "inject"))
    with torch.no_grad():
        logits = _run_model(model, base_inputs)
    inj_handle.remove()

    if logits.dim() == 3:
        logits = logits[:, -1, :]
    elif logits.dim() > 2:
        logits = logits.view(-1, logits.shape[-1])

    lbl = labels.view(-1) if labels.dim() > 1 else labels
    preds = logits.argmax(dim=-1)
    return (preds == lbl).float().mean().item()
