"""CLT Attribution Graphs — cross-layer transcoder feature attribution.

Anthropic, transformer-circuits.pub/2025/attribution-graphs (arXiv:2603.21014)

Builds a feature-level attribution graph using cross-layer transcoders (CLTs).
Each transcoder at layer l maps residual-stream state → sparse features →
layer output reconstruction.  Attribution scores between features at layer l
and layer l+1 use the EAP approximation:

    score(fi → fj) = Σ_{batch, pos} delta_fi · grad_fj

where delta_fi = f_i(corrupted) − f_i(clean) and grad_fj = ∂metric/∂fj.

The clean forward pass splices each transcoder losslessly so that gradients
flow through the feature activations:

    output_spliced = decode(encode(x)) + sg(output − decode(encode(x)))

where sg = stop_gradient (.detach()).  This makes encode(x) part of the
computation graph without altering the numeric values seen by later layers.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

_Inputs = Tensor | dict[str, Any]

__all__ = ["CLTNode", "CLTEdge", "CLTGraphResult", "CLTGraphRunner"]


@dataclass(frozen=True, order=True)
class CLTNode:
    """A feature node in the CLT attribution graph."""
    layer: int
    feature: int


@dataclass(frozen=True, order=True)
class CLTEdge:
    """A directed edge between features at consecutive layers."""
    src: CLTNode
    dst: CLTNode


@dataclass
class CLTGraphResult:
    """Attribution graph output from :class:`CLTGraphRunner`.

    Attributes:
        scores:      ``{CLTEdge: float}`` — EAP attribution score per edge.
        node_scores: ``{CLTNode: float}`` — per-feature importance (delta * grad,
                     summed over batch/pos).
        n_layers:    Number of layers with transcoders.
        n_features:  ``n_features[i]`` = number of features at ``layer_order[i]``.
        layer_order: Sorted list of layer indices that have transcoders.
    """
    scores: dict[CLTEdge, float]
    node_scores: dict[CLTNode, float]
    n_layers: int
    n_features: list[int]
    layer_order: list[int]

    def ranked(self) -> list[tuple[CLTEdge, float]]:
        """All edges sorted by |score| descending."""
        return sorted(self.scores.items(), key=lambda kv: abs(kv[1]), reverse=True)

    def top_k(self, k: int) -> list[tuple[CLTEdge, float]]:
        """Top-k edges by |score|."""
        return self.ranked()[:k]

    def threshold(self, tau: float) -> list[CLTEdge]:
        """Edges with |score| >= tau."""
        return [e for e, s in self.scores.items() if abs(s) >= tau]

    def to_markdown(self, *, top_k: int = 20) -> str:
        lines = ["## CLT Attribution Graph", ""]
        lines.append(f"- Layers: {self.layer_order}")
        lines.append(f"- Features per layer: {self.n_features}")
        lines.append(f"- Total edges scored: {len(self.scores)}")
        lines.append("")
        ranked = self.top_k(top_k)
        lines.append(f"### Top-{len(ranked)} Edges by |score|")
        lines.append("")
        lines.append("| rank | src layer | src feat | dst layer | dst feat | score |")
        lines.append("| ---: | ---: | ---: | ---: | ---: | ---: |")
        for i, (edge, score) in enumerate(ranked, 1):
            lines.append(
                f"| {i} | {edge.src.layer} | {edge.src.feature}"
                f" | {edge.dst.layer} | {edge.dst.feature} | {score:.4g} |"
            )
        return "\n".join(lines)


class CLTGraphRunner:
    """Builds a CLT attribution graph across model layers.

    For each consecutive pair of layers ``(l, l+1)`` that both have transcoders,
    scores every feature-to-feature edge using the EAP approximation.  The clean
    forward pass splices each transcoder losslessly so PyTorch's autograd gives
    ``f.grad = ∂metric/∂f`` after ``loss.backward()``.

    Args:
        model:             PyTorch model.  Must expose a ``layers`` ModuleList
                           (accessed as ``model.layers`` or ``model.model.layers``).
        layer_transcoders: ``{layer_idx: transcoder}`` — one entry per layer to
                           instrument.  Each transcoder must implement
                           ``encode(x: Tensor) → Tensor`` (features from the
                           layer's residual-stream input) and
                           ``decode(f: Tensor) → Tensor`` (reconstruction in
                           residual-stream space).

    Reference: Anthropic, "Attribution Graphs", transformer-circuits.pub/2025/
    attribution-graphs; arXiv:2603.21014.
    """

    def __init__(
        self,
        model: nn.Module,
        layer_transcoders: dict[int, Any],
    ) -> None:
        self.model = model
        self.layer_transcoders = dict(layer_transcoders)
        self._layers = self._locate_layers(model)

    @staticmethod
    def _locate_layers(model: nn.Module) -> nn.ModuleList:
        for attr in ("layers",):
            sub = getattr(model, attr, None)
            if isinstance(sub, nn.ModuleList):
                return sub
        sub = getattr(model, "model", None)
        if sub is not None:
            layers = getattr(sub, "layers", None)
            if isinstance(layers, nn.ModuleList):
                return layers
        raise ValueError(
            "CLTGraphRunner: cannot locate a 'layers' ModuleList. "
            "The model must expose model.layers or model.model.layers."
        )

    def _run_model(self, inputs: _Inputs) -> Tensor:
        if isinstance(inputs, dict):
            return self.model(**inputs)
        return self.model(inputs)

    def _collect_corrupted_features(self, inputs: _Inputs) -> dict[int, Tensor]:
        f_store: dict[int, Tensor] = {}
        handles = []
        for layer_idx, tc in self.layer_transcoders.items():
            module = self._layers[layer_idx]

            def _hook(mod, inp, output, _li=layer_idx, _tc=tc):  # noqa: ARG001
                with torch.no_grad():
                    f_store[_li] = _tc.encode(inp[0].detach())

            handles.append(module.register_forward_hook(_hook))
        try:
            with torch.no_grad():
                self._run_model(inputs)
        finally:
            for h in handles:
                h.remove()
        return f_store

    def _collect_clean_features(
        self, inputs: _Inputs, metric: Callable[[Tensor], Tensor]
    ) -> dict[int, Tensor]:
        """Lossless-splice forward pass; populates ``f.grad`` via backward."""
        f_store: dict[int, Tensor] = {}
        handles = []
        for layer_idx, tc in self.layer_transcoders.items():
            module = self._layers[layer_idx]

            def _hook(mod, inp, output, _li=layer_idx, _tc=tc):  # noqa: ARG001
                x = inp[0]
                f = _tc.encode(x)
                # Parameters are grad-disabled during the clean pass; f must
                # explicitly opt-in to grad tracking so retain_grad() succeeds
                # and backward populates f.grad.
                f.requires_grad_(True)
                f.retain_grad()
                f_store[_li] = f
                recon = _tc.decode(f)
                # Lossless: gradient flows through recon (→ f); error is stopped.
                spliced = recon + (output.detach() - recon.detach())
                if isinstance(output, tuple):
                    return (spliced,) + output[1:]
                return spliced

            handles.append(module.register_forward_hook(_hook))
        try:
            out = self._run_model(inputs)
            loss = metric(out)
            loss.backward()
        finally:
            for h in handles:
                h.remove()
        return f_store

    def run(
        self,
        clean_inputs: _Inputs,
        corrupted_inputs: _Inputs,
        metric: Callable[[Tensor], Tensor],
        *,
        score_threshold: float = 0.0,
    ) -> CLTGraphResult:
        """Build the CLT attribution graph.

        Args:
            clean_inputs:     Clean model inputs.
            corrupted_inputs: Corrupted model inputs.
            metric:           ``metric(logits) → scalar tensor``.
            score_threshold:  Omit edges with |score| < threshold (default 0 = keep all).

        Returns:
            :class:`CLTGraphResult`.
        """
        model = self.model
        was_training = model.training
        orig_rg = {n: p.requires_grad for n, p in model.named_parameters()}
        try:
            model.eval()

            # Pass 1: corrupted features, no grad
            f_corr = self._collect_corrupted_features(corrupted_inputs)

            # Pass 2: clean features with lossless splice; disable param grads so
            # backward only populates feature .grad, not param .grad
            for p in model.parameters():
                p.requires_grad_(False)
            f_clean = self._collect_clean_features(clean_inputs, metric)

            # Score
            sorted_layers = sorted(self.layer_transcoders.keys())
            scores: dict[CLTEdge, float] = {}
            node_scores: dict[CLTNode, float] = {}
            n_features: list[int] = []

            for i, layer_l in enumerate(sorted_layers):
                f_cl = f_clean.get(layer_l)
                f_co = f_corr.get(layer_l)
                if f_cl is None or f_co is None:
                    n_features.append(0)
                    continue

                n_feat_l = f_cl.shape[-1]
                n_features.append(n_feat_l)
                delta_l = (f_co.detach() - f_cl.detach()).to(torch.float32)

                # Node scores: Σ_{batch,pos} delta_fi * grad_fi
                if f_cl.grad is not None:
                    grad_l = f_cl.grad.to(torch.float32)
                    ns_vec = (delta_l * grad_l).reshape(-1, n_feat_l).sum(dim=0)
                    for fi in range(n_feat_l):
                        node_scores[CLTNode(layer_l, fi)] = float(ns_vec[fi].item())

                # Edge scores to next layer
                if i + 1 >= len(sorted_layers):
                    continue
                layer_l1 = sorted_layers[i + 1]
                f_cl1 = f_clean.get(layer_l1)
                if f_cl1 is None or f_cl1.grad is None:
                    continue

                grad_l1 = f_cl1.grad.to(torch.float32)
                n_feat_l1 = grad_l1.shape[-1]
                d_flat = delta_l.reshape(-1, n_feat_l)       # (B·P, F_l)
                g_flat = grad_l1.reshape(-1, n_feat_l1)      # (B·P, F_{l+1})
                score_mat = d_flat.T @ g_flat                 # (F_l, F_{l+1})

                for fi in range(n_feat_l):
                    for fj in range(n_feat_l1):
                        s = float(score_mat[fi, fj].item())
                        if abs(s) >= score_threshold:
                            scores[CLTEdge(CLTNode(layer_l, fi), CLTNode(layer_l1, fj))] = s

            return CLTGraphResult(
                scores=scores,
                node_scores=node_scores,
                n_layers=len(sorted_layers),
                n_features=n_features,
                layer_order=sorted_layers,
            )

        finally:
            for n, p in model.named_parameters():
                p.requires_grad_(orig_rg[n])
            if was_training:
                model.train()
            else:
                model.eval()
