"""RelP — Relevance Propagation attribution for transformer circuits.

FarnoushRJ et al., "RelP: Better Attribution for Transformers through
Relevance Propagation", arXiv:2508.21258.

ReLPRunner replaces EAP's gradient term with an LRP-epsilon coefficient that
weights each writer's contribution by its fraction of the clean residual stream.
Same cost as EAP (2 forward passes + 1 backward); Pearson correlation to
ground-truth activation patching = 0.956 vs 0.006 for EAP on GPT-2 IOI.
Returns EAPResult for drop-in compatibility with EAPRunner.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor

from circuitry.patching.eap import EAPResult, EAPRunner
from circuitry.patching.graph import Edge, EdgeGraph

_Inputs = Tensor | dict[str, Any]


def _score_edges_relp(
    graph: EdgeGraph,
    act_clean: Tensor,        # (batch, pos, |writers|, d_model)
    act_corrupted: Tensor,    # (batch, pos, |writers|, d_model)
    grad_clean: Tensor,       # (batch, pos, |readers|, d_model)
    *,
    eps: float = 1e-6,
) -> EAPResult:
    """RelP edge scoring using LRP-epsilon residual-stream coefficients.

    score[w, r] = Σ_{b,p,d} delta_w[d] * lrp_coeff_w[d] * grad_r[d]

    where:
      delta_w      = act_corrupted[w] - act_clean[w]  (same as EAP)
      lrp_coeff_w  = act_clean[w] / (|sum_w' act_clean[w']| + eps)
                   = each writer's normalised fraction of the clean residual
      grad_r        = gradient of metric w.r.t. reader r's input (same as EAP)

    The LRP coefficient weights the activation delta by how strongly that
    writer contributed to the clean residual, making the score sensitive to
    writers that actually mattered in the clean computation.
    """
    delta = act_corrupted - act_clean                           # (b, p, W, d)
    x_clean = act_clean.sum(dim=2, keepdim=True)               # (b, p, 1, d)
    lrp_coeff = act_clean / (x_clean.abs() + eps)              # (b, p, W, d)
    weighted_delta = delta * lrp_coeff                         # (b, p, W, d)

    score_mat = torch.einsum("bpwd,bprd->wr", weighted_delta, grad_clean)

    scores: dict[Edge, float] = {}
    for e in graph.edges:
        w = graph.writer_index(e.writer)
        r = graph.reader_index(e.reader, e.slot)
        scores[e] = float(score_mat[w, r].item())

    return EAPResult(graph, scores)


class ReLPRunner(EAPRunner):
    """RelP attribution runner (drop-in replacement for EAPRunner).

    Inherits all of EAPRunner's graph-building and data-collection machinery;
    overrides only the scoring to use the LRP-epsilon residual-stream coefficient.

    For the TransformerLens backend or ``ig_steps > 1``, falls back to vanilla
    EAP scoring (LRP-TL and LRP-IG are not yet implemented).

    Args:
        model:    The PyTorch model to attribute.
        resolver: Site resolver (``HFSiteResolver`` or ``TLSiteResolver``).
        eps:      LRP epsilon denominator stabiliser (default 1e-6).

    Example::

        runner = ReLPRunner(model, resolver)
        result = runner.run(clean_inputs, corrupted_inputs, metric)
        # result is an EAPResult — same API as EAPRunner
        top10 = result.top_k(10)

    Reference: arXiv:2508.21258
    """

    def __init__(self, model: Any, resolver: Any = None, *, eps: float = 1e-6) -> None:
        super().__init__(model, resolver)
        self.eps = eps

    def run(
        self,
        clean_inputs: _Inputs,
        corrupted_inputs: _Inputs,
        metric: Callable[[Any], Tensor],
        ig_steps: int = 1,
    ) -> EAPResult:
        """Run RelP attribution and return an EAPResult.

        Falls back to vanilla EAP for the TransformerLens backend or when
        ``ig_steps > 1`` (LRP-IG variant not yet implemented).
        """
        if self._tl or ig_steps > 1:
            return super().run(clean_inputs, corrupted_inputs, metric, ig_steps)

        was_training, orig_rg = self._freeze_eval()
        try:
            with torch.no_grad():
                _, acts_corrupted = self._collect_writer_acts(corrupted_inputs)

            _, acts_clean, grads_clean = self._collect_reader_grads(clean_inputs, metric)

            act_clean_t = self._stack_acts(acts_clean)
            act_corrupted_t = self._stack_acts(acts_corrupted)
            grad_clean_t = self._stack_grads(grads_clean)

            return _score_edges_relp(
                self.graph, act_clean_t, act_corrupted_t, grad_clean_t, eps=self.eps
            )
        finally:
            self._restore(was_training, orig_rg)
