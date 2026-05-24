"""EAP scoring engine + runner. Design spec §3, §5, §7."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from circuitry.patching.graph import Edge, EdgeGraph


@dataclass
class EAPResult:
    graph: EdgeGraph
    scores: dict[Edge, float]

    def ranked(self) -> list[tuple[Edge, float]]:
        return sorted(self.scores.items(), key=lambda kv: abs(kv[1]), reverse=True)

    def top_k(self, n: int) -> list[tuple[Edge, float]]:
        return self.ranked()[:n]

    def threshold(self, tau: float) -> list[Edge]:
        return [e for e, s in self.scores.items() if abs(s) >= tau]


def score_edges(
    graph: EdgeGraph,
    act_clean: Tensor,        # (batch, pos, |writers|, d_model)
    act_corrupted: Tensor,    # (batch, pos, |writers|, d_model)
    grad_clean: Tensor,       # (batch, pos, |readers|, d_model)
) -> EAPResult:
    """Analytic EAP scoring: score[w,r] = sum_{b,p,d} (act_corr-act_clean)[w] * grad[r]."""
    delta = act_corrupted - act_clean
    # (|writers|, |readers|): sum over batch, pos, d_model
    score_mat = torch.einsum("bpwd,bprd->wr", delta, grad_clean)
    scores: dict[Edge, float] = {}
    for e in graph.edges:
        w = graph.writer_index(e.writer)
        r = graph.reader_index(e.reader, e.slot)
        scores[e] = float(score_mat[w, r].item())
    return EAPResult(graph, scores)
