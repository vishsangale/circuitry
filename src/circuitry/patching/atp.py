"""AtP* node attribution. Design spec docs/superpowers/specs/2026-05-24-atp-design.md."""
from __future__ import annotations

from dataclasses import dataclass

from circuitry.patching.graph import Node

_ATTN_SLOTS = ("q", "k", "v")


@dataclass(frozen=True)
class AtPNode:
    """An attributable activation: a graph Node plus an optional attention slot."""
    node: Node
    slot: str | None = None  # "q"/"k"/"v" for attn_head, else None


def enumerate_nodes(n_layers: int, n_heads: int, d_mlp: int | None = None) -> list[AtPNode]:
    nodes: list[AtPNode] = [AtPNode(Node("embed"), None)]
    for L in range(n_layers):
        for h in range(n_heads):
            for slot in _ATTN_SLOTS:
                nodes.append(AtPNode(Node("attn_head", L, h), slot))
        nodes.append(AtPNode(Node("mlp", L), None))
        if d_mlp is not None:
            for nidx in range(d_mlp):
                nodes.append(AtPNode(Node("mlp_neuron", L, neuron=nidx), None))
    return nodes


@dataclass
class AtPResult:
    scores: dict[AtPNode, float]

    def ranked(self) -> list[tuple[AtPNode, float]]:
        return sorted(self.scores.items(), key=lambda kv: abs(kv[1]), reverse=True)

    def top_k(self, n: int) -> list[tuple[AtPNode, float]]:
        return self.ranked()[:n]

    def threshold(self, tau: float) -> list[AtPNode]:
        return [n for n, s in self.scores.items() if abs(s) >= tau]
