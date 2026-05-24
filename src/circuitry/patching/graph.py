"""EAP edge graph: nodes, edges, causal enumeration. Design spec §2."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import torch
from torch import Tensor

Slot = Literal["q", "k", "v", "mlp_in", "logits_in"]
_ATTN_SLOTS: tuple[Slot, ...] = ("q", "k", "v")


@dataclass(frozen=True)
class Node:
    kind: str                 # "embed" | "attn_head" | "mlp" | "logits"
    layer: int | None = None
    head: int | None = None


@dataclass(frozen=True)
class Edge:
    writer: Node
    reader: Node
    slot: Slot


def _order(node: Node, n_layers: int) -> int:
    """Forward-order rank. Heads in one attn block share a rank (parallel)."""
    if node.kind == "embed":
        return 0
    if node.kind == "attn_head":
        return 1 + 2 * node.layer
    if node.kind == "mlp":
        return 2 + 2 * node.layer
    if node.kind == "logits":
        return 1 + 2 * n_layers
    raise ValueError(f"unknown node kind {node.kind!r}")


@dataclass
class EdgeGraph:
    n_layers: int
    n_heads: int
    writers: list[Node]
    readers: list[tuple[Node, Slot]]
    edges: list[Edge]
    _w_idx: dict[Node, int] = field(default_factory=dict)
    _r_idx: dict[tuple[Node, Slot], int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._w_idx = {n: i for i, n in enumerate(self.writers)}
        self._r_idx = {rs: i for i, rs in enumerate(self.readers)}

    def writer_index(self, node: Node) -> int:
        return self._w_idx[node]

    def reader_index(self, node: Node, slot: Slot) -> int:
        return self._r_idx[(node, slot)]

    def valid_mask(self) -> Tensor:
        mask = torch.zeros(len(self.writers), len(self.readers), dtype=torch.bool)
        for e in self.edges:
            mask[self.writer_index(e.writer), self.reader_index(e.reader, e.slot)] = True
        return mask


def build_graph(n_layers: int, n_heads: int) -> EdgeGraph:
    writers: list[Node] = [Node("embed")]
    readers: list[tuple[Node, Slot]] = []
    for L in range(n_layers):
        for h in range(n_heads):
            head = Node("attn_head", L, h)
            writers.append(head)
            for slot in _ATTN_SLOTS:
                readers.append((head, slot))
        mlp = Node("mlp", L)
        writers.append(mlp)
        readers.append((mlp, "mlp_in"))
    logits = Node("logits")
    readers.append((logits, "logits_in"))

    edges: list[Edge] = []
    for w in writers:
        wo = _order(w, n_layers)
        for rnode, slot in readers:
            if wo < _order(rnode, n_layers):
                edges.append(Edge(w, rnode, slot))
    return EdgeGraph(n_layers, n_heads, writers, readers, edges)
