"""Edge Pruning — gradient-based joint circuit discovery.

Bhaskar & Wettig, "Finding Transformer Circuits with Edge Pruning",
NeurIPS 2024. https://arxiv.org/abs/2406.16778
"""
from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from circuitry.patching.eap import EAPResult, EAPRunner
from circuitry.patching.graph import (
    Edge,
    EdgeGraph,
    _node_from_dict,
    _node_str,
    _node_to_dict,
    build_graph,
    edge_sort_key,
)

_Inputs = Tensor | dict[str, Any]


@dataclass
class EdgePruningResult:
    circuit: list[Edge]             # edges with active masks (m_e = 1)
    removed_edges: list[Edge]       # edges with inactive masks (m_e = 0)
    mask_logits: dict[Edge, float]  # final z_e values at convergence
    eap_scores: dict[Edge, float]   # initial EAP scores (for reference)
    lambda_l0: float                # regularization coefficient used
    n_steps_run: int
    graph: EdgeGraph                # full edge graph

    def n_circuit(self) -> int:
        return len(self.circuit)

    def circuit_graph(self) -> EdgeGraph:
        """Return a subgraph containing only the circuit edges."""
        kept = set(self.circuit)
        sub = [e for e in self.graph.edges if e in kept]
        return EdgeGraph(
            self.graph.n_layers, self.graph.n_heads,
            self.graph.writers, self.graph.readers, sub,
        )

    def ranked(self) -> list[tuple[Edge, float]]:
        """All edges sorted by |mask_logit| desc."""
        return sorted(self.mask_logits.items(), key=lambda kv: abs(kv[1]), reverse=True)

    def to_markdown(self, *, top_k: int | None = None) -> str:
        """Render a markdown summary with the circuit edge table."""
        n_total = len(self.circuit) + len(self.removed_edges)
        lines = ["## Edge Pruning Circuit", ""]
        lines.append(f"- Graph: {self.graph.n_layers} layers, {self.graph.n_heads} heads")
        lines.append(f"- Circuit edges: {self.n_circuit()} / {n_total}")
        lines.append(f"- lambda_l0: {self.lambda_l0}")
        lines.append(f"- Steps run: {self.n_steps_run}")
        lines.append("")
        edges = sorted(self.circuit, key=edge_sort_key)
        show = edges[:top_k] if top_k is not None else edges
        lines.append(f"### Circuit Edges ({self.n_circuit()} kept)")
        lines.append("")
        lines.append("| writer | slot | reader | logit |")
        lines.append("| --- | --- | --- | ---: |")
        for edge in show:
            logit = self.mask_logits.get(edge, float("nan"))
            lines.append(
                f"| `{_node_str(edge.writer)}` | {edge.slot}"
                f" | `{_node_str(edge.reader)}` | {logit:.4g} |"
            )
        if top_k is not None and len(edges) > top_k:
            lines.append(f"| … | … | … | ({len(edges) - top_k} more) |")
        return "\n".join(lines)

    def to_json(self) -> str:
        data = {
            "kind": "edge_pruning",
            "n_layers": self.graph.n_layers,
            "n_heads": self.graph.n_heads,
            "lambda_l0": self.lambda_l0,
            "n_steps_run": self.n_steps_run,
            "circuit": [
                {"writer": _node_to_dict(e.writer), "reader": _node_to_dict(e.reader), "slot": e.slot}
                for e in sorted(self.circuit, key=edge_sort_key)
            ],
            "mask_logits": [
                {"writer": _node_to_dict(e.writer), "reader": _node_to_dict(e.reader),
                 "slot": e.slot, "logit": v}
                for e, v in sorted(self.mask_logits.items(), key=lambda kv: -abs(kv[1]))
            ],
        }
        return json.dumps(data, indent=2)

    @classmethod
    def from_json(cls, text: str) -> "EdgePruningResult":
        """Deserialize from to_json() output."""
        data = json.loads(text)
        graph = build_graph(data["n_layers"], data["n_heads"])
        edge_lookup: dict[tuple, Edge] = {
            (e.writer, e.reader, e.slot): e for e in graph.edges
        }

        def _parse_edge(d: dict) -> Edge:
            writer = _node_from_dict(d["writer"])
            reader = _node_from_dict(d["reader"])
            slot = d["slot"]
            return edge_lookup.get((writer, reader, slot)) or Edge(writer, reader, slot)

        circuit = [_parse_edge(d) for d in data.get("circuit", [])]
        circuit_set = set(circuit)
        removed = [e for e in graph.edges if e not in circuit_set]

        mask_logits: dict[Edge, float] = {}
        for row in data.get("mask_logits", []):
            e = _parse_edge(row)
            mask_logits[e] = row["logit"]
        # Fill in any missing edges with default inactive logit
        for e in graph.edges:
            if e not in mask_logits:
                mask_logits[e] = -10.0

        return cls(
            circuit=circuit,
            removed_edges=removed,
            mask_logits=mask_logits,
            eap_scores={},  # not stored in JSON
            lambda_l0=data.get("lambda_l0", 0.0),
            n_steps_run=data.get("n_steps_run", 0),
            graph=graph,
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json())

    @classmethod
    def load(cls, path: str | Path) -> "EdgePruningResult":
        return cls.from_json(Path(path).read_text())


class EdgePruningRunner:
    """Joint edge-mask optimization via gradient descent.

    Uses EAPRunner to compute initial per-edge scores, then optimizes
    soft binary mask parameters z_e with a temperature-annealed sigmoid
    and L0 regularization.
    """

    def __init__(self, model: nn.Module, resolver: Any = None) -> None:
        self._eap = EAPRunner(model, resolver)

    def run(
        self,
        clean_inputs: _Inputs,
        corrupted_inputs: _Inputs,
        metric: Callable[[Any], Tensor],
        *,
        lambda_l0: float = 0.01,
        n_steps: int = 200,
        lr: float = 0.05,
        temperature_init: float = 2.0,
        temperature_final: float = 0.1,
        ig_steps: int = 1,
        candidate_edges: list[Edge] | None = None,
    ) -> EdgePruningResult:
        """Discover a sparse circuit via joint mask optimization.

        Args:
            clean_inputs:      clean prompt pair input.
            corrupted_inputs:  corrupted prompt pair input.
            metric:            callable (logits) -> scalar loss.
            lambda_l0:         L0 regularization strength (higher = sparser).
            n_steps:           gradient descent steps.
            lr:                Adam learning rate.
            temperature_init:  initial sigmoid temperature (softer = smoother).
            temperature_final: final temperature (lower = more binary).
            ig_steps:          EAP IG steps for initial score computation.
            candidate_edges:   if set, only optimize masks for these edges
                               (others are excluded from the circuit).
        """
        # Step 1: Compute EAP scores as gradient signal
        eap_result = self._eap.run(clean_inputs, corrupted_inputs, metric, ig_steps=ig_steps)

        # Step 2: Select candidate edges
        all_edges = list(eap_result.scores.keys())
        if candidate_edges is not None:
            cand_set = set(candidate_edges)
            edges = [e for e in all_edges if e in cand_set]
        else:
            edges = all_edges

        if not edges:
            # Degenerate case: no edges to prune
            return EdgePruningResult(
                circuit=[], removed_edges=all_edges,
                mask_logits={e: -10.0 for e in all_edges},
                eap_scores=dict(eap_result.scores),
                lambda_l0=lambda_l0, n_steps_run=0,
                graph=eap_result.graph,
            )

        # Step 3: Initialize mask parameters
        scores_t = torch.tensor([abs(eap_result.scores[e]) for e in edges], dtype=torch.float32)
        z = torch.zeros(len(edges), dtype=torch.float32, requires_grad=True)
        optimizer = torch.optim.Adam([z], lr=lr)

        # Step 4: Temperature-annealed optimization loop
        log_ratio = math.log(temperature_final / temperature_init) if temperature_init != temperature_final else 0.0

        for step in range(n_steps):
            T = temperature_init * math.exp(log_ratio * step / max(n_steps - 1, 1))
            soft_m = torch.sigmoid(z / T)

            # Linearized task loss: active edges (m_e ~= 1) should have high |score|
            task_loss = -(soft_m * scores_t).sum()
            l0_loss = lambda_l0 * soft_m.sum()
            loss = task_loss + l0_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Step 5: Extract hard circuit
        with torch.no_grad():
            final_soft = torch.sigmoid(z)
            active = final_soft > 0.5
            z_vals = z.tolist()

        circuit = [e for e, a in zip(edges, active.tolist()) if a]
        circuit_set = set(circuit)
        removed = [e for e in all_edges if e not in circuit_set]
        mask_logits = {e: z_val for e, z_val in zip(edges, z_vals)}
        # Excluded candidates get logit = -10
        for e in all_edges:
            if e not in mask_logits:
                mask_logits[e] = -10.0

        return EdgePruningResult(
            circuit=circuit,
            removed_edges=removed,
            mask_logits=mask_logits,
            eap_scores=dict(eap_result.scores),
            lambda_l0=lambda_l0,
            n_steps_run=n_steps,
            graph=eap_result.graph,
        )
