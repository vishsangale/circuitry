"""EAP scoring engine + runner. Design spec §3, §5, §7."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from circuitry.patching.graph import Edge, EdgeGraph, Node, build_graph


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


class EAPRunner:
    """Run EAP on an MLP-only (embed + MLP layers) model and optionally verify
    against brute-force node patching.

    Model interface expected (HF-like, no attention):
      model.embed_tokens            — nn.Embedding
      model.layers[L].mlp.up_proj   — first linear of MLP block
      model.layers[L].mlp.down_proj — last linear of MLP block
      model.lm_head                 — nn.Linear

    Note: ``model.layers[L].mlp`` is expected to be a bare ``nn.Module`` container
    that is NOT directly called in ``forward`` — the model calls ``up_proj`` and
    ``down_proj`` explicitly.  Hooks are therefore attached to ``up_proj``/
    ``down_proj`` rather than to the container.
    """

    def __init__(self, model: nn.Module, resolver: Any = None) -> None:
        self.model = model
        self.resolver = resolver  # unused on MLP-only path; reserved for Task 4+
        n_layers = len(model.layers)  # type: ignore[arg-type]
        self.graph = build_graph(n_layers, n_heads=0)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _freeze_eval(self) -> tuple[bool, dict[str, bool]]:
        """Set eval mode and freeze params. Returns (was_training, orig_requires_grad)."""
        was_training = self.model.training
        orig_rg: dict[str, bool] = {}
        for name, p in self.model.named_parameters():
            orig_rg[name] = p.requires_grad
            p.requires_grad_(False)
        self.model.eval()
        return was_training, orig_rg

    def _restore(self, was_training: bool, orig_rg: dict[str, bool]) -> None:
        if was_training:
            self.model.train()
        else:
            self.model.eval()
        for name, p in self.model.named_parameters():
            if name in orig_rg:
                p.requires_grad_(orig_rg[name])

    def _collect_writer_acts(self, input_ids: Tensor) -> tuple[Tensor, dict[Node, Tensor]]:
        """Forward pass collecting writer activations. Returns (logits, acts_by_node).

        Writer activations (residual contributions):
          embed  — output of embed_tokens  (b, s, d)
          mlp(L) — output of layers[L].mlp.down_proj  (b, s, d)
        """
        acts: dict[Node, Tensor] = {}
        handles: list[Any] = []

        embed_node = Node("embed")

        def _embed_hook(module: nn.Module, inputs: tuple, output: Tensor) -> None:
            acts[embed_node] = output.detach()

        handles.append(self.model.embed_tokens.register_forward_hook(_embed_hook))

        for L, block in enumerate(self.model.layers):
            mlp_node = Node("mlp", L)

            def _down_proj_hook(
                module: nn.Module, inputs: tuple, output: Tensor,
                _n: Node = mlp_node,
            ) -> None:
                acts[_n] = output.detach()

            # down_proj output IS the MLP residual contribution
            handles.append(block.mlp.down_proj.register_forward_hook(_down_proj_hook))

        try:
            logits = self.model(input_ids)
        finally:
            for h in handles:
                h.remove()

        return logits, acts

    def _collect_reader_grads(
        self, input_ids: Tensor, metric: Callable[[Tensor], Tensor]
    ) -> tuple[Tensor, dict[Node, Tensor], dict[Node, Tensor]]:
        """Forward + backward pass collecting writer acts AND reader-input grads.

        Reader input grads (component-only gradient — bypasses excluded):
          mlp_in(L)  — grad w.r.t. what up_proj reads (clone of residual)
          logits_in  — grad w.r.t. what lm_head reads (clone of residual)

        A forward-pre-hook on each projection module returns a distinct clone of
        the incoming residual.  up_proj / lm_head compute on the clone; the
        residual bypass (x + mlp_out) continues with the ORIGINAL x, so the
        clone's gradient is COMPONENT-ONLY (no bypass double-counting).

        Returns (logits, writer_acts, reader_grads).
        """
        writer_acts: dict[Node, Tensor] = {}
        reader_clones: dict[Node, Tensor] = {}
        reader_grads: dict[Node, Tensor] = {}
        handles: list[Any] = []

        embed_node = Node("embed")
        logits_node = Node("logits")

        # ---- writer hooks on embed + down_proj ----
        def _embed_writer_hook(module: nn.Module, inputs: tuple, output: Tensor) -> None:
            writer_acts[embed_node] = output.detach()

        handles.append(self.model.embed_tokens.register_forward_hook(_embed_writer_hook))

        for L, block in enumerate(self.model.layers):
            mlp_node = Node("mlp", L)

            def _down_proj_writer_hook(
                module: nn.Module, inputs: tuple, output: Tensor,
                _n: Node = mlp_node,
            ) -> None:
                writer_acts[_n] = output.detach()

            handles.append(block.mlp.down_proj.register_forward_hook(_down_proj_writer_hook))

        # ---- reader pre-hooks: clone the residual at each projection input ----
        # _split_reader_input returns a leaf clone whose .grad (after backward) is
        # the COMPONENT-ONLY gradient for that reader.  The bypass uses the original
        # x, so there is no double-counting.  No down_proj post-hook needed.
        for L, block in enumerate(self.model.layers):
            mlp_reader_node = Node("mlp", L)

            def _up_proj_reader_pre_hook(
                module: nn.Module, args: tuple,
                _n: Node = mlp_reader_node,
            ) -> tuple:
                x = args[0]
                x_clone = x.clone().requires_grad_(True)
                x_clone.retain_grad()
                reader_clones[_n] = x_clone
                return (x_clone,) + args[1:]

            handles.append(block.mlp.up_proj.register_forward_pre_hook(_up_proj_reader_pre_hook))

        def _lm_head_reader_pre_hook(module: nn.Module, args: tuple) -> tuple:
            x = args[0]
            x_clone = x.clone().requires_grad_(True)
            x_clone.retain_grad()
            reader_clones[logits_node] = x_clone
            return (x_clone,) + args[1:]

        handles.append(self.model.lm_head.register_forward_pre_hook(_lm_head_reader_pre_hook))

        try:
            logits = self.model(input_ids)
            metric(logits).backward()
        finally:
            for h in handles:
                h.remove()

        # Collect component-only reader grads from the clones.
        for node, t in reader_clones.items():
            if t.grad is not None:
                reader_grads[node] = t.grad.detach()

        return logits, writer_acts, reader_grads

    def _stack_acts(self, acts: dict[Node, Tensor]) -> Tensor:
        """Stack writer activation tensors into (batch, pos, |writers|, d_model)."""
        tensors = [acts[n] for n in self.graph.writers]
        return torch.stack(tensors, dim=2)  # (b, s, W, d)

    def _stack_grads(self, grads: dict[Node, Tensor]) -> Tensor:
        """Stack reader gradient tensors into (batch, pos, |readers|, d_model)."""
        tensors = [grads[rnode] for rnode, _slot in self.graph.readers]
        return torch.stack(tensors, dim=2)  # (b, s, R, d)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        clean_inputs: Tensor,
        corrupted_inputs: Tensor,
        metric: Callable[[Tensor], float],
        ig_steps: int = 1,
    ) -> EAPResult:
        """Compute EAP edge scores.

        Steps:
          1. Corrupted forward → cache corrupted writer activations.
          2. Clean forward + backward → cache clean writer activations + reader grads.
          3. score_edges().
        """
        was_training, orig_rg = self._freeze_eval()
        try:
            # Step 1: corrupted forward (no grad needed)
            with torch.no_grad():
                _, acts_corrupted = self._collect_writer_acts(corrupted_inputs)

            # Step 2: clean forward + backward (need grads for reader inputs)
            _, acts_clean, grads_clean = self._collect_reader_grads(clean_inputs, metric)

            # Step 3: build stacked tensors and score
            act_clean_t = self._stack_acts(acts_clean)
            act_corrupted_t = self._stack_acts(acts_corrupted)
            grad_clean_t = self._stack_grads(grads_clean)

            return score_edges(self.graph, act_clean_t, act_corrupted_t, grad_clean_t)
        finally:
            self._restore(was_training, orig_rg)

    def bruteforce_edge_scores(
        self,
        clean_inputs: Tensor,
        corrupted_inputs: Tensor,
        metric: Callable[[Tensor], Tensor],
    ) -> dict[Edge, float]:
        """Exact per-edge brute force.  For each edge (u -> v, slot): add
        delta_act_u = act_corrupted_u - act_clean_u to v's PROJECTION INPUT
        (up_proj for mlp_in, lm_head for logits_in — the same clone point
        used in ``_collect_reader_grads``), measure metric(patched) - metric(clean).

        Bypass is NOT touched (no down_proj post-hook), matching the
        component-only gradient captured in ``run()``.

        On a fully linear model this must equal the analytic EAP score to
        floating-point precision because EAP's first-order approximation is
        exact when there are no nonlinearities.
        """
        was_training, orig_rg = self._freeze_eval()
        try:
            with torch.no_grad():
                # Baseline clean metric (no hooks)
                clean_logits = self.model(clean_inputs)
                clean_metric = metric(clean_logits).item()

                # Cache corrupted writer activations (same as run() step 1)
                _, acts_corrupted = self._collect_writer_acts(corrupted_inputs)

                # Cache clean writer activations so we can compute delta
                _, acts_clean = self._collect_writer_acts(clean_inputs)

            # delta per writer node: (b, s, d)
            delta: dict[Node, Tensor] = {
                n: acts_corrupted[n] - acts_clean[n] for n in self.graph.writers
            }

            scores: dict[Edge, float] = {}

            for edge in self.graph.edges:
                writer_node = edge.writer
                reader_node = edge.reader
                slot = edge.slot
                d_act = delta[writer_node]  # (b, s, d)
                handles: list[Any] = []

                if slot == "mlp_in":
                    # Patch ONLY the projection input (up_proj pre-hook).
                    # The residual bypass (x + mlp_out) is untouched — matching
                    # the component-only gradient captured in _collect_reader_grads.
                    L = reader_node.layer
                    block = self.model.layers[L]

                    def _up_proj_add_hook(
                        module: nn.Module, args: tuple,
                        _d: Tensor = d_act,
                    ) -> tuple:
                        return (args[0] + _d,) + args[1:]

                    handles.append(
                        block.mlp.up_proj.register_forward_pre_hook(_up_proj_add_hook)
                    )

                elif slot == "logits_in":
                    # Patch lm_head's input (forward pre-hook) — same point as the
                    # clone in _collect_reader_grads.
                    def _lm_head_add_hook(
                        module: nn.Module, args: tuple,
                        _d: Tensor = d_act,
                    ) -> tuple:
                        return (args[0] + _d,) + args[1:]

                    handles.append(
                        self.model.lm_head.register_forward_pre_hook(_lm_head_add_hook)
                    )

                try:
                    with torch.no_grad():
                        patched_logits = self.model(clean_inputs)
                        patched_metric = metric(patched_logits).item()
                finally:
                    for h in handles:
                        h.remove()

                scores[edge] = patched_metric - clean_metric

            return scores
        finally:
            self._restore(was_training, orig_rg)
