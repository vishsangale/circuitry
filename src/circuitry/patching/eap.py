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
        self, input_ids: Tensor, metric: Callable[[Tensor], float]
    ) -> tuple[Tensor, dict[Node, Tensor], dict[Node, Tensor]]:
        """Forward + backward pass collecting writer acts AND reader-input grads.

        Reader input grads (gradient of metric w.r.t. residual stream at reader input):
          mlp_in(L)  — grad w.r.t. residual stream entering up_proj at layer L
          logits_in  — grad w.r.t. residual stream entering lm_head

        Returns (logits, writer_acts, reader_grads).
        """
        writer_acts: dict[Node, Tensor] = {}
        reader_input_tensors: dict[Node, Tensor] = {}
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

        # ---- reader pre-hooks on up_proj (mlp_in) and lm_head (logits_in) ----
        # Strategy: detach the residual stream entering each mlp's up_proj so
        # that tensor becomes a leaf node; backward() then deposits the gradient
        # in .grad.  We must NOT detach the lm_head input — if we did, gradients
        # could not propagate upstream through it to reach the detached mlp leaves.
        # Instead we capture the lm_head input gradient via register_hook on the
        # non-leaf tensor.
        for L, block in enumerate(self.model.layers):
            mlp_reader_node = Node("mlp", L)

            def _up_proj_reader_pre_hook(
                module: nn.Module, args: tuple,
                _n: Node = mlp_reader_node,
            ) -> tuple:
                x = args[0]
                x_intercepted = x.detach().requires_grad_(True)
                reader_input_tensors[_n] = x_intercepted
                return (x_intercepted,) + args[1:]

            handles.append(block.mlp.up_proj.register_forward_pre_hook(_up_proj_reader_pre_hook))

        def _lm_head_reader_pre_hook(module: nn.Module, args: tuple) -> tuple:
            x = args[0]
            # Do NOT detach — leaving x connected lets gradients propagate back
            # through it to the detached mlp-reader leaf tensors above.
            # We capture the gradient here via a tensor hook.
            def _save_logits_grad(g: Tensor) -> None:
                reader_grads[logits_node] = g.detach()

            x.register_hook(_save_logits_grad)
            reader_input_tensors[logits_node] = x
            return args

        handles.append(self.model.lm_head.register_forward_pre_hook(_lm_head_reader_pre_hook))

        try:
            logits = self.model(input_ids)
            # Build a differentiable scalar from the non-detached logits path.
            # logit_diff() calls .detach() internally so we must build our own
            # scalar that keeps gradients flowing to the intercepted tensors.
            logits_f = logits.float()
            if logits_f.ndim == 3:
                logits_f = logits_f[:, -1, :]  # use last sequence position
            scalar = (logits_f[:, 0] - logits_f[:, 1]).mean()
            scalar.backward()
        finally:
            for h in handles:
                h.remove()

        # Collect mlp reader grads from leaf tensors (populated by autograd via .grad)
        for node, t in reader_input_tensors.items():
            if t.is_leaf and t.grad is not None:
                reader_grads[node] = t.grad.detach()
        # logits_node grad was already saved via register_hook above

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

    def bruteforce_node_scores(
        self,
        clean_inputs: Tensor,
        corrupted_inputs: Tensor,
        metric: Callable[[Tensor], float],
    ) -> dict[Node, float]:
        """Brute-force node patching: for each writer node u, patch u's residual
        contribution from clean to corrupted and measure the metric delta.

        Returns {node: metric(patched) - metric(clean)} for all writer nodes.
        """
        was_training, orig_rg = self._freeze_eval()
        try:
            with torch.no_grad():
                # Baseline: clean metric
                clean_logits = self.model(clean_inputs)
                clean_metric = metric(clean_logits)

                # Cache corrupted writer activations
                _, acts_corrupted = self._collect_writer_acts(corrupted_inputs)

            deltas: dict[Node, float] = {}

            for writer_node in self.graph.writers:
                corr_act = acts_corrupted[writer_node]  # (b, s, d)
                handles: list[Any] = []

                if writer_node.kind == "embed":
                    def _embed_patch_hook(
                        module: nn.Module, inputs: tuple, output: Tensor,
                        _val: Tensor = corr_act,
                    ) -> Tensor:
                        return _val

                    handles.append(
                        self.model.embed_tokens.register_forward_hook(_embed_patch_hook)
                    )
                elif writer_node.kind == "mlp":
                    L = writer_node.layer
                    block = self.model.layers[L]

                    def _down_proj_patch_hook(
                        module: nn.Module, inputs: tuple, output: Tensor,
                        _val: Tensor = corr_act,
                    ) -> Tensor:
                        return _val

                    # Patch down_proj output = the mlp residual contribution
                    handles.append(
                        block.mlp.down_proj.register_forward_hook(_down_proj_patch_hook)
                    )

                try:
                    with torch.no_grad():
                        patched_logits = self.model(clean_inputs)
                        patched_metric = metric(patched_logits)
                finally:
                    for h in handles:
                        h.remove()

                deltas[writer_node] = patched_metric - clean_metric

            return deltas
        finally:
            self._restore(was_training, orig_rg)
