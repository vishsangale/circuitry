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
        # Goal: capture grad_v = d(metric)/d(residual_stream_at_v) for each reader v,
        # treating the residual at v as a fully independent variable.  This gradient
        # captures ALL downstream effects: through v's MLP AND through the bypass path
        # where the residual flows directly to subsequent layers.
        #
        # Implementation:
        #   For non-leaf residuals (layers > 0, already requires_grad=True):
        #     Call retain_grad() on x and return args unchanged.  Autograd fills x.grad.
        #
        #   For the frozen-param embed output (layer 0, requires_grad=False):
        #     We must make the tensor a leaf *and* ensure the bypass path also flows
        #     through the leaf.  Strategy:
        #       1. up_proj pre-hook: create x_leaf = x.detach().clone().requires_grad_(True)
        #          and return (x_leaf,) so up_proj uses it.
        #       2. down_proj post-hook: add (x_leaf - x_original) to down_proj's output.
        #          Since x_leaf and x_original have the same numeric values, this adds
        #          nothing numerically, but it routes the residual-bypass gradient path
        #          through x_leaf in the computation graph.  After this, the expression
        #          x_original + down_proj_out + (x_leaf - x_original) = x_leaf + mlp_out,
        #          so backward accumulates d(metric)/d(x_leaf) from both the MLP path
        #          and the bypass path — which is the correct total gradient.
        for L, block in enumerate(self.model.layers):
            mlp_reader_node = Node("mlp", L)
            _hook_state: dict[str, Tensor] = {}  # shared between pre- and post-hook

            def _up_proj_reader_pre_hook(
                module: nn.Module, args: tuple,
                _n: Node = mlp_reader_node,
                _state: dict[str, Tensor] = _hook_state,
            ) -> tuple:
                x = args[0]
                if x.requires_grad:
                    # Non-leaf (layer ≥ 1): retain_grad so autograd deposits .grad.
                    x.retain_grad()
                    reader_input_tensors[_n] = x
                    return args  # pass through unchanged
                else:
                    # Leaf without grad (embed output at layer 0): create a leaf that
                    # captures the full gradient including the bypass.
                    x_leaf = x.detach().clone().requires_grad_(True)
                    _state["x_leaf"] = x_leaf
                    _state["x_orig"] = x
                    reader_input_tensors[_n] = x_leaf
                    return (x_leaf,) + args[1:]

            def _down_proj_bypass_grad_hook(
                module: nn.Module, inputs: tuple, output: Tensor,
                _state: dict[str, Tensor] = _hook_state,
            ) -> Tensor | None:
                if "x_leaf" in _state:
                    # Add the bypass term so backward sees the residual path too.
                    bypass = _state["x_leaf"] - _state["x_orig"]
                    _state.clear()
                    return output + bypass
                return None  # no-op if the pre-hook didn't populate state

            handles.append(block.mlp.up_proj.register_forward_pre_hook(_up_proj_reader_pre_hook))
            handles.append(block.mlp.down_proj.register_forward_hook(_down_proj_bypass_grad_hook))

        def _lm_head_reader_pre_hook(module: nn.Module, args: tuple) -> tuple:
            x = args[0]
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

        # Collect mlp reader grads (.grad is populated for leaf tensors and tensors
        # for which retain_grad() was called; guard the access to suppress PyTorch's
        # "non-leaf .grad won't be populated" warning).
        for node, t in reader_input_tensors.items():
            if (t.is_leaf or t.retains_grad) and t.grad is not None:
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

    def bruteforce_edge_scores(
        self,
        clean_inputs: Tensor,
        corrupted_inputs: Tensor,
        metric: Callable[[Tensor], float],
    ) -> dict[Edge, float]:
        """Exact per-edge brute force.  For each edge (u -> v, slot): add
        delta_act_u = act_corrupted_u - act_clean_u to v's residual input
        (the same hook point used for reader-gradient capture in ``run``),
        measure metric(patched) - metric(clean).

        On a fully linear model this must equal the analytic EAP score to
        floating-point precision because EAP's first-order approximation is
        exact when there are no nonlinearities.
        """
        was_training, orig_rg = self._freeze_eval()
        try:
            with torch.no_grad():
                # Baseline clean metric (no hooks)
                clean_logits = self.model(clean_inputs)
                clean_metric = metric(clean_logits)

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
                    # Reader is mlp(L).  Physically, the edge (u → mlp_L) represents
                    # the contribution of writer u to the FULL residual stream entering
                    # layer L.  In the residual-stream forward:
                    #
                    #   x_after_L = x_before_L + down_proj_L(up_proj_L(x_before_L))
                    #
                    # Adding delta to x_before_L affects BOTH the MLP computation
                    # (up_proj receives x_before + delta) AND the residual bypass
                    # (x_after_L gains an extra delta).  To simulate this with hooks
                    # we must hook two points:
                    #   1. up_proj pre-hook: add delta to the up_proj argument.
                    #   2. down_proj post-hook: add delta to down_proj's output.
                    # Together these produce x_after = (x + delta) + mlp_out(x + delta),
                    # matching the true residual-stream patch.  The gradient at
                    # x_before_L = d(metric)/d(residual at that point), which also
                    # sees both paths, so analytic == brute-force on a linear model.
                    L = reader_node.layer
                    block = self.model.layers[L]

                    def _up_proj_add_hook(
                        module: nn.Module, args: tuple,
                        _d: Tensor = d_act,
                    ) -> tuple:
                        x = args[0] + _d
                        return (x,) + args[1:]

                    def _down_proj_bypass_hook(
                        module: nn.Module, inputs: tuple, output: Tensor,
                        _d: Tensor = d_act,
                    ) -> Tensor:
                        return output + _d

                    handles.append(
                        block.mlp.up_proj.register_forward_pre_hook(_up_proj_add_hook)
                    )
                    handles.append(
                        block.mlp.down_proj.register_forward_hook(_down_proj_bypass_hook)
                    )

                elif slot == "logits_in":
                    # Reader is logits; hook point is lm_head forward pre-hook —
                    # identical to _collect_reader_grads.
                    def _lm_head_add_hook(
                        module: nn.Module, args: tuple,
                        _d: Tensor = d_act,
                    ) -> tuple:
                        x = args[0] + _d
                        return (x,) + args[1:]

                    handles.append(
                        self.model.lm_head.register_forward_pre_hook(_lm_head_add_hook)
                    )

                try:
                    with torch.no_grad():
                        patched_logits = self.model(clean_inputs)
                        patched_metric = metric(patched_logits)
                finally:
                    for h in handles:
                        h.remove()

                scores[edge] = patched_metric - clean_metric

            return scores
        finally:
            self._restore(was_training, orig_rg)
