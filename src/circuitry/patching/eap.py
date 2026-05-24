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
    """Run EAP on a model (embed + optional attention + MLP layers) and optionally
    verify against brute-force node patching.

    Model interface expected (HF-like):
      model.embed_tokens                         — nn.Embedding
      model.layers[L].self_attn.q_proj           — attention query projection
      model.layers[L].self_attn.k_proj           — attention key projection
      model.layers[L].self_attn.v_proj           — attention value projection
      model.layers[L].self_attn.o_proj           — attention output projection
      model.layers[L].mlp.up_proj                — first linear of MLP block
      model.layers[L].mlp.down_proj              — last linear of MLP block
      model.lm_head                              — nn.Linear

    If ``resolver`` is provided and has ``n_heads > 0``, attention head writers
    (z @ W_O contributions) and attention reader slots (q/k/v) are included.

    Note: ``model.layers[L].mlp`` is expected to be a bare ``nn.Module`` container
    that is NOT directly called in ``forward`` — the model calls ``up_proj`` and
    ``down_proj`` explicitly.  Hooks are therefore attached to ``up_proj``/
    ``down_proj`` rather than to the container.
    """

    def __init__(self, model: nn.Module, resolver: Any = None) -> None:
        self.model = model
        self.resolver = resolver
        n_layers = len(model.layers)  # type: ignore[arg-type]
        n_heads = getattr(resolver, "n_heads", 0) if resolver is not None else 0
        self.n_heads = n_heads
        self.head_dim = (resolver.d_model // resolver.n_heads) if resolver is not None and n_heads > 0 else None
        self.graph = build_graph(n_layers, n_heads)

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
          embed          — output of embed_tokens  (b, s, d)
          attn_head(L,h) — z[h] @ W_O[h], the head-h contribution to the residual (b, s, d)
          mlp(L)         — output of layers[L].mlp.down_proj  (b, s, d)
        """
        acts: dict[Node, Tensor] = {}
        handles: list[Any] = []

        embed_node = Node("embed")

        def _embed_hook(module: nn.Module, inputs: tuple, output: Tensor) -> None:
            acts[embed_node] = output.detach()

        handles.append(self.model.embed_tokens.register_forward_hook(_embed_hook))

        for L, block in enumerate(self.model.layers):
            # Attention head writer hooks (if attention is present)
            if self.n_heads > 0:
                n_heads = self.n_heads
                head_dim = self.head_dim

                def _o_proj_pre_hook(
                    module: nn.Module, args: tuple,
                    _L: int = L,
                    _n_heads: int = n_heads,
                    _head_dim: int = head_dim,
                ) -> None:
                    # args[0] is the input to o_proj: shape (b, s, n_heads*head_dim)
                    z = args[0].detach()
                    b, s, _ = z.shape
                    z_heads = z.reshape(b, s, _n_heads, _head_dim)
                    W_O = module.weight  # (d_model, n_heads*head_dim)
                    for h in range(_n_heads):
                        z_h = z_heads[:, :, h, :]  # (b, s, head_dim)
                        W_O_h = W_O[:, h * _head_dim:(h + 1) * _head_dim]  # (d_model, head_dim)
                        # contribution = z_h @ W_O_h.T = z_h @ W_O[h].T
                        # W_O is (d_model, head_dim), so z_h @ W_O_h.T => (b,s,d_model)
                        contrib = z_h @ W_O_h.T  # (b, s, d_model)
                        acts[Node("attn_head", _L, h)] = contrib

                handles.append(block.self_attn.o_proj.register_forward_pre_hook(_o_proj_pre_hook))

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

    @staticmethod
    def _backmap_qkv_grad(dL_dhead: Tensor, W_proj_head: Tensor, ln_scale: float = 1.0) -> Tensor:
        """Back-map a per-head projection gradient to residual space.

        dL_dhead: (b, s, head_dim) — gradient of metric w.r.t. head h's proj output
        W_proj_head: (head_dim, d_model) — head h's slice of q/k/v_proj.weight
        ln_scale: LayerNorm scale (1.0 here; Task 5 inserts real RMSNorm scale)
        returns: (b, s, d_model)
        """
        return (dL_dhead @ W_proj_head) * ln_scale

    def _collect_reader_grads(
        self, input_ids: Tensor, metric: Callable[[Tensor], Tensor]
    ) -> tuple[Tensor, dict[Node, Tensor], dict[tuple[Node, str], Tensor]]:
        """Forward + backward pass collecting writer acts AND reader-input grads.

        Reader input grads (component-only gradient — bypasses excluded):
          (mlp(L), "mlp_in")       — grad w.r.t. what up_proj reads (clone of residual)
          (logits, "logits_in")    — grad w.r.t. what lm_head reads (clone of residual)
          (attn_head(L,h), slot)   — back-mapped grad from q/k/v_proj output (d_model)

        A forward-pre-hook on each projection module returns a distinct clone of
        the incoming residual.  up_proj / lm_head compute on the clone; the
        residual bypass (x + mlp_out) continues with the ORIGINAL x, so the
        clone's gradient is COMPONENT-ONLY (no bypass double-counting).

        For attention readers: the q/k/v_proj OUTPUT has retain_grad() set.
        After backward, the grad of the proj output for head h is sliced, then
        back-mapped to residual space via W_proj[h].

        Returns (logits, writer_acts, reader_grads).
        reader_grads keys are (Node, slot_str).
        """
        writer_acts: dict[Node, Tensor] = {}
        reader_clones: dict[tuple[Node, str], Tensor] = {}
        # For attention readers, we store projection output tensors (with retain_grad)
        proj_outputs: dict[tuple[int, str], Tensor] = {}  # (L, slot) -> proj output
        reader_grads: dict[tuple[Node, str], Tensor] = {}
        handles: list[Any] = []

        embed_node = Node("embed")
        logits_node = Node("logits")

        # ---- writer hooks on embed + o_proj + down_proj ----
        def _embed_writer_hook(module: nn.Module, inputs: tuple, output: Tensor) -> None:
            writer_acts[embed_node] = output.detach()

        handles.append(self.model.embed_tokens.register_forward_hook(_embed_writer_hook))

        for L, block in enumerate(self.model.layers):
            # Attention head writer hooks
            if self.n_heads > 0:
                n_heads = self.n_heads
                head_dim = self.head_dim

                def _o_proj_pre_hook_writer(
                    module: nn.Module, args: tuple,
                    _L: int = L,
                    _n_heads: int = n_heads,
                    _head_dim: int = head_dim,
                ) -> None:
                    z = args[0].detach()
                    b, s, _ = z.shape
                    z_heads = z.reshape(b, s, _n_heads, _head_dim)
                    W_O = module.weight  # (d_model, n_heads*head_dim)
                    for h in range(_n_heads):
                        z_h = z_heads[:, :, h, :]  # (b, s, head_dim)
                        W_O_h = W_O[:, h * _head_dim:(h + 1) * _head_dim]  # (d_model, head_dim)
                        contrib = z_h @ W_O_h.T  # (b, s, d_model)
                        writer_acts[Node("attn_head", _L, h)] = contrib

                handles.append(block.self_attn.o_proj.register_forward_pre_hook(_o_proj_pre_hook_writer))

                # Reader hooks: clone the proj INPUT with requires_grad so the
                # proj OUTPUT is in the autograd graph (non-leaf but differentiable).
                # We then retain_grad on the output to capture dL/d(proj_out).
                for slot, proj in [("q", block.self_attn.q_proj),
                                    ("k", block.self_attn.k_proj),
                                    ("v", block.self_attn.v_proj)]:
                    def _proj_pre_hook_reader(
                        module: nn.Module, args: tuple,
                        _L: int = L, _slot: str = slot,
                    ) -> tuple:
                        x = args[0]
                        x_clone = x.clone().requires_grad_(True)
                        return (x_clone,) + args[1:]

                    def _proj_output_hook(
                        module: nn.Module, inputs: tuple, output: Tensor,
                        _L: int = L, _slot: str = slot,
                    ) -> None:
                        output.retain_grad()
                        proj_outputs[(_L, _slot)] = output

                    handles.append(proj.register_forward_pre_hook(_proj_pre_hook_reader))
                    handles.append(proj.register_forward_hook(_proj_output_hook))

            mlp_node = Node("mlp", L)

            def _down_proj_writer_hook(
                module: nn.Module, inputs: tuple, output: Tensor,
                _n: Node = mlp_node,
            ) -> None:
                writer_acts[_n] = output.detach()

            handles.append(block.mlp.down_proj.register_forward_hook(_down_proj_writer_hook))

        # ---- reader pre-hooks: clone the residual at each projection input ----
        for L, block in enumerate(self.model.layers):
            mlp_reader_node = Node("mlp", L)

            def _up_proj_reader_pre_hook(
                module: nn.Module, args: tuple,
                _n: Node = mlp_reader_node,
            ) -> tuple:
                x = args[0]
                x_clone = x.clone().requires_grad_(True)
                x_clone.retain_grad()
                reader_clones[(_n, "mlp_in")] = x_clone
                return (x_clone,) + args[1:]

            handles.append(block.mlp.up_proj.register_forward_pre_hook(_up_proj_reader_pre_hook))

        def _lm_head_reader_pre_hook(module: nn.Module, args: tuple) -> tuple:
            x = args[0]
            x_clone = x.clone().requires_grad_(True)
            x_clone.retain_grad()
            reader_clones[(logits_node, "logits_in")] = x_clone
            return (x_clone,) + args[1:]

        handles.append(self.model.lm_head.register_forward_pre_hook(_lm_head_reader_pre_hook))

        try:
            logits = self.model(input_ids)
            metric(logits).backward()
        finally:
            for h in handles:
                h.remove()

        # Collect component-only reader grads from the MLP/logits clones.
        for key, t in reader_clones.items():
            if t.grad is not None:
                reader_grads[key] = t.grad.detach()

        # Collect attention reader grads: back-map from proj output grads to residual space.
        # Always populate all attention reader slots — zero if the proj isn't in the
        # computation graph (e.g. q/k with fixed attention pattern).
        if self.n_heads > 0:
            n_heads = self.n_heads
            head_dim = self.head_dim
            for L, block in enumerate(self.model.layers):
                for slot, proj in [("q", block.self_attn.q_proj),
                                    ("k", block.self_attn.k_proj),
                                    ("v", block.self_attn.v_proj)]:
                    proj_out = proj_outputs.get((L, slot))
                    W_proj = proj.weight  # (n_heads*head_dim, d_model)
                    # Determine batch/seq shape from input_ids
                    b, s = input_ids.shape
                    d_model = W_proj.shape[1]
                    if proj_out is None or proj_out.grad is None:
                        # q/k unused in forward → zero gradient in residual space
                        for h in range(n_heads):
                            reader_grads[(Node("attn_head", L, h), slot)] = torch.zeros(
                                b, s, d_model, dtype=W_proj.dtype
                            )
                        continue
                    dL_dproj = proj_out.grad  # (b, s, n_heads*head_dim)
                    dL_dproj_heads = dL_dproj.reshape(b, s, n_heads, head_dim)
                    for h in range(n_heads):
                        dL_dproj_h = dL_dproj_heads[:, :, h, :]  # (b, s, head_dim)
                        W_proj_h = W_proj[h * head_dim:(h + 1) * head_dim, :]  # (head_dim, d_model)
                        grad_resid = self._backmap_qkv_grad(dL_dproj_h.detach(), W_proj_h.detach(), 1.0)
                        reader_grads[(Node("attn_head", L, h), slot)] = grad_resid

        return logits, writer_acts, reader_grads

    def _stack_acts(self, acts: dict[Node, Tensor]) -> Tensor:
        """Stack writer activation tensors into (batch, pos, |writers|, d_model)."""
        tensors = [acts[n] for n in self.graph.writers]
        return torch.stack(tensors, dim=2)  # (b, s, W, d)

    def _stack_grads(self, grads: dict[tuple[Node, str], Tensor]) -> Tensor:
        """Stack reader gradient tensors into (batch, pos, |readers|, d_model)."""
        tensors = [grads[(rnode, slot)] for rnode, slot in self.graph.readers]
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
        """Exact per-edge brute force.  For each edge (u -> v, slot):

        - mlp_in: add delta_act_u to up_proj's INPUT (residual clone point)
        - logits_in: add delta_act_u to lm_head's INPUT
        - q/k/v (attention reader): add (delta_act_u @ W_proj[h].T) to head h's
          slice of the q/k/v_proj OUTPUT (patches only that head's read, matching
          the component-only gradient)

        In all cases we measure metric(patched) - metric(clean).

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

            # delta per writer node: (b, s, d_model)
            delta: dict[Node, Tensor] = {
                n: acts_corrupted[n] - acts_clean[n] for n in self.graph.writers
            }

            scores: dict[Edge, float] = {}

            for edge in self.graph.edges:
                writer_node = edge.writer
                reader_node = edge.reader
                slot = edge.slot
                d_act = delta[writer_node]  # (b, s, d_model)
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

                elif slot in ("q", "k", "v"):
                    # Attention reader: patch head h's slice of the proj output.
                    # Add d_act @ W_proj[h].T to the proj output at head h's positions.
                    # This patches ONLY head h's read of this slot (other heads untouched).
                    # Algebra: dL/dproj_h · (Δact_u @ W_proj[h].T)
                    #          == Δact_u · (dL/dproj_h @ W_proj[h]) — both equal.
                    L = reader_node.layer
                    h = reader_node.head
                    block = self.model.layers[L]
                    n_heads = self.n_heads
                    head_dim = self.head_dim

                    if slot == "q":
                        proj_module = block.self_attn.q_proj
                    elif slot == "k":
                        proj_module = block.self_attn.k_proj
                    else:
                        proj_module = block.self_attn.v_proj

                    W_proj = proj_module.weight  # (n_heads*head_dim, d_model)
                    W_proj_h = W_proj[h * head_dim:(h + 1) * head_dim, :]  # (head_dim, d_model)
                    # delta in proj-output space for head h: (b, s, head_dim)
                    delta_proj_h = d_act @ W_proj_h.T  # (b, s, head_dim)

                    def _proj_output_add_hook(
                        module: nn.Module, inputs: tuple, output: Tensor,
                        _h: int = h,
                        _n_heads: int = n_heads,
                        _head_dim: int = head_dim,
                        _delta: Tensor = delta_proj_h,
                    ) -> Tensor:
                        # output shape: (b, s, n_heads*head_dim)
                        b, s, _ = output.shape
                        out = output.clone().reshape(b, s, _n_heads, _head_dim)
                        out[:, :, _h, :] = out[:, :, _h, :] + _delta
                        return out.reshape(b, s, _n_heads * _head_dim)

                    handles.append(proj_module.register_forward_hook(_proj_output_add_hook))

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

    def writer_activations(self, inputs: Tensor) -> dict[Node, Tensor]:
        """Return writer activations dict for the given inputs (debug/test accessor)."""
        was_training, orig_rg = self._freeze_eval()
        try:
            with torch.no_grad():
                _, acts = self._collect_writer_acts(inputs)
            return acts
        finally:
            self._restore(was_training, orig_rg)
