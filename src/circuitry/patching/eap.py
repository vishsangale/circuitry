"""EAP scoring engine + runner. Design spec §3, §5, §7."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from circuitry.patching.graph import Edge, EdgeGraph, Node, build_graph

# Type alias for flexible inputs: either a raw tensor (toy models) or a dict
# of keyword arguments (HF models, e.g. {"input_ids": tensor}).
_Inputs = Tensor | dict[str, Any]


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

    Model interface expected — supports both flat toys and nested HF models:
      Flat toys: model.layers, model.embed_tokens
      HF Llama:  model.model.layers, model.model.embed_tokens, model.lm_head

    Per-layer (layer module ``block``):
      block.self_attn.{q,k,v,o}_proj  — attention projections
      block.mlp.{up,down}_proj         — MLP projections
      block.input_layernorm            — pre-attn RMSNorm (optional; absent → ln_scale=1)
      block.post_attention_layernorm   — pre-mlp RMSNorm  (optional; absent → ln_scale=1)

    If ``resolver`` is provided and has ``n_heads > 0``, attention head writers
    (z @ W_O contributions) and attention reader slots (q/k/v) are included.

    GQA support: ``resolver`` may expose ``n_kv_heads`` (via model.config) for
    models where k/v projections produce fewer heads than q.

    Inputs to ``run`` / ``bruteforce_edge_scores`` may be a raw Tensor (toy models)
    or a dict (HF models, e.g. ``{"input_ids": tensor}``).
    """

    def __init__(self, model: nn.Module, resolver: Any = None) -> None:
        self.model = model
        self.resolver = resolver
        # Locate layers list (nested HF: model.model.layers, flat toys: model.layers)
        self._layers_list = self._locate_layers(model)
        n_layers = len(self._layers_list)
        n_heads = getattr(resolver, "n_heads", 0) if resolver is not None else 0
        self.n_heads = n_heads
        # GQA: number of key/value heads (defaults to n_heads for MHA)
        self.n_kv_heads: int = n_heads
        if resolver is not None and n_heads > 0:
            cfg = getattr(model, "config", None)
            if cfg is not None:
                self.n_kv_heads = getattr(cfg, "num_key_value_heads", n_heads)
        self.head_dim = (resolver.d_model // resolver.n_heads) if resolver is not None and n_heads > 0 else None
        # RMSNorm eps from config (used for ln_scale; only relevant for HF models)
        cfg = getattr(model, "config", None)
        self._rms_norm_eps: float = getattr(cfg, "rms_norm_eps", 1e-6) if cfg is not None else 1e-6
        self.graph = build_graph(n_layers, n_heads)

    # ------------------------------------------------------------------
    # Module locator helpers (nested HF vs flat toys)
    # ------------------------------------------------------------------

    @staticmethod
    def _locate_layers(model: nn.Module) -> nn.ModuleList:
        """Return the transformer layers list: tries model.model.layers then model.layers."""
        inner = getattr(model, "model", None)
        if inner is not None and hasattr(inner, "layers"):
            return inner.layers  # type: ignore[return-value]
        return model.layers  # type: ignore[return-value]

    def _embed(self) -> nn.Module:
        """Return the embedding module."""
        inner = getattr(self.model, "model", None)
        if inner is not None and hasattr(inner, "embed_tokens"):
            return inner.embed_tokens
        return self.model.embed_tokens  # type: ignore[return-value]

    def _lm_head(self) -> nn.Module:
        """Return the language model head."""
        return self.model.lm_head  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Input dispatch helpers
    # ------------------------------------------------------------------

    def _call_model(self, inputs: _Inputs) -> Any:
        """Call the model with either a tensor or a dict of kwargs."""
        if isinstance(inputs, dict):
            return self.model(**inputs)
        return self.model(inputs)

    def _batch_seq(self, inputs: _Inputs) -> tuple[int, int]:
        """Return (batch, seq) from either a tensor or dict with 'input_ids'."""
        if isinstance(inputs, dict):
            t = inputs["input_ids"]
        else:
            t = inputs
        return t.shape[0], t.shape[1]

    # ------------------------------------------------------------------
    # Static helpers (GQA + back-map)
    # ------------------------------------------------------------------

    @staticmethod
    def _kv_head_for(query_head: int, n_heads: int, n_kv_heads: int) -> int:
        """Map a query head index to its corresponding KV-head index under GQA."""
        return query_head // (n_heads // n_kv_heads)

    @staticmethod
    def _backmap_qkv_grad(
        dL_dhead: Tensor,
        W_proj_head: Tensor,
        ln_scale: float | Tensor = 1.0,
    ) -> Tensor:
        """Back-map a per-head projection gradient to residual space.

        dL_dhead: (b, s, head_dim) — gradient of metric w.r.t. head h's proj output
        W_proj_head: (head_dim, d_model) — head h's slice of q/k/v_proj.weight
        ln_scale: RMSNorm normalizer (scalar 1.0 for flat toys; (b, s, 1) for HF)
        returns: (b, s, d_model)
        """
        return (dL_dhead @ W_proj_head) * ln_scale

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

    def _collect_writer_acts(self, inputs: _Inputs) -> tuple[Any, dict[Node, Tensor]]:
        """Forward pass collecting writer activations. Returns (model_out, acts_by_node).

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

        handles.append(self._embed().register_forward_hook(_embed_hook))

        for L, block in enumerate(self._layers_list):
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
                module: nn.Module, inp: tuple, output: Tensor,
                _n: Node = mlp_node,
            ) -> None:
                acts[_n] = output.detach()

            # down_proj output IS the MLP residual contribution
            handles.append(block.mlp.down_proj.register_forward_hook(_down_proj_hook))

        try:
            out = self._call_model(inputs)
        finally:
            for h in handles:
                h.remove()

        return out, acts

    def _collect_reader_grads(
        self, inputs: _Inputs, metric: Callable[[Any], Tensor]
    ) -> tuple[Any, dict[Node, Tensor], dict[tuple[Node, str], Tensor]]:
        """Forward + backward pass collecting writer acts AND reader-input grads.

        Reader input grads (component-only gradient — bypasses excluded):
          (mlp(L), "mlp_in")       — grad w.r.t. what up_proj reads (clone of residual),
                                     scaled by post_attention_layernorm ln_scale
          (logits, "logits_in")    — grad w.r.t. what lm_head reads (clone of residual)
          (attn_head(L,h), slot)   — back-mapped grad from q/k/v_proj output (d_model),
                                     scaled by input_layernorm ln_scale

        A forward-pre-hook on each projection module returns a distinct clone of
        the incoming residual.  up_proj / lm_head compute on the clone; the
        residual bypass (x + mlp_out) continues with the ORIGINAL x, so the
        clone's gradient is COMPONENT-ONLY (no bypass double-counting).

        For attention readers: the q/k/v_proj OUTPUT has retain_grad() set.
        After backward, the grad of the proj output for head h is sliced, then
        back-mapped to residual space via W_proj[h], scaled by ln_scale from
        input_layernorm (flat toys: ln_scale=1.0 when norm absent).

        For GQA: k/v projections produce n_kv_heads heads; query head h maps to
        kv-head kv_h = _kv_head_for(h, n_heads, n_kv_heads).

        Returns (model_out, writer_acts, reader_grads).
        reader_grads keys are (Node, slot_str).
        """
        writer_acts: dict[Node, Tensor] = {}
        reader_clones: dict[tuple[Node, str], Tensor] = {}
        # For attention readers, store projection output tensors (with retain_grad)
        proj_outputs: dict[tuple[int, str], Tensor] = {}  # (L, slot) -> proj output
        # For RMSNorm ln_scale: keyed by (L, "attn") and (L, "mlp")
        ln_scales_attn: dict[int, Any] = {}   # scalar 1.0 or (b, s, 1) Tensor
        ln_scales_mlp: dict[int, Any] = {}
        reader_grads: dict[tuple[Node, str], Tensor] = {}
        handles: list[Any] = []

        embed_node = Node("embed")
        logits_node = Node("logits")
        eps = self._rms_norm_eps

        # ---- writer hooks on embed + o_proj + down_proj ----
        def _embed_writer_hook(module: nn.Module, inp: tuple, output: Tensor) -> None:
            writer_acts[embed_node] = output.detach()

        handles.append(self._embed().register_forward_hook(_embed_writer_hook))

        for L, block in enumerate(self._layers_list):
            # Hook input_layernorm to capture pre-attn RMSNorm scale (if present)
            attn_norm = getattr(block, "input_layernorm", None)
            if attn_norm is not None:
                def _attn_norm_pre_hook(
                    module: nn.Module, args: tuple, _L: int = L,
                ) -> None:
                    x_resid = args[0]
                    scale = (1.0 / torch.sqrt(
                        x_resid.pow(2).mean(dim=-1, keepdim=True) + eps
                    )).detach()
                    ln_scales_attn[_L] = scale

                handles.append(attn_norm.register_forward_pre_hook(_attn_norm_pre_hook))
            else:
                ln_scales_attn[L] = 1.0

            # Hook post_attention_layernorm to capture pre-mlp RMSNorm scale (if present)
            mlp_norm = getattr(block, "post_attention_layernorm", None)
            if mlp_norm is not None:
                def _mlp_norm_pre_hook(
                    module: nn.Module, args: tuple, _L: int = L,
                ) -> None:
                    x_resid = args[0]
                    scale = (1.0 / torch.sqrt(
                        x_resid.pow(2).mean(dim=-1, keepdim=True) + eps
                    )).detach()
                    ln_scales_mlp[_L] = scale

                handles.append(mlp_norm.register_forward_pre_hook(_mlp_norm_pre_hook))
            else:
                ln_scales_mlp[L] = 1.0

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
                        module: nn.Module, inp: tuple, output: Tensor,
                        _L: int = L, _slot: str = slot,
                    ) -> None:
                        output.retain_grad()
                        proj_outputs[(_L, _slot)] = output

                    handles.append(proj.register_forward_pre_hook(_proj_pre_hook_reader))
                    handles.append(proj.register_forward_hook(_proj_output_hook))

            mlp_node = Node("mlp", L)

            def _down_proj_writer_hook(
                module: nn.Module, inp: tuple, output: Tensor,
                _n: Node = mlp_node,
            ) -> None:
                writer_acts[_n] = output.detach()

            handles.append(block.mlp.down_proj.register_forward_hook(_down_proj_writer_hook))

        # ---- reader pre-hooks: clone the residual at each projection input ----
        for L, block in enumerate(self._layers_list):
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

        handles.append(self._lm_head().register_forward_pre_hook(_lm_head_reader_pre_hook))

        try:
            out = self._call_model(inputs)
            metric(out).backward()
        finally:
            for h in handles:
                h.remove()

        # Collect component-only reader grads from the MLP/logits clones.
        # For mlp_in readers: scale by the post_attention_layernorm ln_scale.
        for key, t in reader_clones.items():
            if t.grad is not None:
                node, slot = key
                grad = t.grad.detach()
                if slot == "mlp_in" and node.layer is not None:
                    ln_s = ln_scales_mlp.get(node.layer, 1.0)
                    grad = grad * ln_s
                reader_grads[key] = grad

        # Collect attention reader grads: back-map from proj output grads to residual space.
        # Always populate all attention reader slots — zero if the proj isn't in the
        # computation graph (e.g. q/k with fixed attention pattern).
        if self.n_heads > 0:
            n_heads = self.n_heads
            n_kv_heads = self.n_kv_heads
            head_dim = self.head_dim
            b, s = self._batch_seq(inputs)
            for L, block in enumerate(self._layers_list):
                ln_s = ln_scales_attn.get(L, 1.0)
                for slot, proj in [("q", block.self_attn.q_proj),
                                    ("k", block.self_attn.k_proj),
                                    ("v", block.self_attn.v_proj)]:
                    proj_out = proj_outputs.get((L, slot))
                    W_proj = proj.weight  # (n_proj_heads*head_dim, d_model)
                    d_model = W_proj.shape[1]
                    # Number of heads this proj produces (n_heads for q, n_kv_heads for k/v)
                    n_proj_heads = n_kv_heads if slot in ("k", "v") else n_heads
                    if proj_out is None or proj_out.grad is None:
                        # proj unused → zero gradient in residual space
                        for h in range(n_heads):
                            reader_grads[(Node("attn_head", L, h), slot)] = torch.zeros(
                                b, s, d_model, dtype=W_proj.dtype
                            )
                        continue
                    dL_dproj = proj_out.grad  # (b, s, n_proj_heads*head_dim)
                    dL_dproj_heads = dL_dproj.reshape(b, s, n_proj_heads, head_dim)
                    for h in range(n_heads):
                        if slot in ("k", "v"):
                            # GQA: query head h reads kv-group kv_h
                            kv_h = self._kv_head_for(h, n_heads, n_kv_heads)
                            dL_dproj_h = dL_dproj_heads[:, :, kv_h, :]
                            W_proj_h = W_proj[kv_h * head_dim:(kv_h + 1) * head_dim, :]
                        else:
                            dL_dproj_h = dL_dproj_heads[:, :, h, :]
                            W_proj_h = W_proj[h * head_dim:(h + 1) * head_dim, :]
                        grad_resid = self._backmap_qkv_grad(
                            dL_dproj_h.detach(), W_proj_h.detach(), ln_s
                        )
                        reader_grads[(Node("attn_head", L, h), slot)] = grad_resid

        return out, writer_acts, reader_grads

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
        clean_inputs: _Inputs,
        corrupted_inputs: _Inputs,
        metric: Callable[[Any], Tensor],
        ig_steps: int = 1,
    ) -> EAPResult:
        """Compute EAP edge scores.

        ``clean_inputs`` / ``corrupted_inputs`` may be a Tensor (toy models)
        or a dict (HF models, e.g. ``{"input_ids": tensor}``).

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
        clean_inputs: _Inputs,
        corrupted_inputs: _Inputs,
        metric: Callable[[Any], Tensor],
    ) -> dict[Edge, float]:
        """Exact per-edge brute force.  For each edge (u -> v, slot):

        - mlp_in: add delta_act_u to up_proj's INPUT (residual clone point)
        - logits_in: add delta_act_u to lm_head's INPUT
        - q/k/v (attention reader): add (delta_act_u @ W_proj[kv_h].T) to kv-head kv_h's
          slice of the q/k/v_proj OUTPUT, where kv_h = _kv_head_for(h, n_heads, n_kv_heads).
          For q slots: patches head h's slice of q_proj output.
          For k/v under GQA: patches kv_h's slice of k/v_proj output.
          This patches ONLY the relevant head's read (other heads untouched).

        In all cases we measure metric(patched) - metric(clean).

        Bypass is NOT touched (no down_proj post-hook), matching the
        component-only gradient captured in ``run()``.

        On a fully linear model this must equal the analytic EAP score to
        floating-point precision because EAP's first-order approximation is
        exact when there are no nonlinearities.

        ``clean_inputs`` / ``corrupted_inputs`` may be a Tensor or a dict.
        """
        was_training, orig_rg = self._freeze_eval()
        try:
            with torch.no_grad():
                # Baseline clean metric (no hooks)
                clean_out = self._call_model(clean_inputs)
                clean_metric = metric(clean_out).item()

                # Cache corrupted writer activations (same as run() step 1)
                _, acts_corrupted = self._collect_writer_acts(corrupted_inputs)

                # Cache clean writer activations so we can compute delta
                _, acts_clean = self._collect_writer_acts(clean_inputs)

            # delta per writer node: (b, s, d_model)
            delta: dict[Node, Tensor] = {
                n: acts_corrupted[n] - acts_clean[n] for n in self.graph.writers
            }

            scores: dict[Edge, float] = {}
            n_heads = self.n_heads
            n_kv_heads = self.n_kv_heads
            head_dim = self.head_dim

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
                    block = self._layers_list[L]

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
                        self._lm_head().register_forward_pre_hook(_lm_head_add_hook)
                    )

                elif slot in ("q", "k", "v"):
                    # Attention reader: patch the relevant head's slice of the proj output.
                    # For q: use query head h directly.
                    # For k/v under GQA: use kv_h = _kv_head_for(h, n_heads, n_kv_heads).
                    # Add d_act @ W_proj[patch_h].T to the proj output at that head's positions.
                    # This patches ONLY that head's read (other heads untouched).
                    L = reader_node.layer
                    h = reader_node.head
                    block = self._layers_list[L]

                    if slot == "q":
                        proj_module = block.self_attn.q_proj
                        patch_h = h
                        n_proj_heads = n_heads
                    elif slot == "k":
                        proj_module = block.self_attn.k_proj
                        patch_h = self._kv_head_for(h, n_heads, n_kv_heads)
                        n_proj_heads = n_kv_heads
                    else:  # slot == "v"
                        proj_module = block.self_attn.v_proj
                        patch_h = self._kv_head_for(h, n_heads, n_kv_heads)
                        n_proj_heads = n_kv_heads

                    W_proj = proj_module.weight  # (n_proj_heads*head_dim, d_model)
                    W_proj_h = W_proj[patch_h * head_dim:(patch_h + 1) * head_dim, :]
                    # delta in proj-output space for patch_h: (b, s, head_dim)
                    delta_proj_h = d_act @ W_proj_h.T  # (b, s, head_dim)

                    def _proj_output_add_hook(
                        module: nn.Module, inp: tuple, output: Tensor,
                        _patch_h: int = patch_h,
                        _n_proj_heads: int = n_proj_heads,
                        _head_dim: int = head_dim,
                        _delta: Tensor = delta_proj_h,
                    ) -> Tensor:
                        # output shape: (b, s, n_proj_heads*head_dim)
                        b, s, _ = output.shape
                        out = output.clone().reshape(b, s, _n_proj_heads, _head_dim)
                        out[:, :, _patch_h, :] = out[:, :, _patch_h, :] + _delta
                        return out.reshape(b, s, _n_proj_heads * _head_dim)

                    handles.append(proj_module.register_forward_hook(_proj_output_add_hook))

                try:
                    with torch.no_grad():
                        patched_out = self._call_model(clean_inputs)
                        patched_metric = metric(patched_out).item()
                finally:
                    for hh in handles:
                        hh.remove()

                scores[edge] = patched_metric - clean_metric

            return scores
        finally:
            self._restore(was_training, orig_rg)

    def writer_activations(self, inputs: _Inputs) -> dict[Node, Tensor]:
        """Return writer activations dict for the given inputs (debug/test accessor).

        ``inputs`` may be a Tensor (toy models) or a dict (HF models).
        """
        was_training, orig_rg = self._freeze_eval()
        try:
            with torch.no_grad():
                _, acts = self._collect_writer_acts(inputs)
            return acts
        finally:
            self._restore(was_training, orig_rg)
