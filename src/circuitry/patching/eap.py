"""EAP scoring engine + runner. Design spec §3, §5, §7."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from circuitry.patching.graph import (
    Edge,
    EdgeGraph,
    Node,
    _node_from_dict,
    _node_str,
    _node_to_dict,
    build_graph,
    edge_sort_key,
)

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

    def to_markdown(self, *, top_k: int = 20) -> str:
        """Render a markdown summary with a top-K edge table."""
        lines = ["## EAP Circuit", ""]
        lines.append(f"- Graph: {self.graph.n_layers} layers, {self.graph.n_heads} heads")
        lines.append(f"- Total edges scored: {len(self.scores)}")
        lines.append("")
        ranked = self.top_k(top_k)
        lines.append(f"### Top-{len(ranked)} Edges by |score|")
        lines.append("")
        lines.append("| rank | writer | slot | reader | score |")
        lines.append("| ---: | --- | --- | --- | ---: |")
        for i, (edge, score) in enumerate(ranked, 1):
            lines.append(
                f"| {i} | `{_node_str(edge.writer)}` | {edge.slot}"
                f" | `{_node_str(edge.reader)}` | {score:.4g} |"
            )
        return "\n".join(lines)

    def to_json(self) -> str:
        """Serialize to JSON (round-trips via from_json())."""
        import json
        data = {
            "kind": "eap",
            "n_layers": self.graph.n_layers,
            "n_heads": self.graph.n_heads,
            "scores": [
                {
                    "writer": _node_to_dict(edge.writer),
                    "reader": _node_to_dict(edge.reader),
                    "slot": edge.slot,
                    "score": score,
                }
                for edge, score in sorted(self.scores.items(), key=lambda kv: -abs(kv[1]))
            ],
        }
        return json.dumps(data, indent=2)

    @classmethod
    def from_json(cls, text: str) -> "EAPResult":
        """Deserialize from JSON produced by to_json()."""
        import json
        data = json.loads(text)
        graph = build_graph(data["n_layers"], data["n_heads"])
        edge_lookup: dict[tuple, Edge] = {
            (e.writer, e.reader, e.slot): e for e in graph.edges
        }
        scores: dict[Edge, float] = {}
        for row in data["scores"]:
            writer = _node_from_dict(row["writer"])
            reader = _node_from_dict(row["reader"])
            slot = row["slot"]
            edge = edge_lookup.get((writer, reader, slot)) or Edge(writer, reader, slot)
            scores[edge] = row["score"]
        return cls(graph=graph, scores=scores)


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

        # Detect TransformerLens path via TLSiteResolver (lazy — no TL import here).
        from circuitry.patching.sites import TLSiteResolver  # local import to avoid cycle
        self._tl: bool = isinstance(resolver, TLSiteResolver)

        if self._tl:
            # TL path: read topology from model.cfg (HookedTransformerConfig)
            cfg = model.cfg  # type: ignore[attr-defined]
            n_layers: int = cfg.n_layers
            n_heads: int = cfg.n_heads
            self.n_heads = n_heads
            self.n_kv_heads: int = n_heads  # TL input hooks expose MHA; no GQA back-map needed
            self.head_dim = getattr(cfg, "d_head", None) or (cfg.d_model // n_heads)
            self._layers_list = None  # unused in TL path
            self._rms_norm_eps: float = 1e-6  # unused in TL path
        else:
            # HF/toy path: locate layers list the existing way
            self._layers_list = self._locate_layers(model)
            n_layers = len(self._layers_list)
            n_heads = getattr(resolver, "n_heads", 0) if resolver is not None else 0
            self.n_heads = n_heads
            # GQA: number of key/value heads (defaults to n_heads for MHA)
            self.n_kv_heads = n_heads
            if resolver is not None and n_heads > 0:
                hf_cfg = getattr(model, "config", None)
                if hf_cfg is not None:
                    self.n_kv_heads = getattr(hf_cfg, "num_key_value_heads", n_heads)
            self.head_dim = resolver.head_dim if (resolver is not None and n_heads > 0) else None
            # RMSNorm eps from config (used for ln_scale; only relevant for HF models)
            hf_cfg = getattr(model, "config", None)
            self._rms_norm_eps = getattr(hf_cfg, "rms_norm_eps", 1e-6) if hf_cfg is not None else 1e-6

        self.graph = build_graph(n_layers, n_heads)

    # ------------------------------------------------------------------
    # Module locator helpers (nested HF vs flat toys)
    # ------------------------------------------------------------------

    @staticmethod
    def _locate_layers(model: nn.Module) -> nn.ModuleList:
        from circuitry.patching._layout import locate_layers
        return locate_layers(model)

    def _embed(self) -> nn.Module:
        from circuitry.patching._layout import locate_embed
        return locate_embed(self.model)

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

    def _collect_reader_inputs(
        self, inputs: _Inputs
    ) -> tuple[dict[Node, Tensor], dict[tuple[Node, str], Tensor]]:
        """No-grad pass that captures writer activations AND the raw (pre-proj) reader
        inputs at each hook point, for use by the IG interpolation loop.

        Returns (writer_acts, reader_inputs) where reader_inputs keys are the
        same (Node, slot) as _collect_reader_grads but values are the actual
        tensor the projection / lm_head reads — i.e., the post-LN residual for
        MLP/logits readers, and the post-LN residual for attn readers (pre q/k/v_proj).
        These are *detached* tensors (no grad).

        Also captures ln_scales_attn / ln_scales_mlp (returned as side-effect via
        the shared mutable dicts passed in; here they are captured into returned dicts).
        """
        writer_acts: dict[Node, Tensor] = {}
        reader_inputs: dict[tuple[Node, str], Tensor] = {}
        ln_scales_attn: dict[int, Any] = {}
        ln_scales_mlp: dict[int, Any] = {}
        handles: list[Any] = []

        embed_node = Node("embed")
        eps = self._rms_norm_eps

        def _embed_writer_hook(module: nn.Module, inp: tuple, output: Tensor) -> None:
            writer_acts[embed_node] = output.detach()

        handles.append(self._embed().register_forward_hook(_embed_writer_hook))

        for L, block in enumerate(self._layers_list):
            # Capture LN scales (same logic as _collect_reader_grads)
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

            # Attention head writer hooks (same as _collect_writer_acts)
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
                    W_O = module.weight
                    for h in range(_n_heads):
                        z_h = z_heads[:, :, h, :]
                        W_O_h = W_O[:, h * _head_dim:(h + 1) * _head_dim]
                        contrib = z_h @ W_O_h.T
                        writer_acts[Node("attn_head", _L, h)] = contrib

                handles.append(block.self_attn.o_proj.register_forward_pre_hook(_o_proj_pre_hook_writer))

                # Capture the raw residual at each proj input (post-LN, pre-proj)
                for slot, proj in [("q", block.self_attn.q_proj),
                                    ("k", block.self_attn.k_proj),
                                    ("v", block.self_attn.v_proj)]:
                    def _proj_reader_input_hook(
                        module: nn.Module, args: tuple,
                        _L: int = L, _slot: str = slot,
                    ) -> None:
                        reader_inputs[(Node("attn_head", _L, 0), _slot, _L)] = args[0].detach()  # type: ignore[index]

                    handles.append(proj.register_forward_pre_hook(_proj_reader_input_hook))

            mlp_node = Node("mlp", L)

            def _down_proj_writer_hook(
                module: nn.Module, inp: tuple, output: Tensor,
                _n: Node = mlp_node,
            ) -> None:
                writer_acts[_n] = output.detach()

            handles.append(block.mlp.down_proj.register_forward_hook(_down_proj_writer_hook))

            # Capture MLP reader input (post-LN residual before up_proj)
            def _up_proj_reader_input_hook(
                module: nn.Module, args: tuple,
                _L: int = L,
            ) -> None:
                reader_inputs[(Node("mlp", _L), "mlp_in")] = args[0].detach()  # type: ignore[index]

            handles.append(block.mlp.up_proj.register_forward_pre_hook(_up_proj_reader_input_hook))

        # Capture logits reader input (residual before lm_head)
        def _lm_head_reader_input_hook(module: nn.Module, args: tuple) -> None:
            reader_inputs[(Node("logits"), "logits_in")] = args[0].detach()  # type: ignore[index]

        handles.append(self._lm_head().register_forward_pre_hook(_lm_head_reader_input_hook))

        try:
            with torch.no_grad():
                self._call_model(inputs)
        finally:
            for h in handles:
                h.remove()

        # Return reader_inputs with standard (Node, slot) keys (drop the extra L for attn)
        # The attn keys were stored with a 3-tuple; normalise to (Node, slot) now.
        # We stored attn reader inputs keyed by (Node("attn_head", L, 0), slot, L) —
        # one entry per (L, slot); per-head sharing is handled later in IG loop.
        normalized: dict[tuple[Node, str], Tensor] = {}
        for key, val in reader_inputs.items():
            if len(key) == 3 and isinstance(key[2], int):
                # attn key: (Node("attn_head", L, 0), slot, L) → ("_attn_in", L, slot)
                _, slot, layer_idx = key
                normalized[("_attn_in", layer_idx, slot)] = val  # type: ignore[assignment]
            else:
                normalized[key] = val  # type: ignore[assignment]

        return writer_acts, normalized, ln_scales_attn, ln_scales_mlp  # type: ignore[return-value]

    def _collect_reader_grads_ig(
        self,
        inputs: _Inputs,
        metric: Callable[[Any], Tensor],
        ig_steps: int,
        clean_reader_inputs: dict,
        corrupted_reader_inputs: dict,
        ln_scales_attn: dict[int, Any],
        ln_scales_mlp: dict[int, Any],
    ) -> dict[tuple[Node, str], Tensor]:
        """Run ig_steps interpolation forward+backward passes and average gradients.

        For each k=1..ig_steps, α=k/ig_steps, each reader uses the "offset + clone"
        pattern to evaluate the gradient AT the interpolated value r_interp while
        maintaining downstream gradient flow:

          actual_x = what the model naturally produces at this reader (from clean inputs)
          r_interp = corr_in + α * (clean_in - corr_in)          [target value]
          shifted   = actual_x + (r_interp − actual_x).detach()  [= r_interp numerically;
                                                                   connected to actual_x's grad graph]
          r_clone   = shifted.clone().requires_grad_(True)        [same pattern as vanilla;
                                                                   non-leaf if shifted has grad]

        This preserves inter-layer gradient flow (gradient at layer L flows back through
        the actual residual stream to layer 0 readers), while the VALUE at each reader is
        exactly r_interp. On a linear model the gradient is constant so IG == vanilla;
        on a nonlinear model the gradient varies, giving a better attribution estimate.

        For ig_steps=1: called only when ig_steps > 1 — the vanilla path handles ig_steps=1.
        Returns averaged reader_grads dict (same keys as _collect_reader_grads).
        """
        n_heads = self.n_heads
        n_kv_heads = self.n_kv_heads
        head_dim = self.head_dim
        b, s = self._batch_seq(inputs)
        logits_node = Node("logits")

        # Accumulate grads over ig_steps
        grad_accum_mlp: dict[tuple[Node, str], Tensor] = {}
        grad_accum_attn: dict[tuple[int, str], Tensor] = {}  # (L, slot) -> summed

        for k in range(1, ig_steps + 1):
            alpha = k / ig_steps
            step_mlp_clones: dict[tuple[Node, str], Tensor] = {}
            step_proj_outputs: dict[tuple[int, str], Tensor] = {}
            step_handles: list[Any] = []

            for L, block in enumerate(self._layers_list):
                if self.n_heads > 0:
                    for slot, proj in [("q", block.self_attn.q_proj),
                                        ("k", block.self_attn.k_proj),
                                        ("v", block.self_attn.v_proj)]:
                        clean_in = clean_reader_inputs.get(("_attn_in", L, slot))
                        corr_in = corrupted_reader_inputs.get(("_attn_in", L, slot))

                        def _proj_pre_hook_ig(
                            module: nn.Module, args: tuple,
                            _clean: Any = clean_in,
                            _corr: Any = corr_in,
                            _alpha: float = alpha,
                        ) -> tuple:
                            actual_x = args[0]
                            if _clean is not None and _corr is not None:
                                # Shift to interpolated value while staying in computation graph
                                r_interp = (_corr + _alpha * (_clean - _corr)).detach()
                                offset = (r_interp - actual_x).detach()
                                actual_x = actual_x + offset  # = r_interp numerically
                            r_clone = actual_x.clone().requires_grad_(True)
                            return (r_clone,) + args[1:]

                        def _proj_output_hook_ig(
                            module: nn.Module, inp: tuple, output: Tensor,
                            _L: int = L, _slot: str = slot,
                            _store: dict = step_proj_outputs,
                        ) -> None:
                            output.retain_grad()
                            _store[(_L, _slot)] = output

                        step_handles.append(proj.register_forward_pre_hook(_proj_pre_hook_ig))
                        step_handles.append(proj.register_forward_hook(_proj_output_hook_ig))

                # MLP reader: shift actual residual to interpolated value, then clone
                mlp_node = Node("mlp", L)
                clean_mlp = clean_reader_inputs.get((mlp_node, "mlp_in"))
                corr_mlp = corrupted_reader_inputs.get((mlp_node, "mlp_in"))

                def _up_proj_ig(
                    module: nn.Module, args: tuple,
                    _node: Node = mlp_node,
                    _clean: Any = clean_mlp,
                    _corr: Any = corr_mlp,
                    _alpha: float = alpha,
                    _store: dict = step_mlp_clones,
                ) -> tuple:
                    actual_x = args[0]
                    if _clean is not None and _corr is not None:
                        r_interp = (_corr + _alpha * (_clean - _corr)).detach()
                        offset = (r_interp - actual_x).detach()
                        actual_x = actual_x + offset  # = r_interp numerically
                    r_clone = actual_x.clone().requires_grad_(True)
                    r_clone.retain_grad()
                    _store[(_node, "mlp_in")] = r_clone
                    return (r_clone,) + args[1:]

                step_handles.append(block.mlp.up_proj.register_forward_pre_hook(_up_proj_ig))

            # Logits reader: shift actual residual to interpolated value, then clone
            clean_logits_in = clean_reader_inputs.get((logits_node, "logits_in"))
            corr_logits_in = corrupted_reader_inputs.get((logits_node, "logits_in"))

            def _lm_head_ig(
                module: nn.Module, args: tuple,
                _clean: Any = clean_logits_in,
                _corr: Any = corr_logits_in,
                _alpha: float = alpha,
                _store: dict = step_mlp_clones,
                _logits_node: Node = logits_node,
            ) -> tuple:
                actual_x = args[0]
                if _clean is not None and _corr is not None:
                    r_interp = (_corr + _alpha * (_clean - _corr)).detach()
                    offset = (r_interp - actual_x).detach()
                    actual_x = actual_x + offset  # = r_interp numerically
                r_clone = actual_x.clone().requires_grad_(True)
                r_clone.retain_grad()
                _store[(_logits_node, "logits_in")] = r_clone
                return (r_clone,) + args[1:]

            step_handles.append(self._lm_head().register_forward_pre_hook(_lm_head_ig))

            try:
                with torch.enable_grad():
                    out = self._call_model(inputs)
                    metric(out).backward()
            finally:
                for h in step_handles:
                    h.remove()

            # Accumulate MLP/logits reader grads (with LN scaling same as vanilla)
            for key, t in step_mlp_clones.items():
                node, slot = key
                if t.grad is not None:
                    g = t.grad.detach()
                    if slot == "mlp_in" and node.layer is not None:
                        ln_s = ln_scales_mlp.get(node.layer, 1.0)
                        g = g * ln_s
                else:
                    # No grad (e.g. disconnected): contribute zero
                    ref = clean_reader_inputs.get(key) or corrupted_reader_inputs.get(key)
                    g = torch.zeros_like(ref) if ref is not None else torch.zeros(b, s, 1)
                if key in grad_accum_mlp:
                    grad_accum_mlp[key] = grad_accum_mlp[key] + g
                else:
                    grad_accum_mlp[key] = g

            # Accumulate attn proj output grads
            for (L_k, slot_k), proj_out in step_proj_outputs.items():
                if proj_out.grad is not None:
                    g = proj_out.grad.detach()
                    if (L_k, slot_k) in grad_accum_attn:
                        grad_accum_attn[(L_k, slot_k)] = grad_accum_attn[(L_k, slot_k)] + g
                    else:
                        grad_accum_attn[(L_k, slot_k)] = g

        # Average gradients
        reader_grads: dict[tuple[Node, str], Tensor] = {}

        for key, accum in grad_accum_mlp.items():
            reader_grads[key] = accum / ig_steps

        # Back-map averaged attn grads to residual space (same logic as _collect_reader_grads)
        if self.n_heads > 0:
            for L, block in enumerate(self._layers_list):
                ln_s = ln_scales_attn.get(L, 1.0)
                for slot, proj in [("q", block.self_attn.q_proj),
                                    ("k", block.self_attn.k_proj),
                                    ("v", block.self_attn.v_proj)]:
                    W_proj = proj.weight  # (n_proj_heads*head_dim, d_model)
                    d_model = W_proj.shape[1]
                    n_proj_heads = n_kv_heads if slot in ("k", "v") else n_heads
                    accum = grad_accum_attn.get((L, slot))
                    if accum is None:
                        # No grad accumulated → zero
                        for h in range(n_heads):
                            reader_grads[(Node("attn_head", L, h), slot)] = torch.zeros(
                                b, s, d_model, dtype=W_proj.dtype
                            )
                        continue
                    avg_proj_grad = accum / ig_steps  # (b, s, n_proj_heads*head_dim)
                    avg_proj_grad_heads = avg_proj_grad.reshape(b, s, n_proj_heads, head_dim)
                    for h in range(n_heads):
                        if slot in ("k", "v"):
                            kv_h = self._kv_head_for(h, n_heads, n_kv_heads)
                            dL_dproj_h = avg_proj_grad_heads[:, :, kv_h, :]
                            W_proj_h = W_proj[kv_h * head_dim:(kv_h + 1) * head_dim, :]
                        else:
                            dL_dproj_h = avg_proj_grad_heads[:, :, h, :]
                            W_proj_h = W_proj[h * head_dim:(h + 1) * head_dim, :]
                        grad_resid = self._backmap_qkv_grad(
                            dL_dproj_h, W_proj_h.detach(), ln_s
                        )
                        reader_grads[(Node("attn_head", L, h), slot)] = grad_resid

        return reader_grads

    # ------------------------------------------------------------------
    # TransformerLens-specific collect methods
    # ------------------------------------------------------------------

    def _collect_writer_acts_tl(self, inputs: Tensor) -> dict[Node, Tensor]:
        """Collect writer activations via TL native hooks (no grad).

        Uses:
          hook_embed                        → Node("embed")
          blocks.{L}.attn.hook_result       → Node("attn_head", L, h)  [h in 0..n_heads-1]
          blocks.{L}.hook_mlp_out           → Node("mlp", L)

        Requires model.set_use_attn_result(True) (called by run()).
        """
        acts: dict[Node, Tensor] = {}
        n_heads = self.n_heads
        n_layers = self.graph.n_layers

        def _embed_hook(tensor: Tensor, hook: Any) -> None:
            acts[Node("embed")] = tensor.detach()

        fwd_hooks: list[tuple[str, Any]] = [("hook_embed", _embed_hook)]

        for L in range(n_layers):
            result_key = f"blocks.{L}.attn.hook_result"
            mlp_out_key = f"blocks.{L}.hook_mlp_out"

            def _result_hook(tensor: Tensor, hook: Any, _L: int = L) -> None:
                # tensor: (b, s, n_heads, d_model)
                for h in range(n_heads):
                    acts[Node("attn_head", _L, h)] = tensor[:, :, h, :].detach()

            def _mlp_out_hook(tensor: Tensor, hook: Any, _L: int = L) -> None:
                acts[Node("mlp", _L)] = tensor.detach()

            fwd_hooks.append((result_key, _result_hook))
            fwd_hooks.append((mlp_out_key, _mlp_out_hook))

        with torch.no_grad():
            self.model.run_with_hooks(inputs, fwd_hooks=fwd_hooks)  # type: ignore[attr-defined]

        return acts

    def _collect_reader_grads_tl(
        self,
        inputs: Tensor,
        metric: Callable[[Any], Tensor],
    ) -> tuple[Any, dict[Node, Tensor], dict[tuple[Node, str], Tensor]]:
        """Forward + backward collecting writer acts AND reader grads via TL hooks.

        Writer acts:
          hook_embed → Node("embed")
          blocks.{L}.attn.hook_result[:,:,h,:] → Node("attn_head", L, h)
          blocks.{L}.hook_mlp_out → Node("mlp", L)

        Reader grads (after backward):
          blocks.{L}.hook_q_input[:,:,h,:].grad → (Node("attn_head",L,h), "q")
          blocks.{L}.hook_k_input[:,:,h,:].grad → (Node("attn_head",L,h), "k")
          blocks.{L}.hook_v_input[:,:,h,:].grad → (Node("attn_head",L,h), "v")
          blocks.{L}.hook_mlp_in.grad            → (Node("mlp",L), "mlp_in")
          blocks.{N-1}.hook_resid_post.grad       → (Node("logits"), "logits_in")

        Reader hooks return a clone with requires_grad_(True) so grad flows back.
        """
        writer_acts: dict[Node, Tensor] = {}
        reader_tensors: dict[tuple[Node, str], Tensor] = {}
        n_heads = self.n_heads
        n_layers = self.graph.n_layers

        # Writer hooks: just capture, no modification
        def _embed_hook(tensor: Tensor, hook: Any) -> None:
            writer_acts[Node("embed")] = tensor.detach()

        fwd_hooks: list[tuple[str, Any]] = [("hook_embed", _embed_hook)]

        for L in range(n_layers):
            result_key = f"blocks.{L}.attn.hook_result"
            mlp_out_key = f"blocks.{L}.hook_mlp_out"

            def _result_hook(tensor: Tensor, hook: Any, _L: int = L) -> None:
                for h in range(n_heads):
                    writer_acts[Node("attn_head", _L, h)] = tensor[:, :, h, :].detach()

            def _mlp_out_hook(tensor: Tensor, hook: Any, _L: int = L) -> None:
                writer_acts[Node("mlp", _L)] = tensor.detach()

            fwd_hooks.append((result_key, _result_hook))
            fwd_hooks.append((mlp_out_key, _mlp_out_hook))

        # Reader hooks: return a differentiable clone so grad flows to it
        for L in range(n_layers):
            for slot in ("q", "k", "v"):
                hook_key = f"blocks.{L}.hook_{slot}_input"

                def _qkv_reader_hook(
                    tensor: Tensor, hook: Any,
                    _L: int = L, _slot: str = slot,
                ) -> Tensor:
                    t = tensor.clone().requires_grad_(True)
                    t.retain_grad()
                    # Store the full (b,s,n_heads,d_model) tensor; slice per-head after bwd
                    reader_tensors[(_L, _slot)] = t
                    return t  # substitute: downstream sees our differentiable clone

                fwd_hooks.append((hook_key, _qkv_reader_hook))

            mlp_in_key = f"blocks.{L}.hook_mlp_in"
            mlp_node = Node("mlp", L)

            def _mlp_in_hook(
                tensor: Tensor, hook: Any, _node: Node = mlp_node,
            ) -> Tensor:
                t = tensor.clone().requires_grad_(True)
                t.retain_grad()
                reader_tensors[(_node, "mlp_in")] = t
                return t

            fwd_hooks.append((mlp_in_key, _mlp_in_hook))

        # Logits reader: residual post at the last block
        last_resid_key = f"blocks.{n_layers - 1}.hook_resid_post"
        logits_node = Node("logits")

        def _logits_reader_hook(tensor: Tensor, hook: Any) -> Tensor:
            t = tensor.clone().requires_grad_(True)
            t.retain_grad()
            reader_tensors[(logits_node, "logits_in")] = t
            return t

        fwd_hooks.append((last_resid_key, _logits_reader_hook))

        with torch.enable_grad():
            logits = self.model.run_with_hooks(inputs, fwd_hooks=fwd_hooks)  # type: ignore[attr-defined]
            metric(logits).backward()

        # Assemble reader grads dict (keyed by (Node, slot))
        reader_grads: dict[tuple[Node, str], Tensor] = {}

        for L in range(n_layers):
            for slot in ("q", "k", "v"):
                full_tensor = reader_tensors.get((L, slot))
                for h in range(n_heads):
                    key = (Node("attn_head", L, h), slot)
                    if full_tensor is not None and full_tensor.grad is not None:
                        # Slice head h from full (b, s, n_heads, d_model) grad
                        reader_grads[key] = full_tensor.grad[:, :, h, :].detach()
                    else:
                        # No gradient (e.g. q/k unused in fixed-pattern attn)
                        b, s = inputs.shape[0], inputs.shape[1]
                        d_model = self.model.cfg.d_model  # type: ignore[attr-defined]
                        reader_grads[key] = torch.zeros(b, s, d_model)

            mlp_node = Node("mlp", L)
            mlp_t = reader_tensors.get((mlp_node, "mlp_in"))
            if mlp_t is not None and mlp_t.grad is not None:
                reader_grads[(mlp_node, "mlp_in")] = mlp_t.grad.detach()
            else:
                b, s = inputs.shape[0], inputs.shape[1]
                d_model = self.model.cfg.d_model  # type: ignore[attr-defined]
                reader_grads[(mlp_node, "mlp_in")] = torch.zeros(b, s, d_model)

        logits_t = reader_tensors.get((logits_node, "logits_in"))
        if logits_t is not None and logits_t.grad is not None:
            reader_grads[(logits_node, "logits_in")] = logits_t.grad.detach()
        else:
            b, s = inputs.shape[0], inputs.shape[1]
            d_model = self.model.cfg.d_model  # type: ignore[attr-defined]
            reader_grads[(logits_node, "logits_in")] = torch.zeros(b, s, d_model)

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

        if self._tl:
            # Enable TL hooks needed for the EAP pass; restore prior values after.
            _prior_attn_result = self.model.cfg.use_attn_result  # type: ignore[attr-defined]
            _prior_split_qkv = self.model.cfg.use_split_qkv_input  # type: ignore[attr-defined]
            _prior_mlp_in = self.model.cfg.use_hook_mlp_in  # type: ignore[attr-defined]
            self.model.set_use_attn_result(True)  # type: ignore[attr-defined]
            self.model.set_use_split_qkv_input(True)  # type: ignore[attr-defined]
            self.model.set_use_hook_mlp_in(True)  # type: ignore[attr-defined]

        try:
            if self._tl:
                # TL path: use native hook-based collectors (ig_steps > 1 not implemented
                # for TL; ig_steps=1 falls through unchanged as vanilla)
                acts_corrupted = self._collect_writer_acts_tl(corrupted_inputs)  # type: ignore[arg-type]
                _, acts_clean, grads_clean = self._collect_reader_grads_tl(clean_inputs, metric)  # type: ignore[arg-type]
            elif ig_steps <= 1:
                # HF/toy path: vanilla EAP (ig_steps=1 or default)
                # Step 1: corrupted forward (no grad needed)
                with torch.no_grad():
                    _, acts_corrupted = self._collect_writer_acts(corrupted_inputs)

                # Step 2: clean forward + backward (need grads for reader inputs)
                _, acts_clean, grads_clean = self._collect_reader_grads(clean_inputs, metric)
            else:
                # HF/toy path: EAP-IG with ig_steps interpolation steps
                # Step 1: cache corrupted writer acts + reader inputs (no grad)
                acts_corrupted, corrupted_reader_inputs, _, _ = \
                    self._collect_reader_inputs(corrupted_inputs)  # type: ignore[misc]

                # Step 2: cache clean writer acts + reader inputs + LN scales (no grad)
                acts_clean, clean_reader_inputs, ln_scales_attn, ln_scales_mlp = \
                    self._collect_reader_inputs(clean_inputs)  # type: ignore[misc]

                # Step 3: IG loop — run ig_steps interpolated forward+backward passes
                grads_clean = self._collect_reader_grads_ig(
                    clean_inputs, metric, ig_steps,
                    clean_reader_inputs, corrupted_reader_inputs,
                    ln_scales_attn, ln_scales_mlp,
                )

            # Step 4: build stacked tensors and score (shared)
            act_clean_t = self._stack_acts(acts_clean)
            act_corrupted_t = self._stack_acts(acts_corrupted)
            grad_clean_t = self._stack_grads(grads_clean)

            return score_edges(self.graph, act_clean_t, act_corrupted_t, grad_clean_t)
        finally:
            self._restore(was_training, orig_rg)
            if self._tl:
                # Restore prior TL config flags
                self.model.set_use_attn_result(_prior_attn_result)  # type: ignore[attr-defined]
                self.model.set_use_split_qkv_input(_prior_split_qkv)  # type: ignore[attr-defined]
                self.model.set_use_hook_mlp_in(_prior_mlp_in)  # type: ignore[attr-defined]

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
