"""AtP* node attribution. Design spec docs/superpowers/specs/2026-05-24-atp-design.md."""
from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from circuitry.patching.graph import Node

_ATTN_SLOTS = ("q", "k", "v")

# Type alias for flexible inputs: either a raw tensor (toy models) or a dict
# of keyword arguments (HF models, e.g. {"input_ids": tensor}).
_Inputs = Tensor | dict[str, Any]


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

    def verify_top_k(
        self,
        k: int,
        clean_inputs: Any,
        corrupted_inputs: Any,
        metric: Any,
        resolver: Any,
        runner: AtPRunner,
    ) -> dict[AtPNode, tuple[float, float]]:
        """For the top-K nodes by |score|, run real patch_site patching (ground truth)
        and return {node: (atp_score, true_patch_effect)}.

        Uses runner.bruteforce_node_scores for ground-truth independent patch_site
        interventions. Note: bruteforce_node_scores uses the HF path (self._layers_list);
        calling this on a TL-backed runner is not supported.
        """
        top_nodes = self.top_k(k)
        nodes = [node for node, _ in top_nodes]
        true_effects = runner.bruteforce_node_scores(
            clean_inputs, corrupted_inputs, metric, nodes=nodes
        )
        result: dict[AtPNode, tuple[float, float]] = {}
        for node, atp_score in top_nodes:
            true_effect = true_effects.get(node, 0.0)
            result[node] = (atp_score, true_effect)
        return result


# ---------------------------------------------------------------------------
# GQA helper (mirrors EAP's _kv_head_for)
# ---------------------------------------------------------------------------

def _kv_head_for(query_head: int, n_heads: int, n_kv_heads: int) -> int:
    """Map a query head index to its corresponding KV-head index under GQA."""
    return query_head // (n_heads // n_kv_heads)


class AtPRunner:
    """Vanilla AtP / AtP* node attribution runner.

    Computes per-node scores: score(node) = Σ(Δact_node ⊙ grad_node) summed
    over ALL dims (positions + features).

    Δact = corrupted_act − clean_act
    grad = ∂metric/∂act via retain_grad on the node's OWN activation tensor

    On a LINEAR model this equals the brute-force patch_site metric delta —
    that's the exact correctness gate.

    Nodes covered:
      embed  — embed_tokens output (d_model)
      mlp(L) — layers[L].mlp.down_proj output (d_model)
      attn_head(L,h) slot v — layers[L].self_attn.v_proj output, head-h slice
      attn_head(L,h) slot q/k:
        qk_fix=True  — attention-pattern recomputation (QK fix); scored in d_model space
                       against grad_attn_out (gradient w.r.t. o_proj output)
        qk_fix=False — vanilla Σ(Δq_h ⊙ grad_q_h) in head_dim space (q_proj output grads)
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
            self.n_kv_heads: int = n_heads  # TL exposes MHA; no GQA back-map needed
            self.head_dim: int | None = cfg.d_model // n_heads
            self.n_layers = n_layers
            self._layers_list = None  # unused in TL path
            self._has_rope: bool = False  # TL handles position encoding internally
        else:
            # HF/toy path: locate layers list the existing way
            self._layers_list = self._locate_layers(model)
            n_layers = len(self._layers_list)
            n_heads = getattr(resolver, "n_heads", 0) if resolver is not None else 0
            self.n_layers = n_layers
            self.n_heads = n_heads
            d_model = getattr(resolver, "d_model", None) if resolver is not None else None
            self.head_dim = (d_model // n_heads) if (d_model is not None and n_heads > 0) else None

            # GQA: number of key/value heads (defaults to n_heads for MHA)
            self.n_kv_heads = n_heads
            if resolver is not None and n_heads > 0:
                hf_cfg = getattr(model, "config", None)
                if hf_cfg is not None:
                    self.n_kv_heads = getattr(hf_cfg, "num_key_value_heads", n_heads)

            # Detect HF model with RoPE (has model.model.rotary_emb)
            inner = getattr(model, "model", None)
            self._has_rope = (inner is not None and hasattr(inner, "rotary_emb"))

    # ------------------------------------------------------------------
    # Module locator helpers
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

    # ------------------------------------------------------------------
    # Input dispatch helpers
    # ------------------------------------------------------------------

    def _call_model(self, inputs: _Inputs) -> Any:
        """Call the model with either a tensor or a dict of kwargs."""
        if isinstance(inputs, dict):
            return self.model(**inputs)
        return self.model(inputs)

    # ------------------------------------------------------------------
    # Freeze / restore helpers
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

    # ------------------------------------------------------------------
    # QK data capture helpers
    # ------------------------------------------------------------------

    def _apply_rope(
        self,
        q_pre: Tensor,       # (b, s, n_heads*head_dim)
        k_pre: Tensor,       # (b, s, n_kv_heads*head_dim)
        position_embeddings: tuple[Tensor, Tensor],
    ) -> tuple[Tensor, Tensor]:
        """Apply rotary position embeddings and return post-RoPE Q, K.

        Returns:
          q_post: (b, n_heads, s, head_dim)
          k_post: (b, n_kv_heads, s, head_dim)
        """
        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb  # lazy import

        n_heads = self.n_heads
        n_kv_heads = self.n_kv_heads
        head_dim = self.head_dim
        cos, sin = position_embeddings
        b, s, _ = q_pre.shape
        q_pre_h = q_pre.view(b, s, n_heads, head_dim).transpose(1, 2)       # (b, n_heads, s, hd)
        k_pre_h = k_pre.view(b, s, n_kv_heads, head_dim).transpose(1, 2)    # (b, n_kv, s, hd)
        q_post, k_post = apply_rotary_pos_emb(q_pre_h, k_pre_h, cos, sin)
        return q_post, k_post

    def _cache_qk_data(
        self,
        inputs: _Inputs,
        *,
        compute_pattern: bool = True,
    ) -> dict[int, dict[str, Any]]:
        """No-grad forward: capture per-layer Q, K, V (post-RoPE if available) and
        the attention pattern (via output_attentions=True on HF models).

        Returns layer_idx → dict with keys:
          "q_post": (b, n_heads, s, head_dim)   — post-RoPE Q (or pre-RoPE if no RoPE)
          "k_post": (b, n_kv_heads, s, head_dim)
          "v":      (b, n_kv_heads, s, head_dim)
          "pattern": (b, n_heads, s, s)  [only if compute_pattern=True and HF model]
          "attn_mask": (b, 1, s, s) or None
        """
        n_heads = self.n_heads
        n_kv_heads = self.n_kv_heads
        head_dim = self.head_dim
        has_rope = self._has_rope

        layer_data: dict[int, dict[str, Any]] = {}
        handles: list[Any] = []

        for L, block in enumerate(self._layers_list):
            if not (n_heads > 0 and head_dim is not None and hasattr(block, "self_attn")):
                continue

            cap: dict[str, Any] = {}
            layer_data[L] = cap
            _L = L

            if has_rope:
                def _self_attn_pre_hook(
                    module: nn.Module, args: tuple, kwargs: dict,
                    _c: dict = cap,
                ) -> None:
                    _c["position_embeddings"] = kwargs.get("position_embeddings")
                    _c["attn_mask"] = kwargs.get("attention_mask")

                handles.append(
                    block.self_attn.register_forward_pre_hook(
                        _self_attn_pre_hook, with_kwargs=True
                    )
                )

            def _q_proj_hook(
                module: nn.Module, inp: tuple, output: Tensor,
                _c: dict = cap,
            ) -> None:
                _c["q_pre"] = output.detach()

            def _k_proj_hook(
                module: nn.Module, inp: tuple, output: Tensor,
                _c: dict = cap,
            ) -> None:
                _c["k_pre"] = output.detach()

            def _v_proj_hook(
                module: nn.Module, inp: tuple, output: Tensor,
                _c: dict = cap, _nkv: int = n_kv_heads, _hd: int = head_dim,
            ) -> None:
                v = output.detach()
                b, s, _ = v.shape
                _c["v"] = v.view(b, s, _nkv, _hd).transpose(1, 2)  # (b, nkv, s, hd)

            handles.append(block.self_attn.q_proj.register_forward_hook(_q_proj_hook))
            handles.append(block.self_attn.k_proj.register_forward_hook(_k_proj_hook))
            handles.append(block.self_attn.v_proj.register_forward_hook(_v_proj_hook))

        # For HF models with output_attentions support, enable it to get patterns
        call_inputs = inputs
        if compute_pattern and has_rope and isinstance(inputs, dict):
            call_inputs = dict(inputs)
            call_inputs["output_attentions"] = True

        try:
            with torch.no_grad():
                out = self._call_model(call_inputs)
        finally:
            for h in handles:
                h.remove()

        # Post-process: apply RoPE and store patterns
        for _L, cap in layer_data.items():
            q_pre = cap.get("q_pre")
            k_pre = cap.get("k_pre")
            if q_pre is None or k_pre is None:
                continue

            if has_rope and "position_embeddings" in cap:
                q_post, k_post = self._apply_rope(q_pre, k_pre, cap["position_embeddings"])
            else:
                # Fallback (no RoPE / toy models): reshape to (b, n_heads, s, hd)
                b, s, _ = q_pre.shape
                q_post = q_pre.view(b, s, n_heads, head_dim).transpose(1, 2)
                k_post = k_pre.view(b, s, n_kv_heads, head_dim).transpose(1, 2)

            cap["q_post"] = q_post
            cap["k_post"] = k_post

        # Attach attention patterns from output_attentions
        if compute_pattern and has_rope and hasattr(out, "attentions") and out.attentions:
            for L, cap in layer_data.items():
                if L < len(out.attentions) and out.attentions[L] is not None:
                    cap["pattern"] = out.attentions[L].detach()

        return layer_data

    # ------------------------------------------------------------------
    # Activation capture helpers
    # ------------------------------------------------------------------

    def _cache_node_acts(
        self,
        inputs: _Inputs,
        capture_intermediates: bool = False,
    ) -> tuple[dict[AtPNode, Tensor], dict[int, Tensor]]:
        """No-grad forward pass caching node activations (detached).

        Captures:
          embed node → embed_tokens output (b, s, d_model)
          mlp(L) node → down_proj output (b, s, d_model)
          attn_head(L,h) v → v_proj output, head-h slice (b, s, head_dim)
          [if capture_intermediates] layer-L down_proj INPUT (b, s, d_mlp)

        Returns (acts, intermediates) where intermediates is layer_idx → tensor.
        """
        acts: dict[AtPNode, Tensor] = {}
        intermediates: dict[int, Tensor] = {}
        handles: list[Any] = []

        embed_node = AtPNode(Node("embed"), None)

        def _embed_hook(module: nn.Module, inp: tuple, output: Tensor) -> None:
            acts[embed_node] = output.detach()

        handles.append(self._embed().register_forward_hook(_embed_hook))

        n_heads = self.n_heads
        head_dim = self.head_dim

        for L, block in enumerate(self._layers_list):
            if n_heads > 0 and head_dim is not None and hasattr(block, "self_attn"):
                _L = L
                _n_heads = n_heads
                _head_dim = head_dim

                def _v_proj_hook(
                    module: nn.Module, inp: tuple, output: Tensor,
                    _L: int = _L, _n_heads: int = _n_heads, _head_dim: int = _head_dim,
                ) -> None:
                    # output: (b, s, n_heads * head_dim) — detach and slice per head
                    v = output.detach()
                    b, s, _ = v.shape
                    v_heads = v.reshape(b, s, _n_heads, _head_dim)
                    for h in range(_n_heads):
                        node = AtPNode(Node("attn_head", _L, h), "v")
                        acts[node] = v_heads[:, :, h, :].clone()

                handles.append(block.self_attn.v_proj.register_forward_hook(_v_proj_hook))

            mlp_node = AtPNode(Node("mlp", L), None)
            _mlp_node = mlp_node

            def _down_proj_hook(
                module: nn.Module, inp: tuple, output: Tensor,
                _n: AtPNode = _mlp_node,
            ) -> None:
                acts[_n] = output.detach()

            handles.append(block.mlp.down_proj.register_forward_hook(_down_proj_hook))

            if capture_intermediates:
                _L = L

                def _down_proj_pre_hook(
                    module: nn.Module, args: tuple,
                    _L: int = _L,
                ) -> None:
                    # args[0] is the down_proj INPUT (b, s, d_mlp) — the MLP intermediate
                    intermediates[_L] = args[0].detach().clone()

                handles.append(block.mlp.down_proj.register_forward_pre_hook(_down_proj_pre_hook))

        try:
            with torch.no_grad():
                self._call_model(inputs)
        finally:
            for h in handles:
                h.remove()

        return acts, intermediates

    def _collect_clean_grads(
        self,
        inputs: _Inputs,
        metric: Callable[[Any], Tensor],
        orig_rg: dict[str, bool],
        capture_intermediates: bool = False,
        capture_qk_grads: bool = False,
    ) -> tuple[Any, dict[AtPNode, Tensor], dict[AtPNode, Tensor | None], dict[int, Tensor], dict[int, Tensor], dict[int, Tensor]]:
        """Forward + backward pass capturing node activations AND their gradients.

        For vanilla AtP, we capture the FULL downstream gradient (including
        residual bypass paths). We do this by seeding requires_grad at the
        ACTIVATION level (embed output) rather than on parameters:

          - The embed forward hook replaces the embed output with a detached
            grad-enabled leaf. This seeds the computation graph without touching
            any parameter's requires_grad state.
          - Grad flows through frozen-param Linear ops to their inputs (freezing
            a param only blocks the PARAM's grad accumulation, not grad flow
            through the op to its inputs). All downstream activations inherit
            requires_grad=True automatically.
          - retain_grad() on downstream activation tensors (mlp out, v_proj out)
            lets us read their .grad after backward.
          - No param ever has requires_grad=True during the backward, so no
            param accumulates a .grad — frozen-model contract is upheld.

        This gives the TOTAL gradient w.r.t. each node activation, not
        component-only (which is what EAP's clone trick gives). On a linear
        model, total == component since there's no nonlinearity.

        If capture_intermediates=True, also registers a forward_pre_hook on
        each block.mlp.down_proj to capture the MLP intermediate (the INPUT to
        down_proj — i.e., the post-activation hidden states used for mlp_neuron
        scoring). The intermediate tensor stays in the autograd graph (no
        clone/detach) so its .grad is populated after backward.

        If capture_qk_grads=True, additionally:
          - retain_grad on q_proj and k_proj outputs (for vanilla q/k scoring)
          - retain_grad on o_proj outputs (grad_attn_out for QK fix scoring)

        Returns (model_out, node_acts, node_grads, intermediates,
                 qproj_grads, oproj_grads).
        qproj_grads: layer_idx → full q_proj output (with .grad after backward)
        oproj_grads: layer_idx → o_proj output tensor (with .grad after backward)
        """
        # Params remain frozen (requires_grad=False) — no param re-enable needed.
        # Grad is seeded at the activation level via the embed hook below.

        node_acts: dict[AtPNode, Tensor] = {}
        intermediates: dict[int, Tensor] = {}
        # Storage for q/k proj outputs and o_proj outputs (for QK fix)
        qproj_outs: dict[int, Tensor] = {}   # layer → q_proj output (with retain_grad)
        kproj_outs: dict[int, Tensor] = {}   # layer → k_proj output (with retain_grad)
        oproj_outs: dict[int, Tensor] = {}   # layer → o_proj output (with retain_grad)
        handles: list[Any] = []

        embed_node = AtPNode(Node("embed"), None)

        def _embed_hook(module: nn.Module, inp: tuple, output: Tensor) -> Tensor:
            # Replace the embed output with a grad-enabled leaf tensor.
            # Values are unchanged (detach preserves values); requires_grad seeds
            # the autograd graph without enabling any parameter grad.
            seeded = output.detach().requires_grad_(True)
            seeded.retain_grad()
            node_acts[embed_node] = seeded
            return seeded

        handles.append(self._embed().register_forward_hook(_embed_hook))

        n_heads = self.n_heads
        head_dim = self.head_dim

        for L, block in enumerate(self._layers_list):
            if n_heads > 0 and head_dim is not None and hasattr(block, "self_attn"):
                _L = L
                _n_heads = n_heads
                _head_dim = head_dim

                def _v_proj_hook_grad(
                    module: nn.Module, inp: tuple, output: Tensor,
                    _L: int = _L, _n_heads: int = _n_heads, _head_dim: int = _head_dim,
                ) -> None:
                    # output: (b, s, n_heads * head_dim)
                    # retain_grad on the FULL output tensor; read per-head grad after bwd
                    output.retain_grad()
                    for h in range(_n_heads):
                        node = AtPNode(Node("attn_head", _L, h), "v")
                        # Store a tuple referencing the SAME tensor + head metadata
                        node_acts[node] = (output, h, _n_heads, _head_dim)  # type: ignore[assignment]

                handles.append(block.self_attn.v_proj.register_forward_hook(_v_proj_hook_grad))

                if capture_qk_grads:
                    _L2 = L

                    def _q_proj_hook_grad(
                        module: nn.Module, inp: tuple, output: Tensor,
                        _L: int = _L2,
                    ) -> None:
                        output.retain_grad()
                        qproj_outs[_L] = output

                    def _k_proj_hook_grad(
                        module: nn.Module, inp: tuple, output: Tensor,
                        _L: int = _L2,
                    ) -> None:
                        output.retain_grad()
                        kproj_outs[_L] = output

                    def _o_proj_hook_grad(
                        module: nn.Module, inp: tuple, output: Tensor,
                        _L: int = _L2,
                    ) -> None:
                        output.retain_grad()
                        oproj_outs[_L] = output

                    handles.append(block.self_attn.q_proj.register_forward_hook(_q_proj_hook_grad))
                    handles.append(block.self_attn.k_proj.register_forward_hook(_k_proj_hook_grad))
                    handles.append(block.self_attn.o_proj.register_forward_hook(_o_proj_hook_grad))

            mlp_node = AtPNode(Node("mlp", L), None)
            _mlp_node = mlp_node

            def _down_proj_hook_grad(
                module: nn.Module, inp: tuple, output: Tensor,
                _n: AtPNode = _mlp_node,
            ) -> None:
                output.retain_grad()
                node_acts[_n] = output

            handles.append(block.mlp.down_proj.register_forward_hook(_down_proj_hook_grad))

            if capture_intermediates:
                _L_inter = L

                def _down_proj_pre_hook(
                    module: nn.Module, args: tuple,
                    _L: int = _L_inter,
                ) -> None:
                    # args[0] is the down_proj INPUT (b, s, d_mlp) — the MLP intermediate.
                    # It inherits requires_grad=True from the embed seed flowing through
                    # frozen up_proj. Call retain_grad so we can read .grad after backward.
                    # Do NOT clone/detach — must stay in the live autograd graph.
                    inter = args[0]
                    inter.retain_grad()
                    intermediates[_L] = inter

                handles.append(block.mlp.down_proj.register_forward_pre_hook(_down_proj_pre_hook))

        try:
            with torch.enable_grad():
                out = self._call_model(inputs)
                metric(out).backward()
        finally:
            for h in handles:
                h.remove()
            # Params were never re-enabled (frozen-model contract). No refreeze needed.

        # Extract gradients
        node_grads: dict[AtPNode, Tensor | None] = {}

        # embed grad
        embed_act = node_acts.get(embed_node)
        if embed_act is not None and isinstance(embed_act, Tensor):
            node_grads[embed_node] = embed_act.grad

        # mlp grads
        for L in range(len(self._layers_list)):
            mlp_node = AtPNode(Node("mlp", L), None)
            mlp_act = node_acts.get(mlp_node)
            if mlp_act is not None and isinstance(mlp_act, Tensor):
                node_grads[mlp_node] = mlp_act.grad

        # v grads (from full v_proj output grad, sliced per head)
        if n_heads > 0 and head_dim is not None:
            for L in range(len(self._layers_list)):
                for h in range(n_heads):
                    node = AtPNode(Node("attn_head", L, h), "v")
                    stored = node_acts.get(node)
                    if stored is not None and isinstance(stored, tuple):
                        v_proj_out, h_idx, _n_heads, _head_dim = stored
                        if v_proj_out.grad is not None:
                            b, s, _ = v_proj_out.shape
                            g = v_proj_out.grad.reshape(b, s, _n_heads, _head_dim)
                            node_grads[node] = g[:, :, h_idx, :].detach().clone()
                        else:
                            node_grads[node] = None
                    else:
                        node_grads[node] = None

        return out, node_acts, node_grads, intermediates, qproj_outs, kproj_outs, oproj_outs

    # ------------------------------------------------------------------
    # QK fix analytic scoring
    # ------------------------------------------------------------------

    def _compute_qk_fix_scores(
        self,
        clean_qk_data: dict[int, dict[str, Any]],
        corr_qk_data: dict[int, dict[str, Any]],
        oproj_outs: dict[int, Tensor],
    ) -> dict[AtPNode, float]:
        """Compute QK-fixed scores for all q/k nodes.

        For each layer L and head h:
          Query node:
            Δq_h = q_corr_post_rope[:, h] - q_clean_post_rope[:, h]  (b, s, head_dim)
            Q_new = Q_clean_h + Δq_h = q_corr_post_rope[:, h]
            scores_new = Q_new @ K_clean_h.T * scaling + attn_mask_h
            pattern_new = softmax(scores_new, dim=-1)
            Δpattern_h = pattern_new - pattern_clean_h
            V_clean_kv = v_clean[:, kv_h]   (b, s, head_dim)
            Δz_h = Δpattern_h @ V_clean_kv  (b, q_pos, head_dim)
            Δhead_out_h = Δz_h @ W_O_h.T   (b, q_pos, d_model)
            grad_attn_out = oproj_outs[L].grad  (b, s, d_model) — shared across heads
            score = (Δhead_out_h * grad_attn_out).sum()

          Key node: symmetric — replace K_clean_h with K_corr_h in scores recomputation.

        GQA: kv_h = _kv_head_for(h, n_heads, n_kv_heads)
        """
        n_heads = self.n_heads
        n_kv_heads = self.n_kv_heads
        head_dim = self.head_dim
        assert head_dim is not None

        scores: dict[AtPNode, float] = {}

        for L in range(self.n_layers):
            clean = clean_qk_data.get(L)
            corr = corr_qk_data.get(L)
            if clean is None or corr is None:
                for h in range(n_heads):
                    scores[AtPNode(Node("attn_head", L, h), "q")] = 0.0
                    scores[AtPNode(Node("attn_head", L, h), "k")] = 0.0
                continue

            q_clean = clean.get("q_post")   # (b, n_heads, s, head_dim)
            k_clean = clean.get("k_post")   # (b, n_kv_heads, s, head_dim)
            v_clean = clean.get("v")        # (b, n_kv_heads, s, head_dim)
            pattern_clean = clean.get("pattern")  # (b, n_heads, s, s)
            attn_mask = clean.get("attn_mask")    # (b, 1, s, s) or None
            q_corr = corr.get("q_post")
            k_corr = corr.get("k_post")

            # Get grad_attn_out: o_proj output gradient, shared across heads
            o_out = oproj_outs.get(L)
            if o_out is None or o_out.grad is None:
                for h in range(n_heads):
                    scores[AtPNode(Node("attn_head", L, h), "q")] = 0.0
                    scores[AtPNode(Node("attn_head", L, h), "k")] = 0.0
                continue

            grad_attn_out = o_out.grad.detach()  # (b, s, d_model)

            # W_O for this layer
            block = self._layers_list[L]
            W_O = block.self_attn.o_proj.weight  # (d_model, n_heads*head_dim)

            scaling = 1.0 / math.sqrt(head_dim)

            # If no pattern_clean (toy model without output_attentions),
            # we need to compute it ourselves from clean Q, K, mask
            # We'll handle this by a unified recompute path

            for h in range(n_heads):
                kv_h = _kv_head_for(h, n_heads, n_kv_heads)
                W_O_h = W_O[:, h * head_dim:(h + 1) * head_dim]  # (d_model, head_dim)

                if (q_clean is None or k_clean is None or v_clean is None
                        or q_corr is None or k_corr is None):
                    scores[AtPNode(Node("attn_head", L, h), "q")] = 0.0
                    scores[AtPNode(Node("attn_head", L, h), "k")] = 0.0
                    continue

                Q_clean_h = q_clean[:, h, :, :]     # (b, s, head_dim)  [note: after transpose→(b,n,s,hd)]
                K_clean_h = k_clean[:, kv_h, :, :]  # (b, s, head_dim)
                V_clean_h = v_clean[:, kv_h, :, :]  # (b, s, head_dim)
                Q_corr_h = q_corr[:, h, :, :]
                K_corr_h = k_corr[:, kv_h, :, :]

                # Compute clean pattern (needed as baseline)
                if pattern_clean is not None:
                    pat_clean_h = pattern_clean[:, h, :, :]  # (b, s, s)
                else:
                    # No output_attentions — recompute from clean Q/K
                    scores_clean = (Q_clean_h @ K_clean_h.transpose(-2, -1)) * scaling
                    if attn_mask is not None:
                        scores_clean = scores_clean + attn_mask[:, 0, :, :]
                    pat_clean_h = torch.softmax(scores_clean.float(), dim=-1).to(Q_clean_h.dtype)

                # ---- QUERY node: replace Q_clean_h with Q_corr_h ----
                scores_q = (Q_corr_h @ K_clean_h.transpose(-2, -1)) * scaling
                if attn_mask is not None:
                    scores_q = scores_q + attn_mask[:, 0, :, :]
                pat_q = torch.softmax(scores_q.float(), dim=-1).to(Q_clean_h.dtype)
                delta_pattern_q = pat_q - pat_clean_h                  # (b, s, s)
                delta_z_q = delta_pattern_q @ V_clean_h                # (b, q, head_dim)
                delta_head_out_q = delta_z_q @ W_O_h.T                # (b, q, d_model)
                score_q = float((delta_head_out_q * grad_attn_out).sum().item())
                scores[AtPNode(Node("attn_head", L, h), "q")] = score_q

                # ---- KEY node: replace K_clean_h with K_corr_h ----
                scores_k = (Q_clean_h @ K_corr_h.transpose(-2, -1)) * scaling
                if attn_mask is not None:
                    scores_k = scores_k + attn_mask[:, 0, :, :]
                pat_k = torch.softmax(scores_k.float(), dim=-1).to(Q_clean_h.dtype)
                delta_pattern_k = pat_k - pat_clean_h                  # (b, s, s)
                delta_z_k = delta_pattern_k @ V_clean_h                # (b, q, head_dim)
                delta_head_out_k = delta_z_k @ W_O_h.T                # (b, q, d_model)
                score_k = float((delta_head_out_k * grad_attn_out).sum().item())
                scores[AtPNode(Node("attn_head", L, h), "k")] = score_k

        return scores

    def _compute_vanilla_qk_scores(
        self,
        clean_qk_data: dict[int, dict[str, Any]],
        corr_qk_data: dict[int, dict[str, Any]],
        qproj_outs: dict[int, Tensor],
        kproj_outs: dict[int, Tensor],
    ) -> dict[AtPNode, float]:
        """Vanilla q/k scoring: Σ(Δq_h ⊙ grad_q_h) in head_dim space.

        Uses the gradient w.r.t. q_proj output (per-head slice) and the
        delta from corrupted minus clean q_proj outputs (pre-RoPE).
        """
        n_heads = self.n_heads
        n_kv_heads = self.n_kv_heads
        head_dim = self.head_dim
        assert head_dim is not None

        scores: dict[AtPNode, float] = {}

        for L in range(self.n_layers):
            clean = clean_qk_data.get(L)
            corr = corr_qk_data.get(L)

            q_proj_out = qproj_outs.get(L)
            k_proj_out = kproj_outs.get(L)

            for h in range(n_heads):
                kv_h = _kv_head_for(h, n_heads, n_kv_heads)

                # Q vanilla
                if (clean is not None and corr is not None
                        and q_proj_out is not None and q_proj_out.grad is not None
                        and "q_pre" in clean and "q_pre" in corr):
                    q_clean_pre = clean["q_pre"]  # (b, s, n_heads*head_dim)
                    q_corr_pre = corr["q_pre"]
                    b, s, _ = q_clean_pre.shape
                    q_c = q_clean_pre.view(b, s, n_heads, head_dim)[:, :, h, :]
                    q_r = q_corr_pre.view(b, s, n_heads, head_dim)[:, :, h, :]
                    delta_q = q_r - q_c  # (b, s, head_dim)
                    grad_q = q_proj_out.grad.reshape(b, s, n_heads, head_dim)[:, :, h, :]
                    scores[AtPNode(Node("attn_head", L, h), "q")] = float(
                        (delta_q * grad_q.detach()).sum().item()
                    )
                else:
                    scores[AtPNode(Node("attn_head", L, h), "q")] = 0.0

                # K vanilla
                if (clean is not None and corr is not None
                        and k_proj_out is not None and k_proj_out.grad is not None
                        and "k_pre" in clean and "k_pre" in corr):
                    k_clean_pre = clean["k_pre"]  # (b, s, n_kv_heads*head_dim)
                    k_corr_pre = corr["k_pre"]
                    b, s, _ = k_clean_pre.shape
                    k_c = k_clean_pre.view(b, s, n_kv_heads, head_dim)[:, :, kv_h, :]
                    k_r = k_corr_pre.view(b, s, n_kv_heads, head_dim)[:, :, kv_h, :]
                    delta_k = k_r - k_c  # (b, s, head_dim)
                    grad_k = k_proj_out.grad.reshape(b, s, n_kv_heads, head_dim)[:, :, kv_h, :]
                    scores[AtPNode(Node("attn_head", L, h), "k")] = float(
                        (delta_k * grad_k.detach()).sum().item()
                    )
                else:
                    scores[AtPNode(Node("attn_head", L, h), "k")] = 0.0

        return scores

    # ------------------------------------------------------------------
    # TransformerLens-specific collect methods
    # ------------------------------------------------------------------

    def _cache_node_acts_tl(self, inputs: Tensor) -> dict[AtPNode, Tensor]:
        """No-grad forward caching node activations via TL native hooks.

        Captures:
          hook_embed            → AtPNode(Node("embed"), None)
          blocks.{L}.hook_mlp_out → AtPNode(Node("mlp", L), None)
          blocks.{L}.attn.hook_v[:,:,h,:] → AtPNode(Node("attn_head",L,h), "v")

        Returns detached activation tensors.
        """
        acts: dict[AtPNode, Tensor] = {}
        n_heads = self.n_heads
        n_layers = self.n_layers

        def _embed_hook(tensor: Tensor, hook: Any) -> None:
            acts[AtPNode(Node("embed"), None)] = tensor.detach()

        fwd_hooks: list[tuple[str, Any]] = [("hook_embed", _embed_hook)]

        for L in range(n_layers):
            mlp_out_key = f"blocks.{L}.hook_mlp_out"

            def _mlp_out_hook(tensor: Tensor, hook: Any, _L: int = L) -> None:
                acts[AtPNode(Node("mlp", _L), None)] = tensor.detach()

            fwd_hooks.append((mlp_out_key, _mlp_out_hook))

            # v per-head: hook_v shape is (b, s, n_heads, head_dim)
            v_key = f"blocks.{L}.attn.hook_v"

            def _v_hook(tensor: Tensor, hook: Any, _L: int = L) -> None:
                for _h in range(n_heads):
                    acts[AtPNode(Node("attn_head", _L, _h), "v")] = (
                        tensor[:, :, _h, :].detach().clone()
                    )

            fwd_hooks.append((v_key, _v_hook))

        with torch.no_grad():
            self.model.run_with_hooks(inputs, fwd_hooks=fwd_hooks)  # type: ignore[attr-defined]

        return acts

    def _collect_clean_grads_tl(
        self,
        inputs: Tensor,
        metric: Callable[[Any], Tensor],
    ) -> tuple[dict[AtPNode, Tensor], dict[AtPNode, Tensor | None], dict[int, Tensor], dict[int, Tensor]]:
        """Forward + backward collecting clean node activations AND their gradients
        via TL native hooks.

        Gradient seeding: hook_embed returns a detached leaf with requires_grad_(True),
        seeding the computation graph at activation level (no param grad leakage).
        retain_grad() on mlp_out, hook_v, hook_q, hook_k tensors lets us read .grad
        after backward.

        Vanilla AtP q/k scoring on TL path:
          hook_q[:,:,h,:] and hook_k[:,:,h,:] carry gradients from the retain_grad
          substitution. Scoring: Σ(Δq_h ⊙ grad_q_h) in head_dim space.
          (Full QK-fix with attention-pattern recomputation is not implemented on the
          TL path; the HF path carries the full QK fix.)

        Returns (node_acts, node_grads, qhook_outs, khook_outs):
          node_acts: AtPNode → Tensor (the live tensor in the autograd graph)
          node_grads: AtPNode → Tensor | None (.grad read after backward)
          qhook_outs: layer_idx → full (b,s,n_heads,head_dim) hook_q output (with retain_grad)
          khook_outs: layer_idx → full (b,s,n_heads,head_dim) hook_k output (with retain_grad)
        """
        node_acts: dict[AtPNode, Tensor] = {}
        qhook_outs: dict[int, Tensor] = {}
        khook_outs: dict[int, Tensor] = {}
        # Storage for v_hook outputs (full tensor, retain_grad, per-layer)
        v_hook_outs: dict[int, Tensor] = {}
        n_heads = self.n_heads
        n_layers = self.n_layers

        embed_node = AtPNode(Node("embed"), None)

        def _embed_hook_grad(tensor: Tensor, hook: Any) -> Tensor:
            # Seed computation graph at activation level — detach then requires_grad
            seeded = tensor.detach().requires_grad_(True)
            seeded.retain_grad()
            node_acts[embed_node] = seeded
            return seeded

        fwd_hooks: list[tuple[str, Any]] = [("hook_embed", _embed_hook_grad)]

        for L in range(n_layers):
            mlp_out_key = f"blocks.{L}.hook_mlp_out"

            def _mlp_out_hook_grad(tensor: Tensor, hook: Any, _L: int = L) -> Tensor:
                tensor.retain_grad()
                node_acts[AtPNode(Node("mlp", _L), None)] = tensor
                return tensor

            fwd_hooks.append((mlp_out_key, _mlp_out_hook_grad))

            # v per-head via hook_v
            v_key = f"blocks.{L}.attn.hook_v"

            def _v_hook_grad(
                tensor: Tensor, hook: Any, _L: int = L,
                _store: dict = v_hook_outs,
            ) -> Tensor:
                tensor.retain_grad()
                _store[_L] = tensor
                # Store per-head refs (the full tensor; we'll read .grad per head later)
                for _h in range(n_heads):
                    node_acts[AtPNode(Node("attn_head", _L, _h), "v")] = (tensor, _h)  # type: ignore[assignment]
                return tensor

            fwd_hooks.append((v_key, _v_hook_grad))

            # q/k hooks for vanilla q/k scoring
            q_key = f"blocks.{L}.attn.hook_q"
            k_key = f"blocks.{L}.attn.hook_k"

            def _q_hook_grad(
                tensor: Tensor, hook: Any, _L: int = L,
                _store: dict = qhook_outs,
            ) -> Tensor:
                tensor.retain_grad()
                _store[_L] = tensor
                return tensor

            def _k_hook_grad(
                tensor: Tensor, hook: Any, _L: int = L,
                _store: dict = khook_outs,
            ) -> Tensor:
                tensor.retain_grad()
                _store[_L] = tensor
                return tensor

            fwd_hooks.append((q_key, _q_hook_grad))
            fwd_hooks.append((k_key, _k_hook_grad))

        with torch.enable_grad():
            logits = self.model.run_with_hooks(inputs, fwd_hooks=fwd_hooks)  # type: ignore[attr-defined]
            metric(logits).backward()

        # Extract gradients
        node_grads: dict[AtPNode, Tensor | None] = {}

        # embed grad
        embed_act = node_acts.get(embed_node)
        if embed_act is not None and isinstance(embed_act, Tensor):
            node_grads[embed_node] = embed_act.grad

        # mlp grads
        for L in range(n_layers):
            mlp_node = AtPNode(Node("mlp", L), None)
            mlp_act = node_acts.get(mlp_node)
            if mlp_act is not None and isinstance(mlp_act, Tensor):
                node_grads[mlp_node] = mlp_act.grad

        # v grads (from full v_hook output grad, sliced per head)
        for L in range(n_layers):
            for h in range(n_heads):
                node = AtPNode(Node("attn_head", L, h), "v")
                stored = node_acts.get(node)
                if stored is not None and isinstance(stored, tuple):
                    v_full, h_idx = stored
                    if v_full.grad is not None:
                        node_grads[node] = v_full.grad[:, :, h_idx, :].detach().clone()
                    else:
                        node_grads[node] = None
                else:
                    node_grads[node] = None

        return node_acts, node_grads, qhook_outs, khook_outs

    def _compute_vanilla_qk_scores_tl(
        self,
        clean_qhook_outs: dict[int, Tensor],
        clean_khook_outs: dict[int, Tensor],
        corr_qhook_outs: dict[int, Tensor],
        corr_khook_outs: dict[int, Tensor],
    ) -> dict[AtPNode, float]:
        """Vanilla q/k scoring on the TL path.

        TL hook_q/hook_k have shape (b, s, n_heads, head_dim).
        score(q_h) = Σ(Δq_h ⊙ grad_q_h)  where Δq_h = corr_q_h − clean_q_h
        score(k_h) = Σ(Δk_h ⊙ grad_k_h)

        NOTE: This is vanilla Δq·grad, NOT the QK fix (attention-pattern recomputation).
        The QK fix is implemented on the HF path only. On TL, q/k scores are a linear
        first-order approximation without the softmax-pattern correction.
        """
        scores: dict[AtPNode, float] = {}
        n_heads = self.n_heads

        for L in range(self.n_layers):
            clean_q = clean_qhook_outs.get(L)
            corr_q = corr_qhook_outs.get(L)
            clean_k = clean_khook_outs.get(L)
            corr_k = corr_khook_outs.get(L)

            for h in range(n_heads):
                # Q score
                if (clean_q is not None and corr_q is not None
                        and clean_q.grad is not None):
                    delta_q = (corr_q[:, :, h, :] - clean_q[:, :, h, :]).detach()
                    grad_q = clean_q.grad[:, :, h, :].detach()
                    scores[AtPNode(Node("attn_head", L, h), "q")] = float(
                        (delta_q * grad_q).sum().item()
                    )
                else:
                    scores[AtPNode(Node("attn_head", L, h), "q")] = 0.0

                # K score
                if (clean_k is not None and corr_k is not None
                        and clean_k.grad is not None):
                    delta_k = (corr_k[:, :, h, :] - clean_k[:, :, h, :]).detach()
                    grad_k = clean_k.grad[:, :, h, :].detach()
                    scores[AtPNode(Node("attn_head", L, h), "k")] = float(
                        (delta_k * grad_k).sum().item()
                    )
                else:
                    scores[AtPNode(Node("attn_head", L, h), "k")] = 0.0

        return scores

    def _run_tl(
        self,
        clean_inputs: Tensor,
        corrupted_inputs: Tensor,
        metric: Callable[[Any], Tensor],
        *,
        graddrop: bool = False,
    ) -> AtPResult:
        """AtP* on a TransformerLens HookedTransformer.

        Vanilla AtP + vanilla q/k scoring (NOT full QK fix — softmax-pattern
        recomputation is not implemented on the TL path; use the HF path for that).

        Steps:
          1. Enable TL hooks needed; restore in finally.
          2. Corrupted forward (no grad): cache node activations.
          3. Clean forward + backward: retain_grad on activations, seed via embed hook.
          4. score(node) = Σ(Δact ⊙ grad); GradDrop: Σ|per-pos contribution|.
          5. q/k: vanilla Σ(Δq_h ⊙ grad_q_h) in head_dim space.
          6. Return AtPResult. neurons=False (TL path, no per-neuron scoring).
        """
        # Save and enable required TL flags
        _prior_attn_result = self.model.cfg.use_attn_result  # type: ignore[attr-defined]
        self.model.set_use_attn_result(True)  # type: ignore[attr-defined]

        try:
            # Step 1: cache corrupted activations (no grad)
            corrupted_acts = self._cache_node_acts_tl(corrupted_inputs)

            # Step 2: cache corrupted q/k hook outputs for vanilla scoring (no-grad pass)
            corr_qhook: dict[int, Tensor] = {}
            corr_khook: dict[int, Tensor] = {}
            n_heads = self.n_heads

            # Actually capture q/k from corrupted pass in a single run
            # TL calls hooks as hook(tensor, hook=hook_point) — use **kwargs to absorb it.
            corr_qk_capture: dict[str, dict[int, Tensor]] = {"q": {}, "k": {}}

            corr_qk_fwd_hooks: list[tuple[str, Any]] = []
            for L in range(self.n_layers):
                def _cq(t: Tensor, _L: int = L, **kwargs: Any) -> None:
                    corr_qk_capture["q"][_L] = t.detach()
                def _ck(t: Tensor, _L: int = L, **kwargs: Any) -> None:
                    corr_qk_capture["k"][_L] = t.detach()
                corr_qk_fwd_hooks.append((f"blocks.{L}.attn.hook_q", _cq))
                corr_qk_fwd_hooks.append((f"blocks.{L}.attn.hook_k", _ck))

            with torch.no_grad():
                self.model.run_with_hooks(corrupted_inputs, fwd_hooks=corr_qk_fwd_hooks)  # type: ignore[attr-defined]

            corr_qhook = corr_qk_capture["q"]
            corr_khook = corr_qk_capture["k"]

            # Step 3: clean forward + backward with grad seeding
            clean_node_acts, clean_node_grads, clean_qhook, clean_khook = (
                self._collect_clean_grads_tl(clean_inputs, metric)
            )

            # Step 4: build clean_acts dict (detached values for Δact computation)
            clean_acts: dict[AtPNode, Tensor] = {}

            embed_node = AtPNode(Node("embed"), None)
            embed_act = clean_node_acts.get(embed_node)
            if embed_act is not None and isinstance(embed_act, Tensor):
                clean_acts[embed_node] = embed_act.detach()

            for L in range(self.n_layers):
                mlp_node = AtPNode(Node("mlp", L), None)
                mlp_act = clean_node_acts.get(mlp_node)
                if mlp_act is not None and isinstance(mlp_act, Tensor):
                    clean_acts[mlp_node] = mlp_act.detach()

            # v clean acts (extract from stored (tensor, h_idx) tuples)
            for L in range(self.n_layers):
                for h in range(n_heads):
                    node = AtPNode(Node("attn_head", L, h), "v")
                    stored = clean_node_acts.get(node)
                    if stored is not None and isinstance(stored, tuple):
                        v_full, h_idx = stored
                        clean_acts[node] = v_full.detach()[:, :, h_idx, :].clone()

            # Step 5: q/k vanilla scoring
            qk_scores = self._compute_vanilla_qk_scores_tl(
                clean_qhook, clean_khook, corr_qhook, corr_khook
            )

            # Step 6: score each node
            scores: dict[AtPNode, float] = {}
            all_nodes = enumerate_nodes(self.n_layers, n_heads, d_mlp=None)

            for atp_node in all_nodes:
                if atp_node.slot in ("q", "k"):
                    scores[atp_node] = qk_scores.get(atp_node, 0.0)
                    continue

                if atp_node.node.kind == "mlp_neuron":
                    # neurons not supported on TL path
                    scores[atp_node] = 0.0
                    continue

                corr_act = corrupted_acts.get(atp_node)
                cln_act = clean_acts.get(atp_node)
                grad = clean_node_grads.get(atp_node)

                if corr_act is None or cln_act is None or grad is None:
                    scores[atp_node] = 0.0
                    continue

                delta = corr_act - cln_act
                if graddrop:
                    per_pos = (delta * grad).sum(dim=tuple(range(2, delta.dim())))
                    score = float(per_pos.abs().sum().item())
                else:
                    score = float((delta * grad).sum().item())
                scores[atp_node] = score

            return AtPResult(scores)
        finally:
            self.model.set_use_attn_result(_prior_attn_result)  # type: ignore[attr-defined]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        clean_inputs: _Inputs,
        corrupted_inputs: _Inputs,
        metric: Callable[[Any], Tensor],
        *,
        neurons: bool = False,
        graddrop: bool = False,
        qk_fix: bool = True,
    ) -> AtPResult:
        """Compute AtP* node scores.

        Steps:
          1. Freeze params + eval (save/restore in try/finally).
          2. Corrupted forward (no grad): cache node activations + QK data.
          3. Clean forward + backward: retain_grad on node activation tensors,
             read .grad after backward. Also captures o_proj grads for QK fix.
          4. Clean forward (no grad): capture clean QK data (Q/K/V/pattern).
          5. score(node) = float((Δact * grad).sum()) — sum over ALL dims.
          6. q/k nodes: QK fix (attn-pattern recomputation) or vanilla Δq·grad.
          7. Return AtPResult over enumerate_nodes.

        TL path: dispatches to _run_tl() for TransformerLens HookedTransformer models.
          - Vanilla AtP on embed/mlp/v nodes.
          - Vanilla q/k scoring (Σ(Δq_h ⊙ grad_q_h)); full QK fix not implemented on TL.
          - neurons=True is ignored on TL (mlp_neuron nodes not scored).
          - qk_fix=True is silently used as vanilla q/k on TL.
        """
        if self._tl:
            was_training, orig_rg = self._freeze_eval()
            try:
                return self._run_tl(
                    clean_inputs,  # type: ignore[arg-type]
                    corrupted_inputs,  # type: ignore[arg-type]
                    metric,
                    graddrop=graddrop,
                )
            finally:
                self._restore(was_training, orig_rg)

        need_qk = self.n_heads > 0 and self.head_dim is not None
        was_training, orig_rg = self._freeze_eval()
        try:
            # Step 1: cache corrupted activations (no grad)
            corrupted_acts, corrupted_intermediates = self._cache_node_acts(
                corrupted_inputs, capture_intermediates=neurons
            )

            # Step 2: cache corrupted QK data (for QK fix and vanilla)
            corr_qk_data: dict[int, dict[str, Any]] = {}
            if need_qk:
                corr_qk_data = self._cache_qk_data(corrupted_inputs, compute_pattern=False)

            # Step 3: clean forward + backward to get grads
            _, clean_node_acts, clean_node_grads, clean_intermediates, qproj_outs, kproj_outs, oproj_outs = (
                self._collect_clean_grads(
                    clean_inputs, metric, orig_rg,
                    capture_intermediates=neurons,
                    capture_qk_grads=need_qk,
                )
            )

            # Step 4: cache clean QK data (post-RoPE Q/K/V + pattern)
            clean_qk_data: dict[int, dict[str, Any]] = {}
            if need_qk:
                clean_qk_data = self._cache_qk_data(clean_inputs, compute_pattern=True)
                # Copy pre-RoPE q_pre/k_pre into clean_qk_data for vanilla scoring
                # (already captured by _cache_qk_data)

            # Step 5: compute clean activations for embed and mlp nodes
            # (we need clean acts to compute Δact = corr - clean)
            # For embed and mlp: node_acts stores the actual tensor; detach to get clean value
            clean_acts: dict[AtPNode, Tensor] = {}

            embed_node = AtPNode(Node("embed"), None)
            embed_act = clean_node_acts.get(embed_node)
            if embed_act is not None and isinstance(embed_act, Tensor):
                clean_acts[embed_node] = embed_act.detach()

            for L in range(self.n_layers):
                mlp_node = AtPNode(Node("mlp", L), None)
                mlp_act = clean_node_acts.get(mlp_node)
                if mlp_act is not None and isinstance(mlp_act, Tensor):
                    clean_acts[mlp_node] = mlp_act.detach()

            # For v nodes: extract the clean act from stored tuple
            if self.n_heads > 0 and self.head_dim is not None:
                for L in range(self.n_layers):
                    for h in range(self.n_heads):
                        node = AtPNode(Node("attn_head", L, h), "v")
                        stored = clean_node_acts.get(node)
                        if stored is not None and isinstance(stored, tuple):
                            v_proj_out, h_idx, _n_heads, _head_dim = stored
                            b, s, _ = v_proj_out.shape
                            v_heads = v_proj_out.detach().reshape(b, s, _n_heads, _head_dim)
                            clean_acts[node] = v_heads[:, :, h_idx, :].clone()

            # Step 6: score each node
            scores: dict[AtPNode, float] = {}
            d_mlp = getattr(self.resolver, "d_mlp", None) if self.resolver is not None else None
            all_nodes = enumerate_nodes(
                self.n_layers, self.n_heads, d_mlp=d_mlp if neurons else None
            )

            # Compute q/k scores once (either QK-fix or vanilla)
            qk_scores: dict[AtPNode, float] = {}
            if need_qk:
                if qk_fix:
                    qk_scores = self._compute_qk_fix_scores(
                        clean_qk_data, corr_qk_data, oproj_outs
                    )
                else:
                    qk_scores = self._compute_vanilla_qk_scores(
                        clean_qk_data, corr_qk_data, qproj_outs, kproj_outs
                    )

            for atp_node in all_nodes:
                if atp_node.slot in ("q", "k"):
                    scores[atp_node] = qk_scores.get(atp_node, 0.0)
                    continue

                inner_node = atp_node.node

                if inner_node.kind == "mlp_neuron":
                    # Neuron-level score: Σ_pos (Δinter_n · grad_inter_n)
                    L = inner_node.layer
                    n = inner_node.neuron
                    clean_inter = clean_intermediates.get(L)
                    corr_inter = corrupted_intermediates.get(L)
                    if clean_inter is None or corr_inter is None or clean_inter.grad is None:
                        scores[atp_node] = 0.0
                        continue
                    delta_n = (corr_inter - clean_inter.detach())[..., n]
                    grad_n = clean_inter.grad[..., n]
                    if graddrop:
                        # per-position contribution is already a scalar (feature dim selected)
                        per_pos = delta_n * grad_n  # (b, s)
                        scores[atp_node] = float(per_pos.abs().sum().item())
                    else:
                        scores[atp_node] = float((delta_n * grad_n).sum().item())
                    continue

                corr_act = corrupted_acts.get(atp_node)
                cln_act = clean_acts.get(atp_node)
                grad = clean_node_grads.get(atp_node)

                if corr_act is None or cln_act is None or grad is None:
                    scores[atp_node] = 0.0
                    continue

                delta = corr_act - cln_act
                if graddrop:
                    # GradDrop: Σ_pos |c[pos]| where c[pos] = Σ_features Δact[pos]·grad[pos]
                    # Sum over feature dims (all dims after batch+seq) to get per-position scalar,
                    # then take abs before summing over positions.
                    per_pos = (delta * grad).sum(dim=tuple(range(2, delta.dim())))  # (b, s)
                    score = float(per_pos.abs().sum().item())
                else:
                    score = float((delta * grad).sum().item())
                scores[atp_node] = score

            return AtPResult(scores)
        finally:
            self._restore(was_training, orig_rg)

    def bruteforce_node_scores(
        self,
        clean_inputs: _Inputs,
        corrupted_inputs: _Inputs,
        metric: Callable[[Any], Tensor],
        nodes: list[AtPNode],
    ) -> dict[AtPNode, float]:
        """Independent ground truth: for each node, REPLACE its activation with
        the corrupted value via a forward hook, run clean inputs, measure metric delta.

        This is a real forward intervention — completely independent of the analytic
        path. On a linear model, this must equal the AtP score at 1e-4.

        Replace semantics (not add): the node's activation is substituted wholesale
        with its corrupted-forward value. This matches "what if this node had fired
        as in the corrupted run?"

        q nodes: patch q_proj output head-h slice clean→corrupted (real forward intervention).
        k nodes: patch k_proj output head-h slice clean→corrupted (real forward intervention).
        """
        n_heads = self.n_heads
        n_kv_heads = self.n_kv_heads
        head_dim = self.head_dim
        was_training, orig_rg = self._freeze_eval()
        try:
            with torch.no_grad():
                # Baseline: clean metric with no patching
                clean_out = self._call_model(clean_inputs)
                clean_metric = metric(clean_out).item()

                # Cache corrupted activations for all nodes (including intermediates
                # in case neuron nodes are present in the list)
                has_neuron_nodes = any(n.node.kind == "mlp_neuron" for n in nodes)
                corrupted_acts, corrupted_intermediates = self._cache_node_acts(
                    corrupted_inputs, capture_intermediates=has_neuron_nodes
                )

                # For q/k nodes: cache corrupted q_proj/k_proj outputs
                corr_qk_data: dict[int, dict[str, Any]] = {}
                has_qk_nodes = any(n.slot in ("q", "k") for n in nodes)
                if has_qk_nodes and n_heads > 0 and head_dim is not None:
                    corr_qk_data = self._cache_qk_data(corrupted_inputs, compute_pattern=False)

            scores: dict[AtPNode, float] = {}

            for atp_node in nodes:
                inner_node = atp_node.node
                handles: list[Any] = []

                if atp_node.slot == "q" and inner_node.kind == "attn_head":
                    # Brute-force q node: replace head h's q_proj output with corrupted value
                    L = inner_node.layer
                    h = inner_node.head
                    cap = corr_qk_data.get(L)
                    if cap is None or "q_pre" not in cap:
                        scores[atp_node] = 0.0
                        continue
                    q_corr_pre = cap["q_pre"]  # (b, s, n_heads*head_dim)
                    block = self._layers_list[L]
                    _h = h
                    _n_heads = n_heads
                    _hd = head_dim
                    _corr = q_corr_pre

                    def _q_proj_replace_hook(
                        module: nn.Module, inp: tuple, output: Tensor,
                        _h: int = _h, _nh: int = _n_heads, _hd: int = _hd,
                        _c: Tensor = _corr,
                    ) -> Tensor:
                        b, s, _ = output.shape
                        out = output.clone().reshape(b, s, _nh, _hd)
                        corr_h = _c.view(b, s, _nh, _hd)
                        out[:, :, _h, :] = corr_h[:, :, _h, :]
                        return out.reshape(b, s, _nh * _hd)

                    handles.append(
                        block.self_attn.q_proj.register_forward_hook(_q_proj_replace_hook)
                    )

                elif atp_node.slot == "k" and inner_node.kind == "attn_head":
                    # Brute-force k node: replace kv_h's k_proj output with corrupted value
                    L = inner_node.layer
                    h = inner_node.head
                    kv_h = _kv_head_for(h, n_heads, n_kv_heads)
                    cap = corr_qk_data.get(L)
                    if cap is None or "k_pre" not in cap:
                        scores[atp_node] = 0.0
                        continue
                    k_corr_pre = cap["k_pre"]  # (b, s, n_kv_heads*head_dim)
                    block = self._layers_list[L]
                    _kv_h = kv_h
                    _n_kv = n_kv_heads
                    _hd = head_dim
                    _corr = k_corr_pre

                    def _k_proj_replace_hook(
                        module: nn.Module, inp: tuple, output: Tensor,
                        _kv_h: int = _kv_h, _nkv: int = _n_kv, _hd: int = _hd,
                        _c: Tensor = _corr,
                    ) -> Tensor:
                        b, s, _ = output.shape
                        out = output.clone().reshape(b, s, _nkv, _hd)
                        corr_kv = _c.view(b, s, _nkv, _hd)
                        out[:, :, _kv_h, :] = corr_kv[:, :, _kv_h, :]
                        return out.reshape(b, s, _nkv * _hd)

                    handles.append(
                        block.self_attn.k_proj.register_forward_hook(_k_proj_replace_hook)
                    )

                elif inner_node.kind == "mlp_neuron":
                    # Patch the MLP intermediate (down_proj INPUT) for neuron n only.
                    # Use a forward_pre_hook so we're patching exactly the same point
                    # that the analytic path reads (down_proj INPUT = mlp_neuron site).
                    L = inner_node.layer
                    n = inner_node.neuron
                    corr_inter = corrupted_intermediates.get(L)
                    if corr_inter is None:
                        scores[atp_node] = 0.0
                        continue
                    block = self._layers_list[L]
                    _corr_inter = corr_inter
                    _n = n

                    def _neuron_pre_hook(
                        module: nn.Module, args: tuple,
                        _c: Tensor = _corr_inter, _idx: int = _n,
                    ) -> tuple:
                        # Replace neuron _idx in the intermediate with the corrupted value
                        inter = args[0].clone()
                        inter[..., _idx] = _c[..., _idx]
                        return (inter,)

                    handles.append(
                        block.mlp.down_proj.register_forward_pre_hook(_neuron_pre_hook)
                    )

                elif inner_node.kind == "embed":
                    corr_act = corrupted_acts.get(atp_node)
                    if corr_act is None:
                        scores[atp_node] = 0.0
                        continue
                    _corr = corr_act

                    def _embed_replace_hook(
                        module: nn.Module, inp: tuple, output: Tensor,
                        _c: Tensor = _corr,
                    ) -> Tensor:
                        return _c

                    handles.append(
                        self._embed().register_forward_hook(_embed_replace_hook)
                    )

                elif inner_node.kind == "mlp":
                    corr_act = corrupted_acts.get(atp_node)
                    if corr_act is None:
                        scores[atp_node] = 0.0
                        continue
                    L = inner_node.layer
                    _corr = corr_act
                    block = self._layers_list[L]

                    def _mlp_replace_hook(
                        module: nn.Module, inp: tuple, output: Tensor,
                        _c: Tensor = _corr,
                    ) -> Tensor:
                        return _c

                    handles.append(
                        block.mlp.down_proj.register_forward_hook(_mlp_replace_hook)
                    )

                elif inner_node.kind == "attn_head" and atp_node.slot == "v":
                    corr_act = corrupted_acts.get(atp_node)
                    if corr_act is None:
                        scores[atp_node] = 0.0
                        continue
                    L = inner_node.layer
                    h = inner_node.head
                    _corr_head = corr_act  # (b, s, head_dim)
                    _n_heads = self.n_heads
                    _head_dim = self.head_dim
                    block = self._layers_list[L]
                    if not hasattr(block, "self_attn"):
                        scores[atp_node] = 0.0
                        continue

                    def _v_proj_replace_hook(
                        module: nn.Module, inp: tuple, output: Tensor,
                        _h: int = h, _n_heads: int = _n_heads, _head_dim: int = _head_dim,
                        _c: Tensor = _corr_head,
                    ) -> Tensor:
                        # Replace head h's slice in the v_proj output
                        b, s, _ = output.shape
                        out = output.clone().reshape(b, s, _n_heads, _head_dim)
                        out[:, :, _h, :] = _c
                        return out.reshape(b, s, _n_heads * _head_dim)

                    handles.append(
                        block.self_attn.v_proj.register_forward_hook(_v_proj_replace_hook)
                    )

                else:
                    scores[atp_node] = 0.0
                    continue

                try:
                    with torch.no_grad():
                        patched_out = self._call_model(clean_inputs)
                        patched_metric = metric(patched_out).item()
                finally:
                    for hh in handles:
                        hh.remove()

                scores[atp_node] = patched_metric - clean_metric

            return scores
        finally:
            self._restore(was_training, orig_rg)

    def qk_operand_shapes(
        self,
        clean_inputs: _Inputs,
        corrupted_inputs: _Inputs,
        metric: Callable[[Any], Tensor],
    ) -> dict[tuple[int, int], tuple[int, int]]:
        """Debug accessor: for each (layer, head), return (Δhead_out_dim, grad_dim).

        Both must be d_model. This validates that the QK fix operates in a
        consistent d_model space — Δhead_out is projected to d_model via W_O,
        and grad_attn_out is the d_model gradient w.r.t. the o_proj output.

        Returns dict[(L, h): (Δhead_out_dim, grad_dim)].
        """
        was_training, orig_rg = self._freeze_eval()
        try:
            # Run QK fix to get operand dimensions
            # We only need the o_proj output grad and W_O shapes
            _, _, _, _, _, _, oproj_outs = self._collect_clean_grads(
                clean_inputs, metric, orig_rg,
                capture_intermediates=False,
                capture_qk_grads=True,
            )

            result: dict[tuple[int, int], tuple[int, int]] = {}
            n_heads = self.n_heads

            for L in range(self.n_layers):
                block = self._layers_list[L]
                if not hasattr(block, "self_attn"):
                    continue
                o_out = oproj_outs.get(L)
                if o_out is None:
                    continue
                # grad_dim = d_model (from o_proj output shape)
                grad_dim = o_out.shape[-1]
                # Δhead_out_dim = d_model (W_O_h: (d_model, head_dim), Δz_h: (b, q, head_dim))
                W_O = block.self_attn.o_proj.weight  # (d_model, n_heads*head_dim)
                delta_head_out_dim = W_O.shape[0]  # d_model
                for h in range(n_heads):
                    result[(L, h)] = (delta_head_out_dim, grad_dim)

            return result
        finally:
            self._restore(was_training, orig_rg)
