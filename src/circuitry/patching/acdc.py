"""ACDC: greedy circuit discovery via corrupted-resample set-ablation. Spec §2–§6."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from circuitry.patching.eap import EAPRunner
from circuitry.patching.graph import (
    Edge,
    EdgeGraph,
    Node,
)

_Inputs = Tensor | dict[str, Any]


@dataclass
class ACDCResult:
    kept_edges: list[Edge]
    removed_edges: list[Edge]
    final_kl: float
    graph: EdgeGraph

    def n_kept(self) -> int:
        return len(self.kept_edges)

    def circuit_graph(self) -> EdgeGraph:
        kept = set(self.kept_edges)
        sub = [e for e in self.graph.edges if e in kept]
        return EdgeGraph(self.graph.n_layers, self.graph.n_heads,
                         self.graph.writers, self.graph.readers, sub)


def _logits_of(out: Any) -> Tensor:
    return out.logits if hasattr(out, "logits") else out


class ACDCRunner:
    """Discover a minimal circuit by greedy reverse-topo edge pruning.

    Composes EAPRunner for the graph, module locators, GQA head-mapping, and
    corrupted writer-activation caching.  Adds the set-ablation forward (live
    capture + pre-LN per-head injection).  Forward-only — no gradients.
    """

    def __init__(self, model: nn.Module, resolver: Any = None) -> None:
        self.model = model
        self.resolver = resolver
        self._eap = EAPRunner(model, resolver)
        self.graph = self._eap.graph
        self._tl = self._eap._tl
        self.n_heads = self._eap.n_heads
        self.n_kv_heads = self._eap.n_kv_heads
        self.head_dim = self._eap.head_dim

    # ------------------------------------------------------------------
    # Locators (HF / toy).  TL handled in its own forward (later task).
    # ------------------------------------------------------------------

    def _final_norm(self) -> nn.Module | None:
        inner = getattr(self.model, "model", None)
        if inner is not None and hasattr(inner, "norm"):
            return inner.norm
        return getattr(self.model, "norm", None)

    # ------------------------------------------------------------------
    # Corrupted writer-activation cache (reuse EAP's collector)
    # ------------------------------------------------------------------

    def _cache_corrupted_acts(self, corrupted_inputs: _Inputs) -> dict[Node, Tensor]:
        """Cache each writer's corrupted-run residual contribution (once).

        TL writer contributions come from ``attn.hook_result`` / ``hook_mlp_out``
        / ``hook_embed`` — the same hooks the TL live-capture forward reads, so
        ``corr_act`` and ``live`` live in the same space.
        """
        if self._tl:
            prior = self.model.cfg.use_attn_result
            self.model.set_use_attn_result(True)
            try:
                return self._eap._collect_writer_acts_tl(corrupted_inputs)
            finally:
                self.model.set_use_attn_result(prior)
        return self._eap.writer_activations(corrupted_inputs)

    # ------------------------------------------------------------------
    # The set-ablation forward (HF / toy path).  Returns logits.
    # ------------------------------------------------------------------

    def _run_capturing_live(
        self,
        clean_inputs: _Inputs,
        removed: set[Edge],
        corr_act: dict[Node, Tensor],
        live_out: dict[Node, Tensor] | None = None,
    ) -> Tensor:
        """One forward on clean_inputs with `removed` edges ablated.

        Captures each writer's LIVE contribution as the pass proceeds, and at
        each reader injects Σ (corr_act[u] − live[u]) over its removed incoming
        edges, pre-LN, per head.  If `live_out` is given it is filled with the
        captured live contributions (used by the live-capture isolation test).
        """
        n_heads, n_kv, hd = self.n_heads, self.n_kv_heads, self.head_dim
        live: dict[Node, Tensor] = {} if live_out is None else live_out
        resid_pre: dict[int, Tensor] = {}
        handles: list[Any] = []

        # removed incoming writers, keyed by (reader_node, slot)
        inc: dict[tuple[Node, str], list[Node]] = {}
        for e in removed:
            inc.setdefault((e.reader, e.slot), []).append(e.writer)

        def delta_sum(writers: list[Node]) -> Tensor | None:
            total: Tensor | None = None
            for u in writers:
                d = corr_act[u] - live[u]
                total = d if total is None else total + d
            return total

        layers = self._eap._layers_list
        embed = self._eap._embed()
        lm_head = self._eap._lm_head()

        # --- live: embed ---
        def _embed_hook(m: nn.Module, i: tuple, o: Tensor) -> None:
            live[Node("embed")] = o.detach()
        handles.append(embed.register_forward_hook(_embed_hook))

        for L, block in enumerate(layers):
            norm_attn = getattr(block, "input_layernorm", None)

            # resid_pre capture: norm input if present, else first proj to fire.
            if norm_attn is not None:
                def _cap_norm(m: nn.Module, args: tuple, _L: int = L) -> None:
                    resid_pre[_L] = args[0].detach()
                handles.append(norm_attn.register_forward_pre_hook(_cap_norm))

            if n_heads > 0:
                attn = block.self_attn
                for slot, proj in [("q", attn.q_proj), ("k", attn.k_proj),
                                   ("v", attn.v_proj)]:
                    # guarded resid_pre capture (first proj to fire wins) when no norm
                    def _cap_proj(m: nn.Module, args: tuple, _L: int = L) -> None:
                        if _L not in resid_pre:
                            resid_pre[_L] = args[0].detach()
                    handles.append(proj.register_forward_pre_hook(_cap_proj))

                    # per-head rebuild of the proj OUTPUT
                    def _rebuild(
                        m: nn.Module, args: tuple, output: Tensor,
                        _L: int = L, _slot: str = slot, _norm: Any = norm_attn,
                    ) -> Tensor:
                        rp = resid_pre[_L]
                        W = m.weight                              # (n_proj_heads*hd, d_model)
                        n_proj_heads = n_kv if _slot in ("k", "v") else n_heads
                        b, s, _ = output.shape
                        out = output.clone().reshape(b, s, n_proj_heads, hd)
                        for ph in range(n_proj_heads):
                            if _slot in ("k", "v"):
                                writers: list[Node] = []
                                for h in range(n_heads):
                                    if self._eap._kv_head_for(h, n_heads, n_kv) == ph:
                                        writers += inc.get((Node("attn_head", _L, h), _slot), [])
                                writers = list(dict.fromkeys(writers))  # dedup, ordered
                            else:
                                writers = inc.get((Node("attn_head", _L, ph), _slot), [])
                            d = delta_sum(writers)
                            if d is None:
                                continue                          # kept: natural output
                            x = rp + d
                            if _norm is not None:
                                x = _norm(x)
                            W_ph = W[ph * hd:(ph + 1) * hd, :]
                            out[:, :, ph, :] = x @ W_ph.T
                        return out.reshape(b, s, n_proj_heads * hd)
                    handles.append(proj.register_forward_hook(_rebuild))

                # live: per-head z @ W_O at o_proj input
                def _o_pre(m: nn.Module, args: tuple, _L: int = L) -> None:
                    z = args[0].detach()
                    b, s, _ = z.shape
                    zh = z.reshape(b, s, n_heads, hd)
                    W_O = m.weight                                # (d_model, n_heads*hd)
                    for h in range(n_heads):
                        live[Node("attn_head", _L, h)] = zh[:, :, h, :] @ W_O[:, h * hd:(h + 1) * hd].T
                handles.append(attn.o_proj.register_forward_pre_hook(_o_pre))

            # mlp_in injection (pre-LN if norm present, else at first MLP proj)
            mlp_node = Node("mlp", L)
            norm_mlp = getattr(block, "post_attention_layernorm", None)

            def _mlp_inj(m: nn.Module, args: tuple, _node: Node = mlp_node) -> tuple | None:
                d = delta_sum(inc.get((_node, "mlp_in"), []))
                if d is None:
                    return None
                return (args[0] + d,) + args[1:]

            if norm_mlp is not None:
                handles.append(norm_mlp.register_forward_pre_hook(_mlp_inj))
            else:
                handles.append(block.mlp.up_proj.register_forward_pre_hook(_mlp_inj))

            # live: mlp contribution = down_proj output
            def _mlp_live(m: nn.Module, i: tuple, o: Tensor, _node: Node = mlp_node) -> None:
                live[_node] = o.detach()
            handles.append(block.mlp.down_proj.register_forward_hook(_mlp_live))

        # logits injection (pre-final-norm if present, else at lm_head input)
        log_node = Node("logits")

        def _logits_inj(m: nn.Module, args: tuple) -> tuple | None:
            d = delta_sum(inc.get((log_node, "logits_in"), []))
            if d is None:
                return None
            return (args[0] + d,) + args[1:]

        final_norm = self._final_norm()
        target = final_norm if final_norm is not None else lm_head
        handles.append(target.register_forward_pre_hook(_logits_inj))

        try:
            with torch.no_grad():
                out = self._eap._call_model(clean_inputs)
        finally:
            for h in handles:
                h.remove()
        return _logits_of(out)
