"""ACDC: greedy circuit discovery via corrupted-resample set-ablation. Spec §2–§6."""
from __future__ import annotations

from collections.abc import Callable
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
    edge_sort_key,
    reverse_topo_readers,
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

    @staticmethod
    def _slice_pos(logits: Tensor, position: int | None) -> Tensor:
        """Slice to a single token position (keeping a length-1 axis) or pass through."""
        if position is None or logits.ndim < 3:
            return logits
        return logits[:, position:position + 1, :] if position != -1 else logits[:, -1:, :]

    def _recovery_kl(self, circuit_logits: Tensor, clean_logits: Tensor,
                     position: int | None = -1) -> float:
        """KL(circuit ‖ clean) at `position` (default last token). Reuses core."""
        from circuitry.core.patching import kl_divergence
        p = self._slice_pos(circuit_logits, position)
        q = self._slice_pos(clean_logits, position)
        return kl_divergence(p, q)

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

    # ------------------------------------------------------------------
    # Set-ablation forward: TransformerLens path.
    # ------------------------------------------------------------------

    def _run_capturing_live_tl(
        self,
        clean_inputs: Tensor,
        removed: set[Edge],
        corr_act: dict[Node, Tensor],
    ) -> Tensor:
        """TL set-ablation forward. Uses split q/k/v inputs so per-head injection
        is native; TL applies LN after these hook points (pre-LN injection)."""
        n_heads = self.n_heads
        n_layers = self.graph.n_layers
        live: dict[Node, Tensor] = {}

        # Build map: (reader, slot) -> [writers whose edges are removed]
        inc: dict[tuple[Node, str], list[Node]] = {}
        for e in removed:
            inc.setdefault((e.reader, e.slot), []).append(e.writer)

        def delta_for(reader: Node, slot: str) -> Tensor | None:
            total: Tensor | None = None
            for u in inc.get((reader, slot), []):
                d = corr_act[u] - live[u]
                total = d if total is None else total + d
            return total

        fwd_hooks: list[tuple[str, Any]] = []

        # Writer capture: embed
        def _embed(t: Tensor, hook: Any) -> None:
            live[Node("embed")] = t.detach()
        fwd_hooks.append(("hook_embed", _embed))

        for L in range(n_layers):
            # Writer capture: per-head attention result contributions
            def _res(t: Tensor, hook: Any, _L: int = L) -> None:
                # t: (b, s, n_heads, d_model)
                for h in range(n_heads):
                    live[Node("attn_head", _L, h)] = t[:, :, h, :].detach()
            fwd_hooks.append((f"blocks.{L}.attn.hook_result", _res))

            # Writer capture: MLP output
            def _mlpout(t: Tensor, hook: Any, _L: int = L) -> None:
                live[Node("mlp", _L)] = t.detach()
            fwd_hooks.append((f"blocks.{L}.hook_mlp_out", _mlpout))

            # Reader injection: q/k/v inputs (per-head, pre-LN in TL ordering)
            for slot in ("q", "k", "v"):
                def _qkv(
                    t: Tensor, hook: Any, _L: int = L, _slot: str = slot,
                ) -> Tensor:
                    # t: (b, s, n_heads, d_model) — TL split q/k/v input
                    out = t.clone()
                    for h in range(n_heads):
                        d = delta_for(Node("attn_head", _L, h), _slot)
                        if d is not None:
                            out[:, :, h, :] = out[:, :, h, :] + d
                    return out
                fwd_hooks.append((f"blocks.{L}.hook_{slot}_input", _qkv))

            # Reader injection: MLP input
            def _mlpin(t: Tensor, hook: Any, _L: int = L) -> Tensor:
                d = delta_for(Node("mlp", _L), "mlp_in")
                return t if d is None else t + d
            fwd_hooks.append((f"blocks.{L}.hook_mlp_in", _mlpin))

        # Reader injection: logits (pre-final-LN at last resid_post)
        def _resid_post(t: Tensor, hook: Any) -> Tensor:
            d = delta_for(Node("logits"), "logits_in")
            return t if d is None else t + d
        fwd_hooks.append((f"blocks.{n_layers - 1}.hook_resid_post", _resid_post))

        prior_r = self.model.cfg.use_attn_result
        prior_q = self.model.cfg.use_split_qkv_input
        prior_m = self.model.cfg.use_hook_mlp_in
        self.model.set_use_attn_result(True)
        self.model.set_use_split_qkv_input(True)
        self.model.set_use_hook_mlp_in(True)
        try:
            with torch.no_grad():
                logits = self.model.run_with_hooks(clean_inputs, fwd_hooks=fwd_hooks)
        finally:
            self.model.set_use_attn_result(prior_r)
            self.model.set_use_split_qkv_input(prior_q)
            self.model.set_use_hook_mlp_in(prior_m)
        return logits

    # ------------------------------------------------------------------
    # Dispatch forward (HF / toy path; TL is a later task).
    # ------------------------------------------------------------------

    def _forward(self, clean_inputs: _Inputs, removed: set[Edge],
                 corr_act: dict[Node, Tensor]) -> Tensor:
        if self._tl:
            return self._run_capturing_live_tl(clean_inputs, removed, corr_act)
        return self._run_capturing_live(clean_inputs, removed, corr_act)

    def _incoming_in_order(
        self,
        reader: Node,
        slot: str,
        ordering: str,
        eap_scores: dict[Edge, float] | None,
    ) -> list[Edge]:
        edges = [e for e in self.graph.edges if e.reader == reader and e.slot == slot]
        if ordering == "eap" and eap_scores is not None:
            return sorted(edges, key=lambda e: (abs(eap_scores.get(e, 0.0)), edge_sort_key(e)))
        return sorted(edges, key=edge_sort_key)

    def run(
        self,
        clean_inputs: _Inputs,
        corrupted_inputs: _Inputs,
        tau: float,
        ordering: str | None = None,
        eap_scores: dict[Edge, float] | None = None,
        position: int | None = -1,
        metric: Callable[[Tensor, Tensor], float] | None = None,
    ) -> ACDCResult:
        """Greedy reverse-topo edge pruning. Returns the surviving circuit."""
        if ordering is None:
            ordering = "eap" if eap_scores is not None else "topo"

        corr_act = self._cache_corrupted_acts(corrupted_inputs)
        with torch.no_grad():
            full_clean_logits = _logits_of(self._eap._call_model(clean_inputs))

        def recovery(circuit_logits: Tensor) -> float:
            if metric is not None:
                return float(metric(circuit_logits, full_clean_logits))
            return self._recovery_kl(circuit_logits, full_clean_logits, position)

        removed: set[Edge] = set()
        current = 0.0
        for reader, slot in reverse_topo_readers(self.graph):
            for edge in self._incoming_in_order(reader, slot, ordering, eap_scores):
                removed.add(edge)
                logits = self._forward(clean_inputs, removed, corr_act)
                new_kl = recovery(logits)
                if new_kl - current < tau:
                    current = new_kl
                else:
                    removed.discard(edge)

        kept = [e for e in self.graph.edges if e not in removed]
        return ACDCResult(kept, sorted(removed, key=edge_sort_key), current, self.graph)

    def sweep(
        self,
        clean_inputs: _Inputs,
        corrupted_inputs: _Inputs,
        taus: list[float],
        ordering: str | None = None,
        eap_scores: dict[Edge, float] | None = None,
        position: int | None = -1,
        metric: Callable[[Tensor, Tensor], float] | None = None,
    ) -> list[tuple[float, int, float]]:
        """Run ACDC at each τ; return the Pareto frontier [(τ, n_kept, final_kl)]."""
        out: list[tuple[float, int, float]] = []
        for tau in taus:
            r = self.run(clean_inputs=clean_inputs, corrupted_inputs=corrupted_inputs,
                         tau=tau, ordering=ordering, eap_scores=eap_scores,
                         position=position, metric=metric)
            out.append((tau, r.n_kept(), r.final_kl))
        return out
