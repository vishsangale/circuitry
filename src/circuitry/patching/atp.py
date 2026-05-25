"""AtP* node attribution. Design spec docs/superpowers/specs/2026-05-24-atp-design.md."""
from __future__ import annotations

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


class AtPRunner:
    """Vanilla AtP node attribution runner.

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
      attn_head(L,h) slot q/k — placeholder 0.0 (Task 5 adds QK fix)
    """

    def __init__(self, model: nn.Module, resolver: Any = None) -> None:
        self.model = model
        self.resolver = resolver

        # Locate layers list (support both model.model.layers and model.layers)
        self._layers_list = self._locate_layers(model)
        n_layers = len(self._layers_list)
        n_heads = getattr(resolver, "n_heads", 0) if resolver is not None else 0
        self.n_layers = n_layers
        self.n_heads = n_heads
        d_model = getattr(resolver, "d_model", None) if resolver is not None else None
        self.head_dim = (d_model // n_heads) if (d_model is not None and n_heads > 0) else None

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
    # Activation capture helpers
    # ------------------------------------------------------------------

    def _cache_node_acts(self, inputs: _Inputs) -> dict[AtPNode, Tensor]:
        """No-grad forward pass caching node activations (detached).

        Captures:
          embed node → embed_tokens output (b, s, d_model)
          mlp(L) node → down_proj output (b, s, d_model)
          attn_head(L,h) v → v_proj output, head-h slice (b, s, head_dim)
        """
        acts: dict[AtPNode, Tensor] = {}
        handles: list[Any] = []

        embed_node = AtPNode(Node("embed"), None)

        def _embed_hook(module: nn.Module, inp: tuple, output: Tensor) -> None:
            acts[embed_node] = output.detach()

        handles.append(self._embed().register_forward_hook(_embed_hook))

        n_heads = self.n_heads
        head_dim = self.head_dim

        for L, block in enumerate(self._layers_list):
            if n_heads > 0 and head_dim is not None:
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

        try:
            with torch.no_grad():
                self._call_model(inputs)
        finally:
            for h in handles:
                h.remove()

        return acts

    def _collect_clean_grads(
        self,
        inputs: _Inputs,
        metric: Callable[[Any], Tensor],
        orig_rg: dict[str, bool],
    ) -> tuple[Any, dict[AtPNode, Tensor], dict[AtPNode, Tensor | None]]:
        """Forward + backward pass capturing node activations AND their gradients.

        For vanilla AtP, we capture the FULL downstream gradient (including
        residual bypass paths). We do this by:
          - Temporarily re-enabling requires_grad on all params so the computation
            graph is built (no param values are changed; we just need grad flow).
          - Hooking each node's output, calling retain_grad() on it (works because
            the output is now in the computation graph).
          - After backward, reading .grad from each stored tensor.

        This gives the TOTAL gradient w.r.t. the node activation, not
        component-only (which is what EAP's clone trick gives). On a linear
        model, total == component since there's no nonlinearity.

        Returns (model_out, node_acts, node_grads).
        """
        # Temporarily re-enable requires_grad on params so the computation graph
        # is connected through the frozen model. Param grads themselves are
        # irrelevant (we won't use them) — we only need intermediate act grads.
        for name, p in self.model.named_parameters():
            if orig_rg.get(name, False):
                p.requires_grad_(True)
            # For params that were originally frozen (orig_rg[name] == False):
            # we still need to enable grad so the graph connects through them.
            # Otherwise embed_tokens output has no grad and retain_grad fails.
            p.requires_grad_(True)

        node_acts: dict[AtPNode, Tensor] = {}
        handles: list[Any] = []

        embed_node = AtPNode(Node("embed"), None)

        def _embed_hook(module: nn.Module, inp: tuple, output: Tensor) -> None:
            output.retain_grad()
            node_acts[embed_node] = output

        handles.append(self._embed().register_forward_hook(_embed_hook))

        n_heads = self.n_heads
        head_dim = self.head_dim

        for L, block in enumerate(self._layers_list):
            if n_heads > 0 and head_dim is not None:
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

            mlp_node = AtPNode(Node("mlp", L), None)
            _mlp_node = mlp_node

            def _down_proj_hook_grad(
                module: nn.Module, inp: tuple, output: Tensor,
                _n: AtPNode = _mlp_node,
            ) -> None:
                output.retain_grad()
                node_acts[_n] = output

            handles.append(block.mlp.down_proj.register_forward_hook(_down_proj_hook_grad))

        try:
            with torch.enable_grad():
                out = self._call_model(inputs)
                metric(out).backward()
        finally:
            for h in handles:
                h.remove()
            # Re-freeze all params (back to frozen state before grad collection)
            for _name, p in self.model.named_parameters():
                p.requires_grad_(False)

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

        return out, node_acts, node_grads

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
        """Compute vanilla AtP node scores.

        Steps:
          1. Freeze params + eval (save/restore in try/finally).
          2. Corrupted forward (no grad): cache node activations.
          3. Clean forward + backward: retain_grad on node activation tensors,
             read .grad after backward.
          4. score(node) = float((Δact * grad).sum()) — sum over ALL dims.
          5. q/k nodes: placeholder 0.0 (Task 5 replaces with QK fix).
          6. Return AtPResult over enumerate_nodes.
        """
        was_training, orig_rg = self._freeze_eval()
        try:
            # Step 1: cache corrupted activations (no grad)
            corrupted_acts = self._cache_node_acts(corrupted_inputs)

            # Step 2: clean forward + backward to get grads
            _, clean_node_acts, clean_node_grads = self._collect_clean_grads(
                clean_inputs, metric, orig_rg
            )

            # Step 3: compute clean activations for embed and mlp nodes
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

            # Step 4: score each node
            scores: dict[AtPNode, float] = {}
            d_mlp = getattr(self.resolver, "d_mlp", None) if self.resolver is not None else None
            all_nodes = enumerate_nodes(
                self.n_layers, self.n_heads, d_mlp=d_mlp if neurons else None
            )

            for atp_node in all_nodes:
                if atp_node.slot in ("q", "k"):
                    # Placeholder: Task 5 implements QK fix
                    scores[atp_node] = 0.0
                    continue

                corr_act = corrupted_acts.get(atp_node)
                cln_act = clean_acts.get(atp_node)
                grad = clean_node_grads.get(atp_node)

                if corr_act is None or cln_act is None or grad is None:
                    scores[atp_node] = 0.0
                    continue

                delta = corr_act - cln_act
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

        q/k nodes: return 0.0 (skipped by the test).
        """
        was_training, orig_rg = self._freeze_eval()
        try:
            with torch.no_grad():
                # Baseline: clean metric with no patching
                clean_out = self._call_model(clean_inputs)
                clean_metric = metric(clean_out).item()

                # Cache corrupted activations for all nodes
                corrupted_acts = self._cache_node_acts(corrupted_inputs)

            scores: dict[AtPNode, float] = {}

            for atp_node in nodes:
                if atp_node.slot in ("q", "k"):
                    scores[atp_node] = 0.0
                    continue

                corr_act = corrupted_acts.get(atp_node)
                if corr_act is None:
                    scores[atp_node] = 0.0
                    continue

                # Patch this node's activation with the corrupted value (REPLACE, not add)
                handles: list[Any] = []

                inner_node = atp_node.node

                if inner_node.kind == "embed":
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
                    L = inner_node.layer
                    h = inner_node.head
                    _corr_head = corr_act  # (b, s, head_dim)
                    _n_heads = self.n_heads
                    _head_dim = self.head_dim
                    block = self._layers_list[L]

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
