"""SAE feature→feature edge attribution runner. v1.6.0.

Multi-site error-term splice + per-downstream VJP edge scoring.
Composes (does NOT subclass) SAEFeatureRunner for Stage 1.

Only the HF-eager path is supported (TLSiteResolver → NotImplementedError).
Only resid_post sites are supported (others → NotImplementedError).

Stage B additions:
  - compute_f_per_site: capture feature acts per site (no_grad)
  - _feature_circuit_forward: node-set ablation forward (§4.1)
  - SAEFeatureCircuit.faithfulness / completeness / prune (§4.2)
  - FeatureACDCRunner: greedy node pruning + sweep (§4.3)

See docs/superpowers/specs/2026-05-30-v16-sae-feature-edges-design.md.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from circuitry.patching.atp import AtPNode, AtPResult
from circuitry.patching.graph import Node
from circuitry.patching.sae_features import (
    SAEFeatureRunner,
    _freeze_sae,
    _restore_sae,
    _routed_extract,
    _routed_inject,
)
from circuitry.patching.sites import Site
from circuitry.sae.grad import assert_supported_sae, sae_decompose

# Type alias
_Inputs = Tensor | dict[str, Any]


# ---------------------------------------------------------------------------
# Stage B helpers: feature-acts capture + node-set ablation forward
# ---------------------------------------------------------------------------


def compute_f_per_site(
    model: nn.Module,
    inputs: _Inputs,
    sae_sites: dict[Site, Any],
    resolver: Any,
) -> dict[int, Tensor]:
    """Capture SAE feature activations per site in a single no_grad forward.

    Returns:
        dict mapping site.layer -> f tensor (detached, on SAE device/dtype).
        Used for ablation_value computation (corrupted/zero/mean modes).
    """
    result: dict[int, Tensor] = {}

    handles: list[Any] = []
    for site, sae in sae_sites.items():
        # Route through ResolvedSite (identity for resid_post + position=None)
        resolved = resolver.resolve(model, site)
        layer_mod = resolved.module
        layer = site.layer

        def _hook(
            module: nn.Module,
            inp: Any,
            output: Any,
            _sae: Any = sae,
            _layer: int = layer,
            _resolved: Any = resolved,
        ) -> None:
            a = _routed_extract(_resolved, output).detach()
            a_in = a.to(
                getattr(_sae, "device", a.device),
                getattr(_sae, "dtype", a.dtype),
            )
            with torch.no_grad():
                f = _sae.encode(a_in)
            result[_layer] = f.detach()

        handles.append(layer_mod.register_forward_hook(_hook))

    try:
        if isinstance(inputs, dict):
            with torch.no_grad():
                model(**inputs)
        else:
            with torch.no_grad():
                model(inputs)
    finally:
        for h in handles:
            h.remove()

    return result


def _feature_circuit_forward(
    model: nn.Module,
    inputs: _Inputs,
    sae_sites: dict[Site, Any],
    resolver: Any,
    circuit_nodes: dict[int, set[int]],
    ablation_values: dict[int, Tensor],
    *,
    include_error_node: bool = False,
    error_in_circuit: dict[int, bool] | None = None,
    ablation_mode: str = "corrupted",
    ablation_eps: dict[int, Tensor] | None = None,
) -> Any:
    """Node-set ablation forward (§4.1).

    At each spliced site, replaces NON-circuit feature entries with ablation_values
    before decode. eps is frozen at clean (include_error_node=False) or ablated
    for out-of-circuit error nodes (include_error_node=True).

    Args:
        model:           The model to run.
        inputs:          Model inputs.
        sae_sites:       Ordered dict of Site → SAE.
        resolver:        SiteResolver used to route hooks to the correct submodule.
        circuit_nodes:   dict[layer, set[feature_idx]] — the IN-CIRCUIT features.
        ablation_values: dict[layer, Tensor] — ablation activation (f_corrupt / zeros / mean).
        include_error_node: If True, treat sae_error as a node (§4.4).
        error_in_circuit:   dict[layer, bool] — whether error node is in-circuit per site.
        ablation_mode:   'corrupted' / 'zero' / 'mean' (informational; ablation_values already computed).
        ablation_eps:    dict[layer, Tensor] — eps to use when ablating error nodes (include_error_node=True).

    Returns:
        Model output (may have .logits or be raw tensor).
    """
    handles: list[Any] = []

    # Sort sites in forward order (ascending layer)
    sorted_site_pairs = sorted(sae_sites.items(), key=lambda kv: kv[0].layer)

    for site, sae in sorted_site_pairs:
        layer = site.layer
        # Route through ResolvedSite (identity for resid_post + position=None)
        resolved = resolver.resolve(model, site)
        layer_mod = resolved.module
        # Determine model device/dtype for cast-back
        params = list(layer_mod.parameters())
        if params:
            m_dtype, m_device = params[0].dtype, params[0].device
        else:
            m_dtype, m_device = torch.float32, torch.device("cpu")

        in_circuit = circuit_nodes.get(layer, set())
        abl_val = ablation_values.get(layer)  # may be None for zero-mode

        # Error-node circuit membership
        err_in_circ = True  # default: treat as always in circuit (frozen eps)
        if include_error_node and error_in_circuit is not None:
            err_in_circ = error_in_circuit.get(layer, True)
        abl_eps = ablation_eps.get(layer) if (ablation_eps is not None and not err_in_circ) else None

        def _ablate_hook(
            module: nn.Module,
            inp: Any,
            output: Any,
            _sae: Any = sae,
            _in_circuit: set = in_circuit,
            _abl_val: Tensor | None = abl_val,
            _mdtype: torch.dtype = m_dtype,
            _mdev: Any = m_device,
            _include_err: bool = include_error_node,
            _err_in_circ: bool = err_in_circ,
            _abl_eps: Tensor | None = abl_eps,
            _resolved: Any = resolved,
        ) -> Any:
            a = _routed_extract(_resolved, output).detach()
            a_in = a.to(getattr(_sae, "device", a.device), getattr(_sae, "dtype", a.dtype))

            with torch.no_grad():
                f, x_hat, eps = sae_decompose(_sae, a_in)
                f_ablated = f.clone()

                # Ablate non-circuit features
                n_features = f.shape[-1]
                for feat_i in range(n_features):
                    if feat_i not in _in_circuit:
                        if _abl_val is not None:
                            f_ablated[..., feat_i] = _abl_val[..., feat_i].to(
                                f_ablated.device, f_ablated.dtype
                            )
                        else:
                            f_ablated[..., feat_i] = 0.0

                # Error-node eps handling (§4.4)
                if _include_err and not _err_in_circ and _abl_eps is not None:
                    # Ablate eps to the corrupted/zero/mean value
                    eps = _abl_eps.to(eps.device, eps.dtype)

                recon = _sae.decode(f_ablated) + eps
                recon_cast = recon.to(_mdev, _mdtype)

            return _routed_inject(_resolved, output, recon_cast)

        handles.append(layer_mod.register_forward_hook(_ablate_hook))

    try:
        with torch.no_grad():
            if isinstance(inputs, dict):
                out = model(**inputs)
            else:
                out = model(inputs)
    finally:
        for h in handles:
            h.remove()

    return out


# ---------------------------------------------------------------------------
# SAEFeatureEdge
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SAEFeatureEdge:
    """A directed feature→feature or error→feature edge.

    writer: upstream AtPNode  (sae_feature or sae_error at layer U)
    reader: downstream AtPNode (sae_feature at layer D)

    Direction note:
      - feature→feature: both writer and reader are sae_feature.  Computed
        always (Stage 2 VJP loop).
      - error→feature: writer is sae_error, reader is sae_feature.  Computed
        when include_error_node=True via an independent err_leaf_U in the
        WRITER clean hook.
      - feature→error: STRUCTURALLY ZERO (downstream eps is a detached leaf —
        no gradient path exists).  NOT computed; not returned.

    No position field in v1.6 — scores are position-aggregated scalars.
    """

    writer: AtPNode
    reader: AtPNode


# ---------------------------------------------------------------------------
# SAEFeatureEdgeGraph
# ---------------------------------------------------------------------------


def _sae_edge_sort_key(edge: SAEFeatureEdge) -> tuple:
    """Deterministic total order: writer layer, writer feat idx, reader layer, reader feat idx."""
    w = edge.writer.node
    r = edge.reader.node
    return (
        w.layer if w.layer is not None else -1,
        w.neuron if w.neuron is not None else -1,
        r.layer if r.layer is not None else -1,
        r.neuron if r.neuron is not None else -1,
    )


class SAEFeatureEdgeGraph:
    """Container for the two-stage SAE edge attribution result.

    sites:      ordered list of (site, sae) pairs in forward order
    survivors:  dict[Site, list[AtPNode]] — top-K nodes per site
    edges:      sorted list of SAEFeatureEdge
    """

    def __init__(
        self,
        sites: list[tuple[Site, Any]],
        survivors: dict[Site, list[AtPNode]],
        edges: list[SAEFeatureEdge],
    ) -> None:
        self.sites = sites
        self.survivors = survivors
        self.edges = edges

    def reverse_topo_site_pairs(
        self,
        layer_pairs: str = "adjacent",
    ) -> list[tuple[tuple[Site, Any], tuple[Site, Any]]]:
        """Enumerate (writer_site, reader_site) pairs in reverse-topological order.

        layer_pairs='adjacent'   → n_sites-1 adjacent pairs
        layer_pairs='all_forward' → n(n-1)/2 all forward pairs
        """
        pairs: list[tuple[tuple[Site, Any], tuple[Site, Any]]] = []
        n = len(self.sites)
        for i in range(n):
            for j in range(i + 1, n):
                if layer_pairs == "adjacent" and j != i + 1:
                    continue
                pairs.append((self.sites[i], self.sites[j]))
        # Reverse topo: later (reader) layers first
        pairs.sort(key=lambda p: -p[1][0].layer)
        return pairs


# ---------------------------------------------------------------------------
# SAEFeatureCircuit
# ---------------------------------------------------------------------------


class SAEFeatureCircuit:
    """Result of a two-stage SAE feature edge attribution run.

    nodes: AtPResult — per-site node scores from Stage 1 (v1.5 unchanged)
    edges: dict[SAEFeatureEdge, float] — edge attribution scores
    graph: SAEFeatureEdgeGraph
    _sae_sites: internal reference for ablation (set by runner)
    _resolver: internal reference for ablation (set by runner)

    Methods ranked/top_k/threshold mirror EAPResult.
    """

    def __init__(
        self,
        nodes: AtPResult,
        edges: dict[SAEFeatureEdge, float],
        graph: SAEFeatureEdgeGraph,
        *,
        model: nn.Module | None = None,
        sae_sites: dict[Site, Any] | None = None,
        resolver: Any | None = None,
    ) -> None:
        self.nodes = nodes
        self.edges = edges
        self.graph = graph
        # Internal references for Stage B ablation methods
        self._model = model
        self._sae_sites = sae_sites
        self._resolver = resolver

    def ranked(self) -> list[tuple[SAEFeatureEdge, float]]:
        return sorted(self.edges.items(), key=lambda kv: abs(kv[1]), reverse=True)

    def top_k(self, n: int) -> list[tuple[SAEFeatureEdge, float]]:
        return self.ranked()[:n]

    def threshold(self, tau: float) -> list[SAEFeatureEdge]:
        return [e for e, s in self.edges.items() if abs(s) >= tau]

    # ------------------------------------------------------------------
    # Internal: extract circuit_nodes set from this circuit
    # ------------------------------------------------------------------

    def _circuit_node_sets(self) -> dict[int, set[int]]:
        """Return dict[layer, set[feature_idx]] for nodes in this circuit.

        Nodes are defined by the union of edge endpoints (writer + reader).
        Does NOT include nodes from self.nodes.scores that are not in any edge —
        this ensures an empty-edge circuit has empty node sets.
        """
        result: dict[int, set[int]] = {}
        if self._sae_sites is not None:
            for site in self._sae_sites:
                result[site.layer] = set()

        # Collect all nodes across edges (edge endpoints = circuit members)
        for edge in self.edges:
            for node_ref in [edge.writer, edge.reader]:
                nd = node_ref.node
                if nd.kind == "sae_feature" and nd.layer is not None and nd.neuron is not None:
                    result.setdefault(nd.layer, set()).add(nd.neuron)
        return result

    def _all_node_sets(self) -> dict[int, set[int]]:
        """Return dict[layer, set[feature_idx]] for ALL survivors (full circuit M)."""
        if self._sae_sites is None:
            return {}
        result: dict[int, set[int]] = {site.layer: set() for site in self._sae_sites}
        for site, survivors in self.graph.survivors.items():
            for atp_node in survivors:
                nd = atp_node.node
                if nd.kind == "sae_feature" and nd.neuron is not None:
                    result.setdefault(site.layer, set()).add(nd.neuron)
        return result

    def _compute_ablation_values(
        self,
        clean: _Inputs,
        corrupted: _Inputs,
        ablation_mode: str,
    ) -> dict[int, Tensor]:
        """Compute ablation_value dict[layer, Tensor] per §4.1."""
        if self._model is None or self._sae_sites is None or self._resolver is None:
            raise RuntimeError(
                "SAEFeatureCircuit must be created by SAEFeatureEdgeRunner.run() "
                "to use faithfulness/completeness/prune."
            )
        f_corrupt = compute_f_per_site(
            self._model, corrupted, self._sae_sites, self._resolver
        )
        if ablation_mode == "corrupted":
            return f_corrupt
        if ablation_mode == "zero":
            return {layer: torch.zeros_like(f) for layer, f in f_corrupt.items()}
        if ablation_mode == "mean":
            result: dict[int, Tensor] = {}
            for layer, f in f_corrupt.items():
                mean_val = f.mean(dim=list(range(f.ndim - 1)), keepdim=True)
                result[layer] = mean_val.expand_as(f)
            return result
        raise ValueError(
            f"Unknown ablation_mode {ablation_mode!r}. "
            "Expected 'corrupted', 'zero', or 'mean'."
        )

    def _m_of(
        self,
        clean: _Inputs,
        corrupted: _Inputs,
        metric: Callable[[Any], Tensor],
        circuit_nodes: dict[int, set[int]],
        ablation_values: dict[int, Tensor],
        *,
        include_error_node: bool = False,
        error_in_circuit: dict[int, bool] | None = None,
        ablation_eps: dict[int, Tensor] | None = None,
    ) -> float:
        """Compute m(circuit_nodes) — scalar metric under node-set ablation.

        Args:
            circuit_nodes:     dict[layer, set[feature_idx]] — the IN-CIRCUIT features.
            ablation_values:   dict[layer, Tensor] — ablation activation per site.
            include_error_node: If True, thread error-node membership into the forward.
            error_in_circuit:  dict[layer, bool] — whether error node is in-circuit per site.
            ablation_eps:      dict[layer, Tensor] — eps ablation values for out-of-circuit
                               error nodes (required when include_error_node=True and
                               error node is out of circuit).
        """
        if self._model is None or self._sae_sites is None or self._resolver is None:
            raise RuntimeError(
                "SAEFeatureCircuit must be created by SAEFeatureEdgeRunner.run() "
                "to use faithfulness/completeness/prune."
            )
        out = _feature_circuit_forward(
            self._model,
            clean,
            self._sae_sites,
            self._resolver,
            circuit_nodes,
            ablation_values,
            include_error_node=include_error_node,
            error_in_circuit=error_in_circuit,
            ablation_eps=ablation_eps,
        )
        with torch.no_grad():
            m = metric(out)
        return float(m)

    # ------------------------------------------------------------------
    # Stage B: faithfulness / completeness / prune
    # ------------------------------------------------------------------

    def _error_circuit_membership(self) -> dict[int, bool]:
        """Return dict[layer, bool] — whether the sae_error node at each layer is in circuit.

        An error node is in-circuit if it appears as a WRITER in any edge.
        """
        if self._sae_sites is None:
            return {}
        in_circuit: dict[int, bool] = {site.layer: False for site in self._sae_sites}
        for edge in self.edges:
            w = edge.writer.node
            if w.kind == "sae_error" and w.layer is not None:
                in_circuit[w.layer] = True
        return in_circuit

    def faithfulness(
        self,
        clean: _Inputs,
        corrupted: _Inputs,
        metric: Callable[[Any], Tensor],
        ablation_mode: str = "corrupted",
        include_error_node: bool = False,
    ) -> float:
        """Faithfulness of this circuit: (m(C) − m(∅)) / (m(M) − m(∅)).

        Per §4.2 (Marks SFC §3.2). m(·) = scalar metric under node-set ablation.
        NOTE: metric-diff faithfulness can exceed [0,1] when features anti-correlate
        with the metric — that is expected, not a bug.

        When include_error_node=True, the error-node membership of the CIRCUIT
        is derived from the edge set (an error node is in-circuit iff it appears
        as a writer in at least one edge).  Out-of-circuit error eps values are
        ablated to ablation_mode.

        The FULL circuit m(M) always uses _all_node_sets() to guarantee
        faithfulness(M) = 1 by construction.

        Args:
            clean:          Clean model inputs.
            corrupted:      Corrupted model inputs.
            metric:         Differentiable scalar metric (same as used for attribution).
            ablation_mode:  'corrupted' (default) / 'zero' / 'mean'.
            include_error_node: If True, thread error-node circuit membership into
                                the ablation forward (§4.4).  feature→error is
                                structurally zero and is never computed.
        """
        ablation_values = self._compute_ablation_values(clean, corrupted, ablation_mode)

        # Build error_in_circuit and ablation_eps when include_error_node=True
        error_in_circuit: dict[int, bool] | None = None
        ablation_eps: dict[int, Tensor] | None = None
        if include_error_node:
            error_in_circuit = self._error_circuit_membership()
            # ablation_eps: for out-of-circuit error nodes use the corrupted eps
            if self._sae_sites is not None and self._resolver is not None and self._model is not None:
                from circuitry.sae.grad import sae_decompose as _sae_decompose
                abl_eps_dict: dict[int, Tensor] = {}
                for site, sae in self._sae_sites.items():
                    layer = site.layer
                    # Route through ResolvedSite (identity for resid_post + position=None)
                    resolved = self._resolver.resolve(self._model, site)
                    layer_mod = resolved.module
                    eps_store: dict[str, Tensor] = {}

                    def _eps_hook(
                        module, inp, output,
                        _sae=sae, _st=eps_store,
                        _resolved=resolved,
                    ) -> None:
                        a = _routed_extract(_resolved, output).detach()
                        a_in = a.to(
                            getattr(_sae, "device", a.device),
                            getattr(_sae, "dtype", a.dtype),
                        )
                        with torch.no_grad():
                            _, _, eps_c = _sae_decompose(_sae, a_in)
                        _st["eps"] = eps_c.detach()

                    _h = layer_mod.register_forward_hook(_eps_hook)
                    try:
                        with torch.no_grad():
                            if isinstance(corrupted, dict):
                                self._model(**corrupted)  # type: ignore[union-attr]
                            else:
                                self._model(corrupted)  # type: ignore[union-attr]
                    finally:
                        _h.remove()
                    if "eps" in eps_store:
                        abl_eps_dict[layer] = eps_store["eps"]
                ablation_eps = abl_eps_dict

        # m(∅): empty circuit — all features ablated
        empty_nodes: dict[int, set[int]] = {layer: set() for layer in ablation_values}
        # For empty circuit: all error nodes are out-of-circuit
        empty_err_in_circ = {layer: False for layer in ablation_values} if include_error_node else None
        m_empty = self._m_of(
            clean, corrupted, metric, empty_nodes, ablation_values,
            include_error_node=include_error_node,
            error_in_circuit=empty_err_in_circ,
            ablation_eps=ablation_eps,
        )

        # m(M): full circuit — all survivors kept; use _all_node_sets() so faithfulness(M)=1
        full_nodes = self._all_node_sets()
        # For full circuit: all error nodes are in-circuit (eps frozen at clean)
        full_err_in_circ = {layer: True for layer in full_nodes} if include_error_node else None
        m_full = self._m_of(
            clean, corrupted, metric, full_nodes, ablation_values,
            include_error_node=include_error_node,
            error_in_circuit=full_err_in_circ,
            ablation_eps=ablation_eps,
        )

        # m(C): this circuit.
        # Circuit node membership is defined as EDGE ENDPOINTS (union of all
        # writer/reader nodes in self.edges).  This is the consistent definition:
        # - For the full circuit M (returned by runner.run()), all survivors should
        #   appear as edge endpoints; faithfulness(M) = 1 by construction when
        #   _circuit_node_sets() == _all_node_sets().
        # - For an empty circuit (edges={}), circuit_nodes = {} → m(C) = m(∅)
        #   → faithfulness(∅) = 0 by construction.
        # - For a partial circuit (0 < |C| < |M|), circuit_nodes ⊂ full_nodes.
        # NOTE: if a survivor has no edges (orphaned), it does not count as a
        # circuit member — faithfulness(C) < 1 in that case.  This is the
        # intended Marks-SFC semantics (edges define membership).
        circuit_nodes = self._circuit_node_sets()
        circuit_err_in_circ = error_in_circuit  # derived from actual edge set
        m_circuit = self._m_of(
            clean, corrupted, metric, circuit_nodes, ablation_values,
            include_error_node=include_error_node,
            error_in_circuit=circuit_err_in_circ,
            ablation_eps=ablation_eps,
        )

        denom = m_full - m_empty
        if abs(denom) < 1e-12:
            return float("nan")
        return (m_circuit - m_empty) / denom

    def completeness(
        self,
        clean: _Inputs,
        corrupted: _Inputs,
        metric: Callable[[Any], Tensor],
        ablation_mode: str = "corrupted",
        include_error_node: bool = False,
    ) -> float:
        """Completeness of this circuit: (m(M\\C) − m(M)) / (m(∅) − m(M)).

        Per §4.2. m(M\\C) = model run with the complement of this circuit.
        NOTE: can exceed [0,1] under anti-correlation — same caveat as faithfulness.

        When include_error_node=True, the error-node membership of the CIRCUIT
        is derived from the edge set (an error node is in-circuit iff it appears
        as a writer in at least one edge).  The full circuit has all error nodes
        in-circuit (eps frozen at clean).

        Args:
            clean:          Clean model inputs.
            corrupted:      Corrupted model inputs.
            metric:         Differentiable scalar metric.
            ablation_mode:  'corrupted' (default) / 'zero' / 'mean'.
            include_error_node: If True, thread error-node circuit membership into
                                the ablation forward (§4.4).
        """
        ablation_values = self._compute_ablation_values(clean, corrupted, ablation_mode)

        full_nodes = self._all_node_sets()
        circuit_nodes = self._circuit_node_sets()

        # M\C: complement of circuit within the full node set
        complement: dict[int, set[int]] = {}
        for layer, all_feats in full_nodes.items():
            in_circ = circuit_nodes.get(layer, set())
            complement[layer] = all_feats - in_circ

        # Build error_in_circuit kwargs when include_error_node=True
        err_kwargs: dict[str, Any] = {}
        if include_error_node:
            err_kwargs["include_error_node"] = True
            # Reuse _error_circuit_membership for the circuit
            err_circ = self._error_circuit_membership()
            err_kwargs["error_in_circuit"] = err_circ
            # For full circuit: all error nodes in-circuit
            full_err_in_circ = {layer: True for layer in full_nodes}
            # For complement: complement of error membership
            comp_err_in_circ = {layer: not v for layer, v in err_circ.items()}
            # For empty: all out of circuit
            empty_err_in_circ = {layer: False for layer in ablation_values}

        # m(∅): empty circuit
        empty_nodes: dict[int, set[int]] = {layer: set() for layer in ablation_values}
        if include_error_node:
            m_empty = self._m_of(
                clean, corrupted, metric, empty_nodes, ablation_values,
                include_error_node=True,
                error_in_circuit=empty_err_in_circ,  # type: ignore[possibly-undefined]
            )
        else:
            m_empty = self._m_of(clean, corrupted, metric, empty_nodes, ablation_values)

        # m(M): full circuit
        if include_error_node:
            m_full = self._m_of(
                clean, corrupted, metric, full_nodes, ablation_values,
                include_error_node=True,
                error_in_circuit=full_err_in_circ,  # type: ignore[possibly-undefined]
            )
        else:
            m_full = self._m_of(clean, corrupted, metric, full_nodes, ablation_values)

        # m(M\C): complement circuit
        if include_error_node:
            m_complement = self._m_of(
                clean, corrupted, metric, complement, ablation_values,
                include_error_node=True,
                error_in_circuit=comp_err_in_circ,  # type: ignore[possibly-undefined]
            )
        else:
            m_complement = self._m_of(clean, corrupted, metric, complement, ablation_values)

        denom = m_empty - m_full
        if abs(denom) < 1e-12:
            return float("nan")
        return (m_complement - m_full) / denom

    def prune(
        self,
        method: str = "threshold",
        tau: float = 0.0,
        ablation_mode: str = "corrupted",
        eap_skip_threshold: float | None = None,
        **kwargs: Any,
    ) -> SAEFeatureCircuit:
        """Greedy circuit pruning.

        Args:
            method:  'threshold' — keep |edge|≥tau (pure edge-score filter).
                     'acdc'      — FeatureACDCRunner greedy node pruning.
                     'both'      — threshold first, then acdc (gives ⊆ threshold result).
            tau:     Threshold for edge filtering (threshold/both) or KL tolerance (acdc/both).
            ablation_mode: 'corrupted' / 'zero' / 'mean'.
            eap_skip_threshold: for acdc — skip nodes with |score|>threshold (assume kept).
            **kwargs: forwarded to FeatureACDCRunner.run() (clean, corrupted, metric required).

        Returns:
            Pruned SAEFeatureCircuit.
        """
        if method not in ("threshold", "acdc", "both"):
            raise ValueError(
                f"prune method must be 'threshold', 'acdc', or 'both', got {method!r}"
            )

        if self._model is None or self._sae_sites is None or self._resolver is None:
            raise RuntimeError(
                "SAEFeatureCircuit must be created by SAEFeatureEdgeRunner.run() "
                "to use prune()."
            )

        result = self

        if method in ("threshold", "both"):
            kept_edges = {e: s for e, s in result.edges.items() if abs(s) >= tau}
            # Reconstruct a pruned circuit with only the kept edges
            # Build survivors from kept edges
            kept_nodes: dict[int, set[int]] = {}
            for edge in kept_edges:
                for nd_ref in [edge.writer, edge.reader]:
                    nd = nd_ref.node
                    if nd.kind == "sae_feature" and nd.layer is not None and nd.neuron is not None:
                        kept_nodes.setdefault(nd.layer, set()).add(nd.neuron)

            new_survivors: dict[Site, list[AtPNode]] = {}
            for site, survivors in result.graph.survivors.items():
                layer = site.layer
                feats = kept_nodes.get(layer, set())
                new_survivors[site] = [
                    n for n in survivors
                    if n.node.neuron is not None and n.node.neuron in feats
                ]

            new_graph = SAEFeatureEdgeGraph(
                sites=result.graph.sites,
                survivors=new_survivors,
                edges=sorted(kept_edges.keys(), key=_sae_edge_sort_key),
            )
            result = SAEFeatureCircuit(
                nodes=result.nodes,
                edges=kept_edges,
                graph=new_graph,
                model=self._model,
                sae_sites=self._sae_sites,
                resolver=self._resolver,
            )

        if method in ("acdc", "both"):
            clean = kwargs.get("clean")
            corrupted = kwargs.get("corrupted")
            metric = kwargs.get("metric")
            if clean is None or corrupted is None or metric is None:
                raise ValueError(
                    "prune(method='acdc'|'both') requires keyword args: "
                    "clean=, corrupted=, metric="
                )
            acdc_runner = FeatureACDCRunner(
                model=self._model,
                sae_sites=self._sae_sites,
                resolver=self._resolver,
            )
            result = acdc_runner.run(
                clean, corrupted, metric,
                tau=tau,
                ablation_mode=ablation_mode,
                eap_skip_threshold=eap_skip_threshold,
                _initial_circuit=result,
            )

        return result


# ---------------------------------------------------------------------------
# SAEFeatureEdgeRunner
# ---------------------------------------------------------------------------


class SAEFeatureEdgeRunner:
    """Feature→feature SAE edge attribution via multi-site error-term splice + VJP.

    Composes (NOT subclasses) SAEFeatureRunner for Stage 1 node scoring.
    Only HF-eager (HFSiteResolver) + resid_post sites supported.
    TransformerLens → NotImplementedError.

    Usage:
        runner = SAEFeatureEdgeRunner(model, {site0: sae0, site1: sae1}, resolver)
        circuit = runner.run(clean, corrupted, metric)
        print(circuit.top_k(20))
    """

    def __init__(
        self,
        model: nn.Module,
        sae_sites: dict[Site, Any],
        resolver: Any,
    ) -> None:
        """
        Args:
            model:      The HF/toy model to analyse.
            sae_sites:  Mapping from Site → SAE object or (release, sae_id) tuple.
            resolver:   An HFSiteResolver. TLSiteResolver → NotImplementedError.
        """
        # Gate: no TL support
        from circuitry.patching.sites import TLSiteResolver
        if isinstance(resolver, TLSiteResolver):
            raise NotImplementedError(
                "SAEFeatureEdgeRunner does not support TransformerLens (TLSiteResolver) "
                "in v1.6.0. Use the HF-eager path (HFSiteResolver)."
            )

        self.model = model
        self.resolver = resolver

        # Resolve (release, sae_id) tuples; gate valid SAE components + validate architecture
        _VALID_SAE_COMPONENTS = {"resid_post", "mlp_out", "attn_out"}
        resolved: dict[Site, Any] = {}
        for site, sae_or_tuple in sae_sites.items():
            if site.component not in _VALID_SAE_COMPONENTS:
                raise NotImplementedError(
                    f"SAEFeatureEdgeRunner supports only {sorted(_VALID_SAE_COMPONENTS)} sites "
                    f"(got {site.component!r}). Per-head/per-neuron sub-slices "
                    "(attn_head_out, mlp_neuron, resid_pre, ...) are not supported."
                )
            if site.position is not None:
                raise NotImplementedError(
                    f"SAEFeatureEdgeRunner does not support positional slicing "
                    f"(site.position={site.position!r}). Only position=None is supported."
                )
            if isinstance(sae_or_tuple, tuple) and len(sae_or_tuple) == 2:
                from circuitry.sae.loader import load_sae
                release, sae_id = sae_or_tuple
                sae = load_sae(release, sae_id)
            else:
                sae = sae_or_tuple
            assert_supported_sae(sae)
            resolved[site] = sae

        # Enforce one SAE site per layer (temporary constraint; multi-site-per-layer is P2b)
        _seen_layers: dict[int, str] = {}
        for site in resolved:
            if site.layer in _seen_layers:
                raise NotImplementedError(
                    f"Multiple SAE sites in layer {site.layer} is not yet supported (P2b). "
                    f"Got {_seen_layers[site.layer]!r} and {site.component!r} in the same layer."
                )
            _seen_layers[site.layer] = site.component

        self._sae_sites = resolved

        # Compose SAEFeatureRunner for Stage 1
        self._stage1_runner = SAEFeatureRunner(model, resolved, resolver)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _call_model(self, inputs: _Inputs) -> Any:
        if isinstance(inputs, dict):
            return self.model(**inputs)
        return self.model(inputs)

    def _locate_layers(self) -> nn.ModuleList:
        return self._stage1_runner._atp._locate_layers(self.model)

    def _sorted_sites(self) -> list[tuple[Site, Any]]:
        """Sites sorted in forward order (ascending layer)."""
        return sorted(self._sae_sites.items(), key=lambda kv: kv[0].layer)

    def _model_dtype_device(self, layer_mod: nn.Module) -> tuple[torch.dtype, Any]:
        params = list(layer_mod.parameters())
        if params:
            return params[0].dtype, params[0].device
        return torch.float32, torch.device("cpu")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        clean_inputs: _Inputs,
        corrupted_inputs: _Inputs,
        metric: Callable[[Any], Tensor],
        *,
        layer_pairs: str = "adjacent",
        top_k_survivors: int = 32,
        max_edges: int | None = None,
        include_error_node: bool = False,
        variant: str = "attrib",
    ) -> SAEFeatureCircuit:
        """Compute feature→feature SAE edge scores.

        Args:
            clean_inputs / corrupted_inputs: model inputs.
            metric: differentiable scalar metric.
            layer_pairs: 'adjacent' (n-1 pairs) or 'all_forward' (n(n-1)/2 pairs).
            top_k_survivors: top-K active features/site to enumerate (Stage 1 cap).
            max_edges: global cap on returned edges (top-|score|).
            include_error_node: if True, sae_error nodes participate as WRITER endpoints
                (error→feature edges).  feature→error is structurally zero (downstream
                eps is a detached leaf) and is never computed.  Default False.
            variant: 'attrib' only in v1.6 Stage A. 'ig'/'exact' → NotImplementedError.

        Returns:
            SAEFeatureCircuit with .nodes (v1.5 scores), .edges, .graph.
        """
        if variant != "attrib":
            raise NotImplementedError(
                f"SAEFeatureEdgeRunner variant={variant!r} is not supported in v1.6.0 Stage A. "
                "Only 'attrib' is available. 'ig' and 'exact' are deferred."
            )
        if layer_pairs not in ("adjacent", "all_forward"):
            raise ValueError(
                f"layer_pairs must be 'adjacent' or 'all_forward', got {layer_pairs!r}"
            )

        was_training, orig_rg = self._stage1_runner._atp._freeze_eval()
        sae_orig_rg: dict[Site, dict[str, bool]] = {}
        for site, sae in self._sae_sites.items():
            sae_orig_rg[site] = _freeze_sae(sae)

        try:
            return self._run_inner(
                clean_inputs, corrupted_inputs, metric,
                layer_pairs=layer_pairs,
                top_k_survivors=top_k_survivors,
                max_edges=max_edges,
                include_error_node=include_error_node,
            )
        finally:
            self._stage1_runner._atp._restore(was_training, orig_rg)
            for site, sae in self._sae_sites.items():
                _restore_sae(sae, sae_orig_rg.get(site, {}))

    def _run_inner(
        self,
        clean_inputs: _Inputs,
        corrupted_inputs: _Inputs,
        metric: Callable[[Any], Tensor],
        *,
        layer_pairs: str,
        top_k_survivors: int,
        max_edges: int | None,
        include_error_node: bool,
    ) -> SAEFeatureCircuit:
        """Inner run (freeze/restore already applied by caller)."""
        sorted_sites = self._sorted_sites()  # forward order

        # ------------------------------------------------------------------
        # STAGE 1: run composed SAEFeatureRunner per-site → node scores
        # Keep top-K active survivors per site.
        # ------------------------------------------------------------------
        node_result = self._stage1_runner.run(
            clean_inputs, corrupted_inputs, metric,
            include_error_node=include_error_node,
            max_features=top_k_survivors,
        )

        # Group survivors by site layer
        site_survivors: dict[int, list[AtPNode]] = {}
        for site, _ in sorted_sites:
            layer = site.layer
            site_survivors[layer] = []

        for atp_node in node_result.scores:
            nd = atp_node.node
            if nd.layer is not None and nd.layer in site_survivors:
                site_survivors[nd.layer].append(atp_node)

        # Build EdgeGraph metadata
        edge_graph = SAEFeatureEdgeGraph(
            sites=sorted_sites,
            survivors={site: site_survivors.get(site.layer, []) for site, _ in sorted_sites},
            edges=[],  # filled below
        )

        # ------------------------------------------------------------------
        # STAGE 2: enumerate site pairs, compute edges via multi-site splice + VJP
        # ------------------------------------------------------------------
        all_edge_scores: dict[SAEFeatureEdge, float] = {}

        # Iterate over site pairs in forward order (writer before reader)
        for i, (writer_site, _writer_sae) in enumerate(sorted_sites):
            for j, (reader_site, _reader_sae) in enumerate(sorted_sites):
                if j <= i:
                    continue
                if layer_pairs == "adjacent" and j != i + 1:
                    continue

                writer_survivors = site_survivors.get(writer_site.layer, [])
                reader_survivors = site_survivors.get(reader_site.layer, [])

                # Skip if no survivors on either side
                if not writer_survivors and not reader_survivors:
                    continue

                pair_edges = self._compute_pair_edges(
                    clean_inputs=clean_inputs,
                    corrupted_inputs=corrupted_inputs,
                    metric=metric,
                    writer_site=writer_site,
                    reader_site=reader_site,
                    writer_survivors=writer_survivors,
                    reader_survivors=reader_survivors,
                    include_error_node=include_error_node,
                )
                all_edge_scores.update(pair_edges)

        # Global max_edges cap (top-|score|)
        if max_edges is not None and len(all_edge_scores) > max_edges:
            sorted_edges = sorted(all_edge_scores.items(), key=lambda kv: abs(kv[1]), reverse=True)
            all_edge_scores = dict(sorted_edges[:max_edges])

        edge_graph.edges = sorted(all_edge_scores.keys(), key=_sae_edge_sort_key)

        return SAEFeatureCircuit(
            nodes=node_result,
            edges=all_edge_scores,
            graph=edge_graph,
            model=self.model,
            sae_sites=self._sae_sites,
            resolver=self.resolver,
        )

    def _compute_pair_edges(
        self,
        clean_inputs: _Inputs,
        corrupted_inputs: _Inputs,
        metric: Callable[[Any], Tensor],
        writer_site: Site,
        reader_site: Site,
        writer_survivors: list[AtPNode],
        reader_survivors: list[AtPNode],
        include_error_node: bool,
    ) -> dict[SAEFeatureEdge, float]:
        """Stage 2 for a single (writer, reader) site pair.

        ONE clean forward with BOTH sites spliced simultaneously:
          - WRITER site: detached-leaf seed (f_U = encode(a).detach().requires_grad_(True))
          - READER site: LIVE non-detached encode (f_D = encode(a_live))
          - eps detached/frozen at BOTH sites (recon = x_hat + eps)

        Then: metric.backward(retain_graph=True) → f_D.grad = gradf_D
        For each downstream survivor j:
            G_j = zeros_like(f_D); G_j[..., j] = gradf_D[..., j]
            vjp_j = autograd.grad(f_D, f_U_leaf, grad_outputs=G_j)[0]
            for each upstream survivor i:
                edge(i→j) = float((Δf_U[..., i] * vjp_j[..., i]).to(float32).sum())
            del vjp_j  # FREE per-j

        Δf_U = f_U_corrupt − f_U_clean (captured via no_grad corrupted forward).
        """
        writer_sae = self._sae_sites[writer_site]
        reader_sae = self._sae_sites[reader_site]

        # Route through ResolvedSite (identity for resid_post + position=None)
        writer_resolved = self.resolver.resolve(self.model, writer_site)
        reader_resolved = self.resolver.resolve(self.model, reader_site)
        writer_layer_mod = writer_resolved.module
        reader_layer_mod = reader_resolved.module

        # ------------------------------------------------------------------
        # Step A: capture f_U_corrupt and f_D_corrupt (no_grad)
        # ------------------------------------------------------------------
        writer_corrupt_store: dict[str, Tensor] = {}
        reader_corrupt_store: dict[str, Tensor] = {}

        def _writer_corr_hook(
            module: nn.Module, inp: Any, output: Any,
            _sae: Any = writer_sae,
            _st: dict = writer_corrupt_store,
            _resolved: Any = writer_resolved,
        ) -> None:
            a = _routed_extract(_resolved, output).detach()
            a_in = a.to(getattr(_sae, "device", a.device), getattr(_sae, "dtype", a.dtype))
            with torch.no_grad():
                f_c, x_hat_c, eps_c = sae_decompose(_sae, a_in)
                _st["f"] = f_c.detach()
                _st["eps"] = eps_c.detach()  # captured for error→feature scoring

        def _reader_corr_hook(
            module: nn.Module, inp: Any, output: Any,
            _sae: Any = reader_sae,
            _st: dict = reader_corrupt_store,
            _resolved: Any = reader_resolved,
        ) -> None:
            a = _routed_extract(_resolved, output).detach()
            a_in = a.to(getattr(_sae, "device", a.device), getattr(_sae, "dtype", a.dtype))
            with torch.no_grad():
                f_c = _sae.encode(a_in)
                _st["f"] = f_c.detach()

        wh_corr = writer_layer_mod.register_forward_hook(_writer_corr_hook)
        rh_corr = reader_layer_mod.register_forward_hook(_reader_corr_hook)
        try:
            with torch.no_grad():
                self._call_model(corrupted_inputs)
        finally:
            wh_corr.remove()
            rh_corr.remove()

        f_U_corrupt = writer_corrupt_store.get("f")
        eps_U_corrupt = writer_corrupt_store.get("eps")  # for error→feature delta
        if f_U_corrupt is None:
            return {}

        # ------------------------------------------------------------------
        # Step B: capture f_U_clean (detached leaf) and f_D_clean (live)
        # via ONE simultaneous spliced clean forward + backward
        # ------------------------------------------------------------------
        writer_clean_store: dict[str, Tensor] = {}
        reader_clean_store: dict[str, Tensor] = {}

        # Determine model dtype/device for cast-back
        w_dtype, w_device = self._model_dtype_device(writer_layer_mod)
        r_dtype, r_device = self._model_dtype_device(reader_layer_mod)

        def _writer_clean_hook(
            module: nn.Module, inp: Any, output: Any,
            _sae: Any = writer_sae,
            _st: dict = writer_clean_store,
            _mdtype: torch.dtype = w_dtype,
            _mdev: Any = w_device,
            _include_err: bool = include_error_node,
            _resolved: Any = writer_resolved,
        ) -> Any:
            """WRITER site: detached-leaf seed so f_U.grad receives the VJP.

            When include_error_node=True, also introduce an independent leaf
            err_leaf_U for the error term so that error→feature edges can be
            computed.  The reconstruction becomes decode(f_U) + err_leaf_U
            instead of decode(f_U) + eps (frozen scalar), giving the error term
            a live gradient path to f_D for the VJP.
            """
            a = _routed_extract(_resolved, output)
            a_in = a.detach().to(getattr(_sae, "device", a.device), getattr(_sae, "dtype", a.dtype))
            # Detached-leaf seed (§2.1 WRITER construction)
            f_U = _sae.encode(a_in).detach().requires_grad_(True)
            f_U.retain_grad()
            x_hat = _sae.decode(f_U)
            eps = (a_in - x_hat).detach()  # frozen clean eps

            if _include_err:
                # Independent leaf for the error term — receives VJP from f_D
                err_leaf_U = eps.detach().clone().requires_grad_(True)
                err_leaf_U.retain_grad()
                recon = x_hat + err_leaf_U
                _st["err_leaf_U"] = err_leaf_U
            else:
                recon = x_hat + eps

            _st["f_U"] = f_U
            _st["eps"] = eps
            recon_cast = recon.to(_mdev, _mdtype)
            return _routed_inject(_resolved, output, recon_cast)

        def _reader_clean_hook(
            module: nn.Module, inp: Any, output: Any,
            _sae: Any = reader_sae,
            _st: dict = reader_clean_store,
            _mdtype: torch.dtype = r_dtype,
            _mdev: Any = r_device,
            _resolved: Any = reader_resolved,
        ) -> Any:
            """READER site: LIVE non-detached encode so grad flows from metric to f_D."""
            a = _routed_extract(_resolved, output)
            a_in = a.to(getattr(_sae, "device", a.device), getattr(_sae, "dtype", a.dtype))
            # NOTE: a_in is NOT detached — live activation flows into encode
            f_D, x_hat, eps = sae_decompose(_sae, a_in)
            # eps is already detached by sae_decompose (frozen at clean)
            f_D.retain_grad()
            recon = x_hat + eps
            _st["f_D"] = f_D
            _st["eps"] = eps
            recon_cast = recon.to(_mdev, _mdtype)
            return _routed_inject(_resolved, output, recon_cast)

        wh_clean = writer_layer_mod.register_forward_hook(_writer_clean_hook)
        rh_clean = reader_layer_mod.register_forward_hook(_reader_clean_hook)
        try:
            with torch.enable_grad():
                out = self._call_model(clean_inputs)
                m = metric(out)
                if not (isinstance(m, Tensor) and m.requires_grad):
                    raise RuntimeError(
                        "metric must return a differentiable Tensor. "
                        "Use a logit_diff_t-style metric, not a float."
                    )
                m.backward(retain_graph=True)
        finally:
            wh_clean.remove()
            rh_clean.remove()

        f_U_leaf = writer_clean_store.get("f_U")
        f_D_live = reader_clean_store.get("f_D")
        err_leaf_U = writer_clean_store.get("err_leaf_U")  # only present when include_error_node

        if f_U_leaf is None or f_D_live is None:
            return {}

        if f_D_live.grad is None:
            raise RuntimeError(
                "f_D.grad is None after backward — the metric must be differentiable "
                f"and connected to both SAE sites. Writer: {writer_site}, Reader: {reader_site}."
            )

        # ------------------------------------------------------------------
        # Step C: compute Δf_U = f_U_corrupt − f_U_clean (fp32, device-aligned)
        # ------------------------------------------------------------------
        f_U_clean_fp32 = f_U_leaf.detach().float()
        f_U_corrupt_fp32 = f_U_corrupt.to(f_U_clean_fp32.device, torch.float32)

        if f_U_corrupt_fp32.shape != f_U_clean_fp32.shape:
            raise ValueError(
                f"Shape mismatch for writer site {writer_site}: "
                f"clean shape={f_U_clean_fp32.shape}, corrupt shape={f_U_corrupt_fp32.shape}. "
                "Ensure clean and corrupted inputs have the same sequence length."
            )

        delta_f_U = f_U_corrupt_fp32 - f_U_clean_fp32  # corrupt − clean

        # Δeps_U for error→feature scoring (only needed when include_error_node=True)
        delta_eps_U: Tensor | None = None
        if include_error_node and eps_U_corrupt is not None and err_leaf_U is not None:
            eps_U_clean_fp32 = err_leaf_U.detach().float()
            eps_U_corrupt_fp32 = eps_U_corrupt.to(eps_U_clean_fp32.device, torch.float32)
            delta_eps_U = eps_U_corrupt_fp32 - eps_U_clean_fp32  # corrupt − clean

        # ------------------------------------------------------------------
        # Step D: extract downstream survivors (feature indices and error flag)
        # ------------------------------------------------------------------
        downstream_feat_indices: list[int] = []
        # Note: error-node edge endpoints deferred to Stage B (include_error_node hook)
        for atp_node in reader_survivors:
            nd = atp_node.node
            if nd.kind == "sae_feature" and nd.neuron is not None:
                downstream_feat_indices.append(nd.neuron)

        upstream_feat_indices: list[int] = []
        for atp_node in writer_survivors:
            nd = atp_node.node
            if nd.kind == "sae_feature" and nd.neuron is not None:
                upstream_feat_indices.append(nd.neuron)

        gradf_D = f_D_live.grad  # shape same as f_D_live; not yet fp32

        edge_scores: dict[SAEFeatureEdge, float] = {}

        # Determine grad sources for VJP:  always f_U_leaf; optionally err_leaf_U
        # We compute vjp w.r.t. both in a single autograd.grad call when possible.
        grad_inputs_list: list[Tensor] = [f_U_leaf]
        if include_error_node and err_leaf_U is not None:
            grad_inputs_list.append(err_leaf_U)

        # For each downstream survivor j, compute VJP and dot with Δf_U[:, i]
        for j in downstream_feat_indices:
            G_j = torch.zeros_like(f_D_live)
            G_j[..., j] = gradf_D[..., j]

            try:
                vjp_results = torch.autograd.grad(
                    f_D_live, grad_inputs_list,
                    grad_outputs=G_j,
                    retain_graph=True,
                    allow_unused=True,
                )
            except RuntimeError:
                # f_D not connected to grad inputs (e.g. no path exists)
                del G_j
                continue

            vjp_j = vjp_results[0]
            vjp_err_j = vjp_results[1] if len(vjp_results) > 1 else None

            if vjp_j is None and vjp_err_j is None:
                del G_j
                continue

            # component=None for resid_post (preserves v1.6 node identity); explicit for others
            _r_comp = reader_site.component if reader_site.component != "resid_post" else None
            _w_comp = writer_site.component if writer_site.component != "resid_post" else None
            reader_node = AtPNode(Node("sae_feature", layer=reader_site.layer, neuron=j, component=_r_comp))

            # feature→feature edges
            if vjp_j is not None:
                # Slice to upstream survivors immediately (memory discipline)
                vjp_j_fp32 = vjp_j.to(torch.float32)

                for i in upstream_feat_indices:
                    score = float(
                        (delta_f_U[..., i] * vjp_j_fp32[..., i].to(delta_f_U.device)).to(torch.float32).sum()
                    )
                    writer_node = AtPNode(Node("sae_feature", layer=writer_site.layer, neuron=i, component=_w_comp))
                    edge = SAEFeatureEdge(writer=writer_node, reader=reader_node)
                    edge_scores[edge] = score

                del vjp_j, vjp_j_fp32

            # error→feature edges (include_error_node=True only)
            if include_error_node and vjp_err_j is not None and delta_eps_U is not None:
                vjp_err_j_fp32 = vjp_err_j.to(torch.float32)
                # Score: Σ_pos (Δeps_U * vjp_err_j) summed over all positions
                # delta_eps_U has same shape as eps (batch, seq, d_model); vjp_err_j
                # has the same shape.  We sum the element-wise product over all positions.
                score_err = float(
                    (delta_eps_U.to(vjp_err_j_fp32.device) * vjp_err_j_fp32).to(torch.float32).sum()
                )
                error_writer_node = AtPNode(Node("sae_error", layer=writer_site.layer, component=_w_comp))
                err_edge = SAEFeatureEdge(writer=error_writer_node, reader=reader_node)
                edge_scores[err_edge] = score_err
                del vjp_err_j, vjp_err_j_fp32

            del G_j  # FREE per-j (memory discipline)

        return edge_scores

    def bruteforce_feature_edge_scores(
        self,
        clean_inputs: _Inputs,
        corrupted_inputs: _Inputs,
        metric: Callable[[Any], Tensor],
        edges: list[SAEFeatureEdge],
    ) -> dict[SAEFeatureEdge, float]:
        """Independent ground truth for edge scores.

        For each edge (U:i → D:j): patch ONLY upstream feature i clean→corrupted
        in the spliced forward, measure induced Δ(f_D[j]) · gradf_D[j]
        (FEATURE-level effect, NOT metric-level).

        eps frozen at clean at EVERY site.
        NEVER derived from the analytic scores — completely independent path.
        """
        was_training, orig_rg = self._stage1_runner._atp._freeze_eval()
        sae_orig_rg: dict[Site, dict[str, bool]] = {}
        for site, sae in self._sae_sites.items():
            sae_orig_rg[site] = _freeze_sae(sae)

        try:
            return self._bruteforce_inner(clean_inputs, corrupted_inputs, metric, edges)
        finally:
            self._stage1_runner._atp._restore(was_training, orig_rg)
            for site, sae in self._sae_sites.items():
                _restore_sae(sae, sae_orig_rg.get(site, {}))

    def _bruteforce_inner(
        self,
        clean_inputs: _Inputs,
        corrupted_inputs: _Inputs,
        metric: Callable[[Any], Tensor],
        edges: list[SAEFeatureEdge],
    ) -> dict[SAEFeatureEdge, float]:
        """Inner bruteforce (freeze/restore already applied by caller).

        Handles both feature→feature and error→feature edges:
          - feature→feature: patch f_U[i] clean→corrupted, measure Δf_D[j]·gradf_D[j].
          - error→feature: patch eps_U clean→corrupted (eps_U_corrupt) in spliced forward
            while keeping f_U clean, measure Δf_D[j]·gradf_D[j].
        """
        # Separate edges by writer kind
        # pair_to_feat_edges: (w_layer, w_comp, r_layer, r_comp) → [feature→feature edges]
        # pair_to_err_edges:  (w_layer, w_comp, r_layer, r_comp) → [error→feature edges]
        # In P2a one-component-per-layer is enforced, so layer alone uniquely identifies a site.
        # Keys include component for forward-compatibility with P2b (multi-site-per-layer).
        pair_to_feat_edges: dict[tuple, list[SAEFeatureEdge]] = {}
        pair_to_err_edges: dict[tuple, list[SAEFeatureEdge]] = {}

        for edge in edges:
            w_node = edge.writer.node
            r_node = edge.reader.node
            w_layer = w_node.layer
            r_layer = r_node.layer
            if w_layer is None or r_layer is None:
                continue
            # component stored on node; None means resid_post for legacy compatibility
            w_comp = w_node.component or "resid_post"
            r_comp = r_node.component or "resid_post"
            key = (w_layer, w_comp, r_layer, r_comp)
            if w_node.kind == "sae_error":
                pair_to_err_edges.setdefault(key, []).append(edge)
            else:
                pair_to_feat_edges.setdefault(key, []).append(edge)

        # Merge for site-pair iteration (process feature and error edges per pair)
        all_pairs: set[tuple] = set(pair_to_feat_edges) | set(pair_to_err_edges)
        pair_to_edges: dict[tuple, list[SAEFeatureEdge]] = {}
        for key in all_pairs:
            pair_to_edges[key] = (
                pair_to_feat_edges.get(key, []) + pair_to_err_edges.get(key, [])
            )

        result: dict[SAEFeatureEdge, float] = {}

        for (w_layer, w_comp, r_layer, r_comp), pair_edges in pair_to_edges.items():
            # Find the sites by (layer, component) — component None → resid_post
            writer_site = next(
                (s for s in self._sae_sites
                 if s.layer == w_layer and s.component == w_comp), None
            )
            reader_site = next(
                (s for s in self._sae_sites
                 if s.layer == r_layer and s.component == r_comp), None
            )
            if writer_site is None or reader_site is None:
                for e in pair_edges:
                    result[e] = 0.0
                continue

            writer_sae = self._sae_sites[writer_site]
            reader_sae = self._sae_sites[reader_site]
            # Route through ResolvedSite (identity for resid_post + position=None)
            writer_resolved = self.resolver.resolve(self.model, writer_site)
            reader_resolved = self.resolver.resolve(self.model, reader_site)
            writer_layer_mod = writer_resolved.module
            reader_layer_mod = reader_resolved.module

            w_dtype, w_device = self._model_dtype_device(writer_layer_mod)
            r_dtype, r_device = self._model_dtype_device(reader_layer_mod)

            # ------------------------------------------------------------------
            # A: Capture f_U_corrupt AND eps_U_corrupt
            # ------------------------------------------------------------------
            wu_store: dict[str, Tensor] = {}

            def _wu_hook(
                module: nn.Module, inp: Any, output: Any,
                _sae: Any = writer_sae, _st: dict = wu_store,
                _resolved: Any = writer_resolved,
            ) -> None:
                a = _routed_extract(_resolved, output).detach()
                a_in = a.to(getattr(_sae, "device", a.device), getattr(_sae, "dtype", a.dtype))
                with torch.no_grad():
                    f_c, _x_hat_c, eps_c = sae_decompose(_sae, a_in)
                    _st["f"] = f_c.detach()
                    _st["eps"] = eps_c.detach()

            h_wu = writer_layer_mod.register_forward_hook(_wu_hook)
            try:
                with torch.no_grad():
                    self._call_model(corrupted_inputs)
            finally:
                h_wu.remove()

            f_U_corrupt = wu_store.get("f")
            eps_U_corrupt_bf = wu_store.get("eps")  # corrupted eps for error-writer bruteforce
            if f_U_corrupt is None:
                for e in pair_edges:
                    result[e] = 0.0
                continue

            # ------------------------------------------------------------------
            # B: Baseline spliced clean forward:
            #    capture f_D_clean + gradf_D via retain_grad + backward
            # ------------------------------------------------------------------
            wd_store: dict[str, Tensor] = {}
            rd_store: dict[str, Tensor] = {}

            def _wd_baseline_hook(
                module: nn.Module, inp: Any, output: Any,
                _sae: Any = writer_sae, _st: dict = wd_store,
                _mdtype: torch.dtype = w_dtype, _mdev: Any = w_device,
                _resolved: Any = writer_resolved,
            ) -> Any:
                """Writer splice (frozen-leaf): so we can later patch f[i]."""
                a = _routed_extract(_resolved, output)
                a_in = a.detach().to(getattr(_sae, "device", a.device), getattr(_sae, "dtype", a.dtype))
                f_W = _sae.encode(a_in).detach().requires_grad_(True)
                f_W.retain_grad()
                x_hat = _sae.decode(f_W)
                eps = (a_in - x_hat).detach()
                recon_cast = (x_hat + eps).to(_mdev, _mdtype)
                _st["f_W"] = f_W
                _st["x_hat_clean"] = x_hat.detach()
                _st["eps_clean"] = eps
                return _routed_inject(_resolved, output, recon_cast)

            def _rd_baseline_hook(
                module: nn.Module, inp: Any, output: Any,
                _sae: Any = reader_sae, _st: dict = rd_store,
                _mdtype: torch.dtype = r_dtype, _mdev: Any = r_device,
                _resolved: Any = reader_resolved,
            ) -> Any:
                """Reader splice (live): f_D in graph for retain_grad."""
                a = _routed_extract(_resolved, output)
                a_in = a.to(getattr(_sae, "device", a.device), getattr(_sae, "dtype", a.dtype))
                f_D, x_hat, eps = sae_decompose(_sae, a_in)
                f_D.retain_grad()
                recon_cast = (x_hat + eps).to(_mdev, _mdtype)
                _st["f_D"] = f_D
                return _routed_inject(_resolved, output, recon_cast)

            wh_bl = writer_layer_mod.register_forward_hook(_wd_baseline_hook)
            rh_bl = reader_layer_mod.register_forward_hook(_rd_baseline_hook)
            try:
                with torch.enable_grad():
                    out_bl = self._call_model(clean_inputs)
                    m_bl = metric(out_bl)
                    m_bl.backward(retain_graph=False)
            finally:
                wh_bl.remove()
                rh_bl.remove()

            f_D_baseline = rd_store.get("f_D")
            f_W_baseline = wd_store.get("f_W")
            x_hat_W_clean = wd_store.get("x_hat_clean")
            eps_W_clean = wd_store.get("eps_clean")

            if f_D_baseline is None or f_D_baseline.grad is None:
                for e in pair_edges:
                    result[e] = 0.0
                continue

            gradf_D = f_D_baseline.grad.detach()  # shape of f_D

            # Group feature-writer edges by upstream feature index
            # Group error-writer edges separately
            by_upstream: dict[int, list[SAEFeatureEdge]] = {}
            error_writer_edges: list[SAEFeatureEdge] = []
            for e in pair_edges:
                if e.writer.node.kind == "sae_error":
                    error_writer_edges.append(e)
                else:
                    ui = e.writer.node.neuron
                    if ui is not None:
                        by_upstream.setdefault(ui, []).append(e)

            # ------------------------------------------------------------------
            # C: For each upstream feature i, patch f_U[i] = f_U_corrupt[i],
            #    re-run, capture f_D_patched, compute Δf_D[j] · gradf_D[j]
            # ------------------------------------------------------------------
            for i, i_edges in by_upstream.items():
                # Build patched writer state
                f_W_patched = f_W_baseline.detach().clone()
                f_W_patched[..., i] = f_U_corrupt[..., i].to(f_W_patched.device, f_W_patched.dtype)

                # Re-run with patched writer and live reader
                rd_patch_store: dict[str, Tensor] = {}

                def _wd_patch_hook(
                    module: nn.Module, inp: Any, output: Any,
                    _sae: Any = writer_sae,
                    _f_patched: Tensor = f_W_patched,
                    _x_hat_clean: Tensor = x_hat_W_clean,
                    _eps_clean: Tensor = eps_W_clean,
                    _mdtype: torch.dtype = w_dtype,
                    _mdev: Any = w_device,
                    _resolved: Any = writer_resolved,
                ) -> Any:
                    """Inject patched reconstruction."""
                    recon_cast = (_sae.decode(_f_patched) + _eps_clean).to(_mdev, _mdtype)
                    return _routed_inject(_resolved, output, recon_cast)

                def _rd_patch_hook(
                    module: nn.Module, inp: Any, output: Any,
                    _sae: Any = reader_sae,
                    _st: dict = rd_patch_store,
                    _mdtype: torch.dtype = r_dtype,
                    _mdev: Any = r_device,
                    _resolved: Any = reader_resolved,
                ) -> Any:
                    """Capture patched f_D."""
                    a = _routed_extract(_resolved, output)
                    a_in = a.detach().to(
                        getattr(_sae, "device", a.device), getattr(_sae, "dtype", a.dtype)
                    )
                    with torch.no_grad():
                        f_D_p = _sae.encode(a_in)
                    _st["f_D_patched"] = f_D_p.detach()
                    # Also splice losslessly (eps frozen at clean from current input)
                    f_dp_leaf, x_hat_p, eps_p = sae_decompose(_sae, a_in)
                    recon_cast = (x_hat_p + eps_p).to(_mdev, _mdtype)
                    return _routed_inject(_resolved, output, recon_cast)

                wh_p = writer_layer_mod.register_forward_hook(_wd_patch_hook)
                rh_p = reader_layer_mod.register_forward_hook(_rd_patch_hook)
                try:
                    with torch.no_grad():
                        self._call_model(clean_inputs)
                finally:
                    wh_p.remove()
                    rh_p.remove()

                f_D_patched = rd_patch_store.get("f_D_patched")
                if f_D_patched is None:
                    for e in i_edges:
                        result[e] = 0.0
                    continue

                # Δf_D = f_D_patched - f_D_baseline (fp32 device-aligned)
                f_D_base_fp32 = f_D_baseline.detach().float()
                f_D_patch_fp32 = f_D_patched.to(f_D_base_fp32.device, torch.float32)
                delta_f_D = f_D_patch_fp32 - f_D_base_fp32

                gradf_D_fp32 = gradf_D.to(f_D_base_fp32.device, torch.float32)

                for e in i_edges:
                    j = e.reader.node.neuron
                    if j is None:
                        result[e] = 0.0
                        continue
                    # Feature-level score: Δf_D[j] · gradf_D[j] summed over positions
                    score = float(
                        (delta_f_D[..., j] * gradf_D_fp32[..., j]).to(torch.float32).sum()
                    )
                    result[e] = score

            # ------------------------------------------------------------------
            # D: Error-writer bruteforce
            #    For sae_error@U → sae_feature@D[j]:
            #    patch eps_U clean→corrupted (i.e. use eps_U_corrupt instead of
            #    eps_U_clean in the writer splice), keep f_U at its clean value,
            #    measure Δf_D[j] · gradf_D[j].
            # ------------------------------------------------------------------
            if error_writer_edges and eps_U_corrupt_bf is not None:
                # Capture eps_U_clean from the baseline writer store
                eps_W_clean_for_err = wd_store.get("eps_clean")
                if eps_W_clean_for_err is None:
                    # Fall back: compute via sae_decompose on clean inputs
                    eps_clean_store_err: dict[str, Tensor] = {}

                    def _eps_clean_hook(
                        module: nn.Module, inp: Any, output: Any,
                        _sae: Any = writer_sae, _st: dict = eps_clean_store_err,
                        _resolved: Any = writer_resolved,
                    ) -> None:
                        a = _routed_extract(_resolved, output).detach()
                        a_in = a.to(getattr(_sae, "device", a.device), getattr(_sae, "dtype", a.dtype))
                        with torch.no_grad():
                            _f, _x, eps_c = sae_decompose(_sae, a_in)
                        _st["eps"] = eps_c.detach()

                    _h_ec = writer_layer_mod.register_forward_hook(_eps_clean_hook)
                    try:
                        with torch.no_grad():
                            self._call_model(clean_inputs)
                    finally:
                        _h_ec.remove()
                    eps_W_clean_for_err = eps_clean_store_err.get("eps")

                if eps_W_clean_for_err is None:
                    for e in error_writer_edges:
                        result[e] = 0.0
                else:
                    # Re-run with eps_U patched to corrupted; f_U kept at clean decode
                    eps_U_corrupt_dev = eps_U_corrupt_bf.to(
                        eps_W_clean_for_err.device, eps_W_clean_for_err.dtype
                    )
                    # Patched clean reconstruction at writer: decode(f_U_clean) + eps_U_corrupt
                    f_W_clean_for_err = f_W_baseline.detach() if f_W_baseline is not None else None

                    rd_err_store: dict[str, Tensor] = {}

                    def _wd_err_patch_hook(
                        module: nn.Module, inp: Any, output: Any,
                        _sae: Any = writer_sae,
                        _f_clean: Tensor | None = f_W_clean_for_err,
                        _eps_corrupt: Tensor = eps_U_corrupt_dev,
                        _mdtype: torch.dtype = w_dtype,
                        _mdev: Any = w_device,
                        _resolved: Any = writer_resolved,
                    ) -> Any:
                        """Inject error-patched reconstruction: decode(f_U_clean) + eps_U_corrupt."""
                        a = _routed_extract(_resolved, output)
                        a_in = a.detach().to(getattr(_sae, "device", a.device), getattr(_sae, "dtype", a.dtype))
                        if _f_clean is not None:
                            x_hat = _sae.decode(_f_clean.to(a_in.device, a_in.dtype))
                        else:
                            f_c = _sae.encode(a_in)
                            x_hat = _sae.decode(f_c)
                        recon_cast = (x_hat + _eps_corrupt.to(x_hat.device, x_hat.dtype)).to(_mdev, _mdtype)
                        return _routed_inject(_resolved, output, recon_cast)

                    def _rd_err_patch_hook(
                        module: nn.Module, inp: Any, output: Any,
                        _sae: Any = reader_sae,
                        _st: dict = rd_err_store,
                        _mdtype: torch.dtype = r_dtype,
                        _mdev: Any = r_device,
                        _resolved: Any = reader_resolved,
                    ) -> Any:
                        """Capture patched f_D for error-writer patch."""
                        a = _routed_extract(_resolved, output)
                        a_in = a.detach().to(
                            getattr(_sae, "device", a.device), getattr(_sae, "dtype", a.dtype)
                        )
                        with torch.no_grad():
                            f_D_p = _sae.encode(a_in)
                        _st["f_D_patched"] = f_D_p.detach()
                        f_dp_leaf, x_hat_p, eps_p = sae_decompose(_sae, a_in)
                        recon_cast = (x_hat_p + eps_p).to(_mdev, _mdtype)
                        return _routed_inject(_resolved, output, recon_cast)

                    wh_ep = writer_layer_mod.register_forward_hook(_wd_err_patch_hook)
                    rh_ep = reader_layer_mod.register_forward_hook(_rd_err_patch_hook)
                    try:
                        with torch.no_grad():
                            self._call_model(clean_inputs)
                    finally:
                        wh_ep.remove()
                        rh_ep.remove()

                    f_D_err_patched = rd_err_store.get("f_D_patched")
                    if f_D_err_patched is None:
                        for e in error_writer_edges:
                            result[e] = 0.0
                    else:
                        f_D_base_fp32_err = f_D_baseline.detach().float()
                        f_D_ep_fp32 = f_D_err_patched.to(f_D_base_fp32_err.device, torch.float32)
                        delta_f_D_err = f_D_ep_fp32 - f_D_base_fp32_err
                        gradf_D_err_fp32 = gradf_D.to(f_D_base_fp32_err.device, torch.float32)

                        for e in error_writer_edges:
                            j = e.reader.node.neuron
                            if j is None:
                                result[e] = 0.0
                                continue
                            score_e = float(
                                (delta_f_D_err[..., j] * gradf_D_err_fp32[..., j]).to(torch.float32).sum()
                            )
                            result[e] = score_e

        return result


# ---------------------------------------------------------------------------
# FeatureACDCRunner — greedy node-pruning (§4.3)
# ---------------------------------------------------------------------------


class FeatureACDCRunner:
    """Greedy reverse-topological NODE pruning for SAE feature circuits.

    Maps onto the ACDCRunner control-flow pattern (accept removal if
    KL_new − KL_current < tau) but operates at the NODE level over SAE
    feature sites rather than the edge/head level.

    Stage 1: SAEFeatureEdgeRunner.run() → node scores + visit order.
    Greedy pruning: reverse-topo over sites (later layers first),
    within a site weakest |AtP node score| first.

    Usage:
        runner = FeatureACDCRunner(model, sae_sites, resolver)
        circuit = runner.run(clean, corrupted, metric, tau=0.05)
        table   = runner.sweep(clean, corrupted, metric, taus=[0.01, 0.05, 0.1])
    """

    def __init__(
        self,
        model: nn.Module,
        sae_sites: dict[Site, Any],
        resolver: Any,
    ) -> None:
        """
        Args:
            model:      The HF/toy model.
            sae_sites:  Mapping Site → SAE (pre-resolved; no (release, sae_id) tuples here).
            resolver:   HFSiteResolver (used to build SAEFeatureEdgeRunner for Stage 1).
        """
        if resolver is None:
            raise ValueError("resolver must be provided to FeatureACDCRunner.")
        self.model = model
        self._sae_sites = sae_sites
        self._resolver = resolver

        # Build edge runner for Stage 1
        self._edge_runner = SAEFeatureEdgeRunner(model, sae_sites, resolver)

    def _sorted_sites(self) -> list[tuple[Site, Any]]:
        """Sites in forward order (ascending layer)."""
        return sorted(self._sae_sites.items(), key=lambda kv: kv[0].layer)

    def _call_model(self, inputs: _Inputs) -> Any:
        if isinstance(inputs, dict):
            return self.model(**inputs)
        return self.model(inputs)

    @staticmethod
    def _logits_of(out: Any) -> Any:
        return out.logits if hasattr(out, "logits") else out

    def _recovery_kl(self, circuit_logits: Any, clean_logits: Any) -> float:
        """KL(circuit ‖ clean) at last token position."""
        from circuitry.core.patching import kl_divergence
        lc = self._logits_of(circuit_logits)
        lq = self._logits_of(clean_logits)
        # Use last token position
        if lc.ndim >= 3:
            lc = lc[:, -1:, :]
            lq = lq[:, -1:, :]
        return kl_divergence(lc, lq)

    def run(
        self,
        clean: _Inputs,
        corrupted: _Inputs,
        metric: Callable[[Any], Tensor],
        *,
        tau: float = 0.05,
        ablation_mode: str = "corrupted",
        eap_skip_threshold: float | None = None,
        include_error_node: bool = False,
        top_k_survivors: int = 32,
        _initial_circuit: SAEFeatureCircuit | None = None,
    ) -> SAEFeatureCircuit:
        """Greedy node pruning. Returns the pruned SAEFeatureCircuit.

        Stage 1: run SAEFeatureEdgeRunner to get node scores (unless _initial_circuit given).
        Greedy: reverse-topo over sites, weakest |AtP node score| first within a site.
        Accept removal if KL_new − KL_current < tau.

        NOTE: eap_skip_threshold — nodes with |score| above this are kept without testing.

        Args:
            clean / corrupted:   Model inputs.
            metric:              Scalar metric (must return a Tensor).
            tau:                 KL tolerance for accepting node removal.
            ablation_mode:       'corrupted' / 'zero' / 'mean'.
            eap_skip_threshold:  Skip test for nodes with |score| > threshold.
            include_error_node:  Include sae_error as a node (§4.4).
            top_k_survivors:     Top-K survivors for Stage 1 (ignored if _initial_circuit given).
            _initial_circuit:    Pre-computed circuit to prune (skip Stage 1).

        Returns:
            SAEFeatureCircuit with pruned node set + induced edges.
        """
        # ------------------------------------------------------------------
        # Stage 1: Get node scores for initial circuit
        # ------------------------------------------------------------------
        if _initial_circuit is not None:
            base_circuit = _initial_circuit
        elif self._edge_runner is not None:
            base_circuit = self._edge_runner.run(
                clean, corrupted, metric,
                top_k_survivors=top_k_survivors,
                include_error_node=include_error_node,
            )
        else:
            raise RuntimeError("No edge runner available and no _initial_circuit provided.")

        # Build node score lookup: (layer, feat_idx) → |score|
        # nodes.scores is a dict[AtPNode, float]
        node_scores: dict[tuple[int, int], float] = {}
        for atp_node, score in base_circuit.nodes.scores.items():
            nd = atp_node.node
            if nd.kind == "sae_feature" and nd.layer is not None and nd.neuron is not None:
                node_scores[(nd.layer, nd.neuron)] = float(abs(score))

        # ------------------------------------------------------------------
        # Build reverse-topo order: later layers first, then weakest score first within site
        # ------------------------------------------------------------------
        sorted_sites = self._sorted_sites()
        # Reverse topo: later layers first
        rev_sorted_sites = list(reversed(sorted_sites))

        # Collect all current kept nodes per site (from base_circuit survivors)
        kept_nodes: dict[int, set[int]] = {}
        for site, survivors in base_circuit.graph.survivors.items():
            layer = site.layer
            kept_nodes[layer] = set()
            for atp_node in survivors:
                nd = atp_node.node
                if nd.kind == "sae_feature" and nd.neuron is not None:
                    kept_nodes[layer].add(nd.neuron)

        # ------------------------------------------------------------------
        # Compute ablation values once (corrupted/zero/mean)
        # ------------------------------------------------------------------
        ablation_values = base_circuit._compute_ablation_values(clean, corrupted, ablation_mode)

        # ------------------------------------------------------------------
        # Clean model forward for KL baseline
        # ------------------------------------------------------------------
        with torch.no_grad():
            clean_out = self._call_model(clean)

        current_kl = 0.0  # KL of full circuit vs clean ≈ 0

        # ------------------------------------------------------------------
        # Greedy pruning loop
        # ------------------------------------------------------------------
        for site, _sae in rev_sorted_sites:
            layer = site.layer
            feats_in_layer = list(kept_nodes.get(layer, set()))

            # Sort by weakest |score| first (ascending)
            feats_in_layer.sort(key=lambda fi: node_scores.get((layer, fi), 0.0))

            for feat_i in feats_in_layer:
                score_i = node_scores.get((layer, feat_i), 0.0)

                # eap_skip_threshold: keep high-score nodes without testing
                if eap_skip_threshold is not None and score_i > eap_skip_threshold:
                    continue

                # Tentatively remove feat_i
                kept_nodes[layer].discard(feat_i)

                # Run ablation forward with the tentative kept set
                circuit_out = _feature_circuit_forward(
                    self.model,
                    clean,
                    self._sae_sites,
                    self._resolver,
                    circuit_nodes=kept_nodes,
                    ablation_values=ablation_values,
                    include_error_node=include_error_node,
                )

                new_kl = self._recovery_kl(circuit_out, clean_out)

                # Accept removal if KL increase is within tolerance
                if new_kl - current_kl < tau:
                    current_kl = new_kl
                    # feat_i stays removed
                else:
                    # Reject: put feat_i back
                    kept_nodes[layer].add(feat_i)

        # ------------------------------------------------------------------
        # Build pruned SAEFeatureCircuit from kept_nodes
        # ------------------------------------------------------------------
        # Induce edges: keep only edges where both writer and reader are in kept_nodes
        kept_edges: dict[SAEFeatureEdge, float] = {}
        for edge, score in base_circuit.edges.items():
            w = edge.writer.node
            r = edge.reader.node
            w_in = (
                w.layer is not None
                and w.neuron is not None
                and w.neuron in kept_nodes.get(w.layer, set())
            )
            r_in = (
                r.layer is not None
                and r.neuron is not None
                and r.neuron in kept_nodes.get(r.layer, set())
            )
            if w_in and r_in:
                kept_edges[edge] = score

        # Build new survivors
        new_survivors: dict[Site, list[AtPNode]] = {}
        for site, survivors in base_circuit.graph.survivors.items():
            layer = site.layer
            feats = kept_nodes.get(layer, set())
            new_survivors[site] = [
                n for n in survivors
                if n.node.kind == "sae_feature"
                and n.node.neuron is not None
                and n.node.neuron in feats
            ]

        new_graph = SAEFeatureEdgeGraph(
            sites=base_circuit.graph.sites,
            survivors=new_survivors,
            edges=sorted(kept_edges.keys(), key=_sae_edge_sort_key),
        )

        return SAEFeatureCircuit(
            nodes=base_circuit.nodes,
            edges=kept_edges,
            graph=new_graph,
            model=self.model,
            sae_sites=self._sae_sites,
            resolver=self._resolver,
        )

    def sweep(
        self,
        clean: _Inputs,
        corrupted: _Inputs,
        metric: Callable[[Any], Tensor],
        taus: list[float],
        ablation_mode: str = "corrupted",
        eap_skip_threshold: float | None = None,
        include_error_node: bool = False,
        top_k_survivors: int = 32,
    ) -> list[tuple[float, int, float]]:
        """Run greedy node pruning at each tau; return Pareto frontier.

        Returns:
            List of (tau, n_kept_nodes, final_kl) tuples.

        NOTE: greedy, so final circuit KL may exceed tau — standard ACDC behavior.
        Each run is independent (Stage 1 repeated per tau).
        """
        out: list[tuple[float, int, float]] = []
        for tau in taus:
            circuit = self.run(
                clean, corrupted, metric,
                tau=tau,
                ablation_mode=ablation_mode,
                eap_skip_threshold=eap_skip_threshold,
                include_error_node=include_error_node,
                top_k_survivors=top_k_survivors,
            )
            # Count kept nodes (unique (layer, feat) pairs across all sites)
            kept: set[tuple[int, int]] = set()
            for _site, survivors in circuit.graph.survivors.items():
                for atp_node in survivors:
                    nd = atp_node.node
                    if nd.kind == "sae_feature" and nd.layer is not None and nd.neuron is not None:
                        kept.add((nd.layer, nd.neuron))
            n_kept = len(kept)

            # Compute final KL of the pruned circuit
            ablation_values = circuit._compute_ablation_values(clean, corrupted, ablation_mode)
            with torch.no_grad():
                clean_out = self._call_model(clean)

            # Build circuit_nodes from survivors
            circuit_nodes: dict[int, set[int]] = {
                site.layer: set() for site, _ in self._sorted_sites()
            }
            for (layer, feat) in kept:
                circuit_nodes.setdefault(layer, set()).add(feat)

            circuit_out = _feature_circuit_forward(
                self.model,
                clean,
                self._sae_sites,
                self._resolver,
                circuit_nodes=circuit_nodes,
                ablation_values=ablation_values,
                include_error_node=include_error_node,
            )
            final_kl = self._recovery_kl(circuit_out, clean_out)

            out.append((tau, n_kept, final_kl))
        return out
