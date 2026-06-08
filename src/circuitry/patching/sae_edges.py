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

import warnings
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
# Step 0 helpers: composite (layer, component) keying  (spec §4 / P2b)
# ---------------------------------------------------------------------------

_COMPONENT_OFFSET: dict[str, int] = {"attn_out": 0, "mlp_out": 1, "resid_post": 2}


def _site_key(site: Any) -> tuple[int, str]:
    """Canonical composite key from a Site object."""
    return (site.layer, site.component)


def _node_site_key(node: Any) -> tuple[int, str]:
    """Canonical composite key from a graph Node (component None == resid_post)."""
    return (node.layer, node.component or "resid_post")


def _site_rank(site: Any) -> int:
    """Forward-position rank: attn_out@L=3L, mlp_out@L=3L+1, resid_post@L=3L+2."""
    return 3 * site.layer + _COMPONENT_OFFSET[site.component]


def _is_always_connected(writer_site: Any, reader_site: Any) -> bool:
    """Return True iff this (writer, reader) pair is ALWAYS forward-connected.

    resid_post → resid_post across any layers is structurally guaranteed connected
    in every transformer (the residual stream flows through every layer).
    Pairs involving mlp_out or attn_out may be legitimately disconnected on
    parallel-attention architectures (attn and mlp both read from x, not each other).
    """
    return writer_site.component == "resid_post" and reader_site.component == "resid_post"


# ---------------------------------------------------------------------------
# Stage B helpers: feature-acts capture + node-set ablation forward
# ---------------------------------------------------------------------------


def compute_f_per_site(
    model: nn.Module,
    inputs: _Inputs,
    sae_sites: dict[Site, Any],
    resolver: Any,
    *,
    return_x_hat: bool = False,
) -> (
    dict[tuple[int, str], Tensor]
    | tuple[dict[tuple[int, str], Tensor], dict[tuple[int, str], Tensor]]
):
    """Capture SAE feature activations (and optionally reconstructions) per site
    in a single no_grad forward.

    Uses ``sae_decompose`` (paired encode→decode) so that stateful SAEs with
    ``normalize_activations='layer_norm'`` store the correct normalization statistics
    during encode and decode immediately in the same call — never caching ``f`` and
    decoding later (see ``sae/grad.py`` §92-108).

    Args:
        return_x_hat: When ``False`` (default) return only ``f_dict`` (the
            historical contract). When ``True`` also return the paired
            same-call reconstructions ``x_hat_dict`` — required by the
            stateful-SAE faithfulness path (F3 fix).

    Returns:
        ``f_dict`` keyed by (site.layer, site.component): feature activations
        (detached, on SAE device/dtype). If ``return_x_hat`` is True, returns
        ``(f_dict, x_hat_dict)`` where ``x_hat_dict`` holds the SAE
        reconstructions in model space (used by ``_feature_circuit_forward``
        via ``ablation_x_hat`` to decode correctly for stateful SAEs).
        Composite key supports multiple sites per layer (P2b).
    """
    from circuitry.sae.grad import sae_decompose as _sae_decompose_capture

    f_result: dict[tuple[int, str], Tensor] = {}
    x_hat_result: dict[tuple[int, str], Tensor] = {}

    handles: list[Any] = []
    for site, sae in sae_sites.items():
        # Route through ResolvedSite (identity for resid_post + position=None)
        resolved = resolver.resolve(model, site)
        layer_mod = resolved.module
        site_k = _site_key(site)

        def _hook(
            module: nn.Module,
            inp: Any,
            output: Any,
            _sae: Any = sae,
            _site_k: tuple[int, str] = site_k,
            _resolved: Any = resolved,
        ) -> None:
            a = _routed_extract(_resolved, output).detach()
            a_in = a.to(
                getattr(_sae, "device", a.device),
                getattr(_sae, "dtype", a.dtype),
            )
            with torch.no_grad():
                # Paired encode→decode: sae_decompose ensures stateful SAEs
                # store the correct norm stats during encode and decode immediately.
                f, x_hat, _ = _sae_decompose_capture(_sae, a_in)
            f_result[_site_k] = f.detach()
            x_hat_result[_site_k] = x_hat.detach()

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

    if return_x_hat:
        return f_result, x_hat_result
    return f_result


def _feature_circuit_forward(
    model: nn.Module,
    inputs: _Inputs,
    sae_sites: dict[Site, Any],
    resolver: Any,
    circuit_nodes: dict[tuple[int, str], set[int]],
    ablation_values: dict[tuple[int, str], Tensor],
    *,
    include_error_node: bool = False,
    error_in_circuit: dict[tuple[int, str], bool] | None = None,
    ablation_mode: str = "corrupted",
    ablation_eps: dict[tuple[int, str], Tensor] | None = None,
    ablation_x_hat: dict[tuple[int, str], Tensor] | None = None,
) -> Any:
    """Node-set ablation forward (§4.1 / §4.4).

    At each spliced site, replaces NON-circuit feature entries with ablation_values
    before decode. eps is frozen at clean (include_error_node=False) or ablated
    for out-of-circuit error nodes (include_error_node=True).

    All dicts are keyed by (layer, component) — the composite key (§4.4).
    This ensures two same-layer sites do NOT share or overwrite each other's
    ablation decomposition.

    Args:
        model:           The model to run.
        inputs:          Model inputs.
        sae_sites:       Ordered dict of Site → SAE.
        resolver:        SiteResolver used to route hooks to the correct submodule.
        circuit_nodes:   dict[(layer,component), set[feature_idx]] — IN-CIRCUIT features.
        ablation_values: dict[(layer,component), Tensor] — ablation feature activations per site.
        include_error_node: If True, treat sae_error as a node (§4.4).
        error_in_circuit:   dict[(layer,component), bool] — error node in-circuit per site.
        ablation_mode:   'corrupted' / 'zero' / 'mean' (informational; ablation_values already computed).
        ablation_eps:    dict[(layer,component), Tensor] — eps ablation values for out-of-circuit
                         error nodes (include_error_node=True).
        ablation_x_hat:  dict[(layer,component), Tensor] — SAE reconstructions (x_hat) from the
                         ablation context (e.g. corrupted forward), paired with ablation_values.
                         When provided AND all site features are non-circuit, the hook uses
                         ``ablation_x_hat + eps_clean`` directly instead of
                         ``sae.decode(f_ablated)`` — this ensures stateful SAEs with
                         ``normalize_activations='layer_norm'`` decode in the correct context
                         (F3 fix).  For non-stateful SAEs the two paths are byte-identical.

    Returns:
        Model output (may have .logits or be raw tensor).
    """
    handles: list[Any] = []

    # Sort sites in forward order (forward-position rank)
    sorted_site_pairs = sorted(sae_sites.items(), key=lambda kv: _site_rank(kv[0]))

    for site, sae in sorted_site_pairs:
        sk = _site_key(site)
        # Route through ResolvedSite (identity for resid_post + position=None)
        resolved = resolver.resolve(model, site)
        layer_mod = resolved.module
        # Determine model device/dtype for cast-back.
        # On the TL path, HookPoint.parameters() == [], so the params-fallback
        # silently downcasts.  Read from model.cfg instead.
        from circuitry.patching.sites import TLSiteResolver
        if isinstance(resolver, TLSiteResolver):
            m_dtype = model.cfg.dtype
            m_device = torch.device(model.cfg.device)
        else:
            params = list(layer_mod.parameters())
            if params:
                m_dtype, m_device = params[0].dtype, params[0].device
            else:
                m_dtype, m_device = torch.float32, torch.device("cpu")

        in_circuit = circuit_nodes.get(sk, set())
        abl_val = ablation_values.get(sk)  # may be None for zero-mode
        # F3 fix: pre-paired x_hat from ablation context (e.g. corrupted forward).
        # Only useful when all features at this site are non-circuit — see hook below.
        abl_x_hat = ablation_x_hat.get(sk) if ablation_x_hat is not None else None

        # Error-node circuit membership — per (layer, component)
        err_in_circ = True  # default: treat as always in circuit (frozen eps)
        if include_error_node and error_in_circuit is not None:
            err_in_circ = error_in_circuit.get(sk, True)
        abl_eps = ablation_eps.get(sk) if (ablation_eps is not None and not err_in_circ) else None

        def _ablate_hook(
            module: nn.Module,
            inp: Any,
            output: Any,
            _sae: Any = sae,
            _in_circuit: set = in_circuit,
            _abl_val: Tensor | None = abl_val,
            _abl_x_hat: Tensor | None = abl_x_hat,
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

                # F3 fix: when all features are non-circuit AND a pre-paired x_hat is
                # available (from compute_f_per_site's same-call sae_decompose), use it
                # directly instead of sae.decode(f_ablated).  For stateful SAEs
                # (normalize_activations='layer_norm'), decode must be paired with its
                # encode — the clean encode above already set sae._norm_mean/_norm_std to
                # CLEAN stats, so decoding corrupted f_ablated would use wrong stats.
                # Using the pre-paired abl_x_hat avoids this mismatch.
                # For non-stateful SAEs: abl_x_hat == sae.decode(f_ablated) → byte-identical.
                if _abl_x_hat is not None and len(_in_circuit) == 0:
                    x_hat_recon = _abl_x_hat.to(f_ablated.device, f_ablated.dtype)
                else:
                    x_hat_recon = _sae.decode(f_ablated)

                recon = x_hat_recon + eps
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

        layer_pairs='adjacent'   → consecutive pairs in forward-position order
        layer_pairs='all_forward' → all pairs where rank(writer) < rank(reader)

        Sites are ordered by _site_rank (attn_out@L < mlp_out@L < resid_post@L).
        """
        pairs: list[tuple[tuple[Site, Any], tuple[Site, Any]]] = []
        n = len(self.sites)
        for i in range(n):
            for j in range(i + 1, n):
                # Pair is valid iff rank(writer) < rank(reader)
                if _site_rank(self.sites[i][0]) >= _site_rank(self.sites[j][0]):
                    continue
                if layer_pairs == "adjacent" and j != i + 1:
                    continue
                pairs.append((self.sites[i], self.sites[j]))
        # Reverse topo: higher-rank (later) reader first
        pairs.sort(key=lambda p: -_site_rank(p[1][0]))
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
        position_scores: dict[SAEFeatureEdge, Tensor] | None = None,
    ) -> None:
        self.nodes = nodes
        self.edges = edges
        self.graph = graph
        # Per-position scores: set when SAEFeatureEdgeRunner.run(per_position=True).
        # dict[SAEFeatureEdge, Tensor] where each tensor has shape (seq_len,).
        # position_scores[e].sum() == edges[e] (within float32 rounding).
        self.position_scores = position_scores
        # Internal references for Stage B ablation methods
        self._model = model
        self._sae_sites = sae_sites
        self._resolver = resolver
        # Populated by _compute_ablation_values for 'corrupted' mode:
        # SAE reconstruction (x_hat) from the corrupted forward,
        # used by faithfulness/completeness for the F3 stateful-SAE fix.
        self._ablation_x_hat: dict[tuple[int, str], Tensor] | None = None

    def ranked(self) -> list[tuple[SAEFeatureEdge, float]]:
        return sorted(self.edges.items(), key=lambda kv: abs(kv[1]), reverse=True)

    def top_k(self, n: int) -> list[tuple[SAEFeatureEdge, float]]:
        return self.ranked()[:n]

    def threshold(self, tau: float) -> list[SAEFeatureEdge]:
        return [e for e, s in self.edges.items() if abs(s) >= tau]

    def top_positions(self, edge: SAEFeatureEdge, k: int = 5) -> list[tuple[int, float]]:
        """Return the top-k sequence positions by absolute per-position score for *edge*.

        Returns a list of (position_index, score) pairs sorted by |score| descending.
        Raises KeyError if *edge* is not in position_scores (run with per_position=True first).
        Raises ValueError if position_scores is None.
        """
        if self.position_scores is None:
            raise ValueError(
                "position_scores is None — re-run SAEFeatureEdgeRunner.run() with per_position=True"
            )
        pos_tensor = self.position_scores[edge]
        scores_list = [(int(i), float(v)) for i, v in enumerate(pos_tensor.cpu())]
        scores_list.sort(key=lambda iv: abs(iv[1]), reverse=True)
        return scores_list[:k]

    # ------------------------------------------------------------------
    # Internal: extract circuit_nodes set from this circuit
    # ------------------------------------------------------------------

    def _circuit_node_sets(self) -> dict[tuple[int, str], set[int]]:
        """Return dict[(layer,component), set[feature_idx]] for nodes in this circuit.

        Nodes are defined by the union of edge endpoints (writer + reader).
        Does NOT include nodes from self.nodes.scores that are not in any edge —
        this ensures an empty-edge circuit has empty node sets.
        Keyed by composite (layer, component) so same-layer sites remain distinct (§4.2).
        """
        result: dict[tuple[int, str], set[int]] = {}
        if self._sae_sites is not None:
            for site in self._sae_sites:
                result[_site_key(site)] = set()

        # Collect all nodes across edges (edge endpoints = circuit members)
        for edge in self.edges:
            for node_ref in [edge.writer, edge.reader]:
                nd = node_ref.node
                if nd.kind == "sae_feature" and nd.layer is not None and nd.neuron is not None:
                    nk = _node_site_key(nd)
                    result.setdefault(nk, set()).add(nd.neuron)
        return result

    def _all_node_sets(self) -> dict[tuple[int, str], set[int]]:
        """Return dict[(layer,component), set[feature_idx]] for ALL survivors (full circuit M).
        Keyed by composite (layer, component) so same-layer sites remain distinct (§4.2).
        """
        if self._sae_sites is None:
            return {}
        result: dict[tuple[int, str], set[int]] = {_site_key(site): set() for site in self._sae_sites}
        for site, survivors in self.graph.survivors.items():
            sk = _site_key(site)
            for atp_node in survivors:
                nd = atp_node.node
                if nd.kind == "sae_feature" and nd.neuron is not None:
                    result.setdefault(sk, set()).add(nd.neuron)
        return result

    def _compute_ablation_values(
        self,
        clean: _Inputs,
        corrupted: _Inputs,
        ablation_mode: str,
    ) -> dict[tuple[int, str], Tensor]:
        """Compute ablation_value dict[(layer,component), Tensor] per §4.1/§4.4.

        Returns:
            dict mapping (site.layer, site.component) -> ablation tensor.
            For 'corrupted' mode: this is f_corrupt from compute_f_per_site.
            Also populates self._ablation_x_hat with the SAE reconstruction dict
            for 'corrupted' mode (None for 'zero'/'mean') — used by faithfulness/
            completeness for the F3 stateful-SAE fix.
        """
        if self._model is None or self._sae_sites is None or self._resolver is None:
            raise RuntimeError(
                "SAEFeatureCircuit must be created by SAEFeatureEdgeRunner.run() "
                "to use faithfulness/completeness/prune."
            )
        f_corrupt, x_hat_corrupt = compute_f_per_site(
            self._model, corrupted, self._sae_sites, self._resolver, return_x_hat=True
        )
        if ablation_mode == "corrupted":
            self._ablation_x_hat = x_hat_corrupt
            return f_corrupt
        self._ablation_x_hat = None
        if ablation_mode == "zero":
            return {sk: torch.zeros_like(f) for sk, f in f_corrupt.items()}
        if ablation_mode == "mean":
            result: dict[tuple[int, str], Tensor] = {}
            for sk, f in f_corrupt.items():
                mean_val = f.mean(dim=list(range(f.ndim - 1)), keepdim=True)
                result[sk] = mean_val.expand_as(f)
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
        circuit_nodes: dict[tuple[int, str], set[int]],
        ablation_values: dict[tuple[int, str], Tensor],
        *,
        include_error_node: bool = False,
        error_in_circuit: dict[tuple[int, str], bool] | None = None,
        ablation_eps: dict[tuple[int, str], Tensor] | None = None,
        ablation_x_hat: dict[tuple[int, str], Tensor] | None = None,
    ) -> float:
        """Compute m(circuit_nodes) — scalar metric under node-set ablation.

        Args:
            circuit_nodes:     dict[(layer,component), set[feature_idx]] — IN-CIRCUIT features.
            ablation_values:   dict[(layer,component), Tensor] — ablation feature activations per site.
            include_error_node: If True, thread error-node membership into the forward.
            error_in_circuit:  dict[(layer,component), bool] — error node in-circuit per site.
            ablation_eps:      dict[(layer,component), Tensor] — eps ablation values for out-of-circuit
                               error nodes (required when include_error_node=True and
                               error node is out of circuit).
            ablation_x_hat:    dict[(layer,component), Tensor] — SAE reconstructions from the
                               ablation context (from compute_f_per_site). Passed to
                               _feature_circuit_forward for the F3 stateful-SAE fix.
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
            ablation_x_hat=ablation_x_hat,
        )
        with torch.no_grad():
            m = metric(out)
        return float(m)

    # ------------------------------------------------------------------
    # Stage B: faithfulness / completeness / prune
    # ------------------------------------------------------------------

    def _error_circuit_membership(self) -> dict[tuple[int, str], bool]:
        """Return dict[(layer,component), bool] — whether the sae_error node at each site is in circuit.

        An error node is in-circuit if it appears as a WRITER in any edge.
        Keyed by composite (layer, component) for multi-site-per-layer support (§4.2).
        """
        if self._sae_sites is None:
            return {}
        in_circuit: dict[tuple[int, str], bool] = {_site_key(site): False for site in self._sae_sites}
        for edge in self.edges:
            w = edge.writer.node
            if w.kind == "sae_error" and w.layer is not None:
                wk = _node_site_key(w)
                in_circuit[wk] = True
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
        # _compute_ablation_values populates self._ablation_x_hat for 'corrupted' mode (F3 fix).
        ablation_x_hat = self._ablation_x_hat

        # Build error_in_circuit and ablation_eps when include_error_node=True
        error_in_circuit: dict[tuple[int, str], bool] | None = None
        ablation_eps: dict[tuple[int, str], Tensor] | None = None
        if include_error_node:
            error_in_circuit = self._error_circuit_membership()
            # ablation_eps: for out-of-circuit error nodes use the corrupted eps
            if self._sae_sites is not None and self._resolver is not None and self._model is not None:
                from circuitry.sae.grad import sae_decompose as _sae_decompose
                abl_eps_dict: dict[tuple[int, str], Tensor] = {}
                for site, sae in self._sae_sites.items():
                    sk = _site_key(site)
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
                        abl_eps_dict[sk] = eps_store["eps"]
                ablation_eps = abl_eps_dict

        # m(∅): empty circuit — all features ablated
        empty_nodes: dict[tuple[int, str], set[int]] = {sk: set() for sk in ablation_values}
        # For empty circuit: all error nodes are out-of-circuit
        empty_err_in_circ = {sk: False for sk in ablation_values} if include_error_node else None
        m_empty = self._m_of(
            clean, corrupted, metric, empty_nodes, ablation_values,
            include_error_node=include_error_node,
            error_in_circuit=empty_err_in_circ,
            ablation_eps=ablation_eps,
            ablation_x_hat=ablation_x_hat,
        )

        # m(M): full circuit — all survivors kept; use _all_node_sets() so faithfulness(M)=1
        full_nodes = self._all_node_sets()
        # For full circuit: all error nodes are in-circuit (eps frozen at clean)
        full_err_in_circ = {sk: True for sk in full_nodes} if include_error_node else None
        m_full = self._m_of(
            clean, corrupted, metric, full_nodes, ablation_values,
            include_error_node=include_error_node,
            error_in_circuit=full_err_in_circ,
            ablation_eps=ablation_eps,
            ablation_x_hat=ablation_x_hat,
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
        circuit_nodes_c = self._circuit_node_sets()
        circuit_err_in_circ = error_in_circuit  # derived from actual edge set
        m_circuit = self._m_of(
            clean, corrupted, metric, circuit_nodes_c, ablation_values,
            include_error_node=include_error_node,
            error_in_circuit=circuit_err_in_circ,
            ablation_eps=ablation_eps,
            ablation_x_hat=ablation_x_hat,
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
        # _compute_ablation_values populates self._ablation_x_hat for 'corrupted' mode (F3 fix).
        ablation_x_hat_compl = self._ablation_x_hat

        full_nodes = self._all_node_sets()
        circuit_nodes_c = self._circuit_node_sets()

        # M\C: complement of circuit within the full node set (keyed by composite key)
        complement: dict[tuple[int, str], set[int]] = {}
        for sk, all_feats in full_nodes.items():
            in_circ = circuit_nodes_c.get(sk, set())
            complement[sk] = all_feats - in_circ

        # Build error_in_circuit and ablation_eps when include_error_node=True
        ablation_eps_compl: dict[tuple[int, str], Tensor] | None = None
        if include_error_node:
            # Reuse _error_circuit_membership for the circuit
            err_circ = self._error_circuit_membership()
            # For full circuit: all error nodes in-circuit
            full_err_in_circ: dict[tuple[int, str], bool] = {sk: True for sk in full_nodes}
            # For complement: complement of error membership
            comp_err_in_circ: dict[tuple[int, str], bool] = {sk: not v for sk, v in err_circ.items()}
            # For empty: all out of circuit
            empty_err_in_circ: dict[tuple[int, str], bool] = {sk: False for sk in ablation_values}

            # ablation_eps: for out-of-circuit error nodes use the corrupted eps
            # (mirrors faithfulness() ~line 571-606 — F11 fix)
            if self._sae_sites is not None and self._resolver is not None and self._model is not None:
                from circuitry.sae.grad import sae_decompose as _sae_decompose_compl
                abl_eps_dict_compl: dict[tuple[int, str], Tensor] = {}
                for site, sae in self._sae_sites.items():
                    sk = _site_key(site)
                    resolved = self._resolver.resolve(self._model, site)
                    layer_mod = resolved.module
                    eps_store: dict[str, Tensor] = {}

                    def _eps_hook_compl(
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
                            _, _, eps_c = _sae_decompose_compl(_sae, a_in)
                        _st["eps"] = eps_c.detach()

                    _h = layer_mod.register_forward_hook(_eps_hook_compl)
                    try:
                        with torch.no_grad():
                            if isinstance(corrupted, dict):
                                self._model(**corrupted)  # type: ignore[union-attr]
                            else:
                                self._model(corrupted)  # type: ignore[union-attr]
                    finally:
                        _h.remove()
                    if "eps" in eps_store:
                        abl_eps_dict_compl[sk] = eps_store["eps"]
                ablation_eps_compl = abl_eps_dict_compl

        # m(∅): empty circuit
        empty_nodes: dict[tuple[int, str], set[int]] = {sk: set() for sk in ablation_values}
        if include_error_node:
            m_empty = self._m_of(
                clean, corrupted, metric, empty_nodes, ablation_values,
                include_error_node=True,
                error_in_circuit=empty_err_in_circ,  # type: ignore[possibly-undefined]
                ablation_eps=ablation_eps_compl,
                ablation_x_hat=ablation_x_hat_compl,
            )
        else:
            m_empty = self._m_of(
                clean, corrupted, metric, empty_nodes, ablation_values,
                ablation_x_hat=ablation_x_hat_compl,
            )

        # m(M): full circuit
        if include_error_node:
            m_full = self._m_of(
                clean, corrupted, metric, full_nodes, ablation_values,
                include_error_node=True,
                error_in_circuit=full_err_in_circ,  # type: ignore[possibly-undefined]
                ablation_eps=ablation_eps_compl,
                ablation_x_hat=ablation_x_hat_compl,
            )
        else:
            m_full = self._m_of(
                clean, corrupted, metric, full_nodes, ablation_values,
                ablation_x_hat=ablation_x_hat_compl,
            )

        # m(M\C): complement circuit
        if include_error_node:
            m_complement = self._m_of(
                clean, corrupted, metric, complement, ablation_values,
                include_error_node=True,
                error_in_circuit=comp_err_in_circ,  # type: ignore[possibly-undefined]
                ablation_eps=ablation_eps_compl,
                ablation_x_hat=ablation_x_hat_compl,
            )
        else:
            m_complement = self._m_of(
                clean, corrupted, metric, complement, ablation_values,
                ablation_x_hat=ablation_x_hat_compl,
            )

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
            # Build survivors from kept edges — keyed by composite (layer, component)
            kept_nodes: dict[tuple[int, str], set[int]] = {}
            for edge in kept_edges:
                for nd_ref in [edge.writer, edge.reader]:
                    nd = nd_ref.node
                    if nd.kind == "sae_feature" and nd.layer is not None and nd.neuron is not None:
                        nk = _node_site_key(nd)
                        kept_nodes.setdefault(nk, set()).add(nd.neuron)

            new_survivors: dict[Site, list[AtPNode]] = {}
            for site, survivors in result.graph.survivors.items():
                sk = _site_key(site)
                feats = kept_nodes.get(sk, set())
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
    HF-eager (HFSiteResolver) and TransformerLens (TLSiteResolver) are supported (v1.7 P3).
    Supported components: resid_post, mlp_out, attn_out.

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
            resolver:   An HFSiteResolver or TLSiteResolver.
        """
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

    def _sorted_sites(self) -> list[tuple[Site, Any]]:
        """Sites sorted in forward-position order (by _site_rank: attn_out < mlp_out < resid_post within a layer)."""
        return sorted(self._sae_sites.items(), key=lambda kv: _site_rank(kv[0]))

    def _model_dtype_device(self, layer_mod: nn.Module) -> tuple[torch.dtype, Any]:
        # On the TL path, HookPoint.parameters() == [], so the params-fallback
        # silently downcasts (e.g. fp64 → fp32).  Read from model.cfg instead.
        from circuitry.patching.sites import TLSiteResolver
        if isinstance(self.resolver, TLSiteResolver):
            return self.model.cfg.dtype, torch.device(self.model.cfg.device)
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
        n_ig_steps: int = 0,
        per_position: bool = False,
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
            variant: 'attrib' (default) or 'ig' (integrated gradients, full EAP-IG).
                'exact'/other → NotImplementedError.
                'ig': interpolates the writer leaf f_U_clean→f_U_corrupt (Δf=f_corrupt−f_clean,
                sign-consistent with attrib); reader stays LIVE; eps frozen at both sites.
                Cost = N× attrib; peak memory == attrib (one vjp_j alive at a time).
            n_ig_steps: number of IG integration steps (variant='ig' only). 0 → default of 32.
            per_position: if True, also compute per-sequence-position edge scores and store
                them in SAEFeatureCircuit.position_scores as dict[SAEFeatureEdge, Tensor]
                where each Tensor has shape (seq_len,).  position_scores[e].sum() == edges[e]
                within float32 rounding.  Default False (no overhead).

        Returns:
            SAEFeatureCircuit with .nodes (v1.5 scores), .edges, .graph,
            and .position_scores when per_position=True.
        """
        if variant not in ("attrib", "ig"):
            raise NotImplementedError(
                f"SAEFeatureEdgeRunner variant={variant!r} is not supported. "
                "Supported values: 'attrib', 'ig'."
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
                variant=variant,
                n_ig_steps=n_ig_steps,
                per_position=per_position,
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
        variant: str = "attrib",
        n_ig_steps: int = 0,
        per_position: bool = False,
    ) -> SAEFeatureCircuit:
        """Inner run (freeze/restore already applied by caller)."""
        sorted_sites = self._sorted_sites()  # forward order

        # ------------------------------------------------------------------
        # STAGE 1: run composed SAEFeatureRunner per-site → node scores
        # Keep top-K active survivors per site.
        # NOTE: stage1 always uses 'attrib' for node scoring regardless of edge variant —
        # the edge variant only affects Stage 2 (edge computation).
        # ------------------------------------------------------------------
        node_result = self._stage1_runner.run(
            clean_inputs, corrupted_inputs, metric,
            include_error_node=include_error_node,
            max_features=top_k_survivors,
        )

        # Group survivors by composite (layer, component) site key
        site_survivors: dict[tuple[int, str], list[AtPNode]] = {}
        for site, _ in sorted_sites:
            site_survivors[_site_key(site)] = []

        for atp_node in node_result.scores:
            nd = atp_node.node
            if nd.layer is not None:
                nk = _node_site_key(nd)
                if nk in site_survivors:
                    site_survivors[nk].append(atp_node)

        # Build EdgeGraph metadata
        edge_graph = SAEFeatureEdgeGraph(
            sites=sorted_sites,
            survivors={site: site_survivors.get(_site_key(site), []) for site, _ in sorted_sites},
            edges=[],  # filled below
        )

        # ------------------------------------------------------------------
        # STAGE 2: enumerate site pairs in forward-position order (rank-based)
        # ------------------------------------------------------------------
        all_edge_scores: dict[SAEFeatureEdge, float] = {}
        all_pos_scores: dict[SAEFeatureEdge, Tensor] = {}

        # Iterate over site pairs in forward order (writer rank < reader rank)
        for i, (writer_site, _writer_sae) in enumerate(sorted_sites):
            for j, (reader_site, _reader_sae) in enumerate(sorted_sites):
                # Valid forward edge: rank(writer) < rank(reader)
                if _site_rank(writer_site) >= _site_rank(reader_site):
                    continue
                # 'adjacent' mode: j must immediately follow i in rank order
                if layer_pairs == "adjacent" and j != i + 1:
                    continue

                writer_survivors = site_survivors.get(_site_key(writer_site), [])
                reader_survivors = site_survivors.get(_site_key(reader_site), [])

                # Skip if no survivors on either side
                if not writer_survivors and not reader_survivors:
                    continue

                pair_scalar, pair_pos = self._compute_pair_edges(
                    clean_inputs=clean_inputs,
                    corrupted_inputs=corrupted_inputs,
                    metric=metric,
                    writer_site=writer_site,
                    reader_site=reader_site,
                    writer_survivors=writer_survivors,
                    reader_survivors=reader_survivors,
                    include_error_node=include_error_node,
                    variant=variant,
                    n_ig_steps=n_ig_steps,
                    per_position=per_position,
                )
                all_edge_scores.update(pair_scalar)
                if pair_pos:
                    all_pos_scores.update(pair_pos)

        # Global max_edges cap (top-|score|)
        if max_edges is not None and len(all_edge_scores) > max_edges:
            sorted_edges = sorted(all_edge_scores.items(), key=lambda kv: abs(kv[1]), reverse=True)
            all_edge_scores = dict(sorted_edges[:max_edges])
            # Keep only position_scores for edges that survived the cap
            if all_pos_scores:
                kept_keys = set(all_edge_scores.keys())
                all_pos_scores = {e: v for e, v in all_pos_scores.items() if e in kept_keys}

        edge_graph.edges = sorted(all_edge_scores.keys(), key=_sae_edge_sort_key)

        return SAEFeatureCircuit(
            nodes=node_result,
            edges=all_edge_scores,
            graph=edge_graph,
            position_scores=all_pos_scores if per_position else None,
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
        variant: str = "attrib",
        n_ig_steps: int = 0,
        per_position: bool = False,
    ) -> tuple[dict[SAEFeatureEdge, float], dict[SAEFeatureEdge, Tensor] | None]:
        """Stage 2 for a single (writer, reader) site pair.

        attrib (default):
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

        ig (v1.7 P4, full EAP-IG):
          Wrap Steps B–D in an N-loop. At step k, set the WRITER detached leaf to
          f_U_k = f_U_clean + α_k·Δf_U (α_k=(k-0.5)/N); READER stays LIVE; eps frozen
          at both sites. One backward per step; per-j VJP loop UNCHANGED (one vjp_j
          alive at a time — no dense d_sae×d_sae Jacobian; peak mem == attrib).
          edge(i→j) = Σ_k Δf_U[i]·vjp_j_k[i] / N.
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
            return {}, None

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
            # Guard: only retain_grad if requires_grad is True; on parallel-attention
            # models the reader tensor may not depend on the writer (no causal path) and
            # retain_grad on a leaf-or-no-grad tensor raises RuntimeError.
            if f_D.requires_grad:
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
            return {}, None

        if f_D_live.grad is None:
            # No gradient path from metric to f_D after backward.
            # Distinguish two cases:
            #   1. resid_post → resid_post: always forward-connected in any transformer;
            #      a missing grad here is a BUG (metric not differentiable through the site).
            #   2. Any pair involving mlp_out or attn_out: parallel-attention architectures
            #      may legitimately have no causal path (attn and mlp both read from the same
            #      x, not from each other's outputs).  Warn and return {}.
            if _is_always_connected(writer_site, reader_site):
                raise RuntimeError(
                    f"f_D.grad is None for a resid_post→resid_post pair "
                    f"({writer_site} → {reader_site}). This is a bug: resid_post sites are "
                    "always forward-connected in a transformer. Check that the metric returns "
                    "a differentiable Tensor and that the model is not in no-grad mode."
                )
            warnings.warn(
                f"f_D.grad is None for pair ({writer_site} → {reader_site}). "
                "No causal path detected — this is expected for parallel-attention "
                "architectures where mlp_out/attn_out are not causally connected. "
                "Returning empty edge dict for this pair.",
                stacklevel=3,
            )
            return {}, None

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

        edge_scores: dict[SAEFeatureEdge, float] = {}
        pos_scores: dict[SAEFeatureEdge, Tensor] = {}

        # component labels (None for resid_post to preserve v1.6 identity)
        _r_comp = reader_site.component if reader_site.component != "resid_post" else None
        _w_comp = writer_site.component if writer_site.component != "resid_post" else None

        if variant == "attrib":
            # ----------------------------------------------------------------
            # attrib: single-point AtP (original v1.6 code path)
            # ----------------------------------------------------------------
            gradf_D = f_D_live.grad  # shape same as f_D_live; not yet fp32

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

                reader_node = AtPNode(Node("sae_feature", layer=reader_site.layer, neuron=j, component=_r_comp))

                # feature→feature edges
                if vjp_j is not None:
                    # Slice to upstream survivors immediately (memory discipline)
                    vjp_j_fp32 = vjp_j.to(torch.float32)

                    for i in upstream_feat_indices:
                        product = (delta_f_U[..., i] * vjp_j_fp32[..., i].to(delta_f_U.device)).to(torch.float32)
                        score = float(product.sum())
                        writer_node = AtPNode(Node("sae_feature", layer=writer_site.layer, neuron=i, component=_w_comp))
                        edge = SAEFeatureEdge(writer=writer_node, reader=reader_node)
                        edge_scores[edge] = score
                        if per_position:
                            # Collapse all batch dims, keep seq_len as position axis
                            pos_scores[edge] = product.view(-1, product.shape[-1]).sum(0) if product.dim() > 1 else product.clone()

                    del vjp_j, vjp_j_fp32

                # error→feature edges (include_error_node=True only)
                if include_error_node and vjp_err_j is not None and delta_eps_U is not None:
                    vjp_err_j_fp32 = vjp_err_j.to(torch.float32)
                    product_err = (delta_eps_U.to(vjp_err_j_fp32.device) * vjp_err_j_fp32).to(torch.float32)
                    score_err = float(product_err.sum())
                    error_writer_node = AtPNode(Node("sae_error", layer=writer_site.layer, component=_w_comp))
                    err_edge = SAEFeatureEdge(writer=error_writer_node, reader=reader_node)
                    edge_scores[err_edge] = score_err
                    if per_position:
                        # Sum over feature dim, then collapse batch
                        pos_err = product_err.sum(-1)
                        pos_scores[err_edge] = pos_err.view(-1, pos_err.shape[-1]).sum(0) if pos_err.dim() > 1 else pos_err.clone()
                    del vjp_err_j, vjp_err_j_fp32

                del G_j  # FREE per-j (memory discipline)

        else:
            # ----------------------------------------------------------------
            # ig: full EAP-IG (v1.7 P4)
            #
            # Wrap Steps B–D in an N-loop.  At step k:
            #   f_U_k = f_U_clean + α_k·Δf_U  (interpolate ONLY the writer leaf)
            #   reader stays LIVE; eps frozen at both sites
            #   one backward per step; per-j VJP loop UNCHANGED
            #   (one vjp_j alive at a time — no dense d_sae×d_sae Jacobian)
            # Accumulate edge(i→j) += Σ_k Δf_U[i]·vjp_j_k[i]; divide by N at end.
            # Peak memory == attrib: per-step forward graph freed between steps.
            # ----------------------------------------------------------------
            _n_ig = n_ig_steps if n_ig_steps > 0 else 32

            # f_U_clean in original dtype for stable interpolation
            f_U_clean_t = f_U_leaf.detach()
            f_U_corrupt_t = f_U_corrupt.to(f_U_clean_t.device, f_U_clean_t.dtype)
            delta_f_U_t = f_U_corrupt_t - f_U_clean_t  # same direction as delta_f_U (fp32)

            # F4 fix: pre-compute clean and corrupt eps for error→feature IG path.
            # eps_U_clean is captured from err_leaf_U (the detached eps leaf from the
            # attrib clean forward already run above).  eps_U_corrupt was captured in
            # Step A (_writer_corr_hook stores "eps").  The IG path interpolates the
            # error leaf identically to the feature leaf: err_k = eps_clean + α_k·Δeps.
            eps_U_clean_t: Tensor | None = None
            delta_eps_U_t: Tensor | None = None
            if include_error_node and err_leaf_U is not None and eps_U_corrupt is not None:
                eps_U_clean_t = err_leaf_U.detach()
                eps_U_corrupt_t = eps_U_corrupt.to(eps_U_clean_t.device, eps_U_clean_t.dtype)
                delta_eps_U_t = eps_U_corrupt_t - eps_U_clean_t

            # Accumulate raw sums; divide by N at end
            edge_sum: dict[SAEFeatureEdge, float] = {}
            pos_sum: dict[SAEFeatureEdge, Tensor] = {}

            for k in range(1, _n_ig + 1):
                alpha_k = (k - 0.5) / _n_ig
                # Interpolated writer feature leaf for step k
                f_U_k = (f_U_clean_t + alpha_k * delta_f_U_t).detach().requires_grad_(True)

                # F4 fix: interpolated error leaf for step k (only when include_error_node)
                err_leaf_U_k: Tensor | None = None
                if eps_U_clean_t is not None and delta_eps_U_t is not None:
                    err_leaf_U_k = (
                        eps_U_clean_t + alpha_k * delta_eps_U_t
                    ).detach().requires_grad_(True)
                    err_leaf_U_k.retain_grad()

                # Per-step stores (cleared each iteration)
                writer_k_store: dict[str, Tensor] = {}
                reader_k_store: dict[str, Tensor] = {}

                # Capture f_U_clean_eps for frozen eps computation inside hook.
                # We compute eps_clean once from a_in (see hook below).
                def _writer_k_hook(
                    module: nn.Module, inp: Any, output: Any,
                    _sae: Any = writer_sae,
                    _f_U_k: Tensor = f_U_k,
                    _err_leaf_U_k: Tensor | None = err_leaf_U_k,
                    _include_err: bool = include_error_node,
                    _st: dict = writer_k_store,
                    _mdtype: torch.dtype = w_dtype,
                    _mdev: Any = w_device,
                    _resolved: Any = writer_resolved,
                ) -> Any:
                    """WRITER site (IG step k): inject interpolated f_U_k.

                    When include_error_node=True, the error term uses an interpolated
                    err_leaf_U_k (independent grad leaf) so that error→feature VJPs
                    can be computed — mirrors the attrib _writer_clean_hook construction.
                    When include_error_node=False, eps is frozen at clean (original behaviour).
                    """
                    a = _routed_extract(_resolved, output)
                    a_in = a.detach().to(getattr(_sae, "device", a.device), getattr(_sae, "dtype", a.dtype))
                    # Decode from the interpolated leaf
                    x_hat_k = _sae.decode(_f_U_k)
                    if _include_err and _err_leaf_U_k is not None:
                        # Use the interpolated error leaf — gives grad path for error→feature VJP
                        recon = x_hat_k + _err_leaf_U_k
                        _st["err_leaf_U_k"] = _err_leaf_U_k
                    else:
                        # Compute frozen clean eps: eps = a - decode(encode(a))
                        with torch.no_grad():
                            f_cl = _sae.encode(a_in)
                            x_hat_cl = _sae.decode(f_cl)
                            eps_clean_k = (a_in - x_hat_cl).detach()
                        recon = x_hat_k + eps_clean_k  # eps frozen at clean
                    _st["f_U_k"] = _f_U_k
                    recon_cast = recon.to(_mdev, _mdtype)
                    return _routed_inject(_resolved, output, recon_cast)

                def _reader_k_hook(
                    module: nn.Module, inp: Any, output: Any,
                    _sae: Any = reader_sae,
                    _st: dict = reader_k_store,
                    _mdtype: torch.dtype = r_dtype,
                    _mdev: Any = r_device,
                    _resolved: Any = reader_resolved,
                ) -> Any:
                    """READER site (IG step k): LIVE encode (unchanged from attrib)."""
                    a = _routed_extract(_resolved, output)
                    a_in = a.to(getattr(_sae, "device", a.device), getattr(_sae, "dtype", a.dtype))
                    # Live encode — a_in NOT detached
                    f_D_k, x_hat_k, eps_k = sae_decompose(_sae, a_in)
                    if f_D_k.requires_grad:
                        f_D_k.retain_grad()
                    recon_k = x_hat_k + eps_k  # eps frozen at clean by sae_decompose
                    _st["f_D_k"] = f_D_k
                    recon_cast = recon_k.to(_mdev, _mdtype)
                    return _routed_inject(_resolved, output, recon_cast)

                wh_k = writer_layer_mod.register_forward_hook(_writer_k_hook)
                rh_k = reader_layer_mod.register_forward_hook(_reader_k_hook)
                try:
                    with torch.enable_grad():
                        out_k = self._call_model(clean_inputs)
                        m_k = metric(out_k)
                        if not (isinstance(m_k, Tensor) and m_k.requires_grad):
                            raise RuntimeError(
                                "metric must return a differentiable Tensor. "
                                "Use a logit_diff_t-style metric, not a float."
                            )
                        m_k.backward(retain_graph=True)
                finally:
                    wh_k.remove()
                    rh_k.remove()

                f_D_k_live = reader_k_store.get("f_D_k")
                if f_D_k_live is None or f_D_k_live.grad is None:
                    # Connectivity is a structural property — check it ONCE (k=1) and
                    # apply the same rule as the attrib path:
                    #   resid_post → resid_post: always connected → BUG → raise
                    #   any mlp_out/attn_out involved → legitimately disconnectable → warn+return {}
                    # Do NOT accumulate partial sums then divide by _n_ig (underscales).
                    del out_k, m_k, f_U_k
                    if err_leaf_U_k is not None:
                        del err_leaf_U_k
                    if _is_always_connected(writer_site, reader_site):
                        raise RuntimeError(
                            f"f_D_k.grad is None at IG step k={k} for a resid_post→resid_post pair "
                            f"({writer_site} → {reader_site}). This is a bug: resid_post sites are "
                            "always forward-connected in a transformer. Check that the metric returns "
                            "a differentiable Tensor and that the model is not in no-grad mode."
                        )
                    warnings.warn(
                        f"f_D_k.grad is None at IG step k={k} for pair ({writer_site} → {reader_site}). "
                        "No causal path detected — this is expected for parallel-attention "
                        "architectures where mlp_out/attn_out are not causally connected. "
                        "Returning empty edge dict for this pair.",
                        stacklevel=3,
                    )
                    return {}, None

                gradf_D_k = f_D_k_live.grad

                # Determine VJP grad inputs for this step:
                # always f_U_k; add err_leaf_U_k when include_error_node=True (F4 fix)
                err_leaf_U_k_in_store = writer_k_store.get("err_leaf_U_k")
                grad_inputs_k: list[Tensor] = [f_U_k]
                if include_error_node and err_leaf_U_k_in_store is not None:
                    grad_inputs_k.append(err_leaf_U_k_in_store)

                # Per-j VJP loop (UNCHANGED structure from attrib — one vjp_j alive at a time)
                for j in downstream_feat_indices:
                    G_j = torch.zeros_like(f_D_k_live)
                    G_j[..., j] = gradf_D_k[..., j]

                    try:
                        vjp_results_k = torch.autograd.grad(
                            f_D_k_live, grad_inputs_k,
                            grad_outputs=G_j,
                            retain_graph=True,
                            allow_unused=True,
                        )
                    except RuntimeError:
                        del G_j
                        continue

                    vjp_j_k = vjp_results_k[0]
                    vjp_err_j_k = vjp_results_k[1] if len(vjp_results_k) > 1 else None

                    if vjp_j_k is None and vjp_err_j_k is None:
                        del G_j
                        continue

                    reader_node = AtPNode(Node("sae_feature", layer=reader_site.layer, neuron=j, component=_r_comp))

                    if vjp_j_k is not None:
                        vjp_j_k_fp32 = vjp_j_k.to(torch.float32)

                        for i in upstream_feat_indices:
                            # Accumulate: edge(i→j) += Δf_U[i] · vjp_j_k[i]
                            product_k = (delta_f_U[..., i] * vjp_j_k_fp32[..., i].to(delta_f_U.device)).to(torch.float32)
                            contrib = float(product_k.sum())
                            writer_node = AtPNode(Node("sae_feature", layer=writer_site.layer, neuron=i, component=_w_comp))
                            edge_key = SAEFeatureEdge(writer=writer_node, reader=reader_node)
                            edge_sum[edge_key] = edge_sum.get(edge_key, 0.0) + contrib
                            if per_position:
                                p_k = product_k.view(-1, product_k.shape[-1]).sum(0) if product_k.dim() > 1 else product_k.clone()
                                prev = pos_sum.get(edge_key)
                                pos_sum[edge_key] = p_k if prev is None else prev + p_k

                        del vjp_j_k, vjp_j_k_fp32

                    # F4 fix: accumulate error→feature IG contribution at step k
                    if (
                        include_error_node
                        and vjp_err_j_k is not None
                        and delta_eps_U_t is not None
                    ):
                        vjp_err_j_k_fp32 = vjp_err_j_k.to(torch.float32)
                        product_err_k = (delta_eps_U_t.to(vjp_err_j_k_fp32.device) * vjp_err_j_k_fp32).to(torch.float32)
                        contrib_err = float(product_err_k.sum())
                        error_writer_node = AtPNode(
                            Node("sae_error", layer=writer_site.layer, component=_w_comp)
                        )
                        err_edge_key = SAEFeatureEdge(writer=error_writer_node, reader=reader_node)
                        edge_sum[err_edge_key] = edge_sum.get(err_edge_key, 0.0) + contrib_err
                        if per_position:
                            pos_err_k = product_err_k.sum(-1)
                            p_err_k = pos_err_k.view(-1, pos_err_k.shape[-1]).sum(0) if pos_err_k.dim() > 1 else pos_err_k.clone()
                            prev_err = pos_sum.get(err_edge_key)
                            pos_sum[err_edge_key] = p_err_k if prev_err is None else prev_err + p_err_k
                        del vjp_err_j_k, vjp_err_j_k_fp32

                    del G_j  # FREE per-j

                # Free per-step graph explicitly
                del out_k, m_k, f_U_k, f_D_k_live
                if err_leaf_U_k is not None:
                    del err_leaf_U_k

            # Divide accumulated sums by N to get the IG estimate
            for edge_key, total in edge_sum.items():
                edge_scores[edge_key] = total / _n_ig
            if per_position:
                for edge_key, total_pos in pos_sum.items():
                    pos_scores[edge_key] = total_pos / _n_ig

        return edge_scores, (pos_scores if per_position else None)

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
        """Sites in forward-position order (by _site_rank)."""
        return sorted(self._sae_sites.items(), key=lambda kv: _site_rank(kv[0]))

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
        variant: str = "attrib",
        n_ig_steps: int = 0,
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
            variant:             'attrib' (default) or 'ig' — attribution variant for Stage 1.
            n_ig_steps:          IG integration steps (used when variant='ig').
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
                variant=variant,
                n_ig_steps=n_ig_steps,
            )
        else:
            raise RuntimeError("No edge runner available and no _initial_circuit provided.")

        # Build node score lookup: (layer, component_str, feat_idx) → |score|
        # component_str is node.component or "resid_post" (composite key per §4.2)
        node_scores: dict[tuple[int, str, int], float] = {}
        for atp_node, score in base_circuit.nodes.scores.items():
            nd = atp_node.node
            if nd.kind == "sae_feature" and nd.layer is not None and nd.neuron is not None:
                comp_str = nd.component or "resid_post"
                node_scores[(nd.layer, comp_str, nd.neuron)] = float(abs(score))

        # ------------------------------------------------------------------
        # Build reverse-topo order: higher rank (later) first, weakest score first within site
        # ------------------------------------------------------------------
        sorted_sites = self._sorted_sites()
        # Reverse topo: higher-rank sites first
        rev_sorted_sites = list(reversed(sorted_sites))

        # Collect all current kept nodes per site — composite (layer, component) key
        kept_nodes: dict[tuple[int, str], set[int]] = {}
        for site, survivors in base_circuit.graph.survivors.items():
            sk = _site_key(site)
            kept_nodes[sk] = set()
            for atp_node in survivors:
                nd = atp_node.node
                if nd.kind == "sae_feature" and nd.neuron is not None:
                    kept_nodes[sk].add(nd.neuron)

        # ------------------------------------------------------------------
        # Compute ablation values once (corrupted/zero/mean)
        # ------------------------------------------------------------------
        ablation_values = base_circuit._compute_ablation_values(clean, corrupted, ablation_mode)
        ablation_x_hat_acdc = base_circuit._ablation_x_hat

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
            sk = _site_key(site)
            comp_str = site.component  # "attn_out" / "mlp_out" / "resid_post"
            feats_in_site = list(kept_nodes.get(sk, set()))

            # Sort by weakest |score| first (ascending)
            feats_in_site.sort(key=lambda fi: node_scores.get((site.layer, comp_str, fi), 0.0))

            for feat_i in feats_in_site:
                score_i = node_scores.get((site.layer, comp_str, feat_i), 0.0)

                # eap_skip_threshold: keep high-score nodes without testing
                if eap_skip_threshold is not None and score_i > eap_skip_threshold:
                    continue

                # Tentatively remove feat_i
                kept_nodes[sk].discard(feat_i)

                # Run ablation forward with the tentative kept set
                circuit_out = _feature_circuit_forward(
                    self.model,
                    clean,
                    self._sae_sites,
                    self._resolver,
                    circuit_nodes=kept_nodes,
                    ablation_values=ablation_values,
                    include_error_node=include_error_node,
                    ablation_x_hat=ablation_x_hat_acdc,
                )

                new_kl = self._recovery_kl(circuit_out, clean_out)

                # Accept removal if KL increase is within tolerance
                if new_kl - current_kl < tau:
                    current_kl = new_kl
                    # feat_i stays removed
                else:
                    # Reject: put feat_i back
                    kept_nodes[sk].add(feat_i)

        # ------------------------------------------------------------------
        # Build pruned SAEFeatureCircuit from kept_nodes
        # ------------------------------------------------------------------
        # Induce edges: keep only edges where both writer and reader are in kept_nodes
        kept_edges: dict[SAEFeatureEdge, float] = {}
        for edge, score in base_circuit.edges.items():
            w = edge.writer.node
            r = edge.reader.node
            wk = _node_site_key(w)
            rk = _node_site_key(r)
            w_in = (
                w.layer is not None
                and w.neuron is not None
                and w.neuron in kept_nodes.get(wk, set())
            )
            r_in = (
                r.layer is not None
                and r.neuron is not None
                and r.neuron in kept_nodes.get(rk, set())
            )
            if w_in and r_in:
                kept_edges[edge] = score

        # Build new survivors
        new_survivors: dict[Site, list[AtPNode]] = {}
        for site, survivors in base_circuit.graph.survivors.items():
            sk = _site_key(site)
            feats = kept_nodes.get(sk, set())
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
        variant: str = "attrib",
        n_ig_steps: int = 0,
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
                variant=variant,
                n_ig_steps=n_ig_steps,
            )
            # Count kept nodes (unique (layer, component, feat) tuples across all sites)
            kept: set[tuple[int, str, int]] = set()
            for site, survivors in circuit.graph.survivors.items():
                comp_str = site.component
                for atp_node in survivors:
                    nd = atp_node.node
                    if nd.kind == "sae_feature" and nd.layer is not None and nd.neuron is not None:
                        kept.add((nd.layer, comp_str, nd.neuron))
            n_kept = len(kept)

            # Compute final KL of the pruned circuit
            ablation_values = circuit._compute_ablation_values(clean, corrupted, ablation_mode)
            ablation_x_hat_sweep = circuit._ablation_x_hat
            with torch.no_grad():
                clean_out = self._call_model(clean)

            # Build circuit_nodes from survivors — composite (layer, component) key
            circuit_nodes: dict[tuple[int, str], set[int]] = {
                _site_key(site): set() for site, _ in self._sorted_sites()
            }
            for (layer, comp_str, feat) in kept:
                circuit_nodes.setdefault((layer, comp_str), set()).add(feat)

            circuit_out = _feature_circuit_forward(
                self.model,
                clean,
                self._sae_sites,
                self._resolver,
                circuit_nodes=circuit_nodes,
                ablation_values=ablation_values,
                include_error_node=include_error_node,
                ablation_x_hat=ablation_x_hat_sweep,
            )
            final_kl = self._recovery_kl(circuit_out, clean_out)

            out.append((tau, n_kept, final_kl))
        return out
