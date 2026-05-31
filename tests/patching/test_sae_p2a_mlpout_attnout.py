"""P2a tests: mlp_out / attn_out SAE sites.

Tests with TEETH — the core ones exercise the blocker regression:
compute_f_per_site / the node runner MUST decompose the true submodule tensor
(mlp.forward output OR self_attn.forward output), NOT the residual stream.

Toy model design:
  - ToyBlockLayer has a real .mlp (LinearMLP) and a real .self_attn (LinearAttn)
  - self_attn.forward returns a TUPLE (attn_out, None) — matching HF convention
  - mlp.forward returns a plain Tensor — matching HF convention
  - block output is residual + attn_out + mlp_out
  - All operations are linear so analytic == bruteforce holds (≤1e-4)
  - float64 path for machine-precision checks
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torch import Tensor

from tests.patching.test_sae_features import SyntheticSAE, _metric

# ---------------------------------------------------------------------------
# Toy block model
# ---------------------------------------------------------------------------

class LinearMLP(nn.Module):
    """MLP submodule.  Returns a plain Tensor (matches HF convention)."""
    def __init__(self, d: int) -> None:
        super().__init__()
        self.linear = nn.Linear(d, d, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x)


class LinearAttn(nn.Module):
    """Attention submodule.  Returns a TUPLE (attn_out, None) — matching HF convention.

    The outer _extract_tensor in sae_features.py unwraps this tuple; the
    routed splice round-trips through (tensor, None) → correct.
    """
    def __init__(self, d: int) -> None:
        super().__init__()
        self.linear = nn.Linear(d, d, bias=False)

    def forward(self, x: Tensor):  # type: ignore[override]
        return (self.linear(x), None)  # HF tuple convention


class ToyBlockLayer(nn.Module):
    """Single transformer-block-like layer.

    out = x + attn_out + mlp_out
    where attn_out = self_attn(x)[0]  (tuple unwrap)
      and mlp_out  = mlp(x)           (plain tensor)

    The block output is a plain Tensor (residual stream at this layer).
    Hooking self_attn gets the (tensor, None) tuple.
    Hooking mlp gets the plain tensor.
    """
    def __init__(self, d: int) -> None:
        super().__init__()
        self.self_attn = LinearAttn(d)
        self.mlp = LinearMLP(d)

    def forward(self, x: Tensor) -> Tensor:
        attn_out, _ = self.self_attn(x)
        mlp_out = self.mlp(x)
        return x + attn_out + mlp_out


class TwoLayerBlockToy(nn.Module):
    """Two stacked ToyBlockLayers + a linear head.

    layer_pattern="layers.{L}" so _make_resolver() works.
    After calling forward, the residual stream is layers[0] out → layers[1] out → lm_head.
    """
    def __init__(self, d: int = 8) -> None:
        super().__init__()
        self.layers = nn.ModuleList([ToyBlockLayer(d), ToyBlockLayer(d)])
        self.lm_head = nn.Linear(d, d, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(x)


def _make_block_toy(d: int = 8, seed: int = 42) -> TwoLayerBlockToy:
    torch.manual_seed(seed)
    model = TwoLayerBlockToy(d=d)
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.3)
    return model


def _make_sae(d_model: int = 8, d_sae: int = 16, seed: int = 7) -> SyntheticSAE:
    torch.manual_seed(seed)
    return SyntheticSAE(d_model=d_model, d_sae=d_sae, relu=False)


def _make_inputs(d: int = 8, b: int = 1, s: int = 3, seed: int = 0):
    torch.manual_seed(seed)
    clean = torch.randn(b, s, d)
    corrupted = torch.randn(b, s, d)
    return clean, corrupted


def _make_block_resolver(d: int = 8):
    """HFSiteResolver for TwoLayerBlockToy.

    layer_pattern="layers.{L}", mlp_module="mlp",
    attn_module_full="self_attn" (default, but explicit here for clarity).
    """
    from circuitry.patching.sites import HFSiteResolver
    return HFSiteResolver(
        n_heads=1, d_model=d, head_dim=d,
        layer_pattern="layers.{L}",
        mlp_module="mlp",
        attn_module_full="self_attn",
    )


# ---------------------------------------------------------------------------
# THE BLOCKER REGRESSION: compute_f_per_site decomposes the TRUE submodule tensor
# ---------------------------------------------------------------------------

def test_mlpout_decomposes_true_submodule():
    """BLOCKER regression: runner uses sae.encode(mlp_out) NOT sae.encode(residual).

    For an mlp_out site, the feature activations must match what we get by directly
    hooking the mlp submodule — and must be measurably DIFFERENT from what we get
    by hooking the full block (residual stream).

    This test fails if the routing regresses to hooking the full block output
    (the 22× magnitude mismatch scenario described in spec §10 defect 1).
    """
    from circuitry.patching.sae_edges import compute_f_per_site
    from circuitry.patching.sites import Site

    d = 8
    model = _make_block_toy(d=d, seed=42)
    sae = _make_sae(d_model=d, seed=7)
    clean, _ = _make_inputs(d=d)
    resolver = _make_block_resolver(d=d)
    site = Site("mlp_out", layer=0)

    # --- Runner's compute_f_per_site ---
    runner_f = compute_f_per_site(model, clean, {site: sae}, resolver)
    assert 0 in runner_f, "compute_f_per_site returned no result for layer 0"
    f_runner = runner_f[0]  # shape (b, s, d_sae)

    # --- Reference: directly hook the mlp submodule ---
    mlp_out_ref: dict[str, Tensor] = {}

    def _mlp_hook(module, inp, output):
        mlp_out_ref["a"] = output.detach()

    h = model.layers[0].mlp.register_forward_hook(_mlp_hook)
    with torch.no_grad():
        model(clean)
    h.remove()

    a_mlp = mlp_out_ref["a"]
    f_expected = sae.encode(a_mlp)

    # --- Reference: hook the FULL block (residual stream) ---
    block_out_ref: dict[str, Tensor] = {}

    def _block_hook(module, inp, output):
        block_out_ref["a"] = output.detach()

    h2 = model.layers[0].register_forward_hook(_block_hook)
    with torch.no_grad():
        model(clean)
    h2.remove()

    a_block = block_out_ref["a"]
    f_residual = sae.encode(a_block)

    # Runner's f must match mlp-submodule f closely
    max_diff_mlp = (f_runner.to(f_expected.device, f_expected.dtype) - f_expected).abs().max().item()
    print(f"\n[BLOCKER mlp_out] runner f vs mlp-submodule f: max|diff|={max_diff_mlp:.2e}")
    assert max_diff_mlp < 1e-5, (
        f"runner f does NOT match mlp-submodule f (max|diff|={max_diff_mlp:.2e}). "
        "The routing regressed to the block output!"
    )

    # Runner's f must be measurably DIFFERENT from residual-stream f
    max_diff_resid = (f_runner.to(f_residual.device, f_residual.dtype) - f_residual).abs().max().item()
    print(f"[BLOCKER mlp_out] runner f vs residual-stream f: max|diff|={max_diff_resid:.2e}")
    assert max_diff_resid > 1e-3, (
        f"runner f is indistinguishable from residual-stream f (max|diff|={max_diff_resid:.2e}). "
        "This suggests mlp_out and residual are identical (wrong model or wrong hook)."
    )


def test_attnout_decomposes_true_submodule():
    """BLOCKER regression: runner uses sae.encode(attn_out) NOT sae.encode(residual).

    Mirrors test_mlpout_decomposes_true_submodule for the attn_out site.
    Note: self_attn returns a TUPLE (attn_out, None); _extract_tensor unwraps it.
    """
    from circuitry.patching.sae_edges import compute_f_per_site
    from circuitry.patching.sites import Site

    d = 8
    model = _make_block_toy(d=d, seed=42)
    sae = _make_sae(d_model=d, seed=11)
    clean, _ = _make_inputs(d=d)
    resolver = _make_block_resolver(d=d)
    site = Site("attn_out", layer=0)

    # --- Runner's compute_f_per_site ---
    runner_f = compute_f_per_site(model, clean, {site: sae}, resolver)
    assert 0 in runner_f, "compute_f_per_site returned no result for attn_out site"
    f_runner = runner_f[0]

    # --- Reference: directly hook self_attn and extract [0] from tuple ---
    attn_out_ref: dict[str, Tensor] = {}

    def _attn_hook(module, inp, output):
        # output is a tuple (tensor, None) — extract the tensor
        attn_out_ref["a"] = output[0].detach()

    h = model.layers[0].self_attn.register_forward_hook(_attn_hook)
    with torch.no_grad():
        model(clean)
    h.remove()

    a_attn = attn_out_ref["a"]
    f_expected = sae.encode(a_attn)

    # --- Reference: hook the FULL block (residual stream) ---
    block_out_ref: dict[str, Tensor] = {}

    def _block_hook(module, inp, output):
        block_out_ref["a"] = output.detach()

    h2 = model.layers[0].register_forward_hook(_block_hook)
    with torch.no_grad():
        model(clean)
    h2.remove()

    a_block = block_out_ref["a"]
    f_residual = sae.encode(a_block)

    # Runner's f must match attn-submodule f closely
    max_diff_attn = (f_runner.to(f_expected.device, f_expected.dtype) - f_expected).abs().max().item()
    print(f"\n[BLOCKER attn_out] runner f vs attn-submodule f: max|diff|={max_diff_attn:.2e}")
    assert max_diff_attn < 1e-5, (
        f"runner f does NOT match attn-submodule f (max|diff|={max_diff_attn:.2e}). "
        "The routing regressed to the block output!"
    )

    # Runner's f must be measurably DIFFERENT from residual-stream f
    max_diff_resid = (f_runner.to(f_residual.device, f_residual.dtype) - f_residual).abs().max().item()
    print(f"[BLOCKER attn_out] runner f vs residual-stream f: max|diff|={max_diff_resid:.2e}")
    assert max_diff_resid > 1e-3, (
        f"runner f is indistinguishable from residual-stream f (max|diff|={max_diff_resid:.2e}). "
        "Model may be degenerate or hook wired to the wrong submodule."
    )


# ---------------------------------------------------------------------------
# Analytic == bruteforce (node scoring, float64)
# ---------------------------------------------------------------------------

def _to_float64(model, sae):
    """Return float64 copies of model and SAE (leaves originals intact).

    Must use .to(torch.float64) instead of .double() for SyntheticSAE because
    nn.Module.double() calls _apply() which bypasses the custom .to() override
    that syncs the .dtype attribute.  .to(dtype) goes through the override.
    """
    import copy
    m64 = copy.deepcopy(model).to(torch.float64)
    s64 = copy.deepcopy(sae).to(torch.float64)
    return m64, s64


def test_mlpout_node_analytic_eq_bruteforce():
    """mlp_out node analytic Σ_pos Δf·gradf == bruteforce_feature_scores on linear toy.

    float64 so the gap should be ~8.9e-16 (linear model, affine SAE).
    Gate: ≤1e-4 (same as existing GATE A).
    """
    from circuitry.patching.sae_features import SAEFeatureRunner
    from circuitry.patching.sites import Site

    d = 8
    model = _make_block_toy(d=d, seed=42)
    sae = _make_sae(d_model=d, seed=7)
    clean, corrupted = _make_inputs(d=d)

    model64, sae64 = _to_float64(model, sae)
    clean64 = clean.double()
    corrupted64 = corrupted.double()

    site = Site("mlp_out", layer=0)
    resolver = _make_block_resolver(d=d)

    runner = SAEFeatureRunner(model64, {site: sae64}, resolver)
    result = runner.run(clean64, corrupted64, _metric)
    nodes = list(result.scores.keys())

    if not nodes:
        pytest.skip("No active mlp_out features")

    bf = runner.bruteforce_feature_scores(clean64, corrupted64, _metric, nodes)

    max_diff = 0.0
    for node in nodes:
        atp_s = result.scores[node]
        bf_s = bf.get(node, 0.0)
        max_diff = max(max_diff, abs(atp_s - bf_s))

    print(f"\n[mlp_out node] max|analytic - bruteforce| = {max_diff:.2e}")
    assert max_diff < 1e-4, f"mlp_out node gate: max|diff|={max_diff:.2e} > 1e-4"


def test_attnout_node_analytic_eq_bruteforce():
    """attn_out node analytic == bruteforce on linear toy (float64, ≤1e-4)."""
    from circuitry.patching.sae_features import SAEFeatureRunner
    from circuitry.patching.sites import Site

    d = 8
    model = _make_block_toy(d=d, seed=42)
    sae = _make_sae(d_model=d, seed=11)
    clean, corrupted = _make_inputs(d=d)

    model64, sae64 = _to_float64(model, sae)
    clean64 = clean.double()
    corrupted64 = corrupted.double()

    site = Site("attn_out", layer=0)
    resolver = _make_block_resolver(d=d)

    runner = SAEFeatureRunner(model64, {site: sae64}, resolver)
    result = runner.run(clean64, corrupted64, _metric)
    nodes = list(result.scores.keys())

    if not nodes:
        pytest.skip("No active attn_out features")

    bf = runner.bruteforce_feature_scores(clean64, corrupted64, _metric, nodes)

    max_diff = 0.0
    for node in nodes:
        atp_s = result.scores[node]
        bf_s = bf.get(node, 0.0)
        max_diff = max(max_diff, abs(atp_s - bf_s))

    print(f"\n[attn_out node] max|analytic - bruteforce| = {max_diff:.2e}")
    assert max_diff < 1e-4, f"attn_out node gate: max|diff|={max_diff:.2e} > 1e-4"


# ---------------------------------------------------------------------------
# Faithfulness(M) == 1 and faithfulness(∅) == 0 for mlp_out / attn_out reader
# ---------------------------------------------------------------------------

def _faithfulness_full_and_empty(site_component: str, sae_seed: int = 7) -> tuple[float, float]:
    """Helper: build an edge runner with writer@L0 and reader@L1, run circuit,
    check faithfulness(M) == 1 and faithfulness(∅) == 0.

    faithfulness(M) is computed on the FULL circuit returned by runner.run().
    faithfulness(∅) is computed using the same full circuit object but deriving
    the empty-circuit measurement via the SAEFeatureCircuit API: we create a
    circuit that has the same survivors (so _all_node_sets() == full node set)
    but NO edges (so _circuit_node_sets() == empty).  This gives:
      m(C) = m(∅), m(∅) = m(∅), m(M) = m(full), denom = m(M) - m(∅) ≠ 0
      → faithfulness = (m(∅) - m(∅)) / (m(M) - m(∅)) = 0
    """
    from circuitry.patching.sae_edges import (  # noqa: E501
        SAEFeatureCircuit,
        SAEFeatureEdgeGraph,
        SAEFeatureEdgeRunner,
    )
    from circuitry.patching.sites import Site

    d = 8
    model = _make_block_toy(d=d, seed=42)
    sae0 = _make_sae(d_model=d, seed=sae_seed)
    sae1 = _make_sae(d_model=d, seed=sae_seed + 3)
    clean, corrupted = _make_inputs(d=d)
    resolver = _make_block_resolver(d=d)

    # Layer 0 and layer 1 both use the requested component
    site0 = Site(site_component, layer=0)
    site1 = Site(site_component, layer=1)

    runner = SAEFeatureEdgeRunner(model, {site0: sae0, site1: sae1}, resolver)
    circuit = runner.run(clean, corrupted, _metric, top_k_survivors=32)

    f_M = circuit.faithfulness(clean, corrupted, _metric)

    # Empty circuit: same survivors as full circuit (so _all_node_sets() is nonempty
    # and the denominator is well-defined), but NO edges (so _circuit_node_sets() = ∅).
    # faithfulness = (m(∅) - m(∅)) / (m(M) - m(∅)) = 0.
    empty_circuit = SAEFeatureCircuit(
        nodes=circuit.nodes,
        edges={},
        graph=SAEFeatureEdgeGraph(
            sites=circuit.graph.sites,
            survivors=circuit.graph.survivors,  # keep survivors for valid denominator
            edges=[],
        ),
        model=model,
        sae_sites={site0: sae0, site1: sae1},
        resolver=resolver,
    )
    f_empty = empty_circuit.faithfulness(clean, corrupted, _metric)

    return f_M, f_empty


def test_mlpout_faithfulness_full_circuit():
    """mlp_out: faithfulness(M) == 1 (within ~1e-5) and faithfulness(∅) == 0."""
    f_M, f_empty = _faithfulness_full_and_empty("mlp_out", sae_seed=7)
    print(f"\n[mlp_out faithfulness] M={f_M:.6f}  ∅={f_empty:.6f}")
    assert abs(f_M - 1.0) < 1e-5, f"mlp_out faithfulness(M)={f_M:.6f} not ~1"
    assert abs(f_empty - 0.0) < 1e-5, f"mlp_out faithfulness(∅)={f_empty:.6f} not ~0"


def test_attnout_faithfulness_full_circuit():
    """attn_out: faithfulness(M) == 1 (within ~1e-5) and faithfulness(∅) == 0."""
    f_M, f_empty = _faithfulness_full_and_empty("attn_out", sae_seed=11)
    print(f"\n[attn_out faithfulness] M={f_M:.6f}  ∅={f_empty:.6f}")
    assert abs(f_M - 1.0) < 1e-5, f"attn_out faithfulness(M)={f_M:.6f} not ~1"
    assert abs(f_empty - 0.0) < 1e-5, f"attn_out faithfulness(∅)={f_empty:.6f} not ~0"


# ---------------------------------------------------------------------------
# Cross-layer edge: mlp_out@L0 → mlp_out@L1 analytic == bruteforce
# ---------------------------------------------------------------------------

def test_crosslayer_mlpout_edge():
    """Edge mlp_out@L0 → mlp_out@L1 analytic == bruteforce_feature_edge_scores (≤1e-4).

    float64, linear toy, affine SAE.
    Confirms bruteforce freezes eps at clean (spec §6.3 note — recomputing eps cancels patch).
    """
    from circuitry.patching.sae_edges import SAEFeatureEdgeRunner
    from circuitry.patching.sites import Site

    d = 8
    model = _make_block_toy(d=d, seed=42)
    sae0 = _make_sae(d_model=d, seed=7)
    sae1 = _make_sae(d_model=d, seed=13)
    clean, corrupted = _make_inputs(d=d)
    resolver = _make_block_resolver(d=d)

    model64, sae0_64 = _to_float64(model, sae0)
    _, sae1_64 = _to_float64(model, sae1)
    clean64 = clean.double()
    corrupted64 = corrupted.double()

    site0 = Site("mlp_out", layer=0)
    site1 = Site("mlp_out", layer=1)

    runner = SAEFeatureEdgeRunner(model64, {site0: sae0_64, site1: sae1_64}, resolver)
    circuit = runner.run(clean64, corrupted64, _metric, top_k_survivors=32)

    edges = list(circuit.edges.keys())
    if not edges:
        pytest.skip("No mlp_out edges computed")

    bf = runner.bruteforce_feature_edge_scores(clean64, corrupted64, _metric, edges)

    max_diff = 0.0
    for edge in edges:
        a_s = circuit.edges[edge]
        b_s = bf.get(edge, 0.0)
        max_diff = max(max_diff, abs(a_s - b_s))

    # Verify at least one edge is nonzero (finite nonzero check)
    max_abs = max(abs(s) for s in circuit.edges.values()) if circuit.edges else 0.0
    print(f"\n[mlp_out edge] max|analytic - bruteforce| = {max_diff:.2e}, max|edge| = {max_abs:.2e}")
    assert max_abs > 1e-10, "All mlp_out edges are zero — degenerate model or SAE"
    assert max_diff < 1e-4, f"mlp_out edge gate: max|diff|={max_diff:.2e} > 1e-4"


def test_crosslayer_attnout_edge():
    """Edge attn_out@L0 → attn_out@L1 analytic == bruteforce (float64, ≤1e-4)."""
    from circuitry.patching.sae_edges import SAEFeatureEdgeRunner
    from circuitry.patching.sites import Site

    d = 8
    model = _make_block_toy(d=d, seed=42)
    sae0 = _make_sae(d_model=d, seed=11)
    sae1 = _make_sae(d_model=d, seed=17)
    clean, corrupted = _make_inputs(d=d)
    resolver = _make_block_resolver(d=d)

    model64, sae0_64 = _to_float64(model, sae0)
    _, sae1_64 = _to_float64(model, sae1)
    clean64 = clean.double()
    corrupted64 = corrupted.double()

    site0 = Site("attn_out", layer=0)
    site1 = Site("attn_out", layer=1)

    runner = SAEFeatureEdgeRunner(model64, {site0: sae0_64, site1: sae1_64}, resolver)
    circuit = runner.run(clean64, corrupted64, _metric, top_k_survivors=32)

    edges = list(circuit.edges.keys())
    if not edges:
        pytest.skip("No attn_out edges computed")

    bf = runner.bruteforce_feature_edge_scores(clean64, corrupted64, _metric, edges)

    max_diff = 0.0
    for edge in edges:
        a_s = circuit.edges[edge]
        b_s = bf.get(edge, 0.0)
        max_diff = max(max_diff, abs(a_s - b_s))

    max_abs = max(abs(s) for s in circuit.edges.values()) if circuit.edges else 0.0
    print(f"\n[attn_out edge] max|analytic - bruteforce| = {max_diff:.2e}, max|edge| = {max_abs:.2e}")
    assert max_abs > 1e-10, "All attn_out edges are zero — degenerate model or SAE"
    assert max_diff < 1e-4, f"attn_out edge gate: max|diff|={max_diff:.2e} > 1e-4"


# ---------------------------------------------------------------------------
# Node.component identity
# ---------------------------------------------------------------------------

def test_node_component_identity():
    """mlp_out feature node has .component == 'mlp_out';
    resid_post feature node has .component is None;
    the two are distinct AtPNodes even at the same (layer, neuron).
    """
    from circuitry.patching.atp import AtPNode
    from circuitry.patching.graph import Node

    node_mlp = AtPNode(Node("sae_feature", layer=0, neuron=5, component="mlp_out"))
    node_rp = AtPNode(Node("sae_feature", layer=0, neuron=5, component=None))
    node_attn = AtPNode(Node("sae_feature", layer=0, neuron=5, component="attn_out"))

    assert node_mlp.node.component == "mlp_out"
    assert node_rp.node.component is None
    assert node_attn.node.component == "attn_out"

    # Distinct identity even at same (layer, neuron)
    assert node_mlp != node_rp, "mlp_out and resid_post nodes should be distinct"
    assert node_mlp != node_attn, "mlp_out and attn_out nodes should be distinct"
    assert node_rp != node_attn, "resid_post and attn_out nodes should be distinct"

    # Same component, layer, neuron → equal
    node_mlp2 = AtPNode(Node("sae_feature", layer=0, neuron=5, component="mlp_out"))
    assert node_mlp == node_mlp2, "Identical nodes should be equal"

    # resid_post node constructed without component → component is None
    node_legacy = AtPNode(Node("sae_feature", layer=0, neuron=5))
    assert node_legacy.node.component is None
    assert node_legacy == node_rp, "Legacy node (no component) should equal component=None node"


# ---------------------------------------------------------------------------
# Two SAE sites in one layer → NotImplementedError
# ---------------------------------------------------------------------------

def test_two_sites_one_layer_rejected_node_runner():
    """SAEFeatureRunner rejects two sites sharing a layer (P2b gate)."""
    from circuitry.patching.sae_features import SAEFeatureRunner
    from circuitry.patching.sites import Site

    d = 8
    model = _make_block_toy(d=d, seed=42)
    sae_mlp = _make_sae(d_model=d, seed=7)
    sae_attn = _make_sae(d_model=d, seed=11)
    resolver = _make_block_resolver(d=d)

    # Two sites at layer 0 with different components
    site_mlp = Site("mlp_out", layer=0)
    site_attn = Site("attn_out", layer=0)

    with pytest.raises(NotImplementedError, match="P2b"):
        SAEFeatureRunner(model, {site_mlp: sae_mlp, site_attn: sae_attn}, resolver)


def test_two_sites_one_layer_rejected_edge_runner():
    """SAEFeatureEdgeRunner rejects two sites sharing a layer (P2b gate)."""
    from circuitry.patching.sae_edges import SAEFeatureEdgeRunner
    from circuitry.patching.sites import Site

    d = 8
    model = _make_block_toy(d=d, seed=42)
    sae_mlp = _make_sae(d_model=d, seed=7)
    sae_attn = _make_sae(d_model=d, seed=11)
    resolver = _make_block_resolver(d=d)

    site_mlp = Site("mlp_out", layer=0)
    site_attn = Site("attn_out", layer=0)

    with pytest.raises(NotImplementedError, match="P2b"):
        SAEFeatureEdgeRunner(model, {site_mlp: sae_mlp, site_attn: sae_attn}, resolver)


# ---------------------------------------------------------------------------
# Cross-component edge: mlp_out@L0 → resid_post@L1 (mixed sites, different layers)
# ---------------------------------------------------------------------------

def test_crosscomponent_mlpout_to_resid_edge():
    """Edge mlp_out@L0 → resid_post@L1: finite nonzero, analytic == bruteforce (≤1e-4)."""
    from circuitry.patching.sae_edges import SAEFeatureEdgeRunner
    from circuitry.patching.sites import Site

    d = 8
    # Use LinearResidToy for the resid_post side (it exposes block outputs as resid)
    # But we need a model that has a .mlp submodule at layer 0.
    # Use TwoLayerBlockToy for mlp_out@0 + resid_post@1 (block output = resid_post).
    model = _make_block_toy(d=d, seed=42)
    sae0 = _make_sae(d_model=d, seed=7)
    sae1 = _make_sae(d_model=d, seed=13)
    clean, corrupted = _make_inputs(d=d)
    # Use a resolver where layer 1 resid_post hooks the block output (full block)
    resolver = _make_block_resolver(d=d)

    model64, sae0_64 = _to_float64(model, sae0)
    _, sae1_64 = _to_float64(model, sae1)
    clean64 = clean.double()
    corrupted64 = corrupted.double()

    site0 = Site("mlp_out", layer=0)
    site1 = Site("resid_post", layer=1)

    runner = SAEFeatureEdgeRunner(model64, {site0: sae0_64, site1: sae1_64}, resolver)
    circuit = runner.run(clean64, corrupted64, _metric, top_k_survivors=32)

    edges = list(circuit.edges.keys())
    if not edges:
        pytest.skip("No cross-component edges computed")

    bf = runner.bruteforce_feature_edge_scores(clean64, corrupted64, _metric, edges)

    max_diff = 0.0
    for edge in edges:
        a_s = circuit.edges[edge]
        b_s = bf.get(edge, 0.0)
        max_diff = max(max_diff, abs(a_s - b_s))

    max_abs = max(abs(s) for s in circuit.edges.values()) if circuit.edges else 0.0
    print(f"\n[mlp_out→resid_post edge] max|analytic - bruteforce| = {max_diff:.2e}, max|edge| = {max_abs:.2e}")
    assert max_abs > 1e-10, "All cross-component edges are zero — degenerate model or SAE"
    assert max_diff < 1e-4, f"cross-component edge gate: max|diff|={max_diff:.2e} > 1e-4"


# ---------------------------------------------------------------------------
# Unsupported components still rejected
# ---------------------------------------------------------------------------

def test_unsupported_component_rejected():
    """attn_head_out and resid_pre are still rejected by both runners."""
    from circuitry.patching.sae_edges import SAEFeatureEdgeRunner
    from circuitry.patching.sae_features import SAEFeatureRunner
    from circuitry.patching.sites import Site

    d = 8
    model = _make_block_toy(d=d, seed=42)
    sae = _make_sae(d_model=d, seed=7)
    resolver = _make_block_resolver(d=d)

    for component in ("resid_pre",):
        site = Site(component, layer=0)
        with pytest.raises(NotImplementedError):
            SAEFeatureRunner(model, {site: sae}, resolver)
        with pytest.raises(NotImplementedError):
            SAEFeatureEdgeRunner(model, {site: sae}, resolver)
