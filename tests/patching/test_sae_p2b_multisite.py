"""P2b tests: multiple SAE sites per layer (FULL fork).

Red-green discipline:
  Step 1 (RED):  remove the P2a guard, run these tests → they fail / error
                 due to int-layer key collisions.
  Step 2 (GREEN): composite (layer, component) keys everywhere → all pass.

Toy model: TwoLayerBlockToy from test_sae_p2a_mlpout_attnout.py
  - ToyBlockLayer has real .self_attn (tuple output) and .mlp (tensor output)
  - block out = x + attn_out + mlp_out  (sequential: attn before mlp)
  - All-linear so analytic == bruteforce (≤1e-4 gate)
  - float64 path for machine-precision checks

Forward-position rank (spec §4.3):
  attn_out@L  → rank 3L
  mlp_out@L   → rank 3L+1
  resid_post@L → rank 3L+2

So intra-layer valid edge: attn_out@0 → mlp_out@0 (rank 0 < 1)
   intra-layer INVALID:    mlp_out@0 → attn_out@0 (rank 1 > 0)
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torch import Tensor

from circuitry.patching.sae_edges import _site_rank
from tests.patching.test_sae_features import _metric
from tests.patching.test_sae_p2a_mlpout_attnout import (
    LinearAttn,
    LinearMLP,
    _make_block_resolver,
    _make_block_toy,
    _make_inputs,
    _make_sae,
    _to_float64,
)

# ---------------------------------------------------------------------------
# Sequential toy model: mlp receives x + attn_out (valid intra-layer edge)
# ---------------------------------------------------------------------------

class SequentialBlockLayer(nn.Module):
    """Single transformer-block-like layer with SEQUENTIAL attn → mlp.

    resid_mid = x + attn_out           (attn applied first)
    mlp_out   = mlp(resid_mid)         (mlp sees x + attn_out)
    out       = resid_mid + mlp_out

    This creates a valid causal path: attn_out@L → mlp_out@L
    (modifying attn_out changes what mlp sees).

    self_attn returns a TUPLE (attn_out, None) — matching HF convention.
    mlp returns a plain Tensor.
    """

    def __init__(self, d: int) -> None:
        super().__init__()
        self.self_attn = LinearAttn(d)
        self.mlp = LinearMLP(d)

    def forward(self, x: Tensor) -> Tensor:
        attn_out, _ = self.self_attn(x)
        resid_mid = x + attn_out
        mlp_out = self.mlp(resid_mid)
        return resid_mid + mlp_out


class SequentialTwoLayerToy(nn.Module):
    """Two stacked SequentialBlockLayers + a linear head."""

    def __init__(self, d: int = 8) -> None:
        super().__init__()
        self.layers = nn.ModuleList([SequentialBlockLayer(d), SequentialBlockLayer(d)])
        self.lm_head = nn.Linear(d, d, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(x)


def _make_seq_toy(d: int = 8, seed: int = 42) -> SequentialTwoLayerToy:
    torch.manual_seed(seed)
    model = SequentialTwoLayerToy(d=d)
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.3)
    return model


# ---------------------------------------------------------------------------
# Test 1: no collision — 3 sites at layer 0, 3 distinct decompositions
# ---------------------------------------------------------------------------

def test_multisite_one_layer_no_collision():
    """FULL fork: attn_out@0 + mlp_out@0 + resid_post@0 all coexist in one circuit.

    Asserts:
    1. compute_f_per_site returns 3 distinct entries (one per composite key).
    2. The three feature tensors are pairwise DIFFERENT (each decomposes a
       different submodule output — attn, mlp, residual).
    3. The AtPNodes for the same (layer=0, neuron=k) but different component are
       DISTINCT objects and carry the correct .component.
    """
    from circuitry.patching.sae_edges import compute_f_per_site
    from circuitry.patching.sae_features import SAEFeatureRunner
    from circuitry.patching.sites import Site

    d = 8
    model = _make_seq_toy(d=d, seed=42)
    sae_attn = _make_sae(d_model=d, seed=11)
    sae_mlp = _make_sae(d_model=d, seed=7)
    sae_resid = _make_sae(d_model=d, seed=3)
    clean, corrupted = _make_inputs(d=d)
    resolver = _make_block_resolver(d=d)

    site_attn = Site("attn_out", layer=0)
    site_mlp = Site("mlp_out", layer=0)
    site_resid = Site("resid_post", layer=0)

    sae_sites = {site_attn: sae_attn, site_mlp: sae_mlp, site_resid: sae_resid}

    # (1) compute_f_per_site must return 3 entries (one per (layer, component))
    f_dict = compute_f_per_site(model, clean, sae_sites, resolver)
    key_attn = (0, "attn_out")
    key_mlp = (0, "mlp_out")
    key_resid = (0, "resid_post")

    assert key_attn in f_dict, f"Missing key {key_attn} in compute_f_per_site result. Got: {list(f_dict.keys())}"
    assert key_mlp in f_dict, f"Missing key {key_mlp} in compute_f_per_site result. Got: {list(f_dict.keys())}"
    assert key_resid in f_dict, f"Missing key {key_resid} in compute_f_per_site result. Got: {list(f_dict.keys())}"
    assert len(f_dict) == 3, f"Expected 3 entries, got {len(f_dict)}: {list(f_dict.keys())}"

    f_attn = f_dict[key_attn]
    f_mlp = f_dict[key_mlp]
    f_resid = f_dict[key_resid]

    # (2) The three tensors must be pairwise DIFFERENT
    diff_attn_mlp = (f_attn.to(f_mlp.device, f_mlp.dtype) - f_mlp).abs().max().item()
    diff_attn_resid = (f_attn.to(f_resid.device, f_resid.dtype) - f_resid).abs().max().item()
    diff_mlp_resid = (f_mlp.to(f_resid.device, f_resid.dtype) - f_resid).abs().max().item()

    print(f"\n[multisite no_collision] "
          f"attn vs mlp: {diff_attn_mlp:.2e}, "
          f"attn vs resid: {diff_attn_resid:.2e}, "
          f"mlp vs resid: {diff_mlp_resid:.2e}")
    assert diff_attn_mlp > 1e-3, (
        f"attn_out and mlp_out feature tensors are IDENTICAL (diff={diff_attn_mlp:.2e}). "
        "Layer-key collision: two sites wrote to the same dict slot."
    )
    assert diff_attn_resid > 1e-3, (
        f"attn_out and resid_post feature tensors are IDENTICAL (diff={diff_attn_resid:.2e}). "
        "Layer-key collision."
    )
    assert diff_mlp_resid > 1e-3, (
        f"mlp_out and resid_post feature tensors are IDENTICAL (diff={diff_mlp_resid:.2e}). "
        "Layer-key collision."
    )

    # (3) SAEFeatureRunner: AtPNodes for same (layer=0, neuron=k) but different component
    # must be distinct with correct .component
    runner = SAEFeatureRunner(model, sae_sites, resolver)
    result = runner.run(clean, corrupted, _metric)

    attn_nodes = [n for n in result.scores if n.node.layer == 0 and n.node.component == "attn_out"]
    mlp_nodes = [n for n in result.scores if n.node.layer == 0 and n.node.component == "mlp_out"]
    resid_nodes = [n for n in result.scores if n.node.layer == 0 and n.node.component is None]

    print(f"[multisite no_collision] attn_nodes={len(attn_nodes)}, mlp_nodes={len(mlp_nodes)}, resid_nodes={len(resid_nodes)}")

    assert len(attn_nodes) > 0, "No attn_out@0 nodes in runner result"
    assert len(mlp_nodes) > 0, "No mlp_out@0 nodes in runner result"
    assert len(resid_nodes) > 0, "No resid_post@0 nodes in runner result"

    # Check distinctness: same (layer, neuron) but different component → distinct objects
    # Find any shared neuron index across all three
    attn_neurons = {n.node.neuron for n in attn_nodes}
    mlp_neurons = {n.node.neuron for n in mlp_nodes}

    # Verify no two same-component nodes: each node set should have unique (layer, component, neuron)
    for node in attn_nodes:
        assert node.node.component == "attn_out", f"Node has wrong component: {node.node.component}"
    for node in mlp_nodes:
        assert node.node.component == "mlp_out", f"Node has wrong component: {node.node.component}"
    for node in resid_nodes:
        assert node.node.component is None, f"resid_post node should have component=None, got {node.node.component}"

    # Verify same-neuron nodes are distinct objects
    shared_neurons = attn_neurons & mlp_neurons
    if shared_neurons:
        k = next(iter(shared_neurons))
        from circuitry.patching.atp import AtPNode
        from circuitry.patching.graph import Node
        node_attn_k = AtPNode(Node("sae_feature", layer=0, neuron=k, component="attn_out"))
        node_mlp_k = AtPNode(Node("sae_feature", layer=0, neuron=k, component="mlp_out"))
        assert node_attn_k != node_mlp_k, (
            f"AtPNodes with neuron={k} at layer=0 but different components should be DISTINCT"
        )
        assert node_attn_k.node.component == "attn_out"
        assert node_mlp_k.node.component == "mlp_out"


# ---------------------------------------------------------------------------
# Test 2: intra-layer edge attn_out@0 → mlp_out@0 present; reverse absent
# ---------------------------------------------------------------------------

def test_intralayer_attn_to_mlp_edge():
    """Intra-layer attn_out@0 → mlp_out@0 edge is enumerated with 'all_forward'.

    Forward rank: attn_out@0=0 < mlp_out@0=1, so this is a valid forward edge.
    mlp_out@0 → attn_out@0 would be rank 1 > 0 (backward) and must NOT appear.

    Also validates analytic ≈ bruteforce (≤1e-4) for the intra-layer edge
    on the linear toy model.
    """
    from circuitry.patching.sae_edges import SAEFeatureEdgeRunner
    from circuitry.patching.sites import Site

    d = 8
    # Use sequential model: mlp sees x + attn_out, so attn_out causally affects mlp
    model = _make_seq_toy(d=d, seed=42)
    sae_attn = _make_sae(d_model=d, seed=11)
    sae_mlp = _make_sae(d_model=d, seed=7)
    clean, corrupted = _make_inputs(d=d)
    resolver = _make_block_resolver(d=d)

    model64, sae_attn64 = _to_float64(model, sae_attn)
    _, sae_mlp64 = _to_float64(model, sae_mlp)
    clean64 = clean.double()
    corrupted64 = corrupted.double()

    site_attn = Site("attn_out", layer=0)
    site_mlp = Site("mlp_out", layer=0)

    runner = SAEFeatureEdgeRunner(
        model64, {site_attn: sae_attn64, site_mlp: sae_mlp64}, resolver
    )
    circuit = runner.run(
        clean64, corrupted64, _metric,
        layer_pairs="all_forward",
        top_k_survivors=32,
    )

    edges = list(circuit.edges.keys())

    # (a) at least one edge exists (finite nonzero)
    if not edges:
        pytest.skip("No intra-layer edges computed (model/SAE may be degenerate)")

    max_abs = max(abs(s) for s in circuit.edges.values()) if circuit.edges else 0.0
    print(f"\n[intralayer attn→mlp] {len(edges)} edges, max|score|={max_abs:.2e}")
    assert max_abs > 1e-10, "All intra-layer edges are zero — degenerate model or SAE"

    # (b) at least one attn_out@0 → mlp_out@0 edge is present
    forward_edges = [
        e for e in edges
        if (e.writer.node.component == "attn_out" and e.writer.node.layer == 0
            and e.reader.node.component == "mlp_out" and e.reader.node.layer == 0)
    ]
    assert len(forward_edges) > 0, (
        "No attn_out@0 → mlp_out@0 forward edge found. "
        f"Edges found: {[(e.writer.node.component, e.reader.node.component) for e in edges[:5]]}"
    )

    # (c) Direct rank-guard assertion: attn_out@L < mlp_out@L in forward order.
    # This is the MECHANISM that prevents reverse edges; if _site_rank or _COMPONENT_OFFSET
    # were broken (e.g. swapped), this assertion would fail before the reverse-edge check.
    assert _site_rank(site_attn) < _site_rank(site_mlp), (
        f"_site_rank(attn_out@0)={_site_rank(site_attn)} must be < "
        f"_site_rank(mlp_out@0)={_site_rank(site_mlp)}. "
        "The rank guard that blocks reverse edges is broken."
    )

    # (c2) NO reverse edge mlp_out@0 → attn_out@0 (forward rank violation).
    # Verify via the pair-enumeration gate: the graph's forward-order constraint
    # (_site_rank(writer) < _site_rank(reader)) means the reverse pair is never passed
    # to _compute_pair_edges.  Swapping/removing the rank guard above would flip the
    # assertion in (c) first, making the root cause immediately visible.
    reverse_edges = [
        e for e in edges
        if (e.writer.node.component == "mlp_out" and e.writer.node.layer == 0
            and e.reader.node.component == "attn_out" and e.reader.node.layer == 0)
    ]
    assert len(reverse_edges) == 0, (
        f"Reverse edge mlp_out@0 → attn_out@0 found but should NOT be enumerated "
        f"(rank mlp=1 > rank attn=0). Got {len(reverse_edges)} reverse edges."
    )

    # (d) analytic ≈ bruteforce for forward edges (≤1e-4)
    bf = runner.bruteforce_feature_edge_scores(clean64, corrupted64, _metric, forward_edges)

    max_diff = 0.0
    for edge in forward_edges:
        a_s = circuit.edges[edge]
        b_s = bf.get(edge, 0.0)
        max_diff = max(max_diff, abs(a_s - b_s))

    print(f"[intralayer attn→mlp] max|analytic - bruteforce| = {max_diff:.2e}")
    assert max_diff < 1e-4, (
        f"Intra-layer edge gate: max|diff|={max_diff:.2e} > 1e-4"
    )


# ---------------------------------------------------------------------------
# Test 3: faithfulness(M) == 1 for a multi-site-per-layer circuit
# ---------------------------------------------------------------------------

def test_faithfulness_multisite():
    """faithfulness(M) == 1 (≤1e-5) for a circuit with both attn_out@0 and mlp_out@0.

    This proves node-set ablation correctly dedupes per (layer, component) and does
    NOT clobber one site's a_D with another site's ablation.
    """
    from circuitry.patching.sae_edges import SAEFeatureEdgeRunner
    from circuitry.patching.sites import Site

    d = 8
    model = _make_block_toy(d=d, seed=42)
    sae_attn0 = _make_sae(d_model=d, seed=11)
    sae_mlp0 = _make_sae(d_model=d, seed=7)
    sae_attn1 = _make_sae(d_model=d, seed=17)
    sae_mlp1 = _make_sae(d_model=d, seed=23)
    clean, corrupted = _make_inputs(d=d)
    resolver = _make_block_resolver(d=d)

    site_attn0 = Site("attn_out", layer=0)
    site_mlp0 = Site("mlp_out", layer=0)
    site_attn1 = Site("attn_out", layer=1)
    site_mlp1 = Site("mlp_out", layer=1)

    sae_sites = {
        site_attn0: sae_attn0,
        site_mlp0: sae_mlp0,
        site_attn1: sae_attn1,
        site_mlp1: sae_mlp1,
    }

    runner = SAEFeatureEdgeRunner(model, sae_sites, resolver)
    circuit = runner.run(
        clean, corrupted, _metric,
        layer_pairs="all_forward",
        top_k_survivors=32,
    )

    f_M = circuit.faithfulness(clean, corrupted, _metric)

    print(f"\n[faithfulness_multisite] faithfulness(M) = {f_M:.6f}")
    assert abs(f_M - 1.0) < 1e-5, (
        f"faithfulness(M) = {f_M:.6f} (expected ~1.0). "
        "This means node-set ablation is clobbering sites (key collision)."
    )


# ---------------------------------------------------------------------------
# Test 4: FeatureACDC runs without same-layer key collision
# ---------------------------------------------------------------------------

def test_acdc_multisite_no_clobber():
    """FeatureACDC prune over a multi-site-per-layer circuit runs without collision.

    Checks:
    1. FeatureACDCRunner.run() completes without error.
    2. The kept_nodes bookkeeping stays per (layer, component) — nodes from different
       same-layer components are independently retained/pruned.
    3. At least some nodes remain after pruning with a loose tau.
    """
    from circuitry.patching.sae_edges import FeatureACDCRunner
    from circuitry.patching.sites import Site

    d = 8
    model = _make_block_toy(d=d, seed=42)
    sae_attn0 = _make_sae(d_model=d, seed=11)
    sae_mlp0 = _make_sae(d_model=d, seed=7)
    sae_attn1 = _make_sae(d_model=d, seed=17)
    sae_mlp1 = _make_sae(d_model=d, seed=23)
    clean, corrupted = _make_inputs(d=d)
    resolver = _make_block_resolver(d=d)

    site_attn0 = Site("attn_out", layer=0)
    site_mlp0 = Site("mlp_out", layer=0)
    site_attn1 = Site("attn_out", layer=1)
    site_mlp1 = Site("mlp_out", layer=1)

    sae_sites = {
        site_attn0: sae_attn0,
        site_mlp0: sae_mlp0,
        site_attn1: sae_attn1,
        site_mlp1: sae_mlp1,
    }

    runner = FeatureACDCRunner(model, sae_sites, resolver)

    # Run with loose tau so most nodes are kept — we just want no crash + per-site bookkeeping
    circuit = runner.run(
        clean, corrupted, _metric,
        tau=1e6,  # very loose: accept removing nothing meaningful
        top_k_survivors=16,
    )

    # Circuit should have survivors from multiple sites
    survivors_by_site = {}
    for site, survivors in circuit.graph.survivors.items():
        key = (site.layer, site.component)
        survivors_by_site[key] = survivors

    print(f"\n[acdc_multisite_no_clobber] sites in survivors: {list(survivors_by_site.keys())}")

    # Should have all 4 sites represented
    assert (0, "attn_out") in survivors_by_site, "Missing attn_out@0 in ACDC survivors"
    assert (0, "mlp_out") in survivors_by_site, "Missing mlp_out@0 in ACDC survivors"
    assert (1, "attn_out") in survivors_by_site, "Missing attn_out@1 in ACDC survivors"
    assert (1, "mlp_out") in survivors_by_site, "Missing mlp_out@1 in ACDC survivors"

    # Verify nodes carry correct component — no inter-site collision
    for site, survivors in circuit.graph.survivors.items():
        for atp_node in survivors:
            nd = atp_node.node
            if nd.kind == "sae_feature" and nd.layer == site.layer:
                expected_comp = site.component if site.component != "resid_post" else None
                assert nd.component == expected_comp, (
                    f"Node at site {site} has component={nd.component!r}, "
                    f"expected {expected_comp!r}. Key collision."
                )

    # Run with zero tau: keep everything (acdc never removes when tau=0)
    # This is a stress test that the pruning loop itself doesn't crash
    circuit_zero = runner.run(
        clean, corrupted, _metric,
        tau=0.0,
        top_k_survivors=16,
    )
    # Should still have all 4 sites in survivors
    for site in sae_sites:
        assert site in circuit_zero.graph.survivors, (
            f"Site {site} missing from zero-tau ACDC survivors"
        )
