"""Parallel-attention architecture tests (v1.20).

Verifies that SAEFeatureEdgeRunner.run(arch='parallel') proactively skips
same-layer attn_out→mlp_out edges (causally undefined in GPT-J-style models)
while preserving cross-layer and sequential-arch edges.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torch import Tensor

from tests.patching.test_sae_features import (
    LinearResidToy,
    SyntheticSAE,
    _make_clean_corr,
    _make_resolver,
    _metric,
)


# ---------------------------------------------------------------------------
# Unit tests: _is_parallel_intra_layer helper
# ---------------------------------------------------------------------------

def _make_site(component: str, layer: int):
    from circuitry.patching.sites import Site
    return Site(component, layer=layer)


def test_is_parallel_intra_layer_true_for_same_layer_attn_mlp():
    from circuitry.patching.sae_edges import _is_parallel_intra_layer
    w = _make_site("attn_out", 0)
    r = _make_site("mlp_out", 0)
    assert _is_parallel_intra_layer(w, r) is True


def test_is_parallel_intra_layer_false_cross_layer():
    from circuitry.patching.sae_edges import _is_parallel_intra_layer
    w = _make_site("attn_out", 0)
    r = _make_site("mlp_out", 1)
    assert _is_parallel_intra_layer(w, r) is False


def test_is_parallel_intra_layer_false_for_resid_post():
    from circuitry.patching.sae_edges import _is_parallel_intra_layer
    w = _make_site("resid_post", 0)
    r = _make_site("resid_post", 1)
    assert _is_parallel_intra_layer(w, r) is False


def test_is_parallel_intra_layer_false_reversed():
    """mlp_out → attn_out is a backward edge, not a parallel-skip case."""
    from circuitry.patching.sae_edges import _is_parallel_intra_layer
    w = _make_site("mlp_out", 0)
    r = _make_site("attn_out", 0)
    assert _is_parallel_intra_layer(w, r) is False


# ---------------------------------------------------------------------------
# Invalid arch value → ValueError
# ---------------------------------------------------------------------------

def test_invalid_arch_raises():
    d = 8
    torch.manual_seed(1)
    model = LinearResidToy(n_layers=2, d=d)
    torch.manual_seed(2)
    sae0 = SyntheticSAE(d_model=d, d_sae=16)
    torch.manual_seed(3)
    sae1 = SyntheticSAE(d_model=d, d_sae=16)
    from circuitry.patching.sae_edges import SAEFeatureEdgeRunner
    from circuitry.patching.sites import Site
    runner = SAEFeatureEdgeRunner(
        model,
        {Site("resid_post", layer=0): sae0, Site("resid_post", layer=1): sae1},
        _make_resolver(d),
    )
    clean, corr = _make_clean_corr(d=d)
    with pytest.raises(ValueError, match="arch"):
        runner.run(clean, corr, lambda out: _metric(out), arch="unknown")


# ---------------------------------------------------------------------------
# _run_inner skips pair when arch='parallel' (mock _compute_pair_edges)
# ---------------------------------------------------------------------------

def test_parallel_arch_does_not_call_compute_for_intra_layer_attn_mlp():
    """Verify that _compute_pair_edges is never called for attn_out@L→mlp_out@L under arch='parallel'."""
    from unittest.mock import patch
    from circuitry.patching.graph import Node
    from circuitry.patching.sae_edges import SAEFeatureEdgeRunner
    from circuitry.patching.sites import Site
    from circuitry.patching.atp import AtPNode, AtPResult

    d, d_sae = 8, 16
    torch.manual_seed(70)
    model = LinearResidToy(n_layers=2, d=d)
    torch.manual_seed(71)
    sae_a = SyntheticSAE(d_model=d, d_sae=d_sae)
    torch.manual_seed(72)
    sae_m = SyntheticSAE(d_model=d, d_sae=d_sae)

    site_attn = Site("attn_out", layer=0)
    site_mlp = Site("mlp_out", layer=0)
    resolver = _make_resolver(d)

    runner = SAEFeatureEdgeRunner(model, {site_attn: sae_a, site_mlp: sae_m}, resolver)

    called_pairs = []

    def tracking_compute(*, writer_site, reader_site, **kwargs):
        called_pairs.append((writer_site.component, reader_site.component))
        return {}, None

    # Return dummy survivors so the "skip if no survivors" guard doesn't fire —
    # we want to confirm it's the arch skip that prevents the call, not the survivors guard.
    dummy_node_attn = AtPNode(Node("sae_feature", layer=0, neuron=0, component="attn_out"))
    dummy_node_mlp = AtPNode(Node("sae_feature", layer=0, neuron=0, component="mlp_out"))
    dummy_result = AtPResult(scores={dummy_node_attn: 1.0, dummy_node_mlp: 1.0})

    with patch.object(runner, "_compute_pair_edges", side_effect=tracking_compute):
        with patch.object(runner._stage1_runner, "run", return_value=dummy_result):
            try:
                runner.run(
                    torch.zeros(1, 3, d), torch.zeros(1, 3, d),
                    lambda out: out.sum(),
                    arch="parallel",
                )
            except Exception:
                pass  # site resolution may fail; we only care about called_pairs

    # attn_out@0 → mlp_out@0 must NOT appear in called pairs
    assert ("attn_out", "mlp_out") not in called_pairs, (
        "arch='parallel' should skip same-layer attn_out→mlp_out but it was called"
    )


def test_sequential_arch_does_call_compute_for_intra_layer_attn_mlp():
    """With arch='sequential', attn_out@L→mlp_out@L IS passed to _compute_pair_edges."""
    from unittest.mock import patch
    from circuitry.patching.graph import Node
    from circuitry.patching.sae_edges import SAEFeatureEdgeRunner
    from circuitry.patching.sites import Site
    from circuitry.patching.atp import AtPNode, AtPResult

    d, d_sae = 8, 16
    torch.manual_seed(80)
    model = LinearResidToy(n_layers=2, d=d)
    torch.manual_seed(81)
    sae_a = SyntheticSAE(d_model=d, d_sae=d_sae)
    torch.manual_seed(82)
    sae_m = SyntheticSAE(d_model=d, d_sae=d_sae)

    site_attn = Site("attn_out", layer=0)
    site_mlp = Site("mlp_out", layer=0)
    resolver = _make_resolver(d)

    runner = SAEFeatureEdgeRunner(model, {site_attn: sae_a, site_mlp: sae_m}, resolver)

    called_pairs = []

    def tracking_compute(*, writer_site, reader_site, **kwargs):
        called_pairs.append((writer_site.component, reader_site.component))
        return {}, None

    # Return dummy survivors so the "skip if no survivors" guard doesn't fire
    dummy_node_attn = AtPNode(Node("sae_feature", layer=0, neuron=0, component="attn_out"))
    dummy_node_mlp = AtPNode(Node("sae_feature", layer=0, neuron=0, component="mlp_out"))
    dummy_result = AtPResult(scores={dummy_node_attn: 1.0, dummy_node_mlp: 1.0})

    with patch.object(runner, "_compute_pair_edges", side_effect=tracking_compute):
        with patch.object(runner._stage1_runner, "run", return_value=dummy_result):
            try:
                runner.run(
                    torch.zeros(1, 3, d), torch.zeros(1, 3, d),
                    lambda out: out.sum(),
                    arch="sequential",
                )
            except Exception:
                pass

    assert ("attn_out", "mlp_out") in called_pairs, (
        "arch='sequential' should attempt same-layer attn_out→mlp_out"
    )


# ---------------------------------------------------------------------------
# Cross-layer resid_post edges unaffected by arch flag (integration)
# ---------------------------------------------------------------------------

def _make_resid_runner(model, sae0, sae1, d):
    from circuitry.patching.sae_edges import SAEFeatureEdgeRunner
    from circuitry.patching.sites import Site
    return SAEFeatureEdgeRunner(
        model,
        {Site("resid_post", layer=0): sae0, Site("resid_post", layer=1): sae1},
        _make_resolver(d),
    )


def test_cross_layer_edges_same_under_both_arches():
    d, d_sae = 8, 16
    torch.manual_seed(90)
    model = LinearResidToy(n_layers=2, d=d)
    torch.manual_seed(91)
    sae0 = SyntheticSAE(d_model=d, d_sae=d_sae)
    torch.manual_seed(92)
    sae1 = SyntheticSAE(d_model=d, d_sae=d_sae)
    runner = _make_resid_runner(model, sae0, sae1, d)
    clean, corr = _make_clean_corr(d=d)

    c_seq = runner.run(clean, corr, lambda out: _metric(out), arch="sequential")
    c_par = runner.run(clean, corr, lambda out: _metric(out), arch="parallel")

    # resid_post→resid_post cross-layer edges are unaffected by arch
    assert set(c_seq.edges.keys()) == set(c_par.edges.keys())
    for edge in c_seq.edges:
        assert abs(c_seq.edges[edge] - c_par.edges[edge]) < 1e-6


def test_default_arch_matches_sequential():
    d, d_sae = 8, 16
    torch.manual_seed(100)
    model = LinearResidToy(n_layers=2, d=d)
    torch.manual_seed(101)
    sae0 = SyntheticSAE(d_model=d, d_sae=d_sae)
    torch.manual_seed(102)
    sae1 = SyntheticSAE(d_model=d, d_sae=d_sae)
    runner = _make_resid_runner(model, sae0, sae1, d)
    clean, corr = _make_clean_corr(d=d)

    c_def = runner.run(clean, corr, lambda out: _metric(out))
    c_seq = runner.run(clean, corr, lambda out: _metric(out), arch="sequential")
    assert set(c_def.edges.keys()) == set(c_seq.edges.keys())


def test_parallel_arch_with_per_position():
    d, d_sae = 8, 16
    torch.manual_seed(110)
    model = LinearResidToy(n_layers=2, d=d)
    torch.manual_seed(111)
    sae0 = SyntheticSAE(d_model=d, d_sae=d_sae)
    torch.manual_seed(112)
    sae1 = SyntheticSAE(d_model=d, d_sae=d_sae)
    runner = _make_resid_runner(model, sae0, sae1, d)
    clean, corr = _make_clean_corr(d=d)

    circuit = runner.run(clean, corr, lambda out: _metric(out),
                         arch="parallel", per_position=True)
    if circuit.edges:
        assert circuit.position_scores is not None
