"""Tests for core/feature_flow.py — cross-layer feature flow (v1.47)."""
from __future__ import annotations

import math

import pytest
import torch

from circuitry.core.feature_flow import (
    FeatureFlowGraph,
    FlowEdge,
    feature_flow_graph,
    match_features,
)

# ---------------------------------------------------------------------------
# match_features
# ---------------------------------------------------------------------------


class TestMatchFeatures:
    def test_identical_dictionaries_match_identity(self):
        torch.manual_seed(0)
        W = torch.randn(6, 4)
        indices, sims = match_features(W, W)
        assert indices[:, 0].tolist() == list(range(6))
        torch.testing.assert_close(sims[:, 0], torch.ones(6), atol=1e-5, rtol=0)

    def test_permuted_dictionary_recovered(self):
        torch.manual_seed(0)
        W = torch.randn(5, 8)
        perm = torch.tensor([3, 0, 4, 1, 2])
        indices, sims = match_features(W, W[perm])
        # feature i of A sits at position perm^-1[i] in B
        inv = torch.argsort(perm)
        assert indices[:, 0].tolist() == inv.tolist()

    def test_scale_invariant(self):
        torch.manual_seed(0)
        W = torch.randn(4, 6)
        i1, s1 = match_features(W, W * 7.5)
        i2, s2 = match_features(W, W)
        assert i1[:, 0].tolist() == i2[:, 0].tolist()
        torch.testing.assert_close(s1, s2, atol=1e-5, rtol=0)

    def test_top_k_sorted_descending(self):
        torch.manual_seed(0)
        indices, sims = match_features(torch.randn(3, 8), torch.randn(7, 8), k=4)
        assert indices.shape == (3, 4) and sims.shape == (3, 4)
        for row in sims:
            assert row.tolist() == sorted(row.tolist(), reverse=True)

    def test_k_capped_at_n_features_b(self):
        indices, _ = match_features(torch.randn(3, 4), torch.randn(2, 4), k=10)
        assert indices.shape == (3, 2)

    def test_d_model_mismatch_raises(self):
        with pytest.raises(ValueError, match="d_model mismatch"):
            match_features(torch.randn(3, 4), torch.randn(3, 5))

    def test_non_2d_raises(self):
        with pytest.raises(ValueError, match="2-D"):
            match_features(torch.randn(3, 4, 5), torch.randn(3, 4))


# ---------------------------------------------------------------------------
# feature_flow_graph
# ---------------------------------------------------------------------------


def _three_layer_decoders():
    """Layer 0/1 share directions 0..3 (permuted in layer 1); layer 2 keeps
    only direction 0 and introduces orthogonal newcomers."""
    eye = torch.eye(8)
    d0 = eye[:4]                       # features 0-3 = e0..e3
    d1 = eye[[1, 0, 3, 2]]             # permutation of the same directions
    d2 = torch.cat([eye[1:2], eye[5:7]])  # e1 survives; e5, e6 are new
    return [d0, d1, d2]


class TestFeatureFlowGraph:
    def test_edges_and_similarities(self):
        g = feature_flow_graph(_three_layer_decoders(), threshold=0.9)
        assert isinstance(g, FeatureFlowGraph)
        # layer0 -> layer1: perfect permutation, 4 edges at cosine 1
        l01 = {e: s for e, s in g.scores.items() if e.src_layer == 0}
        assert len(l01) == 4
        assert all(s == pytest.approx(1.0) for s in l01.values())
        assert FlowEdge(0, 0, 1, 1) in l01    # e0 moved to slot 1
        # layer1 -> layer2: only e1 (layer-1 slot 0) survives
        l12 = {e: s for e, s in g.scores.items() if e.src_layer == 1}
        assert set(l12) == {FlowEdge(1, 0, 2, 0)}

    def test_threshold_filters(self):
        g_loose = feature_flow_graph(_three_layer_decoders(), threshold=0.0)
        g_tight = feature_flow_graph(_three_layer_decoders(), threshold=0.9)
        assert len(g_loose.scores) > len(g_tight.scores)

    def test_layer_ids_respected(self):
        g = feature_flow_graph(_three_layer_decoders(), layer_ids=[2, 5, 9],
                               threshold=0.9)
        assert g.layer_ids == [2, 5, 9]
        assert all(e.src_layer in (2, 5) for e in g.scores)

    def test_path_from_follows_chain(self):
        g = feature_flow_graph(_three_layer_decoders(), threshold=0.9)
        # e1 lives at: layer0 feature 1 -> layer1 feature 0 -> layer2 feature 0
        path = g.path_from(0, 1)
        assert [(layer, f) for layer, f, _ in path] == [(0, 1), (1, 0), (2, 0)]
        assert path[0][2] == pytest.approx(1.0)
        assert math.isnan(path[-1][2])  # chain ends at the last layer

    def test_path_from_dead_end(self):
        g = feature_flow_graph(_three_layer_decoders(), threshold=0.9)
        # layer1 feature 1 (= e0) has no match in layer 2
        path = g.path_from(1, 1)
        assert [(layer, f) for layer, f, _ in path] == [(1, 1)]
        assert math.isnan(path[0][2])

    def test_born_at(self):
        g = feature_flow_graph(_three_layer_decoders(), threshold=0.9)
        assert g.born_at(0) == [0, 1, 2, 3]      # first layer: all born
        assert g.born_at(1) == []                # all matched from layer 0
        assert g.born_at(2) == [1, 2]            # e5, e6 are newcomers

    def test_born_at_unknown_layer_raises(self):
        g = feature_flow_graph(_three_layer_decoders(), threshold=0.9)
        with pytest.raises(ValueError, match="not in layer_ids"):
            g.born_at(7)

    def test_needs_two_decoders(self):
        with pytest.raises(ValueError, match=">= 2 decoders"):
            feature_flow_graph([torch.randn(3, 4)])

    def test_layer_id_count_mismatch_raises(self):
        with pytest.raises(ValueError, match="layer_ids"):
            feature_flow_graph([torch.randn(3, 4)] * 2, layer_ids=[0])

    def test_to_markdown(self):
        g = feature_flow_graph(_three_layer_decoders(), threshold=0.9)
        md = g.to_markdown(top_k=3)
        assert "## Feature Flow Graph" in md
        assert "| rank |" in md


# ---------------------------------------------------------------------------
# export integration + top-level exports
# ---------------------------------------------------------------------------


def test_flow_graph_exports_to_neuronpedia_and_html():
    from circuitry.patching.export import to_html, to_neuronpedia_graph

    g = feature_flow_graph(_three_layer_decoders(), threshold=0.9)
    np_graph = to_neuronpedia_graph(g, slug="flow", scan="tiny")
    assert len(np_graph["links"]) == len(g.scores)
    ids = {n["node_id"] for n in np_graph["nodes"]}
    assert "0_1_0" in ids and "2_0_0" in ids
    doc = to_html(g, labels={(0, 1): "the e1 feature"})
    assert "the e1 feature" in doc


def test_top_level_exports():
    import circuitry

    for name in ("match_features", "feature_flow_graph",
                 "FeatureFlowGraph", "FlowEdge"):
        assert hasattr(circuitry, name)
        assert name in circuitry.__all__
