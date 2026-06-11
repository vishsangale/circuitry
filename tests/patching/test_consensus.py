"""Tests for patching/consensus.py — cross-method circuit consensus (v1.42)."""
from __future__ import annotations

import pytest

from circuitry.patching.consensus import CircuitConsensus
from circuitry.patching.eap import EAPResult
from circuitry.patching.graph import build_graph


@pytest.fixture
def edge_triplet():
    graph = build_graph(n_layers=1, n_heads=2)
    e = graph.edges
    return e[0], e[1], e[2]


def test_requires_two_circuits(edge_triplet):
    a, _, _ = edge_triplet
    with pytest.raises(ValueError, match=">= 2 circuits"):
        CircuitConsensus({"only": {a}})


def test_agreement_fractions(edge_triplet):
    a, b, c = edge_triplet
    cc = CircuitConsensus({"x": {a, b}, "y": {a, c}, "z": {a}})
    agreement = cc.agreement()
    assert agreement[a] == 1.0
    assert agreement[b] == pytest.approx(1 / 3)
    assert agreement[c] == pytest.approx(1 / 3)


def test_consensus_edges_thresholds(edge_triplet):
    a, b, c = edge_triplet
    cc = CircuitConsensus({"x": {a, b}, "y": {a, c}, "z": {a, b}})
    assert cc.consensus_edges(1.0) == {a}
    assert cc.consensus_edges(2 / 3) == {a, b}
    assert cc.consensus_edges(0.0) == {a, b, c}


def test_pairwise_jaccard(edge_triplet):
    a, b, c = edge_triplet
    cc = CircuitConsensus({"x": {a, b}, "y": {a, c}})
    assert cc.pairwise_jaccard()[("x", "y")] == pytest.approx(1 / 3)


def test_pairwise_jaccard_empty_circuits():
    cc = CircuitConsensus({"x": set(), "y": set()})
    assert cc.pairwise_jaccard()[("x", "y")] == 1.0


def test_from_results_with_tau():
    graph = build_graph(n_layers=1, n_heads=2)
    scores_a = {e: 0.01 * i for i, e in enumerate(graph.edges)}
    scores_b = {e: -0.01 * i for i, e in enumerate(graph.edges)}
    ra = EAPResult(graph=graph, scores=scores_a)
    rb = EAPResult(graph=graph, scores=scores_b)
    cc = CircuitConsensus.from_results([ra, rb], tau=0.05)
    # identical |score| structure -> identical circuits
    assert cc.pairwise_jaccard()[("method_0", "method_1")] == 1.0
    assert cc.circuits["method_0"] == frozenset(ra.threshold(0.05))


def test_from_results_with_top_k(edge_triplet):
    graph = build_graph(n_layers=1, n_heads=2)
    ra = EAPResult(
        graph=graph, scores={e: float(i) for i, e in enumerate(graph.edges)},
    )
    a, b, _ = edge_triplet
    cc = CircuitConsensus.from_results([ra, {a, b}], top_k=2, names=["eap", "manual"])
    assert len(cc.circuits["eap"]) == 2
    assert cc.circuits["manual"] == frozenset({a, b})


def test_from_results_scored_without_rule_raises():
    graph = build_graph(n_layers=1, n_heads=2)
    r = EAPResult(graph=graph, scores={e: 1.0 for e in graph.edges})
    with pytest.raises(ValueError, match="binarization rule"):
        CircuitConsensus.from_results([r, r])


def test_from_results_name_count_mismatch(edge_triplet):
    a, b, _ = edge_triplet
    with pytest.raises(ValueError, match="names for"):
        CircuitConsensus.from_results([{a}, {b}], names=["only_one"])


def test_to_markdown(edge_triplet):
    a, b, _ = edge_triplet
    cc = CircuitConsensus({"eap": {a, b}, "relp": {a}})
    md = cc.to_markdown()
    assert "## Circuit Consensus" in md
    assert "Pairwise Jaccard" in md
    assert "`eap`" in md and "`relp`" in md
    assert "1.00" in md  # edge a full agreement


def test_exports():
    import circuitry
    from circuitry import patching

    assert circuitry.CircuitConsensus is CircuitConsensus
    assert patching.CircuitConsensus is CircuitConsensus
    assert "CircuitConsensus" in circuitry.__all__
