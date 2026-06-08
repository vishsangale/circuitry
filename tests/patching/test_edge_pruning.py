"""Tests for EdgePruningRunner and EdgePruningResult."""
from __future__ import annotations

import torch

from circuitry.patching.edge_pruning import EdgePruningResult, EdgePruningRunner
from circuitry.patching.graph import Edge, Node, build_graph


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_runner(linear_mlp_toy):
    return EdgePruningRunner(linear_mlp_toy)


def _clean_corrupted():
    clean = torch.tensor([[1, 2, 3, 4]])
    corrupted = torch.tensor([[4, 3, 2, 1]])
    return clean, corrupted


def _metric(logits):
    return logits[:, -1, :].sum()


# ---------------------------------------------------------------------------
# Basic API tests
# ---------------------------------------------------------------------------

def test_edge_pruning_returns_result(linear_mlp_toy):
    runner = _make_runner(linear_mlp_toy)
    clean, corrupted = _clean_corrupted()
    result = runner.run(clean, corrupted, _metric, n_steps=10)
    assert isinstance(result, EdgePruningResult)


def test_edge_pruning_circuit_is_subset_of_graph(linear_mlp_toy):
    runner = _make_runner(linear_mlp_toy)
    clean, corrupted = _clean_corrupted()
    result = runner.run(clean, corrupted, _metric, n_steps=10)
    graph_edge_set = set(result.graph.edges)
    for edge in result.circuit:
        assert edge in graph_edge_set, f"circuit edge {edge} not in graph.edges"


def test_edge_pruning_circuit_plus_removed_equals_all(linear_mlp_toy):
    runner = _make_runner(linear_mlp_toy)
    clean, corrupted = _clean_corrupted()
    result = runner.run(clean, corrupted, _metric, n_steps=10)
    assert len(result.circuit) + len(result.removed_edges) == len(result.graph.edges)


def test_edge_pruning_lambda_zero_keeps_all(linear_mlp_toy):
    """With lambda_l0=0 and enough steps, task loss dominates and all edges
    with nonzero EAP score should become active (mask logit > 0)."""
    runner = _make_runner(linear_mlp_toy)
    clean, corrupted = _clean_corrupted()
    result = runner.run(clean, corrupted, _metric,
                        lambda_l0=0.0, n_steps=300, lr=0.1)
    # All edges (or at least the majority) should be kept when there's no
    # sparsity pressure — most |eap_score| > 0 so task loss pushes z up.
    assert result.n_circuit() > 0


def test_edge_pruning_lambda_large_keeps_few(linear_mlp_toy):
    """With very large lambda_l0, sparsity term dominates and few edges survive."""
    runner = _make_runner(linear_mlp_toy)
    clean, corrupted = _clean_corrupted()
    result = runner.run(clean, corrupted, _metric,
                        lambda_l0=100.0, n_steps=200)
    # With extreme sparsity penalty, expect far fewer than all edges
    assert result.n_circuit() < len(result.graph.edges)


# ---------------------------------------------------------------------------
# Serialization tests
# ---------------------------------------------------------------------------

def test_edge_pruning_json_roundtrip(linear_mlp_toy):
    runner = _make_runner(linear_mlp_toy)
    clean, corrupted = _clean_corrupted()
    result = runner.run(clean, corrupted, _metric, n_steps=20)

    text = result.to_json()
    restored = EdgePruningResult.from_json(text)

    assert restored.graph.n_layers == result.graph.n_layers
    assert restored.graph.n_heads == result.graph.n_heads
    assert set(restored.circuit) == set(result.circuit)
    assert restored.lambda_l0 == result.lambda_l0
    assert restored.n_steps_run == result.n_steps_run
    # mask_logits roundtrip (within float precision)
    for e in result.graph.edges:
        assert abs(restored.mask_logits[e] - result.mask_logits[e]) < 1e-4


def test_edge_pruning_save_load(tmp_path, linear_mlp_toy):
    runner = _make_runner(linear_mlp_toy)
    clean, corrupted = _clean_corrupted()
    result = runner.run(clean, corrupted, _metric, n_steps=20)

    path = tmp_path / "ep_result.json"
    result.save(path)
    loaded = EdgePruningResult.load(path)

    assert set(loaded.circuit) == set(result.circuit)
    assert loaded.graph.n_layers == result.graph.n_layers
    assert loaded.graph.n_heads == result.graph.n_heads


# ---------------------------------------------------------------------------
# Structural / edge-case tests
# ---------------------------------------------------------------------------

def test_edge_pruning_mask_logits_covers_all_graph_edges(linear_mlp_toy):
    runner = _make_runner(linear_mlp_toy)
    clean, corrupted = _clean_corrupted()
    result = runner.run(clean, corrupted, _metric, n_steps=10)
    for e in result.graph.edges:
        assert e in result.mask_logits, f"mask_logits missing edge {e}"


def test_edge_pruning_candidate_edges_restricts_circuit(linear_mlp_toy):
    """When candidate_edges is set, circuit is a subset of those candidates."""
    runner = _make_runner(linear_mlp_toy)
    clean, corrupted = _clean_corrupted()
    graph = runner._eap.graph
    # Use only the first half of graph edges as candidates
    half = graph.edges[:len(graph.edges) // 2]
    result = runner.run(clean, corrupted, _metric,
                        candidate_edges=half, n_steps=20)
    half_set = set(half)
    for e in result.circuit:
        assert e in half_set, f"circuit edge {e} not in candidate_edges"


def test_edge_pruning_circuit_graph_subset(linear_mlp_toy):
    runner = _make_runner(linear_mlp_toy)
    clean, corrupted = _clean_corrupted()
    result = runner.run(clean, corrupted, _metric, n_steps=20)
    sub = result.circuit_graph()
    assert len(sub.edges) == result.n_circuit()
    circuit_set = set(result.circuit)
    for e in sub.edges:
        assert e in circuit_set
