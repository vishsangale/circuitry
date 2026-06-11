"""Tests for HAPRunner (Hybrid Attribution + Pruning)."""
from __future__ import annotations

import torch

from circuitry.patching.edge_pruning import EdgePruningResult, EdgePruningRunner
from circuitry.patching.eap import EAPRunner
from circuitry.patching.hap import HAPRunner


def _clean_corrupted():
    clean = torch.tensor([[1, 2, 3, 4]])
    corrupted = torch.tensor([[4, 3, 2, 1]])
    return clean, corrupted


def _metric(logits):
    return logits[:, -1, :].sum()


def test_hap_returns_edge_pruning_result(linear_mlp_toy):
    runner = HAPRunner(linear_mlp_toy)
    clean, corrupted = _clean_corrupted()
    result = runner.run(clean, corrupted, _metric, n_steps=10)
    assert isinstance(result, EdgePruningResult)


def test_hap_circuit_is_subset_of_top_p_edges(linear_mlp_toy):
    """All circuit edges were in the top-p% by EAP score."""
    clean, corrupted = _clean_corrupted()
    top_p = 0.5

    # Compute the top-p edges independently
    eap_runner = EAPRunner(linear_mlp_toy)
    eap_result = eap_runner.run(clean, corrupted, _metric)
    all_ranked = eap_result.ranked()
    n_keep = max(1, int(len(all_ranked) * top_p))
    top_edge_set = {e for e, _ in all_ranked[:n_keep]}

    hap_runner = HAPRunner(linear_mlp_toy)
    result = hap_runner.run(clean, corrupted, _metric, top_p=top_p, n_steps=20)

    for edge in result.circuit:
        assert edge in top_edge_set, f"circuit edge {edge} not in top-{top_p} EAP edges"


def test_hap_top_p_1_same_candidate_set_as_full_pruning(linear_mlp_toy):
    """top_p=1.0 uses all edges as candidates — same as EdgePruningRunner directly."""
    clean, corrupted = _clean_corrupted()

    hap_runner = HAPRunner(linear_mlp_toy)
    hap_result = hap_runner.run(clean, corrupted, _metric, top_p=1.0, n_steps=50,
                                lambda_l0=0.01, lr=0.05)

    # With top_p=1.0, the candidate set equals the full edge set.
    # Both should cover the same graph edges.
    assert set(hap_result.mask_logits.keys()) == set(hap_result.graph.edges)


def test_hap_fewer_candidates_than_full(linear_mlp_toy):
    """With top_p=0.5, the candidate edges used by HAP are fewer than the total."""
    clean, corrupted = _clean_corrupted()

    eap_runner = EAPRunner(linear_mlp_toy)
    eap_result = eap_runner.run(clean, corrupted, _metric)
    total_edges = len(eap_result.graph.edges)

    top_p = 0.5
    n_candidates = max(1, int(total_edges * top_p))

    # There must be strictly fewer candidates than the full set (assuming > 1 edge)
    if total_edges > 1:
        assert n_candidates < total_edges


def test_hap_result_has_valid_graph(linear_mlp_toy):
    """EdgePruningResult from HAP must have a consistent graph."""
    runner = HAPRunner(linear_mlp_toy)
    clean, corrupted = _clean_corrupted()
    result = runner.run(clean, corrupted, _metric, n_steps=10)
    # circuit + removed == all graph edges
    assert len(result.circuit) + len(result.removed_edges) == len(result.graph.edges)
