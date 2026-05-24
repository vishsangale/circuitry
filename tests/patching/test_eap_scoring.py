"""Tests for the EAP scoring engine (stacked caches -> scores). Spec §5."""
from __future__ import annotations

import pytest
import torch

from circuitry.patching.eap import score_edges
from circuitry.patching.graph import Edge, Node, build_graph


def test_score_matches_manual_dot_product():
    torch.manual_seed(0)
    g = build_graph(n_layers=1, n_heads=1)
    b, p, d = 2, 3, 4
    act_clean = torch.randn(b, p, len(g.writers), d)
    act_corrupted = torch.randn(b, p, len(g.writers), d)
    grad = torch.randn(b, p, len(g.readers), d)

    result = score_edges(g, act_clean, act_corrupted, grad)

    # verify one edge by hand
    e = g.edges[0]
    w = g.writer_index(e.writer)
    r = g.reader_index(e.reader, e.slot)
    delta = act_corrupted[:, :, w, :] - act_clean[:, :, w, :]
    expected = float((delta * grad[:, :, r, :]).sum().item())
    assert result.scores[e] == pytest.approx(expected, abs=1e-4)


def test_only_valid_edges_scored():
    g = build_graph(n_layers=2, n_heads=2)
    b, p, d = 1, 2, 4
    a = torch.zeros(b, p, len(g.writers), d)
    grad = torch.zeros(b, p, len(g.readers), d)
    result = score_edges(g, a, a.clone(), grad)
    assert set(result.scores.keys()) == set(g.edges)


def test_ranked_and_top_k_by_abs():
    g = build_graph(n_layers=1, n_heads=1)
    b, p, d = 1, 1, 2
    act_clean = torch.zeros(b, p, len(g.writers), d)
    act_corrupted = torch.zeros(b, p, len(g.writers), d)
    grad = torch.zeros(b, p, len(g.readers), d)
    # craft one strong edge: embed -> logits
    e = Edge(Node("embed"), Node("logits"), "logits_in")
    w = g.writer_index(e.writer)
    r = g.reader_index(e.reader, e.slot)
    act_corrupted[:, :, w, :] = 5.0
    grad[:, :, r, :] = 1.0
    result = score_edges(g, act_clean, act_corrupted, grad)
    ranked = result.ranked()
    assert ranked[0][0] == e            # largest |score| first
    assert result.top_k(1) == [ranked[0]]


def test_threshold_returns_edges_above_abs_tau():
    g = build_graph(n_layers=1, n_heads=1)
    b, p, d = 1, 1, 2
    act_clean = torch.zeros(b, p, len(g.writers), d)
    act_corrupted = torch.zeros(b, p, len(g.writers), d)
    grad = torch.zeros(b, p, len(g.readers), d)
    e = Edge(Node("embed"), Node("logits"), "logits_in")
    act_corrupted[:, :, g.writer_index(e.writer), :] = -5.0  # negative score
    grad[:, :, g.reader_index(e.reader, e.slot), :] = 1.0
    result = score_edges(g, act_clean, act_corrupted, grad)
    circuit = result.threshold(1.0)
    assert e in circuit  # |negative score| >= tau
