"""ACDC recovery metric + greedy loop + ordering + sweep + custom metric."""
from __future__ import annotations

import torch

from circuitry.patching.acdc import ACDCRunner
from circuitry.patching.graph import Edge, Node


def test_recovery_metric_last_token_kl_zero_for_identical(linear_mlp_toy):
    runner = ACDCRunner(linear_mlp_toy)
    logits = torch.randn(1, 4, 5)
    assert runner._recovery_kl(logits, logits, position=-1) == 0.0


def test_recovery_metric_positive_for_different(linear_mlp_toy):
    runner = ACDCRunner(linear_mlp_toy)
    a = torch.randn(1, 4, 5)
    b = torch.randn(1, 4, 5)
    assert runner._recovery_kl(a, b, position=-1) > 0.0


def test_recovery_metric_position_none_averages_all(linear_mlp_toy):
    runner = ACDCRunner(linear_mlp_toy)
    a = torch.randn(1, 4, 5)
    b = torch.randn(1, 4, 5)
    last = runner._recovery_kl(a, b, position=-1)
    allpos = runner._recovery_kl(a, b, position=None)
    assert last != allpos  # different reductions


def test_run_empty_circuit_at_tau_inf_prunes_all(linear_mlp_toy):
    """τ = ∞ accepts every prune → empty circuit."""
    runner = ACDCRunner(linear_mlp_toy)
    clean = torch.tensor([[1, 2, 3, 4]])
    corrupted = torch.tensor([[4, 3, 2, 1]])
    result = runner.run(clean_inputs=clean, corrupted_inputs=corrupted, tau=float("inf"))
    assert result.n_kept() == 0
    assert len(result.removed_edges) == len(runner.graph.edges)


def test_run_tau_zero_keeps_live_circuit(linear_mlp_toy):
    """τ = 0 prunes only edges whose removal does not raise KL; the kept circuit
    still reproduces clean (final_kl ~ 0)."""
    runner = ACDCRunner(linear_mlp_toy)
    clean = torch.tensor([[1, 2, 3, 4]])
    corrupted = torch.tensor([[4, 3, 2, 1]])
    result = runner.run(clean_inputs=clean, corrupted_inputs=corrupted, tau=0.0)
    assert result.final_kl < 1e-4


def test_dead_edge_is_pruned(linear_mlp_toy):
    """When corrupted == clean, every Δact == 0, so every edge is dead and all
    prunes are free (KL stays 0) → empty circuit at any τ > 0."""
    runner = ACDCRunner(linear_mlp_toy)
    clean = torch.tensor([[1, 2, 3, 4]])
    result = runner.run(clean_inputs=clean, corrupted_inputs=clean, tau=1e-9)
    assert result.n_kept() == 0
    assert result.final_kl < 1e-6


def test_topo_ordering_is_deterministic(linear_mlp_toy):
    runner = ACDCRunner(linear_mlp_toy)
    clean = torch.tensor([[1, 2, 3, 4]])
    corrupted = torch.tensor([[4, 3, 2, 1]])
    r1 = runner.run(clean_inputs=clean, corrupted_inputs=corrupted, tau=0.02, ordering="topo")
    r2 = runner.run(clean_inputs=clean, corrupted_inputs=corrupted, tau=0.02, ordering="topo")
    assert set(r1.kept_edges) == set(r2.kept_edges)


def test_eap_ordering_consumes_scores(linear_mlp_toy):
    runner = ACDCRunner(linear_mlp_toy)
    clean = torch.tensor([[1, 2, 3, 4]])
    corrupted = torch.tensor([[4, 3, 2, 1]])
    scores = {e: 0.0 for e in runner.graph.edges}
    r = runner.run(clean_inputs=clean, corrupted_inputs=corrupted, tau=0.02,
                   ordering="eap", eap_scores=scores)
    assert isinstance(r.final_kl, float)
    r_default = runner.run(clean_inputs=clean, corrupted_inputs=corrupted, tau=0.02,
                           eap_scores=scores)  # ordering=None → picks eap
    assert set(r.kept_edges) == set(r_default.kept_edges)


def test_sweep_is_monotone(linear_mlp_toy):
    runner = ACDCRunner(linear_mlp_toy)
    clean = torch.tensor([[1, 2, 3, 4]])
    corrupted = torch.tensor([[4, 3, 2, 1]])
    frontier = runner.sweep(clean_inputs=clean, corrupted_inputs=corrupted,
                            taus=[0.001, 0.05, 1.0, float("inf")])
    assert [t for t, _, _ in frontier] == [0.001, 0.05, 1.0, float("inf")]
    n_kept = [n for _, n, _ in frontier]
    assert all(a >= b for a, b in zip(n_kept, n_kept[1:], strict=False))  # monotone non-increasing
    assert n_kept[-1] == 0  # τ=inf → empty circuit


def test_custom_metric_drives_pruning(linear_mlp_toy):
    runner = ACDCRunner(linear_mlp_toy)
    clean = torch.tensor([[1, 2, 3, 4]])
    corrupted = torch.tensor([[4, 3, 2, 1]])
    calls = {"n": 0}

    def my_metric(circuit_logits, clean_logits):
        calls["n"] += 1
        return float((circuit_logits[:, -1, :] - clean_logits[:, -1, :]).abs().sum())

    r = runner.run(clean_inputs=clean, corrupted_inputs=corrupted, tau=0.1,
                   metric=my_metric)
    assert calls["n"] > 0  # custom metric was used
    assert isinstance(r.final_kl, float)


def test_acdcresult_circuit_graph_subsets(linear_mlp_toy):
    runner = ACDCRunner(linear_mlp_toy)
    clean = torch.tensor([[1, 2, 3, 4]])
    corrupted = torch.tensor([[4, 3, 2, 1]])
    r = runner.run(clean_inputs=clean, corrupted_inputs=corrupted, tau=0.02)
    sub = r.circuit_graph()
    assert len(sub.edges) == r.n_kept()
    assert all(e in set(r.kept_edges) for e in sub.edges)


def test_prunes_dead_edges_keeps_live_path(linear_mlp_toy):
    """Discrimination: ACDC prunes edges from a provably-dead writer while
    keeping a known-live edge. Zero mlp(1)'s down_proj so its residual
    contribution is identically 0 on every input -> Δact[mlp1] == 0 -> its
    outgoing edges are dead. embed differs clean-vs-corrupted, so embed->logits
    is live. At a small τ, ACDC must prune mlp(1)->logits and keep embed->logits.
    """
    m = linear_mlp_toy
    with torch.no_grad():
        m.layers[1].mlp.down_proj.weight.zero_()
    runner = ACDCRunner(m)
    clean = torch.tensor([[1, 2, 3, 4]])
    corrupted = torch.tensor([[4, 3, 2, 1]])
    result = runner.run(clean_inputs=clean, corrupted_inputs=corrupted, tau=0.01)

    dead_edge = Edge(Node("mlp", 1), Node("logits"), "logits_in")
    live_edge = Edge(Node("embed"), Node("logits"), "logits_in")
    assert dead_edge in set(result.removed_edges)   # dead → pruned
    assert live_edge in set(result.kept_edges)       # live → kept
    # a genuine partial circuit (not all-or-nothing):
    assert 0 < result.n_kept() < len(runner.graph.edges)
