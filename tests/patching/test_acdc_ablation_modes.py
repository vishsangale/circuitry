"""Tests for ACDC A4a (ablation_mode) and A4b (eap_skip_threshold) kwargs.

A4a:
  - ablation_mode="corrupted" (default) must reproduce the pre-change result
    exactly (back-compat regression).
  - ablation_mode="zero" runs without error and produces finite scores.
  - ablation_mode="mean" runs without error and produces finite scores.

A4b:
  - eap_skip_threshold=None reproduces today's behavior (test every edge).
  - eap_skip_threshold below all |EAP| scores keeps every edge (skip every
    ablation test → nothing ends up in removed).
  - A mid-range threshold prunes only sub-threshold edges from being tested
    (i.e., edges above the threshold are kept unconditionally).
"""
from __future__ import annotations

import math

import pytest
import torch

from circuitry.patching.acdc import ACDCRunner
from circuitry.patching.graph import Edge

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_runner(linear_mlp_toy):
    return ACDCRunner(linear_mlp_toy)


def _clean_corrupted():
    clean = torch.tensor([[1, 2, 3, 4]])
    corrupted = torch.tensor([[4, 3, 2, 1]])
    return clean, corrupted


# ---------------------------------------------------------------------------
# A4a — ablation_mode back-compat
# ---------------------------------------------------------------------------

def test_ablation_mode_default_is_corrupted(linear_mlp_toy):
    """Calling run() without ablation_mode must equal run(ablation_mode='corrupted')."""
    runner = _make_runner(linear_mlp_toy)
    clean, corrupted = _clean_corrupted()
    tau = 0.02

    r_default = runner.run(clean_inputs=clean, corrupted_inputs=corrupted, tau=tau)
    r_explicit = runner.run(
        clean_inputs=clean, corrupted_inputs=corrupted, tau=tau,
        ablation_mode="corrupted",
    )
    assert set(r_default.kept_edges) == set(r_explicit.kept_edges)
    assert r_default.final_kl == r_explicit.final_kl


def test_ablation_mode_corrupted_back_compat(linear_mlp_toy):
    """ablation_mode='corrupted' must produce the same kept circuit as the
    baseline run() call (no kwargs) — regression guard for A4a."""
    runner = _make_runner(linear_mlp_toy)
    clean, corrupted = _clean_corrupted()
    tau = 0.02

    # Baseline: run with no new kwargs (old call signature)
    baseline = runner.run(clean_inputs=clean, corrupted_inputs=corrupted, tau=tau)
    # New: explicit corrupted mode
    new = runner.run(
        clean_inputs=clean, corrupted_inputs=corrupted, tau=tau,
        ablation_mode="corrupted",
    )
    assert set(baseline.kept_edges) == set(new.kept_edges), (
        "ablation_mode='corrupted' must reproduce the pre-change result exactly"
    )


def test_ablation_mode_zero_runs_and_finite(linear_mlp_toy):
    """ablation_mode='zero' completes without error and produces a finite KL."""
    runner = _make_runner(linear_mlp_toy)
    clean, corrupted = _clean_corrupted()
    result = runner.run(
        clean_inputs=clean, corrupted_inputs=corrupted, tau=0.02,
        ablation_mode="zero",
    )
    assert math.isfinite(result.final_kl), "final_kl must be finite with zero ablation"
    assert isinstance(result.kept_edges, list)
    assert isinstance(result.removed_edges, list)


def test_ablation_mode_mean_runs_and_finite(linear_mlp_toy):
    """ablation_mode='mean' completes without error and produces a finite KL."""
    runner = _make_runner(linear_mlp_toy)
    clean, corrupted = _clean_corrupted()
    result = runner.run(
        clean_inputs=clean, corrupted_inputs=corrupted, tau=0.02,
        ablation_mode="mean",
    )
    assert math.isfinite(result.final_kl), "final_kl must be finite with mean ablation"
    assert isinstance(result.kept_edges, list)
    assert isinstance(result.removed_edges, list)


def test_ablation_mode_invalid_raises(linear_mlp_toy):
    """An unrecognised ablation_mode must raise ValueError naming the bad mode."""
    runner = _make_runner(linear_mlp_toy)
    corrupted = torch.tensor([[1, 2, 3, 4]])
    with pytest.raises(ValueError, match="bad_mode"):
        runner._cache_corrupted_acts(corrupted, ablation_mode="bad_mode")


def test_ablation_mode_zero_produces_zero_corr_acts(linear_mlp_toy):
    """For mode='zero', every cached corrupted activation tensor must be all-zero."""
    runner = _make_runner(linear_mlp_toy)
    corrupted = torch.tensor([[4, 3, 2, 1]])
    corr_act = runner._cache_corrupted_acts(corrupted, ablation_mode="zero")
    for node, act in corr_act.items():
        assert act.abs().max().item() == 0.0, (
            f"Expected zero tensor for node {node} with ablation_mode='zero'"
        )


def test_ablation_mode_mean_produces_constant_acts(linear_mlp_toy):
    """For mode='mean', each activation tensor should be spatially constant
    (all values along batch/seq dims equal the mean)."""
    runner = _make_runner(linear_mlp_toy)
    corrupted = torch.tensor([[4, 3, 2, 1]])
    corr_act = runner._cache_corrupted_acts(corrupted, ablation_mode="mean")
    for node, act in corr_act.items():
        # act shape: (batch, seq, d) or (batch, d).
        # All spatial slices must be identical (broadcast from mean).
        if act.ndim >= 2:
            first = act.flatten(0, -2)[0:1]  # first spatial position
            assert torch.allclose(act.flatten(0, -2), first.expand_as(act.flatten(0, -2))), (
                f"mean-mode activation for {node} is not spatially constant"
            )


# ---------------------------------------------------------------------------
# A4b — eap_skip_threshold
# ---------------------------------------------------------------------------

def test_eap_skip_threshold_none_reproduces_baseline(linear_mlp_toy):
    """eap_skip_threshold=None must reproduce the behavior of not passing it."""
    runner = _make_runner(linear_mlp_toy)
    clean, corrupted = _clean_corrupted()
    scores = {e: float(i) * 0.1 for i, e in enumerate(runner.graph.edges)}
    tau = 0.02

    baseline = runner.run(
        clean_inputs=clean, corrupted_inputs=corrupted, tau=tau,
        eap_scores=scores,
    )
    with_none = runner.run(
        clean_inputs=clean, corrupted_inputs=corrupted, tau=tau,
        eap_scores=scores, eap_skip_threshold=None,
    )
    assert set(baseline.kept_edges) == set(with_none.kept_edges), (
        "eap_skip_threshold=None must reproduce the no-threshold result"
    )


def test_eap_skip_threshold_below_all_keeps_every_edge(linear_mlp_toy):
    """When the threshold is below all |EAP| scores, every edge is skipped
    (treated as kept) — nothing is added to removed, so all edges survive."""
    runner = _make_runner(linear_mlp_toy)
    clean, corrupted = _clean_corrupted()
    tau = float("inf")  # accept any prune — without skips this would prune all

    # Give every edge a high score (1.0) and set the threshold below all of them.
    scores = {e: 1.0 for e in runner.graph.edges}
    threshold = 0.5  # < 1.0, so every edge's |EAP| exceeds it → all skipped

    result = runner.run(
        clean_inputs=clean, corrupted_inputs=corrupted, tau=tau,
        eap_scores=scores, eap_skip_threshold=threshold,
    )
    # All edges skipped → nothing in removed → all kept
    assert result.n_kept() == len(runner.graph.edges), (
        "With threshold below all EAP scores, every edge should be kept (skipped from testing)"
    )
    assert len(result.removed_edges) == 0


def test_eap_skip_threshold_above_all_equals_no_threshold(linear_mlp_toy):
    """When the threshold is above all |EAP| scores, no edge is skipped
    and the result must match eap_skip_threshold=None."""
    runner = _make_runner(linear_mlp_toy)
    clean, corrupted = _clean_corrupted()
    tau = 0.02

    # All edges have |EAP| = 0.01, threshold is 100.0 → no edge is skipped
    scores = {e: 0.01 for e in runner.graph.edges}
    high_threshold = 100.0

    no_threshold = runner.run(
        clean_inputs=clean, corrupted_inputs=corrupted, tau=tau,
        eap_scores=scores, eap_skip_threshold=None,
    )
    with_high = runner.run(
        clean_inputs=clean, corrupted_inputs=corrupted, tau=tau,
        eap_scores=scores, eap_skip_threshold=high_threshold,
    )
    assert set(no_threshold.kept_edges) == set(with_high.kept_edges), (
        "A threshold above all EAP scores should behave identically to no threshold"
    )


def test_eap_skip_threshold_mid_range_prunes_only_sub_threshold(linear_mlp_toy):
    """With a mid-range threshold, edges above the threshold are kept unconditionally
    (not tested); edges below the threshold are tested normally.

    Strategy:
    - Assign some edges score 2.0 (high) and the rest score 0.1 (low).
    - Set threshold = 1.0 (high-score edges skip; low-score edges are tested).
    - Run with tau=inf so any tested edge can be removed.
    - The high-score edges must all be in kept_edges (they were never tested).
    """
    runner = _make_runner(linear_mlp_toy)
    clean, corrupted = _clean_corrupted()
    edges = list(runner.graph.edges)
    tau = float("inf")  # accept any prune for tested edges

    if len(edges) < 2:
        # Skip if too few edges to distinguish categories (shouldn't happen for the toy)
        return

    # Assign alternating scores: first half high, second half low.
    mid = max(1, len(edges) // 2)
    high_edges = set(edges[:mid])
    low_edges = set(edges[mid:])

    scores: dict[Edge, float] = {}
    for e in high_edges:
        scores[e] = 2.0  # above threshold
    for e in low_edges:
        scores[e] = 0.1  # below threshold

    threshold = 1.0

    result = runner.run(
        clean_inputs=clean, corrupted_inputs=corrupted, tau=tau,
        eap_scores=scores, eap_skip_threshold=threshold,
    )

    kept = set(result.kept_edges)
    removed = set(result.removed_edges)

    # High-EAP edges must NOT appear in removed (they were skipped).
    for e in high_edges:
        assert e in kept, (
            f"Edge {e} with |EAP|=2.0 > threshold=1.0 must be kept (skipped from testing)"
        )

    # Low-EAP edges were tested; with tau=inf they can be removed or kept —
    # we only assert that the two sets are disjoint and cover all edges.
    assert kept | removed == set(edges), "kept ∪ removed must equal all edges"
    assert kept & removed == set(), "kept ∩ removed must be empty"


def test_eap_skip_threshold_no_eap_scores_tests_every_edge(linear_mlp_toy):
    """When eap_skip_threshold is set but eap_scores is None, the threshold
    is a no-op (cannot look up scores) — every edge is tested normally."""
    runner = _make_runner(linear_mlp_toy)
    clean, corrupted = _clean_corrupted()
    tau = 0.02

    baseline = runner.run(
        clean_inputs=clean, corrupted_inputs=corrupted, tau=tau,
    )
    with_threshold_no_scores = runner.run(
        clean_inputs=clean, corrupted_inputs=corrupted, tau=tau,
        eap_skip_threshold=0.0,  # would skip everything if scores were present
        eap_scores=None,          # but scores are None → no-op
    )
    assert set(baseline.kept_edges) == set(with_threshold_no_scores.kept_edges), (
        "eap_skip_threshold with no eap_scores must behave identically to no threshold"
    )
