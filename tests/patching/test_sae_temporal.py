"""SAEFeatureTemporalRunner tests (v1.22).

Verifies that SAEFeatureTemporalRunner:
  - returns per-step AtPResult scores for each step
  - computes delta_scores between consecutive steps
  - identifies stable features (active at all steps)
  - identifies step-specific features (active at one step only)
  - top_stable returns the most reliably active features
  - raises ValueError on empty steps or duplicate keys
  - exports correctly from circuitry.patching
"""
from __future__ import annotations

import pytest
import torch

from tests.patching.test_sae_features import (
    LinearResidToy,
    SyntheticSAE,
    _make_clean_corr,
    _make_resolver,
    _metric,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_runner(model, sae, d, layer=0):
    from circuitry.patching.sae_temporal import SAEFeatureTemporalRunner
    from circuitry.patching.sites import Site
    site = Site("resid_post", layer=layer)
    return SAEFeatureTemporalRunner(model, {site: sae}, _make_resolver(d))


# ---------------------------------------------------------------------------
# Basic correctness
# ---------------------------------------------------------------------------


def test_temporal_runner_returns_per_step_scores():
    """Runner returns one AtPResult per step with finite scores."""
    d = 8
    torch.manual_seed(0)
    model = LinearResidToy(n_layers=2, d=d)
    torch.manual_seed(1)
    sae = SyntheticSAE(d_model=d, d_sae=16)

    runner = _make_runner(model, sae, d)
    clean0, corr0 = _make_clean_corr(d=d, seed=10)
    clean1, corr1 = _make_clean_corr(d=d, seed=20)

    result = runner.run(
        steps=[("step0", clean0, corr0), ("step1", clean1, corr1)],
        metric=lambda out: _metric(out),
    )

    assert set(result.step_keys) == {"step0", "step1"}
    assert "step0" in result.scores
    assert "step1" in result.scores

    for key in ["step0", "step1"]:
        for node, score in result.scores[key].scores.items():
            assert torch.isfinite(torch.tensor(score)), (
                f"Non-finite score at {key} for {node}"
            )


def test_temporal_delta_scores_computed():
    """delta_scores[step1] = scores[step1] - scores[step0] for common features."""
    d = 8
    torch.manual_seed(5)
    model = LinearResidToy(n_layers=2, d=d)
    torch.manual_seed(6)
    sae = SyntheticSAE(d_model=d, d_sae=16)

    runner = _make_runner(model, sae, d)
    clean0, corr0 = _make_clean_corr(d=d, seed=30)
    clean1, corr1 = _make_clean_corr(d=d, seed=40)

    result = runner.run(
        steps=[("s0", clean0, corr0), ("s1", clean1, corr1)],
        metric=lambda out: _metric(out),
    )

    assert "s1" in result.delta_scores
    assert "s0" not in result.delta_scores  # first step has no delta

    s0 = result.scores["s0"].scores
    s1 = result.scores["s1"].scores
    delta = result.delta_scores["s1"].scores

    for node in delta:
        expected = s1.get(node, 0.0) - s0.get(node, 0.0)
        assert abs(delta[node] - expected) < 1e-6, (
            f"Delta mismatch for {node}: got {delta[node]:.6f}, expected {expected:.6f}"
        )


def test_temporal_three_steps_two_deltas():
    """Three steps produce two delta entries (steps 1 and 2, not step 0)."""
    d = 8
    torch.manual_seed(7)
    model = LinearResidToy(n_layers=2, d=d)
    torch.manual_seed(8)
    sae = SyntheticSAE(d_model=d, d_sae=16)

    runner = _make_runner(model, sae, d)
    steps = []
    for i in range(3):
        c, r = _make_clean_corr(d=d, seed=50 + i)
        steps.append((i, c, r))

    result = runner.run(steps, metric=lambda out: _metric(out))

    assert list(result.step_keys) == [0, 1, 2]
    assert 0 not in result.delta_scores
    assert 1 in result.delta_scores
    assert 2 in result.delta_scores


# ---------------------------------------------------------------------------
# stable_features / step_specific_features
# ---------------------------------------------------------------------------


def test_temporal_stable_features_subset_of_all():
    """Stable features are a subset of the union of all active features."""
    d = 8
    torch.manual_seed(10)
    model = LinearResidToy(n_layers=2, d=d)
    torch.manual_seed(11)
    sae = SyntheticSAE(d_model=d, d_sae=16)

    runner = _make_runner(model, sae, d)
    clean0, corr0 = _make_clean_corr(d=d, seed=60)
    clean1, corr1 = _make_clean_corr(d=d, seed=70)

    result = runner.run(
        steps=[("a", clean0, corr0), ("b", clean1, corr1)],
        metric=lambda out: _metric(out),
    )

    threshold = 1e-9  # catch anything nonzero
    stable = result.stable_features(threshold=threshold)
    all_a = {n for n, s in result.scores["a"].scores.items() if abs(s) >= threshold}
    all_b = {n for n, s in result.scores["b"].scores.items() if abs(s) >= threshold}
    all_union = all_a | all_b

    for node in stable:
        assert node in all_union, f"Stable feature {node} not in any step"
        assert node in all_a, f"Stable feature {node} missing from step 'a'"
        assert node in all_b, f"Stable feature {node} missing from step 'b'"


def test_temporal_step_specific_features_not_in_other_steps():
    """Step-specific features for step A should not be active above threshold in step B."""
    d = 8
    torch.manual_seed(12)
    model = LinearResidToy(n_layers=2, d=d)
    torch.manual_seed(13)
    sae = SyntheticSAE(d_model=d, d_sae=16)

    runner = _make_runner(model, sae, d)
    clean0, corr0 = _make_clean_corr(d=d, seed=80)
    clean1, corr1 = _make_clean_corr(d=d, seed=90)

    result = runner.run(
        steps=[(0, clean0, corr0), (1, clean1, corr1)],
        metric=lambda out: _metric(out),
    )

    threshold = 0.01
    specific_0 = set(result.step_specific_features(0, threshold=threshold))
    active_1 = {n for n, s in result.scores[1].scores.items() if abs(s) >= threshold}

    overlap = specific_0 & active_1
    assert len(overlap) == 0, (
        f"Step-specific features for step 0 should not be active in step 1, "
        f"but found overlap: {overlap}"
    )


def test_temporal_stable_features_with_high_threshold_is_subset():
    """Raising the threshold can only shrink or keep the stable set."""
    d = 8
    torch.manual_seed(14)
    model = LinearResidToy(n_layers=2, d=d)
    torch.manual_seed(15)
    sae = SyntheticSAE(d_model=d, d_sae=16)

    runner = _make_runner(model, sae, d)
    clean0, corr0 = _make_clean_corr(d=d, seed=100)
    clean1, corr1 = _make_clean_corr(d=d, seed=110)

    result = runner.run(
        steps=[(0, clean0, corr0), (1, clean1, corr1)],
        metric=lambda out: _metric(out),
    )

    stable_low = set(result.stable_features(threshold=1e-9))
    stable_high = set(result.stable_features(threshold=1.0))
    assert stable_high <= stable_low, (
        "Higher threshold should give a subset of stable features at lower threshold"
    )


# ---------------------------------------------------------------------------
# top_stable
# ---------------------------------------------------------------------------


def test_temporal_top_stable_returns_k_or_fewer():
    """top_stable(k) returns at most k entries."""
    d = 8
    torch.manual_seed(16)
    model = LinearResidToy(n_layers=2, d=d)
    torch.manual_seed(17)
    sae = SyntheticSAE(d_model=d, d_sae=16)

    runner = _make_runner(model, sae, d)
    clean0, corr0 = _make_clean_corr(d=d, seed=120)
    clean1, corr1 = _make_clean_corr(d=d, seed=130)

    result = runner.run(
        steps=[("x", clean0, corr0), ("y", clean1, corr1)],
        metric=lambda out: _metric(out),
    )

    top5 = result.top_stable(k=5)
    assert len(top5) <= 5


def test_temporal_top_stable_sorted_descending():
    """top_stable returns entries in descending order of min |score|."""
    d = 8
    torch.manual_seed(18)
    model = LinearResidToy(n_layers=2, d=d)
    torch.manual_seed(19)
    sae = SyntheticSAE(d_model=d, d_sae=16)

    runner = _make_runner(model, sae, d)
    clean0, corr0 = _make_clean_corr(d=d, seed=140)
    clean1, corr1 = _make_clean_corr(d=d, seed=150)

    result = runner.run(
        steps=[(0, clean0, corr0), (1, clean1, corr1)],
        metric=lambda out: _metric(out),
    )

    top = result.top_stable(k=10)
    scores = [s for _, s in top]
    assert scores == sorted(scores, reverse=True), (
        "top_stable should return entries sorted by min |score| descending"
    )


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_temporal_empty_steps_raises():
    d = 8
    torch.manual_seed(20)
    model = LinearResidToy(n_layers=2, d=d)
    torch.manual_seed(21)
    sae = SyntheticSAE(d_model=d, d_sae=16)
    runner = _make_runner(model, sae, d)
    with pytest.raises(ValueError, match="non-empty"):
        runner.run([], metric=lambda out: _metric(out))


def test_temporal_duplicate_keys_raises():
    d = 8
    torch.manual_seed(22)
    model = LinearResidToy(n_layers=2, d=d)
    torch.manual_seed(23)
    sae = SyntheticSAE(d_model=d, d_sae=16)
    runner = _make_runner(model, sae, d)
    clean, corr = _make_clean_corr(d=d)
    with pytest.raises(ValueError, match="unique"):
        runner.run(
            [("same", clean, corr), ("same", clean, corr)],
            metric=lambda out: _metric(out),
        )


def test_temporal_step_specific_unknown_key_raises():
    d = 8
    torch.manual_seed(24)
    model = LinearResidToy(n_layers=2, d=d)
    torch.manual_seed(25)
    sae = SyntheticSAE(d_model=d, d_sae=16)
    runner = _make_runner(model, sae, d)
    clean, corr = _make_clean_corr(d=d)
    result = runner.run([("only", clean, corr)], metric=lambda out: _metric(out))
    with pytest.raises(KeyError):
        result.step_specific_features("nonexistent")


def test_temporal_single_step_no_delta():
    """Single step: delta_scores is empty dict."""
    d = 8
    torch.manual_seed(26)
    model = LinearResidToy(n_layers=2, d=d)
    torch.manual_seed(27)
    sae = SyntheticSAE(d_model=d, d_sae=16)
    runner = _make_runner(model, sae, d)
    clean, corr = _make_clean_corr(d=d)
    result = runner.run([("only", clean, corr)], metric=lambda out: _metric(out))
    assert result.delta_scores == {}
    assert len(result.scores) == 1


# ---------------------------------------------------------------------------
# Top-level import
# ---------------------------------------------------------------------------


def test_temporal_top_level_import():
    from circuitry.patching import SAEFeatureTemporalRunner, TemporalAtPResult
    assert SAEFeatureTemporalRunner is not None
    assert TemporalAtPResult is not None
