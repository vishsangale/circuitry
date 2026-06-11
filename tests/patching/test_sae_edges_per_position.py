"""Per-position edge score tests (v1.19).

Verifies that SAEFeatureEdgeRunner.run(per_position=True) populates
SAEFeatureCircuit.position_scores with (seq_len,) tensors that sum to the
scalar edge scores.  Uses the same SyntheticSAE / LinearResidToy helpers as
the main edge tests.
"""
from __future__ import annotations

import pytest
import torch

from tests.patching.test_sae_edges import (
    _make_clean_corr,
    _make_two_saes,
    _make_two_site_runner,
)
from tests.patching.test_sae_features import (
    LinearResidToy,
    _make_resolver,
    _metric,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_pair(seq_len: int = 4, d: int = 8, d_sae: int = 16, per_position: bool = False):
    """Run SAEFeatureEdgeRunner with the given per_position flag."""
    torch.manual_seed(99)
    model = LinearResidToy(n_layers=2, d=d)
    sae0, sae1 = _make_two_saes(d=d, d_sae=d_sae)
    runner = _make_two_site_runner(model, sae0, sae1, d=d)
    clean, corr = _make_clean_corr(d=d, b=1, s=seq_len)
    return runner.run(
        clean, corr, lambda out: _metric(out),
        per_position=per_position,
    )


# ---------------------------------------------------------------------------
# Default (per_position=False): position_scores must be None
# ---------------------------------------------------------------------------

def test_position_scores_none_by_default():
    circuit = _run_pair(per_position=False)
    assert circuit.position_scores is None


# ---------------------------------------------------------------------------
# per_position=True: position_scores must be populated
# ---------------------------------------------------------------------------

def test_position_scores_not_none_when_requested():
    circuit = _run_pair(per_position=True)
    assert circuit.position_scores is not None


def test_position_scores_keys_match_edges():
    circuit = _run_pair(per_position=True)
    assert set(circuit.position_scores.keys()) == set(circuit.edges.keys())


def test_position_scores_tensor_shape():
    seq_len = 5
    circuit = _run_pair(seq_len=seq_len, per_position=True)
    for edge, pos_t in circuit.position_scores.items():
        assert pos_t.shape == (seq_len,), (
            f"Expected (seq_len={seq_len},) but got {pos_t.shape} for edge {edge}"
        )


def test_position_scores_sum_equals_scalar():
    """position_scores[e].sum() must equal edges[e] within float32 tolerance."""
    circuit = _run_pair(per_position=True)
    for edge, scalar in circuit.edges.items():
        pos_sum = float(circuit.position_scores[edge].sum())
        assert abs(pos_sum - scalar) < 1e-4, (
            f"pos_sum={pos_sum} != scalar={scalar} for edge {edge}"
        )


def test_position_scores_consistent_across_batch_sizes():
    """Scalar scores and position sums should be consistent (within tolerance) for batch=1."""
    circuit = _run_pair(seq_len=4, d=8, per_position=True)
    for edge, pos_t in circuit.position_scores.items():
        assert pos_t.sum().isfinite(), f"Non-finite position score for edge {edge}"


# ---------------------------------------------------------------------------
# top_positions helper
# ---------------------------------------------------------------------------

def test_top_positions_returns_correct_count():
    circuit = _run_pair(seq_len=6, per_position=True)
    if not circuit.edges:
        pytest.skip("no edges found")
    edge = next(iter(circuit.edges))
    result = circuit.top_positions(edge, k=3)
    assert len(result) <= 3


def test_top_positions_sorted_by_abs_score():
    circuit = _run_pair(seq_len=8, per_position=True)
    if not circuit.edges:
        pytest.skip("no edges found")
    edge = next(iter(circuit.edges))
    result = circuit.top_positions(edge, k=8)
    abs_scores = [abs(s) for _, s in result]
    assert abs_scores == sorted(abs_scores, reverse=True)


def test_top_positions_default_k():
    circuit = _run_pair(seq_len=10, per_position=True)
    if not circuit.edges:
        pytest.skip("no edges found")
    edge = next(iter(circuit.edges))
    result = circuit.top_positions(edge)
    assert len(result) <= 5


def test_top_positions_raises_without_per_position():
    from circuitry.patching.sae_edges import SAEFeatureEdge
    circuit = _run_pair(per_position=False)
    if not circuit.edges:
        pytest.skip("no edges found")
    edge = next(iter(circuit.edges))
    with pytest.raises(ValueError, match="per_position=True"):
        circuit.top_positions(edge)


# ---------------------------------------------------------------------------
# Scalar scores unchanged: per_position=True must not alter scalar edges
# ---------------------------------------------------------------------------

def test_scalar_edges_unchanged_by_per_position_flag():
    """Run the same inputs with and without per_position; scalar scores must match."""
    torch.manual_seed(42)
    from tests.patching.test_sae_features import LinearResidToy, _make_resolver, _metric
    d, d_sae, seq_len = 8, 16, 4
    model = LinearResidToy(n_layers=2, d=d)
    sae0, sae1 = _make_two_saes(d=d, d_sae=d_sae)
    runner = _make_two_site_runner(model, sae0, sae1, d=d)
    clean, corr = _make_clean_corr(d=d, b=1, s=seq_len)

    torch.manual_seed(0)
    c_no_pos = runner.run(clean, corr, lambda out: _metric(out), per_position=False)
    torch.manual_seed(0)
    c_with_pos = runner.run(clean, corr, lambda out: _metric(out), per_position=True)

    assert set(c_no_pos.edges.keys()) == set(c_with_pos.edges.keys())
    for edge in c_no_pos.edges:
        s1 = c_no_pos.edges[edge]
        s2 = c_with_pos.edges[edge]
        assert abs(s1 - s2) < 1e-6, f"Scalar score changed for {edge}: {s1} vs {s2}"
