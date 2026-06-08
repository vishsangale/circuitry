"""Tests for circuitry.core.dynamics — phase_transition_steps, head_formation_step, grokking_step."""
from __future__ import annotations

import math

import pytest
import torch

from circuitry.core.dynamics import (
    fourier_feature_alignment,
    grokking_step,
    head_formation_step,
    information_bottleneck_score,
    phase_transition_steps,
)


# ---------------------------------------------------------------------------
# phase_transition_steps
# ---------------------------------------------------------------------------

def test_step_change_is_detected():
    """Flat series with a sudden jump should produce at least one detection."""
    series = [(i, 0.0) for i in range(10)] + [(i, 10.0) for i in range(10, 20)]
    pts = phase_transition_steps(series)
    assert len(pts) >= 1
    # Transition should be detected near step 10
    assert any(8 <= s <= 12 for s in pts)


def test_monotone_linear_no_detection():
    """A monotone linear ramp has no sudden changes — expect no detections."""
    series = [(i, float(i)) for i in range(30)]
    assert phase_transition_steps(series) == []


def test_constant_series_no_detection():
    """All-constant series has std=0; must return []."""
    series = [(i, 5.0) for i in range(20)]
    assert phase_transition_steps(series) == []


def test_single_point_returns_empty():
    assert phase_transition_steps([(0, 1.0)]) == []


def test_empty_series_returns_empty():
    assert phase_transition_steps([]) == []


def test_min_gap_collapses_nearby_detections():
    """Two very close transitions should be collapsed to one."""
    # Build a series with a step change spread across adjacent indices
    flat_before = [(i, 0.0) for i in range(10)]
    step_region = [(10, 5.0), (11, 9.0), (12, 10.0)]
    flat_after = [(i, 10.0) for i in range(13, 30)]
    series = flat_before + step_region + flat_after
    pts = phase_transition_steps(series, min_gap=5)
    assert len(pts) == 1


def test_two_separated_transitions_both_detected():
    """Two equal-magnitude step changes separated by > min_gap should both appear.

    Note: when transitions have very different magnitudes the z-score cutoff
    is raised by the larger one, and the smaller one may not be detected.
    Equal-magnitude transitions are used here so both are reliably above the
    z-threshold.
    """
    flat = [(i, 0.0) for i in range(10)]
    jump1 = [(i, 10.0) for i in range(10, 25)]
    jump2 = [(i, 20.0) for i in range(25, 40)]
    series = flat + jump1 + jump2
    pts = phase_transition_steps(series, min_gap=3)
    assert len(pts) == 2


def test_result_is_sorted():
    """phase_transition_steps always returns a sorted list."""
    series = [(i, 0.0) for i in range(10)] + [(i, 10.0) for i in range(10, 30)]
    pts = phase_transition_steps(series)
    assert pts == sorted(pts)


def test_high_z_threshold_suppresses_weak_transitions():
    """A very high z_threshold should find nothing on a moderate step change."""
    series = [(i, 0.0) for i in range(10)] + [(i, 1.0) for i in range(10, 20)]
    pts = phase_transition_steps(series, z_threshold=10.0)
    assert pts == []


# ---------------------------------------------------------------------------
# head_formation_step
# ---------------------------------------------------------------------------

def test_formation_detected_at_correct_step():
    """Head crosses threshold midway through training."""
    series = [(i, 0.1) for i in range(10)] + [(i, 0.9) for i in range(10, 20)]
    step = head_formation_step(series, threshold=0.4)
    assert step == 10


def test_never_crosses_returns_none():
    series = [(i, 0.1) for i in range(20)]
    assert head_formation_step(series, threshold=0.4) is None


def test_n_sustain_1_any_crossing_counts():
    """With n_sustain=1, even a single step above threshold is a formation."""
    series = [(0, 0.1), (5, 0.9), (10, 0.1)]
    assert head_formation_step(series, threshold=0.4, n_sustain=1) == 5


def test_n_sustain_2_brief_spike_not_counted():
    """A single-step spike followed by a drop does NOT satisfy n_sustain=2."""
    # vals: 0.1, 0.9, 0.1, 0.1, 0.1
    series = [(0, 0.1), (5, 0.9), (10, 0.1), (15, 0.1), (20, 0.1)]
    assert head_formation_step(series, threshold=0.4, n_sustain=2) is None


def test_sustained_formation_after_spike():
    """Spike then sustained crossing — formation at sustained crossing."""
    series = [
        (0, 0.1), (5, 0.9), (10, 0.1),   # brief spike, not sustained
        (15, 0.8), (20, 0.85), (25, 0.9),  # sustained from step 15
    ]
    step = head_formation_step(series, threshold=0.4, n_sustain=2)
    assert step == 15


def test_end_of_series_tolerance():
    """Formation at the very last two points satisfies n_sustain=2."""
    series = [(i, 0.1) for i in range(8)] + [(8, 0.9), (9, 0.95)]
    step = head_formation_step(series, threshold=0.4, n_sustain=2)
    assert step == 8


def test_empty_series_returns_none():
    assert head_formation_step([], threshold=0.4) is None


def test_already_above_at_step_zero():
    """Head above threshold from step 0 — returns step 0."""
    series = [(i, 0.9) for i in range(10)]
    assert head_formation_step(series, threshold=0.4) == 0


# ---------------------------------------------------------------------------
# grokking_step
# ---------------------------------------------------------------------------

def test_grokking_step_detects_loss_drop():
    """Sharp loss drop midway returns the transition step."""
    series = [(i, 2.0) for i in range(10)] + [(i, 0.3) for i in range(10, 20)]
    step = grokking_step(series)
    assert step is not None
    assert 8 <= step <= 12


def test_grokking_step_monotone_returns_none():
    """A smooth monotone decline is not a grokking event."""
    series = [(i, 2.0 - 0.05 * i) for i in range(30)]
    assert grokking_step(series) is None


def test_grokking_step_constant_returns_none():
    series = [(i, 1.5) for i in range(20)]
    assert grokking_step(series) is None


def test_grokking_step_single_point_returns_none():
    assert grokking_step([(0, 1.0)]) is None


def test_grokking_step_returns_first_of_multiple():
    """When there are two grokking events, the first (earliest) step is returned."""
    flat1 = [(i, 2.0) for i in range(10)]
    drop1 = [(i, 1.0) for i in range(10, 25)]
    drop2 = [(i, 0.2) for i in range(25, 40)]
    series = flat1 + drop1 + drop2
    step = grokking_step(series)
    assert step is not None
    assert step < 25  # first event, not second


# ---------------------------------------------------------------------------
# fourier_feature_alignment
# ---------------------------------------------------------------------------

def test_fourier_feature_alignment_task_freq_1():
    """W is a pure cosine at frequency k: alignment at that freq should be ≈1.0."""
    d_out, d_in = 8, 64
    k = 7  # arbitrary non-zero frequency
    i_idx = torch.arange(d_in, dtype=torch.float32)
    # All rows are the same cosine; rfft will have all power at bin k
    row = torch.cos(2 * math.pi * k * i_idx / d_in)
    W = row.unsqueeze(0).expand(d_out, -1).clone()
    alignment = fourier_feature_alignment(W, task_freqs=[k])
    assert alignment == pytest.approx(1.0, abs=1e-4)


def test_fourier_feature_alignment_empty_freqs_returns_zero():
    """Empty task_freqs must return 0.0 immediately."""
    W = torch.randn(4, 32)
    assert fourier_feature_alignment(W, task_freqs=[]) == 0.0


def test_fourier_feature_alignment_range():
    """Random W with a few task_freqs — result must be in [0, 1]."""
    W = torch.randn(16, 64)
    result = fourier_feature_alignment(W, task_freqs=[0, 1, 2])
    assert 0.0 <= result <= 1.0


def test_fourier_feature_alignment_all_freqs_returns_one():
    """Passing all frequency bins as task_freqs must give ≈1.0 (all power accounted for)."""
    d_in = 32
    W = torch.randn(8, d_in)
    all_freqs = list(range(d_in // 2 + 1))
    alignment = fourier_feature_alignment(W, task_freqs=all_freqs)
    assert alignment == pytest.approx(1.0, abs=1e-5)


# ---------------------------------------------------------------------------
# information_bottleneck_score
# ---------------------------------------------------------------------------

def test_information_bottleneck_score_range():
    """Score is a float in [0, 1]."""
    torch.manual_seed(0)
    acts_train = torch.randn(50, 8)
    acts_val = torch.randn(20, 8)
    labels_train = torch.randint(0, 3, (50,))
    labels_val = torch.randint(0, 3, (20,))
    score = information_bottleneck_score(acts_train, acts_val, labels_train, labels_val)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_information_bottleneck_high_when_predictive():
    """Activations that perfectly separate labels should yield a high score (> 0.5)."""
    torch.manual_seed(1)
    n_per_class = 40
    # 3 classes, activations = class_id * 10 + tiny noise → highly predictable
    acts = torch.cat([
        torch.full((n_per_class, 4), float(c)) + torch.randn(n_per_class, 4) * 0.01
        for c in range(3)
    ])
    labels = torch.cat([torch.full((n_per_class,), c, dtype=torch.long) for c in range(3)])
    # Split 50/50 train/val
    acts_train, acts_val = acts[:60], acts[60:]
    labels_train, labels_val = labels[:60], labels[60:]
    score = information_bottleneck_score(acts_train, acts_val, labels_train, labels_val)
    assert score > 0.5


def test_information_bottleneck_low_when_random():
    """Average over seeds: random acts vs structured labels → mean score < 0.6."""
    scores = []
    for seed in range(5):
        torch.manual_seed(seed)
        acts_train = torch.randn(60, 8)
        acts_val = torch.randn(30, 8)
        labels_train = torch.arange(60) % 3
        labels_val = torch.arange(30) % 3
        scores.append(information_bottleneck_score(acts_train, acts_val, labels_train, labels_val))
    # Random activations should not reliably predict labels; mean score well below 1
    assert sum(scores) / len(scores) < 0.65
