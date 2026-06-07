"""Tests for circuitry.core.dynamics — phase_transition_steps and head_formation_step."""
from __future__ import annotations

import pytest

from circuitry.core.dynamics import head_formation_step, phase_transition_steps


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
