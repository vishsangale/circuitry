"""Tests for the v1.14 ## Training Dynamics report section."""
from __future__ import annotations

import json
import pathlib

import pytest

from circuitry.recorder.report import build_report


def _write_metrics(tmp_path: pathlib.Path, scalars: list[dict]) -> pathlib.Path:
    p = tmp_path / "metrics.jsonl"
    p.write_text("\n".join(json.dumps(s) for s in scalars))
    return tmp_path


def _scalar(tag: str, value: float, step: int = 0) -> dict:
    return {"kind": "scalar", "tag": tag, "value": value, "step": step}


def _ind(module: str, head: int, value: float, step: int) -> dict:
    return _scalar(f"activation/induction_score/{module}/head_{head}", value, step)


def _rank(module: str, value: float, step: int) -> dict:
    return _scalar(f"weight/effective_rank/{module}", value, step)


# ---------------------------------------------------------------------------
# Head Formation Events
# ---------------------------------------------------------------------------

def test_section_appears_when_head_forms_during_training(tmp_path):
    """Head below threshold at step 0, above threshold at steps 1-3."""
    scalars = (
        [_ind("m.self_attn", 0, 0.1, step=0)]
        + [_ind("m.self_attn", 0, 0.9, step=s) for s in range(1, 4)]
    )
    _write_metrics(tmp_path, scalars)
    report = build_report(tmp_path).read_text()
    assert "## Training Dynamics" in report
    assert "Head Formation Events" in report


def test_formation_step_value_shown_in_report(tmp_path):
    """The exact step at which formation occurred should appear in the table."""
    scalars = [
        _ind("m.self_attn", 0, 0.05, step=0),
        _ind("m.self_attn", 0, 0.05, step=50),
        _ind("m.self_attn", 0, 0.9, step=100),
        _ind("m.self_attn", 0, 0.85, step=150),
    ]
    _write_metrics(tmp_path, scalars)
    report = build_report(tmp_path).read_text()
    assert "100" in report   # formation step


def test_section_absent_when_head_pre_formed(tmp_path):
    """Head already above threshold at step 0 — no formation event to report."""
    scalars = [_ind("m.self_attn", 0, 0.9, step=s) for s in range(5)]
    _write_metrics(tmp_path, scalars)
    report = build_report(tmp_path).read_text()
    assert "## Training Dynamics" not in report


def test_section_absent_when_head_never_crosses(tmp_path):
    """Head stays below threshold throughout — no formation event."""
    scalars = [_ind("m.self_attn", 0, 0.1, step=s) for s in range(5)]
    _write_metrics(tmp_path, scalars)
    report = build_report(tmp_path).read_text()
    assert "## Training Dynamics" not in report


def test_section_absent_with_single_step(tmp_path):
    """Single-step run has no dynamics — section must not appear."""
    scalars = [_ind("m.self_attn", 0, 0.9, step=0)]
    _write_metrics(tmp_path, scalars)
    report = build_report(tmp_path).read_text()
    assert "## Training Dynamics" not in report


def test_section_absent_in_compact_mode(tmp_path):
    """compact=True suppresses Training Dynamics."""
    scalars = (
        [_ind("m.self_attn", 0, 0.1, step=0)]
        + [_ind("m.self_attn", 0, 0.9, step=s) for s in range(1, 4)]
    )
    _write_metrics(tmp_path, scalars)
    report = build_report(tmp_path, compact=True).read_text()
    assert "## Training Dynamics" not in report


def test_multiple_heads_formation_all_listed(tmp_path):
    """Two heads that form at different steps both appear in the table."""
    # Head 0 forms at step 10; head 1 forms at step 20
    scalars = (
        [_ind("m.self_attn", 0, 0.05, step=s) for s in range(0, 10)]
        + [_ind("m.self_attn", 0, 0.9, step=s) for s in range(10, 25)]
        + [_ind("m.self_attn", 1, 0.05, step=s) for s in range(0, 20)]
        + [_ind("m.self_attn", 1, 0.9, step=s) for s in range(20, 30)]
    )
    _write_metrics(tmp_path, scalars)
    report = build_report(tmp_path).read_text()
    assert "head_0" in report
    assert "head_1" in report


# ---------------------------------------------------------------------------
# Phase Transitions
# ---------------------------------------------------------------------------

def test_phase_transition_detected_for_rank_collapse(tmp_path):
    """Sharp rank drop mid-training should surface as a Phase Transition."""
    scalars = (
        [_rank("layer_0", 15.0, step=s) for s in range(0, 10)]
        + [_rank("layer_0", 3.0, step=s) for s in range(10, 20)]
    )
    _write_metrics(tmp_path, scalars)
    report = build_report(tmp_path).read_text()
    assert "Phase Transitions" in report


def test_phase_transition_absent_for_gradual_rank_decline(tmp_path):
    """A smooth monotone rank decline should not trigger a phase transition."""
    scalars = [_rank("layer_0", 15.0 - float(s), step=s) for s in range(20)]
    _write_metrics(tmp_path, scalars)
    report = build_report(tmp_path).read_text()
    # Phase Transitions section should not appear for a gradual decline
    assert "Phase Transitions" not in report


def test_section_absent_when_no_tracked_metrics(tmp_path):
    """No attention scores and no rank metrics — section absent."""
    scalars = [_scalar("train/loss", 1.0 - 0.01 * s, step=s) for s in range(10)]
    _write_metrics(tmp_path, scalars)
    report = build_report(tmp_path).read_text()
    assert "## Training Dynamics" not in report
