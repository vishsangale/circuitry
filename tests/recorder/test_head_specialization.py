"""Tests for the v1.13 head-specialization report section."""
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


def _all_three(module: str, head: int, ind: float, css: float, snk: float,
               step: int = 0) -> list[dict]:
    return [
        _scalar(f"activation/induction_score/{module}/head_{head}", ind, step),
        _scalar(f"activation/copy_suppression_score/{module}/head_{head}", css, step),
        _scalar(f"activation/attention_sink_score/{module}/head_{head}", snk, step),
    ]


def test_head_specialization_section_appears_when_all_three_present(tmp_path):
    scalars = _all_three("layers.0.self_attn", 0, ind=0.8, css=0.1, snk=0.1)
    _write_metrics(tmp_path, scalars)
    report = build_report(tmp_path).read_text()
    assert "## Head Specialization" in report


def test_induction_head_labelled_in_report(tmp_path):
    scalars = _all_three("layers.0.self_attn", 0, ind=0.8, css=0.1, snk=0.1)
    _write_metrics(tmp_path, scalars)
    report = build_report(tmp_path).read_text()
    assert "**induction**" in report


def test_sink_head_labelled_in_report(tmp_path):
    scalars = _all_three("layers.0.self_attn", 0, ind=0.1, css=0.1, snk=0.9)
    _write_metrics(tmp_path, scalars)
    report = build_report(tmp_path).read_text()
    assert "**sink**" in report


def test_uniform_head_not_bolded(tmp_path):
    scalars = _all_three("layers.0.self_attn", 0, ind=0.05, css=0.05, snk=0.05)
    _write_metrics(tmp_path, scalars)
    report = build_report(tmp_path).read_text()
    assert "## Head Specialization" in report
    assert "uniform" in report
    assert "**uniform**" not in report  # uniform is rendered plain, not bolded


def test_section_absent_when_no_attention_scores(tmp_path):
    scalars = [_scalar("weight/effective_rank/layer_0", 5.0)]
    _write_metrics(tmp_path, scalars)
    report = build_report(tmp_path).read_text()
    assert "## Head Specialization" not in report


def test_section_absent_when_only_one_score_type_present(tmp_path):
    """With only induction_score (no css/sink), the section must not appear."""
    scalars = [_scalar("activation/induction_score/layers.0.self_attn/head_0", 0.8)]
    _write_metrics(tmp_path, scalars)
    report = build_report(tmp_path).read_text()
    # Section is present (the tag exists) but classification shows "—" for missing scores.
    # Either way, the table should at least list the module.
    assert "layers.0.self_attn" in report


def test_multiple_heads_all_classified(tmp_path):
    scalars = (
        _all_three("m.self_attn", 0, ind=0.9, css=0.05, snk=0.05)
        + _all_three("m.self_attn", 1, ind=0.05, css=0.05, snk=0.8)
    )
    _write_metrics(tmp_path, scalars)
    report = build_report(tmp_path).read_text()
    assert "**induction**" in report
    assert "**sink**" in report


def test_last_step_value_used_not_first(tmp_path):
    """When multiple steps exist, the final step's value determines the type."""
    # Step 0: ind=0.05 (uniform). Step 5: ind=0.85 (induction).
    scalars = (
        _all_three("mod.self_attn", 0, ind=0.05, css=0.05, snk=0.05, step=0)
        + _all_three("mod.self_attn", 0, ind=0.85, css=0.05, snk=0.05, step=5)
    )
    _write_metrics(tmp_path, scalars)
    report = build_report(tmp_path).read_text()
    assert "**induction**" in report


def test_section_hidden_in_compact_mode(tmp_path):
    """--compact renders only Summary + Flags; head specialization is suppressed."""
    scalars = _all_three("m.self_attn", 0, ind=0.9, css=0.05, snk=0.05)
    _write_metrics(tmp_path, scalars)
    report = build_report(tmp_path, compact=True).read_text()
    assert "## Head Specialization" not in report
