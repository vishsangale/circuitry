"""Tests for v0.8.0 report-renderer changes: static/dynamic subtitle,
single-step note, HERO_DIAGS filtering, <details> collapse, grad top-K."""
from __future__ import annotations

import json
import pathlib

import pytest

from circuitry.recorder.report import build_report


def _write_metrics(run_dir: pathlib.Path, rows: list[dict]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows)
    )


def test_header_subtitle_static_for_one_step(tmp_path):
    _write_metrics(tmp_path, [
        {"kind": "scalar", "tag": "weight/effective_rank/foo",
         "step": 0, "value": 1.0},
    ])
    out = build_report(tmp_path).read_text()
    assert "# circuitry report — static (1 step)" in out, out


def test_header_subtitle_dynamic_for_multi_step(tmp_path):
    _write_metrics(tmp_path, [
        {"kind": "scalar", "tag": "weight/effective_rank/foo",
         "step": 0, "value": 1.0},
        {"kind": "scalar", "tag": "weight/effective_rank/foo",
         "step": 10, "value": 1.1},
    ])
    out = build_report(tmp_path).read_text()
    assert "# circuitry report — dynamic (2 steps)" in out, out


def test_single_step_note_in_summary(tmp_path):
    _write_metrics(tmp_path, [
        {"kind": "scalar", "tag": "weight/effective_rank/foo",
         "step": 0, "value": 1.0},
    ])
    out = build_report(tmp_path).read_text()
    assert "Single-step run" in out
    assert "Δ uniformly zero" in out


def test_multi_step_omits_single_step_note(tmp_path):
    _write_metrics(tmp_path, [
        {"kind": "scalar", "tag": "weight/effective_rank/foo",
         "step": 0, "value": 1.0},
        {"kind": "scalar", "tag": "weight/effective_rank/foo",
         "step": 10, "value": 1.1},
    ])
    out = build_report(tmp_path).read_text()
    assert "Single-step run" not in out
