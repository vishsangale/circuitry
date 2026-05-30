"""Tests for the ## Flags verdict block and compact mode in build_report (T3)."""
from __future__ import annotations

import json
import pathlib

from circuitry.recorder.report import build_report


def _write(path: pathlib.Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows))


def test_flags_suppressed_single_step(tmp_path):
    """step_count == 1 → ## Flags block must NOT appear."""
    _write(
        tmp_path / "metrics.jsonl",
        [
            {
                "tag": "activation/dead_fraction/mod",
                "value": 0.2,
                "step": 0,
                "kind": "scalar",
            },
        ],
    )
    out = tmp_path / "report.md"
    build_report(tmp_path, out)
    assert "## Flags" not in out.read_text()


def test_flags_fires_dead_fraction(tmp_path):
    """Rising dead_fraction (0.01 → 0.15) over 2 steps → dead_rising flag fires."""
    _write(
        tmp_path / "metrics.jsonl",
        [
            {
                "tag": "activation/dead_fraction/mod",
                "value": 0.01,
                "step": 0,
                "kind": "scalar",
            },
            {
                "tag": "activation/dead_fraction/mod",
                "value": 0.15,
                "step": 1,
                "kind": "scalar",
            },
        ],
    )
    out = tmp_path / "report.md"
    build_report(tmp_path, out)
    md = out.read_text()
    assert "## Flags" in md
    assert "dead_rising" in md


def test_no_false_flags_flat(tmp_path):
    """Flat series (same value at both steps) → ## Flags present but no dead_rising."""
    _write(
        tmp_path / "metrics.jsonl",
        [
            {
                "tag": "activation/dead_fraction/mod",
                "value": 0.10,
                "step": 0,
                "kind": "scalar",
            },
            {
                "tag": "activation/dead_fraction/mod",
                "value": 0.10,
                "step": 1,
                "kind": "scalar",
            },
        ],
    )
    out = tmp_path / "report.md"
    build_report(tmp_path, out)
    md = out.read_text()
    assert "## Flags" in md
    assert "dead_rising" not in md
    assert "no flags" in md.lower()


def test_compact_omits_tables(tmp_path):
    """compact=True → ## Summary present, per-tag section tables absent."""
    _write(
        tmp_path / "metrics.jsonl",
        [
            {
                "tag": "weight/effective_rank/mod",
                "value": 8.0,
                "step": 0,
                "kind": "scalar",
            },
            {
                "tag": "weight/effective_rank/mod",
                "value": 7.0,
                "step": 1,
                "kind": "scalar",
            },
        ],
    )
    out = tmp_path / "report.md"
    build_report(tmp_path, out, compact=True)
    md = out.read_text()
    assert "## Summary" in md
    assert "## weight/effective_rank" not in md


def test_compact_includes_flags(tmp_path):
    """compact=True with rising dead_fraction → ## Flags block is included."""
    _write(
        tmp_path / "metrics.jsonl",
        [
            {
                "tag": "activation/dead_fraction/mod",
                "value": 0.01,
                "step": 0,
                "kind": "scalar",
            },
            {
                "tag": "activation/dead_fraction/mod",
                "value": 0.20,
                "step": 1,
                "kind": "scalar",
            },
        ],
    )
    out = tmp_path / "report.md"
    build_report(tmp_path, out, compact=True)
    md = out.read_text()
    assert "## Summary" in md
    assert "## Flags" in md
    assert "dead_rising" in md
