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


def test_rank_collapse_trend_flag(tmp_path):
    """Declining rank_trajectory below threshold fires rank_collapse_trend."""
    _write(tmp_path / "metrics.jsonl", [
        {"tag": "weight/rank_trajectory/mod", "value": 12.0, "step": 0, "kind": "scalar"},
        {"tag": "weight/rank_trajectory/mod", "value": 6.0,  "step": 1, "kind": "scalar"},
    ])
    out = tmp_path / "report.md"
    build_report(tmp_path, out)
    md = out.read_text()
    assert "## Flags" in md
    assert "rank_collapse_trend" in md


def test_update_delta_vanishing_flag(tmp_path):
    """Near-zero RELATIVE update_delta fires update_delta_vanishing (v1.10:
    the flag keys on the scale-invariant ||ΔW||/||W|| companion)."""
    _write(tmp_path / "metrics.jsonl", [
        {"tag": "weight/update_delta_rel/mod", "value": 1e-7, "step": 0, "kind": "scalar"},
        {"tag": "weight/update_delta_rel/mod", "value": 5e-8, "step": 1, "kind": "scalar"},
    ])
    out = tmp_path / "report.md"
    build_report(tmp_path, out)
    md = out.read_text()
    assert "update_delta_vanishing" in md


def test_healthy_relative_step_keeps_flags_empty(tmp_path):
    """A large-matrix healthy step (large ABSOLUTE ||ΔW|| but small relative)
    must NOT fire the flag — the whole point of the v1.10 scale-invariant fix."""
    _write(tmp_path / "metrics.jsonl", [
        # Absolute delta is large, but the relative companion shows a healthy
        # step well above the vanishing threshold.
        {"tag": "weight/update_delta/mod", "value": 4.0, "step": 0, "kind": "scalar"},
        {"tag": "weight/update_delta/mod", "value": 4.2, "step": 1, "kind": "scalar"},
        {"tag": "weight/update_delta_rel/mod", "value": 0.02, "step": 0, "kind": "scalar"},
        {"tag": "weight/update_delta_rel/mod", "value": 0.02, "step": 1, "kind": "scalar"},
    ])
    out = tmp_path / "report.md"
    build_report(tmp_path, out)
    # Assert precisely on the Flags block (the tmp path contains test names, so
    # a bare substring check would false-positive).
    assert "| — | — | no flags |" in out.read_text()


def test_direction_reversal_flag(tmp_path):
    """Strongly negative direction_cosine fires direction_reversal."""
    _write(tmp_path / "metrics.jsonl", [
        {"tag": "weight/direction_cosine/mod", "value": 0.1,  "step": 0, "kind": "scalar"},
        {"tag": "weight/direction_cosine/mod", "value": -0.8, "step": 1, "kind": "scalar"},
    ])
    out = tmp_path / "report.md"
    build_report(tmp_path, out)
    md = out.read_text()
    assert "direction_reversal" in md


def test_dynamics_tags_in_hero_sections(tmp_path):
    """weight/update_delta and rank_trajectory render in hero (not in <details>)."""
    _write(tmp_path / "metrics.jsonl", [
        {"tag": "weight/update_delta/mod",    "value": 0.1, "step": 0, "kind": "scalar"},
        {"tag": "weight/rank_trajectory/mod", "value": 8.0, "step": 0, "kind": "scalar"},
    ])
    out = tmp_path / "report.md"
    build_report(tmp_path, out)
    md = out.read_text()
    # Hero sections appear before <details>; advanced sections inside <details>.
    assert "## weight/update_delta" in md
    assert "## weight/rank_trajectory" in md
    details_start = md.find("<details>")
    update_delta_start = md.find("## weight/update_delta")
    rank_traj_start = md.find("## weight/rank_trajectory")
    # Sections should appear before <details> (or <details> absent entirely)
    if details_start >= 0:
        assert update_delta_start < details_start
        assert rank_traj_start < details_start
