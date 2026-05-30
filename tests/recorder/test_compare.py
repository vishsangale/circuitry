"""Tests for recorder/compare.py — FamilyDelta, compare_runs, build_compare_report.

JSONL fixture schema (as written by JsonlWriter):
  {"kind": "scalar", "tag": "<tag>", "step": <int>, "value": <float>}
"""
from __future__ import annotations

import json
import math
import pathlib

import pytest

from circuitry.recorder.compare import build_compare_report, compare_runs

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_jsonl(path: pathlib.Path, rows: list[dict]) -> None:
    """Write JSONL rows to *path*, creating parent dirs as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows))


def _make_run(
    run_dir: pathlib.Path,
    tag: str,
    steps: list[tuple[int, float]],
) -> pathlib.Path:
    """Write a metrics.jsonl with a single tag series into *run_dir*."""
    rows = [
        {"kind": "scalar", "tag": tag, "step": step, "value": value}
        for step, value in steps
    ]
    _write_jsonl(run_dir / "metrics.jsonl", rows)
    return run_dir


# ---------------------------------------------------------------------------
# T4 tests
# ---------------------------------------------------------------------------


def test_compare_single_family_delta(tmp_path: pathlib.Path) -> None:
    """delta == last_b - last_a for a single-family comparison."""
    tag = "activation/dead_fraction/mod"
    run_a = _make_run(tmp_path / "a", tag, [(0, 0.10), (1, 0.12)])
    run_b = _make_run(tmp_path / "b", tag, [(0, 0.10), (1, 0.20)])

    deltas = compare_runs(run_a, run_b)

    assert len(deltas) == 1
    fd = deltas[0]
    assert fd.section == "activation/dead_fraction"
    assert fd.diagnostic == "dead_fraction"
    assert abs(fd.last_a - 0.12) < 1e-6
    assert abs(fd.last_b - 0.20) < 1e-6
    assert abs(fd.delta - (0.20 - 0.12)) < 1e-6


def test_compare_missing_family_in_one_run(tmp_path: pathlib.Path) -> None:
    """A family present only in run_a produces NaN last_b in the result."""
    tag = "weight/effective_rank/mod"
    run_a = _make_run(tmp_path / "a", tag, [(0, 10.0)])
    # run_b has a metrics.jsonl but only a DIFFERENT family (not weight/effective_rank)
    # so run_b is non-empty (passes empty-run guard) but missing the target family.
    _write_jsonl(
        tmp_path / "b" / "metrics.jsonl",
        [{"kind": "scalar", "tag": "activation/dead_fraction/mod", "step": 0, "value": 0.5}],
    )
    run_b = tmp_path / "b"

    deltas = compare_runs(run_a, run_b)

    # Two sections: weight/effective_rank (from a) + activation/dead_fraction (from b)
    fd_map = {fd.section: fd for fd in deltas}
    assert "weight/effective_rank" in fd_map
    fd = fd_map["weight/effective_rank"]
    assert not math.isnan(fd.last_a)
    assert math.isnan(fd.last_b)
    assert math.isnan(fd.delta)
    assert fd.trend_b == "flat"  # sentinel for absent side


def test_compare_trend_agreement(tmp_path: pathlib.Path) -> None:
    """Both runs rising → trend_agrees=True."""
    tag = "weight/effective_rank/m"
    run_a = _make_run(tmp_path / "a", tag, [(0, 1.0), (1, 2.0)])
    run_b = _make_run(tmp_path / "b", tag, [(0, 3.0), (1, 4.0)])

    deltas = compare_runs(run_a, run_b)

    assert len(deltas) == 1
    fd = deltas[0]
    assert fd.trend_a == "up"
    assert fd.trend_b == "up"
    assert fd.trend_agrees is True


def test_compare_trend_disagree(tmp_path: pathlib.Path) -> None:
    """run_a rising, run_b falling → trend_agrees=False."""
    tag = "weight/effective_rank/m"
    run_a = _make_run(tmp_path / "a", tag, [(0, 1.0), (1, 2.0)])
    run_b = _make_run(tmp_path / "b", tag, [(0, 4.0), (1, 3.0)])

    deltas = compare_runs(run_a, run_b)

    assert len(deltas) == 1
    fd = deltas[0]
    assert fd.trend_a == "up"
    assert fd.trend_b == "down"
    assert fd.trend_agrees is False


def test_build_compare_report_writes_markdown(tmp_path: pathlib.Path) -> None:
    """Output file contains the required header and both run paths."""
    tag = "weight/effective_rank/m"
    run_a = _make_run(tmp_path / "a", tag, [(0, 8.0)])
    run_b = _make_run(tmp_path / "b", tag, [(0, 9.0)])
    out = tmp_path / "compare.md"

    build_compare_report(run_a, run_b, out_path=out)

    md = out.read_text()
    assert "# circuitry compare" in md
    assert str(run_a) in md
    assert str(run_b) in md
    assert "| family/diagnostic" in md


def test_cli_compare_subcommand(tmp_path: pathlib.Path) -> None:
    """CLI positional args: circuitry compare <run_a> <run_b> --out <path>."""
    tag = "activation/dead_fraction/m"
    run_a = _make_run(tmp_path / "a", tag, [(0, 0.05)])
    run_b = _make_run(tmp_path / "b", tag, [(0, 0.07)])
    out = tmp_path / "compare.md"

    from circuitry.cli.main import main

    rc = main(["compare", str(run_a), str(run_b), "--out", str(out)])
    assert rc == 0
    assert out.exists()
    assert "# circuitry compare" in out.read_text()


def test_compare_missing_metrics_raises(tmp_path: pathlib.Path) -> None:
    """FileNotFoundError when a run directory has no metrics.jsonl."""
    tag = "weight/effective_rank/m"
    run_a = _make_run(tmp_path / "a", tag, [(0, 5.0)])
    run_b = tmp_path / "b"
    run_b.mkdir()  # directory exists but metrics.jsonl is absent

    with pytest.raises(FileNotFoundError, match=str(run_b)):
        compare_runs(run_a, run_b)


def test_compare_result_sorted_by_section(tmp_path: pathlib.Path) -> None:
    """Results are sorted deterministically by section name."""
    for rdir in (tmp_path / "a", tmp_path / "b"):
        _write_jsonl(
            rdir / "metrics.jsonl",
            [
                {"kind": "scalar", "tag": "weight/effective_rank/m", "step": 0, "value": 1.0},
                {"kind": "scalar", "tag": "activation/dead_fraction/m", "step": 0, "value": 0.1},
            ],
        )

    deltas = compare_runs(tmp_path / "a", tmp_path / "b")
    sections = [fd.section for fd in deltas]
    assert sections == sorted(sections)


def test_compare_empty_metrics_raises(tmp_path: pathlib.Path) -> None:
    """ValueError when a run's metrics.jsonl exists but has no scalar rows."""
    tag = "weight/effective_rank/m"
    run_a = _make_run(tmp_path / "a", tag, [(0, 5.0)])
    run_b = tmp_path / "b"
    run_b.mkdir(parents=True, exist_ok=True)
    (run_b / "metrics.jsonl").write_text("")  # file exists but is empty

    with pytest.raises(ValueError, match="no scalar metrics"):
        compare_runs(run_a, run_b)


def test_compare_missing_family_trend_disagrees(tmp_path: pathlib.Path) -> None:
    """A family present only in one run must have trend_agrees=False."""
    # run_a has a flat trend for weight/effective_rank (single point → flat)
    run_a = _make_run(
        tmp_path / "a", "weight/effective_rank/m", [(0, 5.0)]
    )
    # run_b has a different family entirely, so weight/effective_rank is absent
    _write_jsonl(
        tmp_path / "b" / "metrics.jsonl",
        [{"kind": "scalar", "tag": "activation/dead_fraction/m", "step": 0, "value": 0.1}],
    )
    run_b = tmp_path / "b"

    deltas = compare_runs(run_a, run_b)

    fd_map = {fd.section: fd for fd in deltas}
    fd = fd_map["weight/effective_rank"]
    # Absent side has NaN last_b → cannot agree, must be False even if trend_a is "flat"
    assert fd.trend_agrees is False


def test_load_rows_malformed_raises(tmp_path: pathlib.Path) -> None:
    """ValueError with 'malformed' message when metrics.jsonl has a bad line."""
    from circuitry.recorder._metrics import load_rows

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "metrics.jsonl").write_text(
        '{"kind": "scalar", "tag": "weight/effective_rank/m", "step": 0, "value": 1.0}\n'
        "this is not valid json\n"
    )

    with pytest.raises(ValueError, match="malformed"):
        load_rows(run_dir)
