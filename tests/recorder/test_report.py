from __future__ import annotations

import json
import pathlib

from circuitry.recorder.report import build_report


def _write_jsonl(path: pathlib.Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows))


def test_build_report_writes_markdown_with_sections(tmp_path):
    _write_jsonl(tmp_path / "metrics.jsonl", [
        {"tag": "loss", "value": 1.5, "step": 0, "kind": "scalar"},
        {"tag": "loss", "value": 1.0, "step": 1, "kind": "scalar"},
        {"tag": "weight/effective_rank/0", "value": 8.0, "step": 0, "kind": "scalar"},
        {"tag": "weight/effective_rank/0", "value": 7.5, "step": 1, "kind": "scalar"},
    ])
    (tmp_path / "circuitry" / "matched_modules.txt").parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / "circuitry" / "matched_modules.txt").write_text("# hook_point[0]\n0\n2\n")

    out = tmp_path / "inspect" / "report.md"
    build_report(run_dir=tmp_path, out_path=out)

    md = out.read_text()
    assert "# circuitry report" in md
    assert "matched modules" in md.lower()
    assert "loss" in md
    assert "effective_rank" in md


def test_build_report_handles_missing_jsonl(tmp_path):
    out = tmp_path / "report.md"
    build_report(run_dir=tmp_path, out_path=out)
    assert out.exists()
    assert "no metrics found" in out.read_text().lower()


def test_build_report_sections_use_first_two_tag_segments(tmp_path):
    """3+ segment tags should group under `family/diagnostic`, not just
    `family`. Distinct diagnostics within the same family get distinct
    sections."""
    _write_jsonl(tmp_path / "metrics.jsonl", [
        {"tag": "weight/effective_rank/A", "value": 1.0, "step": 0, "kind": "scalar"},
        {"tag": "weight/stable_rank/A", "value": 2.0, "step": 0, "kind": "scalar"},
        {"tag": "weight/effective_rank/B", "value": 3.0, "step": 0, "kind": "scalar"},
    ])
    out = tmp_path / "inspect" / "report.md"
    build_report(run_dir=tmp_path, out_path=out)
    md = out.read_text()
    assert "## weight/effective_rank" in md
    assert "## weight/stable_rank" in md
    # Row identifier under each section is the tag tail, not the full tag.
    er_section = md.split("## weight/effective_rank")[1].split("##")[0]
    assert "`A`" in er_section and "`B`" in er_section
    assert "effective_rank" not in er_section.replace(
        "## weight/effective_rank", ""
    ).split("|")[0]


def test_build_report_includes_delta_column_and_moves_dynamic_first(tmp_path):
    """A row whose value changes across steps (`min != max`) should sort
    before static rows in its section, and the Δ column should reflect the
    spread."""
    _write_jsonl(tmp_path / "metrics.jsonl", [
        # Static — same value at both steps.
        {"tag": "weight/effective_rank/static", "value": 5.0, "step": 0, "kind": "scalar"},
        {"tag": "weight/effective_rank/static", "value": 5.0, "step": 1, "kind": "scalar"},
        # Moving — value changes.
        {"tag": "weight/effective_rank/moving", "value": 1.0, "step": 0, "kind": "scalar"},
        {"tag": "weight/effective_rank/moving", "value": 4.0, "step": 1, "kind": "scalar"},
    ])
    out = tmp_path / "inspect" / "report.md"
    build_report(run_dir=tmp_path, out_path=out)
    md = out.read_text()
    assert "| Δ |" in md
    # `moving` row appears before `static` row in the table.
    section = md.split("## weight/effective_rank")[1]
    moving_pos = section.find("`moving`")
    static_pos = section.find("`static`")
    assert 0 <= moving_pos < static_pos, (
        f"moving@{moving_pos} should precede static@{static_pos}"
    )
    # Static row's Δ cell renders as `—`; moving row's as a number.
    moving_row = next(line for line in section.splitlines() if "`moving`" in line)
    static_row = next(line for line in section.splitlines() if "`static`" in line)
    assert "—" not in moving_row.split("|")[-2]
    assert "—" in static_row.split("|")[-2]


def test_build_report_attach_summary_table_renders(tmp_path):
    """## Attach summary table is rendered after ## Summary when attach_summary.json exists."""
    import json as _json

    _write_jsonl(tmp_path / "metrics.jsonl", [
        {"tag": "weight/effective_rank/0", "value": 8.0, "step": 0, "kind": "scalar"},
    ])
    attach_summary = {
        "hook_points": [
            {"idx": 0, "source": "weight", "label": r"^\d+$",
             "matched": 2, "resolved": 2, "unresolved": 0},
        ],
        "totals": {"matched": 2, "resolved": 2, "unresolved": 0},
    }
    (tmp_path / "circuitry").mkdir(parents=True, exist_ok=True)
    (tmp_path / "circuitry" / "attach_summary.json").write_text(
        _json.dumps(attach_summary)
    )
    out = tmp_path / "inspect" / "report.md"
    build_report(run_dir=tmp_path, out_path=out)
    md = out.read_text()

    assert "## Attach summary" in md
    # Table header columns present
    assert "| hp | source | target | matched | resolved | unresolved |" in md
    # Data row with idx=0
    assert "| 0 | weight |" in md
    # Totals row
    assert "**total**" in md
    assert "| 2 | 2 | 0 |" in md


def test_build_report_no_attach_summary_section_when_file_absent(tmp_path):
    """## Attach summary is silently skipped when attach_summary.json is absent."""
    _write_jsonl(tmp_path / "metrics.jsonl", [
        {"tag": "weight/effective_rank/0", "value": 8.0, "step": 0, "kind": "scalar"},
    ])
    out = tmp_path / "inspect" / "report.md"
    build_report(run_dir=tmp_path, out_path=out)
    md = out.read_text()
    assert "Attach summary" not in md


def test_build_report_summary_block_counts_moving_and_static(tmp_path):
    _write_jsonl(tmp_path / "metrics.jsonl", [
        {"tag": "a/b/c", "value": 1.0, "step": 0, "kind": "scalar"},
        {"tag": "a/b/c", "value": 2.0, "step": 1, "kind": "scalar"},  # moves
        {"tag": "a/b/d", "value": 1.0, "step": 0, "kind": "scalar"},
        {"tag": "a/b/d", "value": 1.0, "step": 1, "kind": "scalar"},  # static
    ])
    out = tmp_path / "inspect" / "report.md"
    build_report(run_dir=tmp_path, out_path=out)
    md = out.read_text()
    assert "## Summary" in md
    assert "**2** scalar tags" in md
    assert "**1** moving" in md
    assert "**1** static" in md
    assert "**2** emit step" in md


def test_delta_column_is_signed_not_unsigned_range(tmp_path):
    """v1.10 polish: a monotonically *decreasing* metric must render a negative
    Δ in the table, not the positive unsigned range (vmax - vmin) it showed
    through v1.9 (which read like an increase)."""
    _write_jsonl(tmp_path / "metrics.jsonl", [
        {"tag": "weight/effective_rank/0", "value": 15.0, "step": 0, "kind": "scalar"},
        {"tag": "weight/effective_rank/0", "value": 5.0, "step": 1, "kind": "scalar"},
    ])
    out = tmp_path / "inspect" / "report.md"
    build_report(run_dir=tmp_path, out_path=out)
    md = out.read_text()
    # The row's Δ cell is last - first = 5 - 15 = -10, NOT the +10 range.
    assert "-10" in md
    assert "| +10 |" not in md and "| 10 |" not in md


def test_delta_column_signed_positive_for_rising_metric(tmp_path):
    _write_jsonl(tmp_path / "metrics.jsonl", [
        {"tag": "activation/dead_fraction/0", "value": 0.1, "step": 0, "kind": "scalar"},
        {"tag": "activation/dead_fraction/0", "value": 0.3, "step": 1, "kind": "scalar"},
    ])
    out = tmp_path / "inspect" / "report.md"
    build_report(run_dir=tmp_path, out_path=out)
    md = out.read_text()
    assert "+0.2" in md


def test_summary_shows_family_tag_counts(tmp_path):
    """## Summary should include a per-top-level-family tag count line."""
    _write_jsonl(tmp_path / "metrics.jsonl", [
        {"tag": "weight/effective_rank/0", "value": 5.0, "step": 0, "kind": "scalar"},
        {"tag": "weight/stable_rank/0", "value": 3.0, "step": 0, "kind": "scalar"},
        {"tag": "activation/dead_fraction/0", "value": 0.1, "step": 0, "kind": "scalar"},
    ])
    out = tmp_path / "inspect" / "report.md"
    build_report(run_dir=tmp_path, out_path=out)
    md = out.read_text()
    assert "Tags by family" in md
    assert "**weight**: 2" in md
    assert "**activation**: 1" in md


def test_grokking_signals_appear_in_training_dynamics(tmp_path):
    """A sharp loss drop should surface in the Grokking Signals sub-table."""
    rows = (
        [{"tag": "train/loss", "value": 2.0, "step": s, "kind": "scalar"}
         for s in range(10)]
        + [{"tag": "train/loss", "value": 0.3, "step": s, "kind": "scalar"}
           for s in range(10, 20)]
    )
    _write_jsonl(tmp_path / "metrics.jsonl", rows)
    out = tmp_path / "inspect" / "report.md"
    build_report(run_dir=tmp_path, out_path=out)
    md = out.read_text()
    assert "Grokking Signals" in md
    assert "loss" in md


def test_grokking_signals_absent_for_monotone_loss(tmp_path):
    """A smoothly declining loss must not produce a Grokking Signals entry."""
    rows = [
        {"tag": "train/loss", "value": 2.0 - 0.05 * s, "step": s, "kind": "scalar"}
        for s in range(30)
    ]
    _write_jsonl(tmp_path / "metrics.jsonl", rows)
    out = tmp_path / "inspect" / "report.md"
    build_report(run_dir=tmp_path, out_path=out)
    md = out.read_text()
    assert "Grokking Signals" not in md
