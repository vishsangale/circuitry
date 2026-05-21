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
