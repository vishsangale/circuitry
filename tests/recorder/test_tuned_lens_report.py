"""Report integration for the v1.10 tuned-lens diagnostic: hero section + flag."""
from __future__ import annotations

import json
import pathlib

from circuitry.recorder.report import HERO_SECTIONS, build_report


def _write(path: pathlib.Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows))


def test_tuned_lens_kl_is_a_hero_section():
    assert "activation/tuned_lens_kl" in HERO_SECTIONS


def test_tuned_lens_not_forming_flag_fires_when_kl_stays_high(tmp_path):
    _write(
        tmp_path / "metrics.jsonl",
        [
            {"tag": "activation/tuned_lens_kl/layers.0", "value": 2.5,
             "step": 0, "kind": "scalar"},
            {"tag": "activation/tuned_lens_kl/layers.0", "value": 2.6,
             "step": 1, "kind": "scalar"},
        ],
    )
    out = tmp_path / "report.md"
    build_report(tmp_path, out)
    md = out.read_text()
    assert "## Flags" in md
    assert "tuned_lens_not_forming" in md


def test_tuned_lens_flag_quiet_when_kl_low(tmp_path):
    _write(
        tmp_path / "metrics.jsonl",
        [
            {"tag": "activation/tuned_lens_kl/layers.0", "value": 0.05,
             "step": 0, "kind": "scalar"},
            {"tag": "activation/tuned_lens_kl/layers.0", "value": 0.02,
             "step": 1, "kind": "scalar"},
        ],
    )
    out = tmp_path / "report.md"
    build_report(tmp_path, out)
    md = out.read_text()
    assert "tuned_lens_not_forming" not in md
