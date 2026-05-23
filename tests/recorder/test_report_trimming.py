"""Tests for v0.8.0 report-renderer changes: static/dynamic subtitle,
single-step note, HERO_DIAGS filtering, <details> collapse, grad top-K."""
from __future__ import annotations

import json
import pathlib

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


HERO_DIAGS = {
    "weight/effective_rank",
    "weight/attention_head_rank",
    "activation/dead_fraction",
    "activation/gate_stats",
}


def _scalar(tag: str, step: int, value: float) -> dict:
    return {"kind": "scalar", "tag": tag, "step": step, "value": value}


def test_hero_diags_render_inline(tmp_path):
    _write_metrics(tmp_path, [
        _scalar("weight/effective_rank/foo", 0, 1.0),
        _scalar("activation/dead_fraction/bar", 0, 0.5),
    ])
    out = build_report(tmp_path).read_text()
    # Hero sections appear as top-level ## headers, not inside <details>.
    assert "## weight/effective_rank" in out
    assert "## activation/dead_fraction" in out
    # No <details> when only hero diagnostics present.
    assert "<details>" not in out


def test_advanced_diags_collapsed_in_details(tmp_path):
    _write_metrics(tmp_path, [
        _scalar("weight/effective_rank/foo", 0, 1.0),
        _scalar("weight/stable_rank/foo", 0, 0.9),
        _scalar("activation/kurtosis/bar", 0, 3.1),
    ])
    out = build_report(tmp_path).read_text()
    assert "## weight/effective_rank" in out
    assert "<details>" in out
    assert "<summary>Advanced metrics</summary>" in out
    # Advanced section names appear INSIDE the details block.
    details_start = out.index("<details>")
    details_end = out.index("</details>", details_start)
    block = out[details_start:details_end]
    assert "weight/stable_rank" in block
    assert "activation/kurtosis" in block


def test_grad_per_param_table_trimmed_to_top_and_bottom_k(tmp_path):
    # 25 distinct grad/per_param tags. Top-K + bottom-K = 20 should
    # render; the middle 5 should be elided.
    rows = []
    for i in range(25):
        rows.append(_scalar(f"grad/per_param/m{i:02d}/norm", 0, float(i)))
    _write_metrics(tmp_path, rows)
    out = build_report(tmp_path).read_text()
    # Top 10 (largest norm): m24..m15. Bottom 10: m09..m00.
    assert "`m24/norm`" in out
    assert "`m00/norm`" in out
    # Middle elided.
    assert "`m12/norm`" not in out
    assert "rows hidden" in out or "more rows" in out  # elision label


def test_grad_global_total_norm_always_inline(tmp_path):
    _write_metrics(tmp_path, [
        _scalar("grad/global/total_norm", 0, 1.5),
        _scalar("weight/effective_rank/foo", 0, 1.0),
    ])
    out = build_report(tmp_path).read_text()
    # total_norm is hero — must NOT be in <details>.
    if "<details>" in out:
        details_start = out.index("<details>")
        details_end = out.index("</details>", details_start)
        block = out[details_start:details_end]
        assert "## grad/global" not in block
        assert "`total_norm`" not in block
    # Hero section header and row are present inline.
    assert "## grad/global" in out
    assert "`total_norm`" in out


def test_logit_lens_kl_is_hero_section(tmp_path):
    _write_metrics(tmp_path, [
        _scalar("weight/effective_rank/foo", 0, 1.0),
        _scalar("activation/logit_lens_kl/layers.0", 0, 0.5),
        _scalar("weight/kurtosis/bar", 0, 3.0),
    ])
    out = build_report(tmp_path).read_text()
    assert "## activation/logit_lens_kl" in out
    if "<details>" in out:
        details_start = out.index("<details>")
        details_end = out.index("</details>", details_start)
        block = out[details_start:details_end]
        assert "activation/logit_lens_kl" not in block


def test_induction_score_is_hero_section(tmp_path):
    _write_metrics(tmp_path, [
        _scalar("weight/effective_rank/foo", 0, 1.0),
        _scalar("activation/induction_score/layers.0.self_attn/head_0", 0, 0.8),
        _scalar("weight/kurtosis/bar", 0, 3.0),
    ])
    out = build_report(tmp_path).read_text()
    assert "## activation/induction_score" in out
    if "<details>" in out:
        details_start = out.index("<details>")
        details_end = out.index("</details>", details_start)
        block = out[details_start:details_end]
        assert "activation/induction_score" not in block


def test_attention_pattern_entropy_is_hero_section(tmp_path):
    _write_metrics(tmp_path, [
        _scalar("weight/effective_rank/foo", 0, 1.0),
        _scalar("activation/attention_pattern_entropy/layers.0.self_attn/head_0", 0, 1.2),
        _scalar("weight/kurtosis/bar", 0, 3.0),
    ])
    out = build_report(tmp_path).read_text()
    assert "## activation/attention_pattern_entropy" in out
    if "<details>" in out:
        details_start = out.index("<details>")
        details_end = out.index("</details>", details_start)
        block = out[details_start:details_end]
        assert "activation/attention_pattern_entropy" not in block


def test_sae_is_hero_section(tmp_path):
    _write_metrics(tmp_path, [
        _scalar("weight/effective_rank/foo", 0, 1.0),
        _scalar("activation/sae/layers.8.mlp/recon_mse", 0, 0.05),
        _scalar("weight/kurtosis/bar", 0, 3.0),
    ])
    out = build_report(tmp_path).read_text()
    assert "## activation/sae" in out
    if "<details>" in out:
        details_start = out.index("<details>")
        details_end = out.index("</details>", details_start)
        block = out[details_start:details_end]
        assert "activation/sae" not in block
