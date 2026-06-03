"""recsys C/D + fingerprint #3: Recipe.forward_fn (non-HF probe forward) and
flexible scan_run checkpoint discovery.

forward_fn lets non-HF models (e.g. SASRec.predict_scores) drive the recorder's
internal probe passes; without it the recorder uses the HF-style call with a
TypeError fallback. The no-config model also gets a one-time warning pointing at
forward_fn. scan_run's discovery accepts explicit paths / globs / (step, path)
pairs so arbitrarily-named, single-snapshot checkpoints can be scanned.
"""
from __future__ import annotations

import logging
import pathlib

import pytest
import torch
import torch.nn as nn

from circuitry.recipes import (
    Recipe,
    _clear_registry_for_tests,
    register_recipe,
)
from circuitry.recorder.hooks import HookPoint, TensorSource
from circuitry.recorder.live import Recorder
from circuitry.recorder.scan import _coerce_checkpoints, scan_run

# --------------------------- forward_fn -----------------------------------

def test_probe_forward_uses_recipe_forward_fn(tmp_path):
    calls = []

    def my_fwd(model, batch):
        calls.append((model, batch))
        return "custom-out"

    recipe = Recipe(name="ff", hook_points=[], forward_fn=my_fwd)
    model = nn.Linear(4, 4)
    rec = Recorder(model, run_dir=tmp_path, recipe=recipe, writer="null")
    out = rec._probe_forward(torch.zeros(2, 4))
    assert out == "custom-out"
    assert len(calls) == 1
    assert calls[0][0] is model


def test_probe_forward_hf_style_default(tmp_path):
    """No forward_fn: a model accepting output_attentions is called HF-style."""

    class _M(nn.Module):
        def forward(self, x, output_attentions=False):
            assert output_attentions is True
            return x * 2

    recipe = Recipe(name="hf", hook_points=[])
    rec = Recorder(_M(), run_dir=tmp_path, recipe=recipe, writer="null")
    out = rec._probe_forward(torch.ones(3))
    assert torch.equal(out, torch.full((3,), 2.0))


def test_probe_forward_typeerror_fallback(tmp_path):
    """A wrapper whose forward lacks output_attentions falls back to model(probe)."""

    class _M(nn.Module):
        def forward(self, x):
            return x + 1

    recipe = Recipe(name="wrap", hook_points=[])
    rec = Recorder(_M(), run_dir=tmp_path, recipe=recipe, writer="null")
    out = rec._probe_forward(torch.zeros(2))
    assert torch.equal(out, torch.ones(2))


def test_no_config_model_warns_and_skips_attention(tmp_path, caplog):
    """A non-HF model (no .config) with attention_pattern_entropy enabled must
    warn once (pointing at forward_fn) and mark attention diagnostics skipped."""
    recipe = Recipe(
        name="noc", hook_points=[],
        activation_diagnostics=["attention_pattern_entropy"],
    )
    rec = Recorder(nn.Linear(4, 4), run_dir=tmp_path, recipe=recipe,
                   writer="null", strict=False)
    with caplog.at_level(logging.WARNING):
        rec.attach()
        rec.detach()
    assert rec._attn_diags_sdpa_skip is True
    msgs = [r.message for r in caplog.records]
    assert any("no resolvable `config`" in m for m in msgs)
    assert any("forward_fn" in m for m in msgs)


# ----------------------- scan checkpoint discovery -------------------------

def test_coerce_explicit_step_path_pairs_sorted(tmp_path):
    pairs = [(5, tmp_path / "b.pt"), (1, tmp_path / "a.pt")]
    out = _coerce_checkpoints(pairs, tmp_path)
    assert [s for s, _ in out] == [1, 5]
    assert out[0][1] == tmp_path / "a.pt"


def test_coerce_path_list_parses_steps(tmp_path):
    paths = ["x/step000010.pt", "y/step000002.pt"]
    out = _coerce_checkpoints(paths, tmp_path)
    assert [s for s, _ in out] == [2, 10]


def test_coerce_single_file_step_zero_when_unnamed(tmp_path):
    out = _coerce_checkpoints("models/final_checkpoint.pt", tmp_path)
    assert out == [(0, pathlib.Path("models/final_checkpoint.pt"))]


def test_coerce_glob_matches_files(tmp_path):
    for n in ("step1.pt", "step2.pt"):
        (tmp_path / n).write_bytes(b"x")
    out = _coerce_checkpoints(str(tmp_path / "*.pt"), tmp_path)
    assert [s for s, _ in out] == [1, 2]


@pytest.fixture
def _demo_recipe():
    _clear_registry_for_tests()
    register_recipe(Recipe(
        name="scan-demo",
        hook_points=[HookPoint(source=TensorSource.WEIGHT, pattern=r"^\d+$")],
        weight_diagnostics=["effective_rank"],
    ))
    yield
    _clear_registry_for_tests()


def _toy() -> nn.Module:
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 4))


def test_scan_run_with_explicit_arbitrary_named_checkpoints(_demo_recipe, tmp_path):
    """fingerprint #3: scan an arbitrarily-named single snapshot that the
    default step*.pt glob would never discover, via explicit (step, path)."""
    ckpt = tmp_path / "mymodel_final.pt"
    torch.save(_toy().state_dict(), ckpt)
    out_dir = tmp_path / "out"

    # Default discovery finds nothing here (no checkpoints/ dir).
    with pytest.raises(FileNotFoundError):
        scan_run(run_dir=tmp_path, recipe="scan-demo", out_dir=out_dir,
                 model_factory=_toy, writer="jsonl")

    # Explicit checkpoints argument scans the snapshot.
    scan_run(run_dir=tmp_path, recipe="scan-demo", out_dir=out_dir,
             model_factory=_toy, writer="jsonl",
             checkpoints=[(42, ckpt)])
    content = (out_dir / "metrics.jsonl").read_text()
    assert "effective_rank" in content
    assert '"step": 42' in content or '"step":42' in content
