"""recsys follow-up #3: TensorSource.NAMED_PARAM reaches a fused parameter that
the WEIGHT source can't — e.g. nn.MultiheadAttention.in_proj_weight (a direct
Parameter on the module, not a child Linear's .weight).
"""
from __future__ import annotations

import pytest
import torch.nn as nn

from circuitry.recipes import Recipe
from circuitry.recorder.hooks import HookPoint, TensorSource, match_modules
from circuitry.recorder.live import Recorder


class _AttnModel(nn.Module):
    def __init__(self, d: int = 8, h: int = 2) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(d, h, batch_first=True)


def test_match_modules_named_param_matches_parameter_names():
    m = _AttnModel()
    hp = HookPoint(source=TensorSource.NAMED_PARAM, pattern=r".*\.in_proj_weight$")
    names = match_modules(m, hp)
    assert names == ["attn.in_proj_weight"]


def test_weight_source_cannot_resolve_fused_in_proj(tmp_path):
    """Motivation: hooking the attn module as WEIGHT can't pick in_proj_weight
    (the module has two 2-D params, so find_primary_weight returns None)."""
    recipe = Recipe(
        name="w-attn",
        hook_points=[HookPoint(source=TensorSource.WEIGHT, pattern=r"^attn$")],
        weight_diagnostics=["effective_rank"],
    )
    rec = Recorder(_AttnModel(), run_dir=tmp_path, recipe=recipe,
                   writer="jsonl", every_n_steps=1, strict=False)
    rec.attach()
    rec.step(0, loss=0.0)
    rec.detach()
    content = (tmp_path / "metrics.jsonl").read_text()
    assert "in_proj_weight" not in content  # the gap NAMED_PARAM fills


def test_named_param_emits_weight_diagnostics_for_in_proj(tmp_path):
    recipe = Recipe(
        name="np-attn",
        hook_points=[HookPoint(source=TensorSource.NAMED_PARAM,
                               pattern=r".*\.in_proj_weight$")],
        weight_diagnostics=["effective_rank", "stable_rank"],
    )
    rec = Recorder(_AttnModel(), run_dir=tmp_path, recipe=recipe,
                   writer="jsonl", every_n_steps=1, strict=True)
    rec.attach()
    rec.step(0, loss=0.0)
    rec.detach()
    content = (tmp_path / "metrics.jsonl").read_text()
    assert "weight/effective_rank/attn.in_proj_weight" in content
    assert "weight/stable_rank/attn.in_proj_weight" in content


def test_named_param_zero_match_raises_under_strict(tmp_path):
    recipe = Recipe(
        name="np-miss",
        hook_points=[HookPoint(source=TensorSource.NAMED_PARAM,
                               pattern=r".*\.no_such_param$")],
        weight_diagnostics=["effective_rank"],
    )
    rec = Recorder(_AttnModel(), run_dir=tmp_path, recipe=recipe,
                   writer="jsonl", strict=True)
    with pytest.raises(RuntimeError, match="matched 0"):
        rec.attach()


def test_named_param_skips_non_2d_param(tmp_path):
    """A NAMED_PARAM matching a 1-D param (bias) is skipped with a warning, not
    fed to the rank diagnostics (which require 2-D)."""
    model = nn.Sequential()
    model.add_module("lin", nn.Linear(4, 4))  # lin.weight (2-D), lin.bias (1-D)
    recipe = Recipe(
        name="np-bias",
        hook_points=[HookPoint(source=TensorSource.NAMED_PARAM,
                               pattern=r".*\.bias$", optional=True)],
        weight_diagnostics=["effective_rank"],
    )
    rec = Recorder(model, run_dir=tmp_path, recipe=recipe, writer="jsonl",
                   strict=False)
    rec.attach()
    rec.step(0, loss=0.0)
    rec.detach()
    content = (tmp_path / "metrics.jsonl").read_text() if (
        tmp_path / "metrics.jsonl").exists() else ""
    # The 1-D bias must not produce a rank diagnostic (it was skipped at attach).
    assert "lin.bias" not in content
