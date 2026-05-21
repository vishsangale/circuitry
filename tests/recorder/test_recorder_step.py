from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from circuitry.recipes import Recipe, _clear_registry_for_tests, register_recipe
from circuitry.recorder.hooks import HookPoint, StepContext, TensorSource
from circuitry.recorder.live import Recorder
from circuitry.writers.base import RecordingWriter


@pytest.fixture(autouse=True)
def _clean():
    _clear_registry_for_tests()
    yield
    _clear_registry_for_tests()


def _toy_model() -> nn.Module:
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 4))


def test_step_writes_weight_diagnostic_scalars(tmp_path):
    register_recipe(Recipe(
        name="w-only",
        hook_points=[HookPoint(source=TensorSource.WEIGHT, pattern=r"^\d+$")],
        weight_diagnostics=["effective_rank", "stable_rank"],
    ))
    writer = RecordingWriter()
    rec = Recorder(_toy_model(), run_dir=tmp_path, recipe="w-only",
                   writer=writer, every_n_steps=1)
    rec.attach()
    rec.step(0, loss=1.0)
    rec.detach()
    tags = {t for t, _, _ in writer.scalars}
    assert any("effective_rank" in t for t in tags)
    assert any("stable_rank" in t for t in tags)
    assert ("loss", 1.0, 0) in writer.scalars


def test_step_respects_every_n_steps(tmp_path):
    register_recipe(Recipe(
        name="every-3",
        hook_points=[HookPoint(source=TensorSource.WEIGHT, pattern=r"^\d+$")],
        weight_diagnostics=["effective_rank"],
    ))
    writer = RecordingWriter()
    rec = Recorder(_toy_model(), run_dir=tmp_path, recipe="every-3",
                   writer=writer, every_n_steps=3)
    rec.attach()
    for s in range(7):
        rec.step(s, loss=float(s))
    rec.detach()
    steps_with_rank = sorted({s for t, _, s in writer.scalars if "effective_rank" in t})
    # Emit steps: 0, 3, 6
    assert steps_with_rank == [0, 3, 6]
    # Loss is recorded every step.
    assert sorted(s for t, _, s in writer.scalars if t == "loss") == list(range(7))


def test_step_runs_activation_diagnostic_after_forward(tmp_path):
    register_recipe(Recipe(
        name="act",
        hook_points=[HookPoint(source=TensorSource.OUTPUT, pattern=r"^0$")],
        activation_diagnostics=["dead_fraction"],
    ))
    model = _toy_model()
    writer = RecordingWriter()
    rec = Recorder(model, run_dir=tmp_path, recipe="act",
                   writer=writer, every_n_steps=1)
    rec.attach()
    _ = model(torch.randn(2, 4))
    rec.step(0)
    rec.detach()
    assert any("dead_fraction" in t for t, _, _ in writer.scalars)


def test_step_runs_custom_diagnostic(tmp_path):
    def custom(ctx: StepContext) -> dict[str, float]:
        return {"my_metric": float(ctx.step + 1)}

    register_recipe(Recipe(
        name="cust",
        hook_points=[HookPoint(source=TensorSource.WEIGHT, pattern=r"^\d+$")],
        custom=[custom],
    ))
    writer = RecordingWriter()
    rec = Recorder(_toy_model(), run_dir=tmp_path, recipe="cust",
                   writer=writer, every_n_steps=1)
    rec.attach()
    rec.step(2)
    rec.detach()
    assert ("custom/my_metric", 3.0, 2) in writer.scalars


def test_step_skips_disabled_diagnostic(tmp_path):
    register_recipe(Recipe(
        name="dis",
        hook_points=[HookPoint(source=TensorSource.WEIGHT, pattern=r"^\d+$")],
        weight_diagnostics=["effective_rank", "stable_rank"],
        enabled={"stable_rank": False},
    ))
    writer = RecordingWriter()
    rec = Recorder(_toy_model(), run_dir=tmp_path, recipe="dis",
                   writer=writer, every_n_steps=1)
    rec.attach()
    rec.step(0)
    rec.detach()
    tags = {t for t, _, _ in writer.scalars}
    assert any("effective_rank" in t for t in tags)
    assert not any("stable_rank" in t for t in tags)


def test_activation_diagnostic_with_every_n_steps_3(tmp_path):
    """Regression: hook capture timing must not depend on stale _current_step.

    With every_n_steps=3, activation tags should appear on steps {0, 3, 6} —
    not on {1, 2, 4, 5} — and must NOT all silently drop because the hook
    gating ran before step() updated _current_step.
    """
    register_recipe(Recipe(
        name="act-every-3",
        hook_points=[HookPoint(source=TensorSource.OUTPUT, pattern=r"^0$")],
        activation_diagnostics=["dead_fraction"],
    ))
    model = _toy_model()
    writer = RecordingWriter()
    rec = Recorder(model, run_dir=tmp_path, recipe="act-every-3",
                   writer=writer, every_n_steps=3)
    rec.attach()
    for s in range(7):
        _ = model(torch.randn(2, 4))
        rec.step(s)
    rec.detach()
    act_steps = sorted({step for tag, _, step in writer.scalars
                        if "dead_fraction" in tag})
    assert act_steps == [0, 3, 6]
