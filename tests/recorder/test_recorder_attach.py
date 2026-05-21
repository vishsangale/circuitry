# tests/recorder/test_recorder_attach.py
from __future__ import annotations

import logging

import pytest
import torch.nn as nn

from circuitry.recipes import Recipe, _clear_registry_for_tests, register_recipe
from circuitry.recorder.hooks import HookPoint, TensorSource
from circuitry.recorder.live import Recorder
from circuitry.writers.base import RecordingWriter


@pytest.fixture(autouse=True)
def _clean():
    _clear_registry_for_tests()
    yield
    _clear_registry_for_tests()


def _toy_model() -> nn.Module:
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 4))


def _register_demo(pattern: str = r"^\d+$", min_matches: int = 0) -> None:
    register_recipe(Recipe(
        name="demo",
        hook_points=[HookPoint(source=TensorSource.WEIGHT, pattern=pattern)],
        weight_diagnostics=["effective_rank"],
        expected_min_matches={pattern: min_matches},
    ))


def test_attach_writes_matched_modules_file(tmp_path):
    _register_demo()
    model = _toy_model()
    rec = Recorder(model, run_dir=tmp_path, recipe="demo",
                   writer=RecordingWriter(), every_n_steps=1)
    rec.attach()
    f = tmp_path / "circuitry" / "matched_modules.txt"
    assert f.exists()
    content = f.read_text()
    assert "0" in content and "2" in content
    rec.detach()


def test_attach_logs_matched_modules_at_info(tmp_path, caplog):
    _register_demo()
    caplog.set_level(logging.INFO, logger="circuitry")
    rec = Recorder(_toy_model(), run_dir=tmp_path, recipe="demo",
                   writer=RecordingWriter(), every_n_steps=1)
    rec.attach()
    rec.detach()
    assert any("matched" in r.message.lower() for r in caplog.records)


def test_attach_raises_on_zero_matches_regardless_of_strict(tmp_path):
    register_recipe(Recipe(
        name="bad",
        hook_points=[HookPoint(source=TensorSource.WEIGHT,
                               pattern=r"this-matches-nothing")],
    ))
    rec = Recorder(_toy_model(), run_dir=tmp_path, recipe="bad",
                   writer=RecordingWriter(), every_n_steps=1, strict=False)
    with pytest.raises(RuntimeError, match="matched 0 modules"):
        rec.attach()


def test_attach_raises_on_min_matches_violation_in_strict_mode(tmp_path):
    _register_demo(pattern=r"^\d+$", min_matches=99)
    rec = Recorder(_toy_model(), run_dir=tmp_path, recipe="demo",
                   writer=RecordingWriter(), every_n_steps=1, strict=True)
    with pytest.raises(RuntimeError, match="expected at least 99"):
        rec.attach()


def test_attach_warns_on_min_matches_violation_in_non_strict_mode(tmp_path, caplog):
    _register_demo(pattern=r"^\d+$", min_matches=99)
    caplog.set_level(logging.WARNING, logger="circuitry")
    rec = Recorder(_toy_model(), run_dir=tmp_path, recipe="demo",
                   writer=RecordingWriter(), every_n_steps=1, strict=False)
    rec.attach()
    rec.detach()
    assert any("expected at least 99" in r.message for r in caplog.records)


def test_detach_removes_all_hooks(tmp_path):
    _register_demo()
    model = _toy_model()
    rec = Recorder(model, run_dir=tmp_path, recipe="demo",
                   writer=RecordingWriter(), every_n_steps=1)
    rec.attach()
    rec.detach()
    post = sum(len(m._forward_hooks) + len(m._forward_pre_hooks)
               + len(m._backward_hooks) for m in model.modules())
    assert post == 0
    # We don't assert pre > 0 — pure-weight recipes may not install hooks.


def test_recorder_noop_on_non_zero_rank(monkeypatch, tmp_path):
    _register_demo()
    monkeypatch.setattr("torch.distributed.is_initialized", lambda: True)
    monkeypatch.setattr("torch.distributed.get_rank", lambda: 1)
    writer = RecordingWriter()
    rec = Recorder(_toy_model(), run_dir=tmp_path, recipe="demo",
                   writer=writer, every_n_steps=1)
    rec.attach()
    rec.step(0, loss=1.0)
    rec.detach()
    assert writer.scalars == []
    assert not (tmp_path / "circuitry" / "matched_modules.txt").exists()
