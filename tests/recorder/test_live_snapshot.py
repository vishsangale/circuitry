"""Tests for the cross-step snapshot holder and weight-dynamics dispatch (v1.3)."""
from __future__ import annotations

import torch
import torch.nn as nn

from circuitry.recipes import Recipe
from circuitry.recorder.hooks import HookPoint, TensorSource
from circuitry.recorder.live import Recorder
from circuitry.writers.null import NullWriter


def _make_recorder(tmp_path, weight_diagnostics, every_n_steps=1):
    model = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 4))
    recipe = Recipe(
        name="__test_snapshot__",
        hook_points=[HookPoint(pattern=r".*", source=TensorSource.WEIGHT)],
        weight_diagnostics=weight_diagnostics,
        activation_diagnostics=[],
        gradient_diagnostics=[],
    )
    rec = Recorder(
        model, run_dir=tmp_path, recipe=recipe,
        writer=NullWriter(), every_n_steps=every_n_steps,
    )
    return model, rec


def _one_step(model, rec, step):
    with torch.no_grad():
        _ = model(torch.randn(2, 8))
    rec.step(step)


# ── Snapshot lifecycle ──────────────────────────────────────────────────────


def test_prev_weights_empty_before_first_emit(tmp_path):
    model, rec = _make_recorder(tmp_path, ["update_delta"])
    rec.attach()
    assert rec._prev_weights == {}
    rec.detach()


def test_prev_weights_populated_after_first_emit(tmp_path):
    model, rec = _make_recorder(tmp_path, ["update_delta"])
    rec.attach()
    _one_step(model, rec, step=0)
    assert len(rec._prev_weights) > 0
    rec.detach()


def test_prev_prev_populated_after_second_emit(tmp_path):
    model, rec = _make_recorder(tmp_path, ["direction_cosine"])
    rec.attach()
    _one_step(model, rec, step=0)
    assert rec._prev_prev_weights == {}
    _one_step(model, rec, step=1)
    assert len(rec._prev_prev_weights) > 0
    rec.detach()


def test_detach_clears_snapshots(tmp_path):
    model, rec = _make_recorder(tmp_path, ["update_delta"])
    rec.attach()
    _one_step(model, rec, step=0)
    _one_step(model, rec, step=1)
    rec.detach()
    assert rec._prev_weights == {}
    assert rec._prev_prev_weights == {}


def test_snapshot_is_cpu_detached(tmp_path):
    model, rec = _make_recorder(tmp_path, ["update_delta"])
    rec.attach()
    _one_step(model, rec, step=0)
    for t in rec._prev_weights.values():
        assert t.device.type == "cpu"
        assert not t.requires_grad
    rec.detach()


# ── First-step guard — no diagnostics emitted on step 0 ──────────────────


class _RecordingWriter:
    def __init__(self):
        self.scalars: list[tuple[str, float, int]] = []
    def add_scalar(self, tag, value, step): self.scalars.append((tag, value, step))
    def add_histogram(self, *a, **k): pass
    def add_image(self, *a, **k): pass
    def add_text(self, *a, **k): pass
    def flush(self): pass
    def close(self): pass


def _make_recorder_recording(tmp_path, weight_diagnostics):
    model = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 4))
    recipe = Recipe(
        name="__test_dyn__",
        hook_points=[HookPoint(pattern=r".*", source=TensorSource.WEIGHT)],
        weight_diagnostics=weight_diagnostics,
        activation_diagnostics=[],
        gradient_diagnostics=[],
    )
    writer = _RecordingWriter()
    rec = Recorder(
        model, run_dir=tmp_path, recipe=recipe,
        writer=writer, every_n_steps=1,
    )
    return model, rec, writer


def test_no_update_delta_on_first_step(tmp_path):
    model, rec, writer = _make_recorder_recording(tmp_path, ["update_delta"])
    rec.attach()
    _one_step(model, rec, step=0)
    tags = [t for t, _, _ in writer.scalars]
    assert not any("update_delta" in t for t in tags)
    rec.detach()


def test_no_direction_cosine_on_first_two_steps(tmp_path):
    model, rec, writer = _make_recorder_recording(tmp_path, ["direction_cosine"])
    rec.attach()
    _one_step(model, rec, step=0)
    _one_step(model, rec, step=1)
    tags = [t for t, _, _ in writer.scalars]
    assert not any("direction_cosine" in t for t in tags)
    rec.detach()


def test_no_rank_trajectory_on_first_step(tmp_path):
    model, rec, writer = _make_recorder_recording(tmp_path, ["rank_trajectory"])
    rec.attach()
    _one_step(model, rec, step=0)
    tags = [t for t, _, _ in writer.scalars]
    assert not any("rank_trajectory" in t for t in tags)
    rec.detach()


# ── Emission after sufficient steps ──────────────────────────────────────


def test_update_delta_emitted_on_second_step(tmp_path):
    model, rec, writer = _make_recorder_recording(tmp_path, ["update_delta"])
    rec.attach()
    _one_step(model, rec, step=0)
    _one_step(model, rec, step=1)
    tags = [t for t, _, _ in writer.scalars]
    assert any("weight/update_delta/" in t for t in tags)
    # Values must be non-negative (L2 norm)
    for t, v, _ in writer.scalars:
        if "update_delta" in t:
            assert v >= 0.0
    rec.detach()


def test_direction_cosine_emitted_on_third_step(tmp_path):
    model, rec, writer = _make_recorder_recording(tmp_path, ["direction_cosine"])
    rec.attach()
    for s in range(3):
        _one_step(model, rec, step=s)
    tags = [t for t, _, _ in writer.scalars]
    assert any("weight/direction_cosine/" in t for t in tags)
    # Cosine in [-1, 1]
    for t, v, _ in writer.scalars:
        if "direction_cosine" in t:
            assert -1.0 - 1e-6 <= v <= 1.0 + 1e-6
    rec.detach()


def test_rank_trajectory_emitted_on_second_step(tmp_path):
    model, rec, writer = _make_recorder_recording(tmp_path, ["rank_trajectory"])
    rec.attach()
    _one_step(model, rec, step=0)
    _one_step(model, rec, step=1)
    tags = [t for t, _, _ in writer.scalars]
    assert any("weight/rank_trajectory/" in t for t in tags)
    # Effective rank must be positive
    for t, v, _ in writer.scalars:
        if "rank_trajectory" in t:
            assert v > 0.0
    rec.detach()


# ── Regression: snapshots must be real copies, not views of live params ──────
# A storage-aliasing snapshot tracks in-place optimizer updates, making
# update_delta / direction_cosine identically zero on CPU training (the
# library's primary path). These tests drive a real optimizer step between
# emits so the alias bug — invisible to the forward-only tests above — fails.


def _train_step(model, rec, opt, step):
    opt.zero_grad()
    loss = model(torch.randn(2, 8)).pow(2).sum()
    loss.backward()
    opt.step()
    rec.step(step)


def test_update_delta_positive_under_training(tmp_path):
    model, rec, writer = _make_recorder_recording(tmp_path, ["update_delta"])
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    rec.attach()
    _train_step(model, rec, opt, step=0)
    _train_step(model, rec, opt, step=1)
    rec.detach()
    deltas = [v for t, v, _ in writer.scalars if "weight/update_delta/" in t]
    assert deltas, "expected update_delta tags after two training steps"
    # A real weight change between emits → strictly positive delta.
    # An aliasing snapshot would report exactly 0.0.
    assert max(deltas) > 0.0


def test_snapshots_distinct_under_training(tmp_path):
    model, rec = _make_recorder(tmp_path, ["update_delta"])
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    rec.attach()
    _train_step(model, rec, opt, step=0)
    _train_step(model, rec, opt, step=1)
    # The two snapshots hold weights from different steps; an aliasing bug would
    # make them identical (both tracking the live, now-updated parameter).
    assert rec._prev_weights and rec._prev_prev_weights
    assert any(
        not torch.equal(rec._prev_weights[k], rec._prev_prev_weights[k])
        for k in rec._prev_weights
    )
    rec.detach()
