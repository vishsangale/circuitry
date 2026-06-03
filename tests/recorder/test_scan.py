from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from circuitry.recipes import Recipe, _clear_registry_for_tests, register_recipe
from circuitry.recorder.hooks import HookPoint, TensorSource
from circuitry.recorder.scan import scan_run


@pytest.fixture(autouse=True)
def _clean():
    _clear_registry_for_tests()
    yield
    _clear_registry_for_tests()


def _register():
    register_recipe(Recipe(
        name="scan-demo",
        hook_points=[HookPoint(source=TensorSource.WEIGHT, pattern=r"^\d+$")],
        weight_diagnostics=["effective_rank"],
    ))


def _register_trajectory():
    register_recipe(Recipe(
        name="scan-traj",
        hook_points=[HookPoint(source=TensorSource.WEIGHT, pattern=r"^\d+$")],
        weight_diagnostics=["effective_rank", "update_delta", "direction_cosine"],
    ))


def _toy(seed: int) -> nn.Module:
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 4))


def test_scan_run_processes_checkpoints(tmp_path):
    _register()
    # Lay down two checkpoint files in conventional locations.
    ckpts = tmp_path / "checkpoints"
    ckpts.mkdir()
    for step, seed in [(100, 0), (200, 1)]:
        torch.save(_toy(seed).state_dict(), ckpts / f"step{step:09d}.pt")

    out_dir = tmp_path / "tb_retro"
    scan_run(run_dir=tmp_path, recipe="scan-demo", out_dir=out_dir,
             model_factory=lambda: _toy(0))

    event_files = list(out_dir.rglob("events.out.tfevents.*"))
    assert event_files


def test_single_snapshot_warns_about_trajectory_diagnostics(tmp_path):
    """fingerprint #5: a 1-checkpoint scan with trajectory diagnostics warns
    that they need >= 2 emitted steps and will be absent."""
    _register_trajectory()
    ckpts = tmp_path / "checkpoints"
    ckpts.mkdir()
    torch.save(_toy(0).state_dict(), ckpts / "step000000100.pt")  # single snapshot
    with pytest.warns(UserWarning, match="trajectory diagnostics"):
        scan_run(run_dir=tmp_path, recipe="scan-traj",
                 out_dir=tmp_path / "out", model_factory=lambda: _toy(0),
                 writer="jsonl")


def test_two_snapshots_no_trajectory_warning(tmp_path, recwarn):
    """With >= 2 checkpoints the trajectory warning must NOT fire."""
    _register_trajectory()
    ckpts = tmp_path / "checkpoints"
    ckpts.mkdir()
    for step, seed in [(100, 0), (200, 1)]:
        torch.save(_toy(seed).state_dict(), ckpts / f"step{step:09d}.pt")
    scan_run(run_dir=tmp_path, recipe="scan-traj",
             out_dir=tmp_path / "out", model_factory=lambda: _toy(0),
             writer="jsonl")
    assert not [w for w in recwarn.list if "trajectory diagnostics" in str(w.message)]


def test_scan_run_raises_when_no_checkpoints(tmp_path):
    _register()
    with pytest.raises(FileNotFoundError):
        scan_run(run_dir=tmp_path, recipe="scan-demo",
                 out_dir=tmp_path / "tb_retro",
                 model_factory=lambda: _toy(0))


def test_scan_run_accepts_jsonl_writer(tmp_path):
    """v0.5.0: scan_run takes an optional `writer` parameter so its output
    can be made build_report-compatible. Default stays "tensorboard"."""
    _register()
    ckpts = tmp_path / "checkpoints"
    ckpts.mkdir()
    for step, seed in [(0, 0), (1, 1)]:
        torch.save(_toy(seed).state_dict(), ckpts / f"step{step:09d}.pt")

    out_dir = tmp_path / "scan_jsonl"
    scan_run(run_dir=tmp_path, recipe="scan-demo", out_dir=out_dir,
             model_factory=lambda: _toy(0), writer="jsonl")
    jsonl = out_dir / "metrics.jsonl"
    assert jsonl.exists()
    content = jsonl.read_text()
    assert "effective_rank" in content
