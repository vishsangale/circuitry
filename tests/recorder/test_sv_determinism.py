"""A1 — Recorder SVD-subsample determinism tests.

The Recorder passes a FIXED ``_SUBSAMPLE_SEED`` to ``singular_values()``, so the
random column/row subsample used for matrices wider than ``max_dim`` is identical
across runs regardless of the global RNG state.

These tests deep-copy a single model (so both runs have byte-identical weights)
but perturb the **global** RNG differently per run. With the seed wiring in place
the subsample uses its own fixed generator and the diagnostics match; if the seed
wiring were removed (unseeded ``randperm``), the differing global RNG would change
the subsample and ``rank_trajectory`` would diverge — so this test has teeth.
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn

from circuitry.recipes import Recipe
from circuitry.recorder.hooks import HookPoint, TensorSource
from circuitry.recorder.live import Recorder


class _RecordingWriter:
    """Minimal MetricWriter that collects scalars in memory."""

    def __init__(self):
        self.scalars: list[tuple[str, float, int]] = []

    def add_scalar(self, tag: str, value: float, step: int) -> None:
        self.scalars.append((tag, float(value), step))

    def add_histogram(self, *a, **k) -> None:
        pass

    def add_image(self, *a, **k) -> None:
        pass

    def add_text(self, *a, **k) -> None:
        pass

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass


def _build_large_linear():
    """nn.Linear(600, 600): weight shape (600, 600), min(shape)=600 > max_dim=512,
    so singular_values() subsamples — the path A1 must make deterministic."""
    torch.manual_seed(0)
    return nn.Linear(600, 600)


def _run_two_step_recorder(tmp_path, model, weight_diagnostics, rng_seed):
    """Run a Recorder for 2 emit steps on ``model``; first perturb the global RNG
    with ``rng_seed`` so the *unseeded* subsample path would draw differently."""
    # Drive the global RNG to a run-specific state. With the A1 fix the subsample
    # uses its own fixed generator and is unaffected; without it the subsample
    # would consume this (differing) global RNG and the runs would diverge.
    torch.manual_seed(rng_seed)
    _ = torch.rand(rng_seed % 64 + 1)

    recipe = Recipe(
        name="__test_sv_det__",
        hook_points=[HookPoint(pattern=r".*", source=TensorSource.WEIGHT)],
        weight_diagnostics=weight_diagnostics,
        activation_diagnostics=[],
        gradient_diagnostics=[],
    )
    writer = _RecordingWriter()
    rec = Recorder(model, run_dir=tmp_path, recipe=recipe, writer=writer, every_n_steps=1)
    rec.attach()

    # Step 0: first emit (rank_trajectory/update_delta skip — no prior snapshot).
    with torch.no_grad():
        _ = model(torch.zeros(2, 600))
    rec.step(step=0)

    # Perturb weights so update_delta is non-zero at step 1 (torch.ones: no RNG).
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.ones_like(p) * 0.01)

    # Step 1: emit (rank_trajectory and update_delta fire).
    with torch.no_grad():
        _ = model(torch.zeros(2, 600))
    rec.step(step=1)

    rec.detach()
    return writer.scalars


def _filter_sorted(scalars, needle):
    return sorted(
        [(t, v, s) for t, v, s in scalars if needle in t],
        key=lambda x: (x[0], x[2]),
    )


def test_rank_trajectory_deterministic_across_runs(tmp_path):
    """rank_trajectory depends on the random subsample. With the fixed seed, two
    runs on identical weights but DIFFERENT global RNG must match exactly; an
    unseeded subsample would diverge."""
    base = _build_large_linear()
    scalars_a = _run_two_step_recorder(
        tmp_path / "run_a", copy.deepcopy(base), ["rank_trajectory"], rng_seed=123
    )
    scalars_b = _run_two_step_recorder(
        tmp_path / "run_b", copy.deepcopy(base), ["rank_trajectory"], rng_seed=98765
    )
    rank_a = _filter_sorted(scalars_a, "rank_trajectory")
    rank_b = _filter_sorted(scalars_b, "rank_trajectory")

    assert len(rank_a) > 0, "expected at least one rank_trajectory scalar"
    assert len(rank_a) == len(rank_b), "both runs should emit the same number of scalars"
    for (tag_a, val_a, step_a), (tag_b, val_b, step_b) in zip(rank_a, rank_b, strict=True):
        assert tag_a == tag_b
        assert step_a == step_b
        assert val_a == val_b, (
            f"rank_trajectory not reproducible at step {step_a}: {val_a} != {val_b} "
            "(seed wiring broken?)"
        )


def test_update_delta_deterministic_across_runs(tmp_path):
    """update_delta uses full-matrix norms (no subsample), so it is RNG-independent
    by construction; this confirms end-to-end reproducibility under a perturbed RNG."""
    base = _build_large_linear()
    scalars_a = _run_two_step_recorder(
        tmp_path / "run_a", copy.deepcopy(base), ["update_delta"], rng_seed=123
    )
    scalars_b = _run_two_step_recorder(
        tmp_path / "run_b", copy.deepcopy(base), ["update_delta"], rng_seed=98765
    )
    delta_a = _filter_sorted(scalars_a, "update_delta")
    delta_b = _filter_sorted(scalars_b, "update_delta")

    assert len(delta_a) > 0, "expected at least one update_delta scalar"
    assert len(delta_a) == len(delta_b)
    for (tag_a, val_a, step_a), (tag_b, val_b, step_b) in zip(delta_a, delta_b, strict=True):
        assert tag_a == tag_b
        assert step_a == step_b
        assert val_a == val_b, (
            f"update_delta not reproducible at step {step_a}: {val_a} != {val_b}"
        )
