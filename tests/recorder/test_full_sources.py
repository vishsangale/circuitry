"""Tests for TensorSource.WEIGHT_FULL / ACTIVATION_FULL (design §11, v1.46).

Single-process semantics (passthrough — identical to WEIGHT / OUTPUT) are
exercised directly; the collective path runs under a real 2-process gloo
group via ``torch.multiprocessing.spawn``: rank 0 must see the cross-rank
concatenated activation batch, rank 1 must write nothing, and the run must
complete without deadlock.
"""
from __future__ import annotations

import os

import pytest
import torch
import torch.multiprocessing as mp
import torch.nn as nn

from circuitry.recipes import Recipe, _clear_registry_for_tests, register_recipe
from circuitry.recorder.hooks import HookPoint, StepContext, TensorSource
from circuitry.recorder.live import Recorder
from circuitry.writers.base import RecordingWriter

D = 6


@pytest.fixture(autouse=True)
def _clean():
    _clear_registry_for_tests()
    yield
    _clear_registry_for_tests()


def _model() -> nn.Module:
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(D, D), nn.ReLU(), nn.Linear(D, D))


def _batch_size_diag(ctx: StepContext) -> dict[str, float]:
    return {
        f"batch_rows/{name}": float(t.shape[0])
        for name, t in ctx.activations.items()
    }


# ---------------------------------------------------------------------------
# Single-process: *_FULL is a pure passthrough
# ---------------------------------------------------------------------------


def test_weight_full_behaves_like_weight_single_process(tmp_path):
    register_recipe(Recipe(
        name="wf",
        hook_points=[HookPoint(source=TensorSource.WEIGHT_FULL, pattern=r"^\d+$")],
        weight_diagnostics=["effective_rank"],
    ))
    writer = RecordingWriter()
    rec = Recorder(_model(), run_dir=tmp_path, recipe="wf",
                   writer=writer, every_n_steps=1)
    rec.attach()
    rec.step(0)
    rec.detach()
    tags = {t for t, _, _ in writer.scalars}
    assert "weight/effective_rank/0" in tags
    assert "weight/effective_rank/2" in tags


def test_activation_full_behaves_like_output_single_process(tmp_path):
    register_recipe(Recipe(
        name="af",
        hook_points=[HookPoint(source=TensorSource.ACTIVATION_FULL, pattern=r"^0$")],
        activation_diagnostics=["dead_fraction"],
        custom=[_batch_size_diag],
    ))
    model = _model()
    writer = RecordingWriter()
    rec = Recorder(model, run_dir=tmp_path, recipe="af",
                   writer=writer, every_n_steps=1)
    rec.attach()
    _ = model(torch.randn(3, D))
    rec.step(0)
    rec.detach()
    tags = {t for t, _, _ in writer.scalars}
    assert any("dead_fraction" in t for t in tags)
    # no gather in single process: local batch of 3 rows
    rows = [v for t, v, _ in writer.scalars if t == "custom/batch_rows/0"]
    assert rows == [3.0]


def test_weight_full_supports_update_delta_across_steps(tmp_path):
    register_recipe(Recipe(
        name="wf-dyn",
        hook_points=[HookPoint(source=TensorSource.WEIGHT_FULL, pattern=r"^\d+$")],
        weight_diagnostics=["update_delta"],
    ))
    model = _model()
    writer = RecordingWriter()
    rec = Recorder(model, run_dir=tmp_path, recipe="wf-dyn",
                   writer=writer, every_n_steps=1)
    rec.attach()
    rec.step(0)
    with torch.no_grad():
        model[0].weight.add_(1.0)
    rec.step(1)
    rec.detach()
    deltas = [v for t, v, _ in writer.scalars
              if t == "weight/update_delta/0" ]
    assert len(deltas) == 1 and deltas[0] > 0


# ---------------------------------------------------------------------------
# Real 2-process gloo group
# ---------------------------------------------------------------------------


def _dist_worker(rank: int, world: int, port: int, results, run_root: str) -> None:
    import torch.distributed as dist

    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world,
    )
    try:
        _clear_registry_for_tests()
        register_recipe(Recipe(
            name="dist-full",
            hook_points=[
                HookPoint(source=TensorSource.ACTIVATION_FULL, pattern=r"^0$"),
                HookPoint(source=TensorSource.WEIGHT_FULL, pattern=r"^\d+$"),
            ],
            weight_diagnostics=["effective_rank"],
            custom=[_batch_size_diag],
        ))
        model = _model()  # seeded -> identical on both ranks (DDP contract)
        writer = RecordingWriter()
        rec = Recorder(model, run_dir=os.path.join(run_root, f"rank{rank}"),
                       recipe="dist-full", writer=writer, every_n_steps=1)
        rec.attach()
        for step in range(2):
            # each rank forwards its own 2-row shard
            _ = model(torch.full((2, D), float(rank)))
            rec.step(step)
        rec.detach()
        results[rank] = {
            "tags": sorted({t for t, _, _ in writer.scalars}),
            "batch_rows": [v for t, v, _ in writer.scalars
                           if t == "custom/batch_rows/0"],
            "rank_dir_exists": os.path.isdir(
                os.path.join(run_root, f"rank{rank}", "circuitry")
            ),
        }
    finally:
        dist.destroy_process_group()
        _clear_registry_for_tests()


@pytest.mark.skipif(
    not torch.distributed.is_available(), reason="torch.distributed unavailable"
)
def test_full_sources_two_process_gloo(tmp_path):
    port = 29612 + (os.getpid() % 500)
    world = 2
    manager = mp.Manager()
    results = manager.dict()
    mp.spawn(
        _dist_worker, args=(world, port, results, str(tmp_path)),
        nprocs=world, join=True,
    )

    assert set(results.keys()) == {0, 1}
    r0, r1 = results[0], results[1]
    # rank 0 saw the gathered batch: 2 rows local + 2 rows from rank 1
    assert r0["batch_rows"] == [4.0, 4.0]  # two emit steps
    # WEIGHT_FULL weight diagnostics emitted on rank 0 (plain replicated params)
    assert "weight/effective_rank/0" in r0["tags"]
    assert r0["rank_dir_exists"] is True
    # rank 1 is a pure participant: nothing written, no run-dir files
    assert r1["tags"] == []
    assert r1["batch_rows"] == []
    assert r1["rank_dir_exists"] is False


def _legacy_worker(rank: int, world: int, port: int, results, run_root: str) -> None:
    """Without *_FULL sources, non-zero ranks must keep the v0.x no-op contract."""
    import torch.distributed as dist

    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world,
    )
    try:
        _clear_registry_for_tests()
        register_recipe(Recipe(
            name="legacy",
            hook_points=[HookPoint(source=TensorSource.WEIGHT, pattern=r"^\d+$")],
            weight_diagnostics=["effective_rank"],
        ))
        model = _model()
        writer = RecordingWriter()
        rec = Recorder(model, run_dir=os.path.join(run_root, f"rank{rank}"),
                       recipe="legacy", writer=writer, every_n_steps=1)
        rec.attach()
        rec.step(0)
        rec.detach()
        results[rank] = sorted({t for t, _, _ in writer.scalars})
    finally:
        dist.destroy_process_group()
        _clear_registry_for_tests()


@pytest.mark.skipif(
    not torch.distributed.is_available(), reason="torch.distributed unavailable"
)
def test_legacy_recipes_keep_rank0_noop_contract(tmp_path):
    port = 29712 + (os.getpid() % 500)
    manager = mp.Manager()
    results = manager.dict()
    mp.spawn(
        _legacy_worker, args=(2, port, results, str(tmp_path)),
        nprocs=2, join=True,
    )
    assert any("effective_rank" in t for t in results[0])
    assert results[1] == []  # rank 1 fully no-op, as before
