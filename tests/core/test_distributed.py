"""Tests for core/distributed.py — design §11 reduce helpers (v1.45).

Single-process semantics are exercised directly; the collective path runs
under a real 2-process gloo group via ``torch.multiprocessing.spawn``.
"""
from __future__ import annotations

import os

import pytest
import torch
import torch.multiprocessing as mp

from circuitry.core.distributed import (
    all_gather_concat,
    full_tensor,
    is_distributed,
    is_main_process,
    world_size,
)

# ---------------------------------------------------------------------------
# Single-process passthrough semantics
# ---------------------------------------------------------------------------


def test_not_distributed_by_default():
    assert not is_distributed()
    assert world_size() == 1
    assert is_main_process()


def test_all_gather_concat_passthrough():
    t = torch.randn(3, 4)
    out = all_gather_concat(t)
    assert out is t  # no copy when not distributed


def test_full_tensor_passthrough_plain_tensor():
    t = torch.randn(2, 2)
    assert full_tensor(t) is t


def test_full_tensor_passthrough_parameter():
    p = torch.nn.Parameter(torch.randn(2, 2))
    assert full_tensor(p) is p


def test_full_tensor_dtensor_duck_type():
    class FakeDTensor:
        def full_tensor(self):
            return torch.ones(4)

    out = full_tensor(FakeDTensor())
    assert torch.equal(out, torch.ones(4))


# ---------------------------------------------------------------------------
# Real 2-process gloo group
# ---------------------------------------------------------------------------


def _worker(rank: int, world: int, port: int, results) -> None:
    import torch.distributed as dist

    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world,
    )
    try:
        local = torch.full((2, 3), float(rank))
        gathered = all_gather_concat(local, dim=0)
        results[rank] = {
            "shape": tuple(gathered.shape),
            "rank_means": [float(gathered[i * 2: (i + 1) * 2].mean()) for i in range(world)],
            "world_size": world_size(),
            "is_main": is_main_process(),
            "is_dist": is_distributed(),
        }
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not torch.distributed.is_available(), reason="torch.distributed unavailable"
)
def test_all_gather_concat_two_process_gloo():
    port = 29512 + (os.getpid() % 500)
    world = 2
    manager = mp.Manager()
    results = manager.dict()
    mp.spawn(_worker, args=(world, port, results), nprocs=world, join=True)

    assert set(results.keys()) == {0, 1}
    for rank in (0, 1):
        r = results[rank]
        assert r["is_dist"] is True
        assert r["world_size"] == 2
        # every rank sees the full concatenation, rank-major
        assert r["shape"] == (4, 3)
        assert r["rank_means"] == [0.0, 1.0]
    assert results[0]["is_main"] is True
    assert results[1]["is_main"] is False


def test_top_level_exports():
    from circuitry.core import distributed

    for name in (
        "is_distributed", "world_size", "is_main_process",
        "all_gather_concat", "full_tensor",
    ):
        assert hasattr(distributed, name)
        assert name in distributed.__all__
