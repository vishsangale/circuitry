from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from circuitry.recipes import _clear_registry_for_tests, get_recipe
from circuitry.recipes.two_tower import register
from circuitry.recorder.live import Recorder
from circuitry.writers.base import RecordingWriter


@pytest.fixture(autouse=True)
def _clean():
    _clear_registry_for_tests()
    register()
    yield
    _clear_registry_for_tests()


class _TwoTower(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.query_tower = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 4))
        self.item_tower = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 4))

    def forward(self, q, i):
        return (self.query_tower(q) * self.item_tower(i)).sum(-1)


def test_two_tower_emits_embedding_alignment(tmp_path):
    model = _TwoTower()
    writer = RecordingWriter()
    rec = Recorder(model, run_dir=tmp_path, recipe="two_tower",
                   writer=writer, every_n_steps=1)
    rec.attach()
    q = torch.randn(8, 4)
    i = torch.randn(8, 4)
    _ = model(q, i)
    rec.step(0)
    rec.detach()
    tags = {t for t, _, _ in writer.scalars}
    assert any("embedding_alignment" in t for t in tags)


def test_two_tower_registered():
    r = get_recipe("two_tower")
    assert r.name == "two_tower"
    assert len(r.custom) >= 1
