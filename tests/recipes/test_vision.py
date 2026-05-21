from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from circuitry.recipes import _clear_registry_for_tests, get_recipe
from circuitry.recipes.vision import register
from circuitry.recorder.live import Recorder
from circuitry.writers.base import RecordingWriter


@pytest.fixture(autouse=True)
def _clean():
    _clear_registry_for_tests()
    register()
    yield
    _clear_registry_for_tests()


class _TinyResNetBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 8, 3, padding=1)
        self.fc1 = nn.Linear(8 * 8 * 8, 10)

    def forward(self, x):
        x = self.conv2(self.conv1(x))
        return self.fc1(x.flatten(1))


def test_vision_recipe_matches_conv_and_fc(tmp_path):
    model = _TinyResNetBlock()
    writer = RecordingWriter()
    rec = Recorder(model, run_dir=tmp_path, recipe="vision",
                   writer=writer, every_n_steps=1)
    rec.attach()
    _ = model(torch.randn(2, 3, 8, 8))
    rec.step(0)
    rec.detach()
    tags = {t for t, _, _ in writer.scalars}
    assert any("conv" in t for t in tags)
    assert any("fc" in t for t in tags)


def test_vision_recipe_is_registered():
    assert "vision" in [r for r in [get_recipe("vision").name]]
