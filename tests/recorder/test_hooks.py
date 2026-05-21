from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from circuitry.recorder.hooks import (
    HookPoint,
    StepContext,
    TensorSource,
    match_modules,
)


def _toy() -> nn.Module:
    return nn.Sequential(
        nn.Linear(4, 8),  # 0
        nn.ReLU(),        # 1
        nn.Linear(8, 4),  # 2
    )


def test_hookpoint_requires_exactly_one_target():
    with pytest.raises(ValueError):
        HookPoint(source=TensorSource.WEIGHT)
    with pytest.raises(ValueError):
        HookPoint(source=TensorSource.WEIGHT, pattern=r".*", modules=[nn.Linear(2, 2)])


def test_match_modules_by_pattern():
    model = _toy()
    hp = HookPoint(source=TensorSource.WEIGHT, pattern=r"^\d+$")
    names = match_modules(model, hp)
    assert set(names) == {"0", "1", "2"}


def test_match_modules_by_explicit_instance():
    model = _toy()
    linear = model[0]
    hp = HookPoint(source=TensorSource.WEIGHT, modules=[linear])
    names = match_modules(model, hp)
    assert names == ["0"]


def test_match_modules_by_selector():
    model = _toy()
    hp = HookPoint(
        source=TensorSource.WEIGHT,
        selector=lambda m: [name for name, _ in m.named_modules() if "Linear" in type(_).__name__],
    )
    names = match_modules(model, hp)
    assert set(names) == {"0", "2"}


def test_step_context_holds_dicts():
    ctx = StepContext(step=5, model=_toy(), activations={}, gradients={}, weights={},
                      loss=0.5, user={"epoch": 1})
    assert ctx.step == 5
    assert ctx.user["epoch"] == 1
