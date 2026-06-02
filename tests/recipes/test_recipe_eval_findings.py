"""Regression tests for the v1.7 real-model evaluation findings in ``recipes/``.

Source: ``docs/observations/2026-05-31-real-model-evaluation.md`` (F33).

RED under current code; flips GREEN once fixed. Marked ``xfail(strict=True)`` so the
suite stays green today. Reproduce with ``--runxfail``.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from circuitry.recipes import _clear_registry_for_tests, get_recipe
from circuitry.recipes.llm import register as _register_llm
from circuitry.recipes.two_tower import register
from circuitry.recorder.live import Recorder
from circuitry.writers.base import RecordingWriter


@pytest.fixture(autouse=True)
def _clean():
    _clear_registry_for_tests()
    register()
    yield
    _clear_registry_for_tests()


# ---------------------------------------------------------------------------
# F9 — condition_number is in core._WEIGHT_DIAGS and design §4.1 but absent
#      from the llm recipe's weight_diagnostics, so default-llm users get no
#      condition_number rows at all. (Once F1 makes condition_number accurate,
#      it should be a default weight diagnostic.) Fix: add it to the recipe, or
#      document the exclusion.
# ---------------------------------------------------------------------------


def test_F9_llm_recipe_includes_condition_number():
    _register_llm()
    recipe = get_recipe("llm")
    assert "condition_number" in recipe.weight_diagnostics, (
        f"condition_number missing from llm recipe weight_diagnostics: "
        f"{recipe.weight_diagnostics}"
    )


# ---------------------------------------------------------------------------
# F33 — embedding_alignment silently returns {} when item_tower is an
#       nn.ModuleList. The two_tower OUTPUT hook matches the `item_tower`
#       module, but a forward hook on an nn.ModuleList never fires (it has no
#       forward), so the item-tower activation is never captured and the
#       custom diagnostic emits nothing — silently, with no warning.
#       Fix: register the output hook on each child, or doc the requirement.
# ---------------------------------------------------------------------------


class _TwoTowerModuleList(nn.Module):
    """A two-tower model whose item_tower is an nn.ModuleList of per-field towers
    (the standard recsys multi-embedding-table layout)."""

    def __init__(self) -> None:
        super().__init__()
        self.query_tower = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 4))
        self.item_tower = nn.ModuleList([nn.Linear(4, 4) for _ in range(2)])

    def forward(self, q, i):
        it = sum(t(i) for t in self.item_tower)
        return (self.query_tower(q) * it).sum(-1)


def test_F33_embedding_alignment_fires_with_modulelist_item_tower(tmp_path):
    model = _TwoTowerModuleList()
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
    assert any("embedding_alignment" in t for t in tags), (
        "embedding_alignment did not emit for an nn.ModuleList item_tower "
        "(the output hook never fired on the ModuleList)"
    )
