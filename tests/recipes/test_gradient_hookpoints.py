"""recsys #5 (+ systemic): a recipe that declares gradient_diagnostics must ship
at least one TensorSource.GRAD hook point.

The recorder populates ``ctx.gradients`` ONLY from GRAD hook points (live.py
step()); `norms_per_param` / `grad_norm_per_module` read `ctx.gradients`. So a
recipe that lists a gradient diagnostic but no GRAD hook point silently emits
zero gradient tags. The SASRec evaluation hit this on `recsys`; `two_tower` and
`vision` had the same latent gap (only `llm` wired a GRAD hook). The guard test
below enforces the invariant for every stock recipe.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from circuitry.recipes import (
    _register_stock_recipes,
    get_recipe,
    list_recipes,
)
from circuitry.recorder.hooks import TensorSource
from circuitry.recorder.live import Recorder


@pytest.fixture(autouse=True)
def _ensure_stock_recipes():
    _register_stock_recipes()
    yield


@pytest.mark.parametrize("name", ["llm", "vision", "two_tower", "recsys"])
def test_recipe_with_gradient_diagnostics_has_grad_hookpoint(name):
    recipe = get_recipe(name)
    if not recipe.gradient_diagnostics:
        pytest.skip(f"{name!r} declares no gradient diagnostics")
    assert any(hp.source is TensorSource.GRAD for hp in recipe.hook_points), (
        f"recipe {name!r} declares gradient_diagnostics "
        f"{recipe.gradient_diagnostics} but has no TensorSource.GRAD hook point — "
        f"ctx.gradients stays empty and gradient tags never emit"
    )


def test_all_stock_recipes_obey_the_grad_invariant():
    """Belt-and-suspenders over the whole registry (catches future recipes)."""
    offenders = []
    for name in list_recipes():
        r = get_recipe(name)
        if r.gradient_diagnostics and not any(
            hp.source is TensorSource.GRAD for hp in r.hook_points
        ):
            offenders.append(name)
    assert not offenders, f"recipes declare gradient diagnostics with no GRAD hook: {offenders}"


class _TinySeqRec(nn.Module):
    """Minimal sequential recommender matching recsys WEIGHT/GRAD patterns:
    encoder.item_emb (embedding anchor) + encoder.ffn.{0,2} (FFN linears)."""

    def __init__(self, n_items: int = 16, d: int = 8) -> None:
        super().__init__()

        class _Enc(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.item_emb = nn.Embedding(n_items, d)
                self.ffn = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))

        self.encoder = _Enc()
        self.out = nn.Linear(d, n_items)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        x = self.encoder.ffn(self.encoder.item_emb(idx))
        return self.out(x)


def test_recsys_emits_gradient_tags_end_to_end(tmp_path):
    """Attach recsys, run forward+backward+step, and assert grad tags land in
    metrics.jsonl. Pre-fix (no GRAD hook) this file contained no grad tags."""
    model = _TinySeqRec()
    rec = Recorder(model, run_dir=tmp_path, recipe="recsys",
                   writer="jsonl", every_n_steps=1, strict=True)
    rec.attach()
    try:
        idx = torch.randint(0, 16, (2, 5))
        logits = model(idx)
        loss = logits.float().pow(2).mean()
        loss.backward()
        rec.step(0, loss=float(loss.detach()))
    finally:
        rec.detach()

    content = (tmp_path / "metrics.jsonl").read_text()
    assert "grad/per_param/" in content
    assert "grad/global/total_norm" in content
