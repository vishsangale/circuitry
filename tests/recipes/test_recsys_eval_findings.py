"""Regression tests for the v1.7 real-model evaluation findings in ``recipes/two_tower``
related to recsys / DLRM architectures.

Source: ``docs/observations/2026-05-31-real-model-evaluation.md`` (F32, F34).
Reproduce with ``--runxfail``.

**F32** (🟠 recipes/two_tower): the two_tower recipe matches *0 modules* on a standard-named
DLRM (``embed_tables`` / ``bottom_mlp`` / ``top_mlp``).  The patterns are name-locked to
``query_tower|item_tower|interaction``.  No ``recsys`` / ``dlrm`` recipe exists.
Fix direction: add a flexible recsys/dlrm recipe (or broaden the two_tower patterns) that
covers the standard DLRM module names.

**F34** (🟡 recipes/two_tower): activation hooks fire only on the **top-level** ``item_tower``
container, not on the individual embedding tables (``item_tower.0``, ``item_tower.1``, …).
The output pattern is ``(query_tower|item_tower)$``; an ``nn.ModuleList`` forward hook
never fires for individual children.  Per-table ``dead_fraction`` / ``participation_ratio``
are therefore never emitted.
Fix direction: extend the output hook pattern to
``(query_tower|item_tower)(\\.\\ d+)?$`` (or equivalent) so per-child hooks fire.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from circuitry.recipes import _clear_registry_for_tests
from circuitry.recipes.two_tower import register
from circuitry.recorder.live import Recorder
from circuitry.writers.base import RecordingWriter

# ---------------------------------------------------------------------------
# Shared autouse fixture — clear and re-register two_tower for each test.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean():
    _clear_registry_for_tests()
    register()
    yield
    _clear_registry_for_tests()


# ---------------------------------------------------------------------------
# Synthetic models
# ---------------------------------------------------------------------------


class _DLRM(nn.Module):
    """Minimal DLRM using the **standard** Facebook-DLRM module names:
    ``embed_tables`` (ModuleList of Embeddings), ``bottom_mlp`` (Sequential),
    and ``top_mlp`` (Sequential).  No ``query_tower`` / ``item_tower`` names.

    Architecture:
      - 4 embedding tables  (vocab 100, dim 8)
      - bottom_mlp: Linear(4→8)→ReLU→Linear(8→8)
      - dot-product interaction (upper-triangle): 5×4/2 = 10 scalars
      - top_mlp:  Linear(18→8)→ReLU→Linear(8→1)
    """

    def __init__(self) -> None:
        super().__init__()
        self.embed_tables = nn.ModuleList([nn.Embedding(100, 8) for _ in range(4)])
        self.bottom_mlp = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 8))
        # 5 vectors (4 embed + 1 bottom) → 5*4/2 = 10 dot pairs + 8 bottom = 18
        self.top_mlp = nn.Sequential(nn.Linear(18, 8), nn.ReLU(), nn.Linear(8, 1))

    def forward(self, dense: torch.Tensor, cat_ids: torch.Tensor) -> torch.Tensor:
        embs = [self.embed_tables[i](cat_ids[:, i]) for i in range(4)]
        bottom = self.bottom_mlp(dense)
        all_vecs = embs + [bottom]
        T = torch.stack(all_vecs, dim=1)  # (B, 5, 8)
        dots = torch.bmm(T, T.transpose(1, 2))
        idx_i, idx_j = zip(*[(i, j) for i in range(5) for j in range(i)], strict=False)
        interact = dots[:, list(idx_i), list(idx_j)]  # (B, 10)
        feat = torch.cat([bottom, interact], dim=1)   # (B, 18)
        return self.top_mlp(feat).squeeze(-1)


class _TwoTowerTableList(nn.Module):
    """A two-tower model whose ``item_tower`` is an ``nn.ModuleList`` of per-field
    ``nn.Embedding`` tables — the standard recsys multi-embedding-table layout.

    ``query_tower`` is a normal ``nn.Sequential`` (matches the recipe).
    ``item_tower.0`` … ``item_tower.3`` are the per-field tables.
    """

    def __init__(self) -> None:
        super().__init__()
        self.query_tower = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 8))
        self.item_tower = nn.ModuleList([nn.Embedding(100, 8) for _ in range(4)])

    def forward(self, q: torch.Tensor, cat_ids: torch.Tensor) -> torch.Tensor:
        qt = self.query_tower(q)
        it = sum(self.item_tower[i](cat_ids[:, i]) for i in range(4))
        return (qt * it).sum(-1)


# ---------------------------------------------------------------------------
# F32 — bare DLRM with standard names gets 0 coverage from the two_tower recipe
# ---------------------------------------------------------------------------


def test_F32_dlrm_standard_names_coverage_nonzero(tmp_path):
    """Attach the recsys/dlrm recipe (or an extended two_tower) to a standard-named DLRM
    and assert that at least one weight or activation tag is emitted.

    Confirmed baseline: two_tower recipe emits **0** tags on this model today
    (both HookPoints matched 0 modules, strict=False skips them with a warning).
    """
    model = _DLRM()
    writer = RecordingWriter()
    rec = Recorder(
        model,
        run_dir=tmp_path,
        recipe="two_tower",
        writer=writer,
        every_n_steps=1,
        strict=False,  # avoid RuntimeError on 0-match; assert on captured count instead
    )
    rec.attach()
    dense = torch.randn(4, 4)
    cat_ids = torch.randint(0, 100, (4, 4))
    out = model(dense, cat_ids)
    out.sum().backward()
    rec.step(0)
    rec.detach()

    tags = {t for t, _, _ in writer.scalars}
    weight_or_act_tags = [
        t for t in tags
        if t.startswith("weight/") or t.startswith("activation/")
    ]
    assert len(weight_or_act_tags) > 0, (
        f"F32: two_tower recipe captured 0 weight/activation tags on a standard-named "
        f"DLRM (embed_tables/bottom_mlp/top_mlp). "
        f"All emitted tags: {sorted(tags)}"
    )


# ---------------------------------------------------------------------------
# F34 — per-table activation tags absent when item_tower is an nn.ModuleList
# ---------------------------------------------------------------------------


def test_F34_per_table_activation_tags_emitted(tmp_path):
    """Attach two_tower to a model whose item_tower is an nn.ModuleList of embedding
    tables and assert per-table activation tags are emitted (e.g. 'activation/dead_fraction/
    item_tower.0').

    Confirmed baseline today:
      - Emitted activation tags: ['activation/dead_fraction/query_tower',
          'activation/participation_ratio/query_tower']  (query side only).
      - No 'item_tower.0' / 'item_tower.1' / … activation tags present at all —
        the output hook matches only the top-level 'item_tower' (ModuleList), whose
        forward hook never fires.
    """
    model = _TwoTowerTableList()
    writer = RecordingWriter()
    rec = Recorder(
        model,
        run_dir=tmp_path,
        recipe="two_tower",
        writer=writer,
        every_n_steps=1,
        strict=False,  # item_tower ModuleList output hook matches 0 children
    )
    rec.attach()
    q = torch.randn(4, 4)
    cat_ids = torch.randint(0, 100, (4, 4))
    out = model(q, cat_ids)
    out.sum().backward()
    rec.step(0)
    rec.detach()

    tags = {t for t, _, _ in writer.scalars}
    per_table_act_tags = [
        t for t in tags
        if "item_tower." in t
        and any(k in t for k in ("dead_fraction", "participation_ratio"))
    ]
    assert len(per_table_act_tags) > 0, (
        f"F34: No per-table activation tags emitted for item_tower.N children. "
        f"Expected tags like 'activation/dead_fraction/item_tower.0'. "
        f"All emitted tags: {sorted(tags)}"
    )
