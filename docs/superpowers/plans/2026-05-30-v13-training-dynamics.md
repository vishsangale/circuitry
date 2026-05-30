# v1.3.0 — Training-dynamics diagnostics implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire three shipped-but-unwired cross-step weight-dynamics primitives
(`update_delta`, `direction_cosine`, `rank_trajectory`) into the live Recorder and the
`llm` recipe; add the cross-step snapshot holder to `Recorder`; surface new flag rules in
`build_report`; update `docs/design.md`.

**Spec:** `docs/superpowers/specs/2026-05-30-v13-training-dynamics-design.md`

**Tech stack:** Python 3.12, PyTorch, pytest, ruff. No new dependencies; no GPU/model downloads in tests.

**Environment:** full venv paths — `venv/bin/python`, `venv/bin/pytest`, `venv/bin/ruff`. Never `source venv/bin/activate`.

**CI invariants (must not break):**
- `tests/test_layering.py` — no new root packages.
- All existing `tests/recorder/test_report.py` tests stay green.
- No `.cuda()` in `core/`; `core/` does not import `recorder/` or `recipes/`.

**Sequencing:**
- T1 (snapshot holder + dispatch) is the critical prereq — must complete before T2 and T3.
- T2 (recipe wiring) and T3 (report surfacing) are independent of each other; can run in
  parallel **after T1**, but both touch different files so no conflict.
- T4 (docs + hygiene) runs last, after T1–T3 complete.

---

## File inventory

| Action | Path |
| --- | --- |
| Modify | `src/circuitry/recorder/live.py` |
| Modify | `src/circuitry/recipes/llm.py` |
| Modify | `src/circuitry/recorder/report.py` |
| Modify | `docs/design.md` |
| Modify | `CHANGELOG.md` |
| Create | `tests/recorder/test_live_snapshot.py` |
| Create | `tests/recipes/test_llm_dynamics.py` |

---

## Task 1 — Cross-step snapshot holder + weight-dynamics dispatch + tests

**Independent. No prereqs. Start immediately. All other tasks depend on this.**

**Files:**
- Modify: `src/circuitry/recorder/live.py`
- Create: `tests/recorder/test_live_snapshot.py`

### New state on `Recorder.__init__`

Add after `self._current_step: int = -1` (current `live.py:154`):

```python
# Cross-step weight snapshots for training-dynamics diagnostics (v1.3).
# Detached CPU copies of ctx.weights from prior emit steps.
# Empty at attach(); populated after each emit; cleared in detach().
self._prev_weights: dict[str, torch.Tensor] = {}
self._prev_prev_weights: dict[str, torch.Tensor] = {}
```

### New import at top of `live.py`

After the existing `from circuitry.core import weight as _w` (line ~19):

```python
from circuitry.core import spectral as _spectral
```

### Snapshot roll in `step()` — after `_run_diagnostics`

Current `live.py:585-587`:
```python
        self._run_diagnostics(ctx)
        # Discard activations now that we've consumed them.
        self._captured_activations.clear()
```

Replace with:
```python
        self._run_diagnostics(ctx)
        # Discard activations now that we've consumed them.
        self._captured_activations.clear()
        # Roll cross-step weight snapshots forward (v1.3 training-dynamics).
        self._prev_prev_weights = self._prev_weights
        self._prev_weights = {
            name: t.detach().cpu() for name, t in ctx.weights.items()
        }
```

### Snapshot clear in `detach()`

In `detach()` (current `live.py:522-530`), after `self._hook_handles.clear()`:

```python
        # Release cross-step snapshots to free RAM (v1.3).
        self._prev_weights.clear()
        self._prev_prev_weights.clear()
```

### Dispatch branches in `_run_diagnostics`

Add three `elif` branches inside `for name in self.recipe.weight_diagnostics:`, after the
`elif name == "sv_histogram":` block (current `live.py:647`), before the `else: fn = _WEIGHT_DIAGS.get(name)` fallthrough:

```python
            elif name == "update_delta":
                if not self._prev_weights:
                    continue  # first emit step — no prior snapshot yet
                deltas = _w.update_delta(ctx.weights, self._prev_weights)
                for mod_name, val in deltas.items():
                    self._writer.add_scalar(
                        f"weight/update_delta/{mod_name}", val, ctx.step
                    )

            elif name == "direction_cosine":
                if not self._prev_weights or not self._prev_prev_weights:
                    continue  # need two prior snapshots; skip first two emit steps
                cosines = _w.direction_cosine(
                    ctx.weights, self._prev_weights, self._prev_prev_weights
                )
                for mod_name, val in cosines.items():
                    self._writer.add_scalar(
                        f"weight/direction_cosine/{mod_name}", val, ctx.step
                    )

            elif name == "rank_trajectory":
                if not self._prev_weights:
                    continue  # first emit step
                # Reuse the per-step SVD cache (_sv) to avoid a redundant SVD.
                # Equivalent to rank_trajectory([prev, now])[-1] but zero extra SVD cost.
                for mod_name, w in ctx.weights.items():
                    rank_now = _w._effective_rank_from_sv(_sv(w))
                    self._writer.add_scalar(
                        f"weight/rank_trajectory/{mod_name}", rank_now, ctx.step
                    )
```

Note: `_sv` is the local closure defined at `live.py:602` (the SVD cache). The `rank_trajectory`
branch accesses it via closure — it is in scope since it's defined earlier in the same
`_run_diagnostics` method.

### Tests — `tests/recorder/test_live_snapshot.py`

```python
"""Tests for the cross-step snapshot holder and weight-dynamics dispatch (v1.3)."""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from circuitry.recorder.live import Recorder
from circuitry.recipes import Recipe
from circuitry.recorder.hooks import HookPoint, TensorSource
from circuitry.writers.null import NullWriter


def _make_recorder(tmp_path, weight_diagnostics, every_n_steps=1):
    model = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 4))
    recipe = Recipe(
        name="__test_snapshot__",
        hook_points=[HookPoint(pattern=r".*", source=TensorSource.WEIGHT)],
        weight_diagnostics=weight_diagnostics,
        activation_diagnostics=[],
        gradient_diagnostics=[],
    )
    rec = Recorder(
        model, run_dir=tmp_path, recipe=recipe,
        writer=NullWriter(), every_n_steps=every_n_steps,
    )
    return model, rec


def _one_step(model, rec, step):
    with torch.no_grad():
        _ = model(torch.randn(2, 8))
    rec.step(step)


# ── Snapshot lifecycle ──────────────────────────────────────────────────────


def test_prev_weights_empty_before_first_emit(tmp_path):
    model, rec = _make_recorder(tmp_path, ["update_delta"])
    rec.attach()
    assert rec._prev_weights == {}
    rec.detach()


def test_prev_weights_populated_after_first_emit(tmp_path):
    model, rec = _make_recorder(tmp_path, ["update_delta"])
    rec.attach()
    _one_step(model, rec, step=0)
    assert len(rec._prev_weights) > 0
    rec.detach()


def test_prev_prev_populated_after_second_emit(tmp_path):
    model, rec = _make_recorder(tmp_path, ["direction_cosine"])
    rec.attach()
    _one_step(model, rec, step=0)
    assert rec._prev_prev_weights == {}
    _one_step(model, rec, step=1)
    assert len(rec._prev_prev_weights) > 0
    rec.detach()


def test_detach_clears_snapshots(tmp_path):
    model, rec = _make_recorder(tmp_path, ["update_delta"])
    rec.attach()
    _one_step(model, rec, step=0)
    _one_step(model, rec, step=1)
    rec.detach()
    assert rec._prev_weights == {}
    assert rec._prev_prev_weights == {}


def test_snapshot_is_cpu_detached(tmp_path):
    model, rec = _make_recorder(tmp_path, ["update_delta"])
    rec.attach()
    _one_step(model, rec, step=0)
    for t in rec._prev_weights.values():
        assert t.device.type == "cpu"
        assert not t.requires_grad
    rec.detach()


# ── First-step guard — no diagnostics emitted on step 0 ──────────────────


class _RecordingWriter:
    def __init__(self):
        self.scalars: list[tuple[str, float, int]] = []
    def add_scalar(self, tag, value, step): self.scalars.append((tag, value, step))
    def add_histogram(self, *a, **k): pass
    def add_image(self, *a, **k): pass
    def add_text(self, *a, **k): pass
    def flush(self): pass
    def close(self): pass


def _make_recorder_recording(tmp_path, weight_diagnostics):
    model = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 4))
    recipe = Recipe(
        name="__test_dyn__",
        hook_points=[HookPoint(pattern=r".*", source=TensorSource.WEIGHT)],
        weight_diagnostics=weight_diagnostics,
        activation_diagnostics=[],
        gradient_diagnostics=[],
    )
    writer = _RecordingWriter()
    rec = Recorder(
        model, run_dir=tmp_path, recipe=recipe,
        writer=writer, every_n_steps=1,
    )
    return model, rec, writer


def test_no_update_delta_on_first_step(tmp_path):
    model, rec, writer = _make_recorder_recording(tmp_path, ["update_delta"])
    rec.attach()
    _one_step(model, rec, step=0)
    tags = [t for t, _, _ in writer.scalars]
    assert not any("update_delta" in t for t in tags)
    rec.detach()


def test_no_direction_cosine_on_first_two_steps(tmp_path):
    model, rec, writer = _make_recorder_recording(tmp_path, ["direction_cosine"])
    rec.attach()
    _one_step(model, rec, step=0)
    _one_step(model, rec, step=1)
    tags = [t for t, _, _ in writer.scalars]
    assert not any("direction_cosine" in t for t in tags)
    rec.detach()


def test_no_rank_trajectory_on_first_step(tmp_path):
    model, rec, writer = _make_recorder_recording(tmp_path, ["rank_trajectory"])
    rec.attach()
    _one_step(model, rec, step=0)
    tags = [t for t, _, _ in writer.scalars]
    assert not any("rank_trajectory" in t for t in tags)
    rec.detach()


# ── Emission after sufficient steps ──────────────────────────────────────


def test_update_delta_emitted_on_second_step(tmp_path):
    model, rec, writer = _make_recorder_recording(tmp_path, ["update_delta"])
    rec.attach()
    _one_step(model, rec, step=0)
    _one_step(model, rec, step=1)
    tags = [t for t, _, _ in writer.scalars]
    assert any("weight/update_delta/" in t for t in tags)
    # Values must be non-negative (L2 norm)
    for t, v, _ in writer.scalars:
        if "update_delta" in t:
            assert v >= 0.0
    rec.detach()


def test_direction_cosine_emitted_on_third_step(tmp_path):
    model, rec, writer = _make_recorder_recording(tmp_path, ["direction_cosine"])
    rec.attach()
    for s in range(3):
        _one_step(model, rec, step=s)
    tags = [t for t, _, _ in writer.scalars]
    assert any("weight/direction_cosine/" in t for t in tags)
    # Cosine in [-1, 1]
    for t, v, _ in writer.scalars:
        if "direction_cosine" in t:
            assert -1.0 - 1e-6 <= v <= 1.0 + 1e-6
    rec.detach()


def test_rank_trajectory_emitted_on_second_step(tmp_path):
    model, rec, writer = _make_recorder_recording(tmp_path, ["rank_trajectory"])
    rec.attach()
    _one_step(model, rec, step=0)
    _one_step(model, rec, step=1)
    tags = [t for t, _, _ in writer.scalars]
    assert any("weight/rank_trajectory/" in t for t in tags)
    # Effective rank must be positive
    for t, v, _ in writer.scalars:
        if "rank_trajectory" in t:
            assert v > 0.0
    rec.detach()
```

### Steps

- [ ] **Step 1: Write the failing tests**

```bash
venv/bin/pytest tests/recorder/test_live_snapshot.py -v
```
Expected: `AttributeError` — `Recorder` has no `_prev_weights`.

- [ ] **Step 2: Add `_prev_weights` / `_prev_prev_weights` to `Recorder.__init__`**

   Insert after `self._current_step: int = -1` in `live.py`.

- [ ] **Step 3: Add `from circuitry.core import spectral as _spectral` import**

   After existing core imports in `live.py`.

- [ ] **Step 4: Add snapshot roll to `step()`**

   After `_captured_activations.clear()` in `live.py:587` area.

- [ ] **Step 5: Add snapshot clear to `detach()`**

   After `self._hook_handles.clear()` in `live.py:523` area.

- [ ] **Step 6: Add three dispatch branches to `_run_diagnostics`**

   After `elif name == "sv_histogram":` block (`live.py:647` area), before the `else: fn = _WEIGHT_DIAGS.get(name)` fallthrough.

- [ ] **Step 7: Run snapshot tests**

```bash
venv/bin/pytest tests/recorder/test_live_snapshot.py -v
```
Expected: all PASS.

- [ ] **Step 8: Run full suite + layering**

```bash
venv/bin/pytest -q
venv/bin/pytest tests/test_layering.py -v
```
Expected: all green.

- [ ] **Step 9: Commit**

```bash
git add src/circuitry/recorder/live.py tests/recorder/test_live_snapshot.py
git commit -m "$(cat <<'EOF'
feat(recorder): cross-step weight snapshot holder + update_delta/direction_cosine/rank_trajectory dispatch

Adds _prev_weights/_prev_prev_weights to Recorder (detached CPU, matched-modules only).
Rolled forward after each emit step; cleared in detach().
First-step/second-step guards suppress cross-step diagnostics until enough snapshots exist.
Wires weight.update_delta, weight.direction_cosine, spectral.rank_trajectory (via SVD cache)
into _run_diagnostics as named special cases parallel to sv_histogram/attention_head_rank.
Tags: weight/update_delta/<module>, weight/direction_cosine/<module>, weight/rank_trajectory/<module>.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

**Acceptance gate:** `venv/bin/pytest -q` green; `venv/bin/pytest tests/test_layering.py -v` passes; `venv/bin/ruff check src/circuitry/recorder/live.py` clean.

---

## Task 2 — `llm` recipe: add three new weight-diagnostic names

**Prereq: T1 complete.**

**Files:**
- Modify: `src/circuitry/recipes/llm.py`
- Create: `tests/recipes/test_llm_dynamics.py`

### Change to `llm.py`

In `RECIPE`, append to `weight_diagnostics` list (current `llm.py:36`):

```python
weight_diagnostics=[
    "effective_rank", "attention_head_rank", "stable_rank",
    "heavy_tail_alpha", "sv_histogram",
    # v1.3 training-dynamics:
    "update_delta", "rank_trajectory", "direction_cosine",
],
```

No other changes to `llm.py`. No new `HookPoint` entries needed — existing WEIGHT hooks
already capture all matched weight matrices per emit step.

### New symbols / tag names

| String added to recipe | Tag emitted (after 2+ emit steps) |
| --- | --- |
| `"update_delta"` | `weight/update_delta/<module_name>` |
| `"rank_trajectory"` | `weight/rank_trajectory/<module_name>` |
| `"direction_cosine"` | `weight/direction_cosine/<module_name>` (after 3+ emit steps) |

### Tests — `tests/recipes/test_llm_dynamics.py`

```python
"""End-to-end test: llm recipe emits v1.3 training-dynamics tags on a toy transformer."""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from circuitry.recorder.live import Recorder
from circuitry.recipes.llm import RECIPE
from circuitry.writers.null import NullWriter


class _RecordingWriter:
    def __init__(self):
        self.scalars: list[tuple[str, float, int]] = []
    def add_scalar(self, tag, value, step): self.scalars.append((tag, value, step))
    def add_histogram(self, *a, **k): pass
    def add_image(self, *a, **k): pass
    def add_text(self, *a, **k): pass
    def flush(self): pass
    def close(self): pass


class _TinyAttn(nn.Module):
    """Minimal attention layer with qkv / o projections."""
    def __init__(self, d=16):
        super().__init__()
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.o_proj = nn.Linear(d, d, bias=False)
    def forward(self, x):
        return self.o_proj(self.v_proj(x) + self.q_proj(x) + self.k_proj(x))


class _TinyMLP(nn.Module):
    def __init__(self, d=16):
        super().__init__()
        self.gate_proj = nn.Linear(d, d * 4, bias=False)
        self.down_proj = nn.Linear(d * 4, d, bias=False)
    def forward(self, x):
        return self.down_proj(torch.relu(self.gate_proj(x)))


class _TinyLayer(nn.Module):
    def __init__(self, d=16):
        super().__init__()
        self.self_attn = _TinyAttn(d)
        self.mlp = _TinyMLP(d)
    def forward(self, x):
        return self.mlp(self.self_attn(x) + x)


class _TinyLM(nn.Module):
    def __init__(self, d=16, n_layers=2):
        super().__init__()
        self.layers = nn.ModuleList([_TinyLayer(d) for _ in range(n_layers)])
    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x


def _run_recorder(tmp_path, n_steps=3):
    model = _TinyLM()
    writer = _RecordingWriter()
    rec = Recorder(
        model, run_dir=tmp_path, recipe=RECIPE,
        writer=writer, every_n_steps=1, strict=False,
    )
    rec.attach()
    for s in range(n_steps):
        with torch.no_grad():
            _ = model(torch.randn(2, 8, 16))
        rec.step(s)
    rec.detach()
    return writer


def test_llm_recipe_emits_update_delta(tmp_path):
    writer = _run_recorder(tmp_path)
    tags = {t for t, _, _ in writer.scalars}
    assert any("weight/update_delta/" in t for t in tags), \
        f"Expected update_delta tags; got: {sorted(tags)[:10]}"


def test_llm_recipe_emits_rank_trajectory(tmp_path):
    writer = _run_recorder(tmp_path)
    tags = {t for t, _, _ in writer.scalars}
    assert any("weight/rank_trajectory/" in t for t in tags)


def test_llm_recipe_emits_direction_cosine(tmp_path):
    # direction_cosine needs 3 emit steps (prev_prev populated after step 1)
    writer = _run_recorder(tmp_path, n_steps=3)
    tags = {t for t, _, _ in writer.scalars}
    assert any("weight/direction_cosine/" in t for t in tags)


def test_update_delta_nonneg(tmp_path):
    writer = _run_recorder(tmp_path)
    for t, v, _ in writer.scalars:
        if "weight/update_delta/" in t:
            assert v >= 0.0


def test_direction_cosine_in_range(tmp_path):
    writer = _run_recorder(tmp_path, n_steps=3)
    for t, v, _ in writer.scalars:
        if "weight/direction_cosine/" in t:
            assert -1.0 - 1e-5 <= v <= 1.0 + 1e-5
```

### Steps

- [ ] **Step 1: Write the failing tests**

```bash
venv/bin/pytest tests/recipes/test_llm_dynamics.py -v
```
Expected: `update_delta` / `rank_trajectory` / `direction_cosine` tags absent → assertions fail.

- [ ] **Step 2: Add three names to `llm.py` `weight_diagnostics`**

- [ ] **Step 3: Run recipe tests + full suite**

```bash
venv/bin/pytest tests/recipes/test_llm_dynamics.py -v
venv/bin/pytest -q
```
Expected: all PASS.

- [ ] **Step 4: Commit**

```bash
git add src/circuitry/recipes/llm.py tests/recipes/test_llm_dynamics.py
git commit -m "$(cat <<'EOF'
feat(recipes): add update_delta / rank_trajectory / direction_cosine to llm recipe

Three v1.3 weight-dynamics diagnostics wired into the stock llm recipe.
No new hook points needed — existing WEIGHT hooks capture the required tensors.
First-step guards in live.py suppress emission until snapshots exist.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

**Acceptance gate:** `venv/bin/pytest -q` green; `venv/bin/ruff check src/circuitry/recipes/llm.py` clean.

---

## Task 3 — Report surfacing: `HERO_SECTIONS` + new `FLAG_RULES`

**Prereq: T1 complete. Independent of T2.**

**Files:**
- Modify: `src/circuitry/recorder/report.py`
- Extend: `tests/recorder/test_report_flags.py` (existing file from v1.2)

### Changes to `report.py`

**1. `HERO_SECTIONS` — add three new entries** (after existing `"activation/sae"` entry,
current `report.py:39`):

```python
HERO_SECTIONS = frozenset({
    "weight/effective_rank",
    "weight/attention_head_rank",
    "activation/dead_fraction",
    "activation/gate_stats",
    "grad/global",
    "activation/logit_lens_kl",
    "activation/induction_score",
    "activation/attention_pattern_entropy",
    "activation/sae",
    # v1.3 training-dynamics:
    "weight/update_delta",
    "weight/rank_trajectory",
    "weight/direction_cosine",
})
```

**2. `FLAG_RULES` — add three new rules** (append after the `"attn_rank_low"` entry,
current `report.py:65-72`):

```python
    (
        "weight/rank_trajectory",
        "rank_collapse_trend",
        lambda last, signed: signed < -1.0 and last < 8.0,
        "rank_trajectory declining (last={last:.2f}, Δ={signed:+.4g})",
    ),
    (
        "weight/update_delta",
        "update_delta_vanishing",
        lambda last, signed: last < 1e-6,
        "update_delta near zero — possible gradient vanishing (last={last:.2g})",
    ),
    (
        "weight/direction_cosine",
        "direction_reversal",
        lambda last, signed: last < -0.5,
        "direction_cosine strongly negative — update direction reversal (last={last:.3f})",
    ),
```

Recall: in `FLAG_RULES` predicates, `signed` = `last - first` (the signed intra-run trend),
NOT the unsigned range delta. This matches the comment at `report.py:47`.

### New flag tests — add to `tests/recorder/test_report_flags.py`

```python
def test_rank_collapse_trend_flag(tmp_path):
    """Declining rank_trajectory below threshold fires rank_collapse_trend."""
    _write(tmp_path / "metrics.jsonl", [
        {"tag": "weight/rank_trajectory/mod", "value": 12.0, "step": 0, "kind": "scalar"},
        {"tag": "weight/rank_trajectory/mod", "value": 6.0,  "step": 1, "kind": "scalar"},
    ])
    out = tmp_path / "report.md"
    build_report(tmp_path, out)
    md = out.read_text()
    assert "## Flags" in md
    assert "rank_collapse_trend" in md


def test_update_delta_vanishing_flag(tmp_path):
    """Near-zero update_delta fires update_delta_vanishing."""
    _write(tmp_path / "metrics.jsonl", [
        {"tag": "weight/update_delta/mod", "value": 1e-8, "step": 0, "kind": "scalar"},
        {"tag": "weight/update_delta/mod", "value": 5e-9, "step": 1, "kind": "scalar"},
    ])
    out = tmp_path / "report.md"
    build_report(tmp_path, out)
    md = out.read_text()
    assert "update_delta_vanishing" in md


def test_direction_reversal_flag(tmp_path):
    """Strongly negative direction_cosine fires direction_reversal."""
    _write(tmp_path / "metrics.jsonl", [
        {"tag": "weight/direction_cosine/mod", "value": 0.1,  "step": 0, "kind": "scalar"},
        {"tag": "weight/direction_cosine/mod", "value": -0.8, "step": 1, "kind": "scalar"},
    ])
    out = tmp_path / "report.md"
    build_report(tmp_path, out)
    md = out.read_text()
    assert "direction_reversal" in md


def test_dynamics_tags_in_hero_sections(tmp_path):
    """weight/update_delta and rank_trajectory render in hero (not in <details>)."""
    _write(tmp_path / "metrics.jsonl", [
        {"tag": "weight/update_delta/mod",    "value": 0.1, "step": 0, "kind": "scalar"},
        {"tag": "weight/rank_trajectory/mod", "value": 8.0, "step": 0, "kind": "scalar"},
    ])
    out = tmp_path / "report.md"
    build_report(tmp_path, out)
    md = out.read_text()
    # Hero sections appear before <details>; advanced sections inside <details>.
    assert "## weight/update_delta" in md
    assert "## weight/rank_trajectory" in md
    details_start = md.find("<details>")
    update_delta_start = md.find("## weight/update_delta")
    rank_traj_start = md.find("## weight/rank_trajectory")
    # Sections should appear before <details> (or <details> absent entirely)
    if details_start >= 0:
        assert update_delta_start < details_start
        assert rank_traj_start < details_start
```

### Steps

- [ ] **Step 1: Write the failing tests** (append to existing `test_report_flags.py`)

```bash
venv/bin/pytest tests/recorder/test_report_flags.py -v -k "rank_collapse or vanishing or reversal or hero"
```
Expected: new tests fail (flags not present); existing tests still pass.

- [ ] **Step 2: Add three entries to `HERO_SECTIONS` in `report.py`**

- [ ] **Step 3: Add three entries to `FLAG_RULES` in `report.py`**

- [ ] **Step 4: Run all reporter tests**

```bash
venv/bin/pytest tests/recorder/test_report_flags.py tests/recorder/test_report.py -v
```
Expected: all PASS (existing tests unbroken).

- [ ] **Step 5: Run full suite**

```bash
venv/bin/pytest -q
```

- [ ] **Step 6: Commit**

```bash
git add src/circuitry/recorder/report.py tests/recorder/test_report_flags.py
git commit -m "$(cat <<'EOF'
feat(recorder): surface v1.3 training-dynamics families in report (HERO_SECTIONS + FLAG_RULES)

Adds weight/update_delta, weight/rank_trajectory, weight/direction_cosine to HERO_SECTIONS
so they appear above <details> in the report.
New FLAG_RULES: rank_collapse_trend, update_delta_vanishing, direction_reversal.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

**Acceptance gate:** `venv/bin/pytest tests/recorder/ -q` green; `venv/bin/ruff check src/circuitry/recorder/report.py` clean.

---

## Task 4 — Hygiene fold-in: docs + CHANGELOG

**Prereq: T1, T2, T3 complete. Run last.**

**Files:**
- Modify: `docs/design.md`
- Modify: `CHANGELOG.md`

### `docs/design.md` changes

**1. Header** — bump `**Last updated:** 2026-05-30` (already at this date; confirm and leave).

**2. §4.1 — primitive catalog additions**

Under `# weight-space`, after `weight.attention_head_rank(...)`, add:

```python
# weight-space dynamics (v1.3 — shipped in core/; now wired live)
weight.update_delta(sd_now: Mapping[str, Tensor], sd_prev: Mapping[str, Tensor]) -> dict[str, float]
weight.direction_cosine(sd_now: Mapping[str, Tensor], sd_prev: Mapping[str, Tensor],
                        sd_prev_prev: Mapping[str, Tensor]) -> dict[str, float]
```

Under `# spectral`, add a note after `spectral.rank_trajectory(...)`:
```
# Note: rank_trajectory is now wired live in the Recorder (v1.3) via the SVD cache.
```

**3. §4.4 — `Recipe` and hook escape hatches**

After the prose for `Recipe.only(names)` (v1.2), add:

> The `Recorder` maintains two internal CPU weight snapshots (`_prev_weights`,
> `_prev_prev_weights`) to support cross-step weight-dynamics primitives. Both are
> empty at `attach()`, populated (detached CPU copies of matched-module weight tensors)
> after each emit step, and cleared in `detach()`. The cross-step diagnostics
> `update_delta`, `direction_cosine`, and `rank_trajectory` silently skip emission on
> the first emit step (or first two, for `direction_cosine`) until enough snapshots
> exist. This is the only internal recorder state change in v1.3; `StepContext` shape
> is unchanged.

**4. §5 — Recipe internals worked example**

Update the `weight_diagnostics` line in the `RECIPE` block to include the three new names:

```python
weight_diagnostics=["effective_rank", "attention_head_rank", "stable_rank",
                    "heavy_tail_alpha", "sv_histogram",
                    "update_delta", "rank_trajectory", "direction_cosine"],
```

### `CHANGELOG.md` entry

Insert at top, after `# Changelog` header and before `## [1.2.0]`:

```markdown
## [1.3.0] — 2026-05-30

### Added
- Cross-step weight snapshot holder in `Recorder` (`_prev_weights`, `_prev_prev_weights`):
  detached CPU copies of matched-module weights from prior emit steps. Populated after each
  emit; cleared in `detach()`. First-step and second-step guards suppress cross-step
  diagnostics until sufficient history exists.
- `weight/update_delta/<module>` live diagnostic: L2 norm of per-module weight delta between
  consecutive emit steps. Wires the existing `core/weight.update_delta` primitive.
- `weight/direction_cosine/<module>` live diagnostic: cosine similarity between two
  consecutive parameter update vectors. Wires `core/weight.direction_cosine`. Emits from
  the third emit step onward (requires two prior snapshots).
- `weight/rank_trajectory/<module>` live diagnostic: effective rank of each weight matrix at
  each emit step, reusing the per-step SVD cache (zero extra SVD cost). Wires the logic of
  `core/spectral.rank_trajectory` inline via the SVD cache.
- `llm` recipe now includes `update_delta`, `rank_trajectory`, `direction_cosine` in
  `weight_diagnostics`.
- `build_report`: `HERO_SECTIONS` extended with `weight/update_delta`,
  `weight/rank_trajectory`, `weight/direction_cosine` (rendered above `<details>`).
- `build_report`: three new `FLAG_RULES` — `rank_collapse_trend` (declining rank_trajectory),
  `update_delta_vanishing` (near-zero update norm), `direction_reversal` (strongly negative
  cosine).

### Documentation
- `docs/design.md §4.1`: added `update_delta` and `direction_cosine` to the public primitive
  catalog; added note that `rank_trajectory` is now wired live.
- `docs/design.md §4.4`: added prose on the cross-step snapshot lifecycle.
- `docs/design.md §5`: updated llm recipe example `weight_diagnostics` list.

### Deferred (Option B — representational drift probe)
- A probe-based representational drift primitive (CKA / cosine of activations vs a stored
  reference) is explicitly deferred to v1.3.1 or v1.4. It requires a second forward pass
  per emit step and must be opt-in (default off) due to the §10 GPU budget constraint.
  The cross-step snapshot infrastructure shipped in this release is the foundation it reuses.
```

### Steps

- [ ] **Step 1: Apply all `design.md` amendments**

- [ ] **Step 2: Write `CHANGELOG.md` entry** (insert at top after header)

- [ ] **Step 3: Run full suite + ruff**

```bash
venv/bin/pytest -q
venv/bin/ruff check src/circuitry/
```
Expected: all green + ruff clean.

- [ ] **Step 4: Commit**

```bash
git add docs/design.md CHANGELOG.md
git commit -m "$(cat <<'EOF'
chore(release): v1.3.0 — training-dynamics diagnostics

design.md: §4.1 catalog adds update_delta/direction_cosine/rank_trajectory note;
§4.4 adds cross-step snapshot lifecycle prose; §5 llm recipe example updated.
CHANGELOG.md: [1.3.0] entry with deferred Option B note.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

**Acceptance gate:** `venv/bin/pytest -q` green; `venv/bin/ruff check src/circuitry/` clean; `git log --oneline -4` shows all four task commits.

---

## Deferred: Option B — representational drift probe

The following work is explicitly out of scope for v1.3.0. It is recorded here so the
design intent is not lost.

**What it is:** A new pure `core` primitive (tentatively `activation.drift_score` or
`core/drift.py:drift_score`) measuring cosine similarity or CKA between activations on a
fixed probe batch now vs a stored reference from an earlier step. Measures representational
geometry change over training — complementary to the weight-space dynamics in Option A.

**Why deferred:**
- Requires a second forward pass per emit step on a probe batch.
- The §10 GPU wall-clock budget is already at +12.66% (roadmap); a mandatory second forward
  would exceed budget at default settings.
- Needs a non-trivial lifecycle: probe batch storage, reference activation capture, a
  `reset_reference()` API, a `max_tokens` cap for the quadratic CKA computation.
- No confirmed consumer in the current user base.

**Implementation path (when ready):**
1. New primitive: `core/activation.py` or `core/drift.py` — `drift_score(acts_now, acts_ref,
   method="cosine"|"cka") -> float`. Pure function; CKA needs a `max_samples` guard.
2. New `Recipe` field: `probe_batch: Tensor | None = None`.
3. New `Recorder` path in `step()`: if `recipe.probe_batch` set and `"drift_probe"` in
   `activation_diagnostics` and `self._enabled("drift_probe")`: run `model(probe_batch)` in
   `torch.no_grad()`, capture hooked activations, compare against `self._reference_acts`.
4. `_reference_acts` captured on first emit or at `attach()` via one extra forward pass.
5. `enabled["drift_probe"] = False` in all stock recipes (opt-in only).
6. Performance budget must be re-benchmarked; document expected overhead in README.

The `_prev_weights`/`_prev_prev_weights` snapshot infrastructure from Task 1 is not
directly reused by Option B, but the design pattern (RAII-clean CPU state, first-step
guard, `detach()` cleanup) is the template to follow.

---

## Self-Review checklist

- **Spec coverage:** T1 → spec §3 (snapshot holder) + §4 (dispatch); T2 → spec §5 (recipe);
  T3 → spec §7 (report surfacing); T4 → spec §6 (§4.1 catalog) + §13 (design.md amendments).
- **Sequencing:** T1 prereq for T2 and T3; T2 and T3 independent of each other; T4 last.
- **No regression:** every task ends with `venv/bin/pytest -q`; T3 explicitly runs `test_report.py`.
- **Layering:** only `spectral` import added to `live.py` (already imports other `core` modules);
  `llm.py` changes are string additions only; `report.py` adds entries to two existing data
  structures; no new root packages.
- **First-step guards:** explicitly tested — `test_no_update_delta_on_first_step`,
  `test_no_direction_cosine_on_first_two_steps`, `test_no_rank_trajectory_on_first_step`.
- **SVD reuse:** `rank_trajectory` branch uses `_sv_cache` via the `_sv(w)` closure — zero
  extra SVD cost confirmed by design (spec §9).
- **CPU-only:** all tests use `NullWriter`/`_RecordingWriter`; no `.cuda()` calls; no downloads.
- **`detach()` cleanup:** tested by `test_detach_clears_snapshots` and
  `test_snapshot_is_cpu_detached`.
