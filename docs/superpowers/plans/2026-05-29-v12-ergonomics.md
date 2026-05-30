# v1.2.0 — Recorder & reporting ergonomics implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver the four v1.2.0 ergonomics components: `Recipe.disable`/`only`; per-family verdict/flag block in `build_report`; compact mode + `--compact` CLI flag; `circuitry compare` subcommand; hygiene fold-ins.

**Spec:** `docs/superpowers/specs/2026-05-29-v12-ergonomics-design.md`

**Tech stack:** Python 3.12, PyTorch, pytest, ruff. No new dependencies; no GPU/model downloads in tests.

**Environment:** full venv paths — `venv/bin/python`, `venv/bin/pytest`, `venv/bin/ruff`. Never `source venv/bin/activate`.

**CI invariants (must not break):**
- `tests/test_layering.py` — no new root packages in import closure.
- `tests/recorder/test_report.py` — all 6 tests stay green after T2 and T3.
- No `.cuda()` in `core/`; `core/` does not import `recorder/` or `recipes/`.

**Sequencing constraint:** T2 must complete before T3 and T4 start (both consume `_metrics.py`).
T3 and T4 each modify `report.py` and `cli/main.py` — they MUST NOT run in parallel.
T5 runs last (finalizes docs after all code is in).

---

## File inventory

| Action | Path |
| --- | --- |
| Modify | `src/circuitry/recipes/__init__.py` |
| Create | `src/circuitry/recorder/_metrics.py` |
| Modify | `src/circuitry/recorder/report.py` |
| Create | `src/circuitry/recorder/compare.py` |
| Modify | `src/circuitry/cli/main.py` |
| Create | `tests/recipes/test_recipe_helpers.py` |
| Create | `tests/recorder/test_metrics.py` |
| Create | `tests/recorder/test_report_flags.py` |
| Create | `tests/recorder/test_compare.py` |
| Modify | `docs/design.md` |
| Modify | `TODO.md` |
| Modify | `pyproject.toml` |
| Modify | `CHANGELOG.md` |

---

## Task 1 — `Recipe.disable(names)` / `Recipe.only(names)` + tests

**Independent.** No prereqs. Safe to start immediately.

**Files:**
- Modify: `src/circuitry/recipes/__init__.py` (after `with_sae`, ~line 70)
- Create: `tests/recipes/test_recipe_helpers.py`

### New symbols

```python
# src/circuitry/recipes/__init__.py

def disable(self, names: list[str]) -> Recipe:
    """Return a new Recipe with each name in *names* disabled.
    names must be a subset of weight_diagnostics + activation_diagnostics + gradient_diagnostics.
    Raises ValueError for unknown names. custom DiagnosticFn callables are unaffected.
    """

def only(self, names: list[str]) -> Recipe:
    """Return a new Recipe running ONLY the diagnostics in *names*.
    The complement of names within the three diagnostic lists is disabled.
    Raises ValueError for unknown names. custom DiagnosticFn callables are unaffected.
    """
```

Both use `dataclasses.replace(self, enabled={...})`, merging into `self.enabled` (latest-wins),
mirroring `with_prefix` (line 43) and `with_sae` (line 70) exactly.

### Implementation notes

- `_all = set(self.weight_diagnostics + self.activation_diagnostics + self.gradient_diagnostics)`
- `disable`: `new_enabled = {**self.enabled, **{n: False for n in names}}`
- `only`: `new_enabled = {**self.enabled, **{n: (n in set(names)) for n in _all}}`
- Fail-fast: `unknown = set(names) - _all; if unknown: raise ValueError(...)`
- `disable([])` and `only([])` are valid (no-op and disable-all respectively).

### Steps

- [ ] **Step 1: Write the failing tests**

```python
# tests/recipes/test_recipe_helpers.py
from __future__ import annotations
import pytest
from circuitry.recipes import get_recipe

# Use the stock "llm" recipe for all tests — no model needed.


def test_disable_sets_enabled_false():
    r = get_recipe("llm")
    r2 = r.disable(["effective_rank"])
    assert r2.enabled.get("effective_rank") is False
    # Other names should not appear as disabled (absent = default True via _enabled).
    assert r2.enabled.get("stable_rank", True) is True


def test_disable_empty_is_noop():
    r = get_recipe("llm")
    r2 = r.disable([])
    assert r2.enabled == r.enabled


def test_disable_unknown_raises():
    r = get_recipe("llm")
    with pytest.raises(ValueError, match="nonexistent_diag"):
        r.disable(["nonexistent_diag"])


def test_only_disables_complement():
    r = get_recipe("llm")
    r2 = r.only(["effective_rank"])
    assert r2.enabled.get("effective_rank") is True
    # Every other diagnostic in the three lists must be False.
    _all = set(r.weight_diagnostics + r.activation_diagnostics + r.gradient_diagnostics)
    for name in _all:
        if name != "effective_rank":
            assert r2.enabled.get(name) is False, f"{name} should be disabled"


def test_only_unknown_raises():
    r = get_recipe("llm")
    with pytest.raises(ValueError, match="no_such"):
        r.only(["no_such"])


def test_only_empty_disables_all():
    r = get_recipe("llm")
    r2 = r.only([])
    _all = set(r.weight_diagnostics + r.activation_diagnostics + r.gradient_diagnostics)
    for name in _all:
        assert r2.enabled.get(name) is False


def test_disable_then_only_composes():
    """Latest-wins: disable heavy_tail_alpha first, then only(["effective_rank"])
    should result in effective_rank=True and everything else False."""
    r = get_recipe("llm")
    r2 = r.disable(["heavy_tail_alpha"]).only(["effective_rank"])
    assert r2.enabled.get("effective_rank") is True
    _all = set(r.weight_diagnostics + r.activation_diagnostics + r.gradient_diagnostics)
    for name in _all:
        if name != "effective_rank":
            assert r2.enabled.get(name) is False


def test_custom_diagnostics_unaffected_by_disable(tmp_path):
    """Custom DiagnosticFn still fires even after disable() on all named diagnostics."""
    import torch
    import torch.nn as nn
    from circuitry import Recorder
    from circuitry.recipes import Recipe
    from circuitry.recorder.hooks import HookPoint, TensorSource
    from circuitry.writers.null import NullWriter

    fired = []

    def _custom(ctx):
        fired.append(ctx.step)
        return {"custom_val": 1.0}

    recipe = Recipe(
        name="__test_custom__",
        hook_points=[HookPoint(pattern=r".*", source=TensorSource.WEIGHT)],
        weight_diagnostics=["effective_rank"],
        custom=[_custom],
    ).disable(["effective_rank"])

    model = nn.Linear(4, 4)
    rec = Recorder(model, run_dir=tmp_path, recipe=recipe, writer=NullWriter(), every_n_steps=1)
    rec.attach()
    with torch.no_grad():
        _ = model(torch.randn(2, 4))
    rec.step(0)
    rec.detach()
    assert fired, "custom DiagnosticFn must fire even after disable()"
```

- [ ] **Step 2: Verify test fails**

```bash
venv/bin/pytest tests/recipes/test_recipe_helpers.py -v
```
Expected: all tests fail (AttributeError: Recipe has no `disable`).

- [ ] **Step 3: Add `disable` and `only` to `src/circuitry/recipes/__init__.py`**

Insert after `with_sae` (~line 70), before `_REGISTRY`:

```python
    def disable(self, names: list[str]) -> "Recipe":
        """Return a new Recipe with each name in *names* disabled.

        *names* must be a subset of the recipe's own
        ``weight_diagnostics + activation_diagnostics + gradient_diagnostics``.
        Raises ``ValueError`` for any name not present in those lists.

        Custom ``DiagnosticFn`` callables are not name-addressable and are
        unaffected by this helper.
        """
        _all = set(
            self.weight_diagnostics + self.activation_diagnostics + self.gradient_diagnostics
        )
        unknown = set(names) - _all
        if unknown:
            raise ValueError(
                f"Recipe {self.name!r}: unknown diagnostic name(s) {sorted(unknown)}. "
                f"Available: {sorted(_all)}"
            )
        new_enabled = {**self.enabled, **{n: False for n in names}}
        return dataclasses.replace(self, enabled=new_enabled)

    def only(self, names: list[str]) -> "Recipe":
        """Return a new Recipe running *only* the diagnostics in *names*.

        Every name in
        ``weight_diagnostics + activation_diagnostics + gradient_diagnostics``
        that is not in *names* is disabled. Raises ``ValueError`` for any name
        not present in those lists.

        Custom ``DiagnosticFn`` callables are unaffected.
        """
        _all = set(
            self.weight_diagnostics + self.activation_diagnostics + self.gradient_diagnostics
        )
        unknown = set(names) - _all
        if unknown:
            raise ValueError(
                f"Recipe {self.name!r}: unknown diagnostic name(s) {sorted(unknown)}. "
                f"Available: {sorted(_all)}"
            )
        new_enabled = {**self.enabled, **{n: (n in set(names)) for n in _all}}
        return dataclasses.replace(self, enabled=new_enabled)
```

- [ ] **Step 4: Run tests + full suite**

```bash
venv/bin/pytest tests/recipes/test_recipe_helpers.py -v
venv/bin/pytest -q
```
Expected: all new tests PASS; full suite green.

- [ ] **Step 5: Commit**

```bash
git add src/circuitry/recipes/__init__.py tests/recipes/test_recipe_helpers.py
git commit -m "$(cat <<'EOF'
feat(recipes): Recipe.disable() / Recipe.only() diagnostic selection helpers

Immutable helpers (dataclasses.replace) mirroring with_prefix/with_sae idiom.
Universe derived from recipe's own three diagnostic lists; fail-fast ValueError
on unknown names; custom DiagnosticFn callables unaffected.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

**Acceptance gate:** `venv/bin/pytest -q` green; `venv/bin/ruff check src/circuitry/recipes/__init__.py` clean.

---

## Task 2 — Mechanical refactor: extract `_group`/`_stats` into `recorder/_metrics.py`

**Prereq for T3 and T4.** Must complete before either starts.
**Constraint:** T3 and T4 both touch `report.py`; run T2 first, then T3, then T4 sequentially.

**Files:**
- Create: `src/circuitry/recorder/_metrics.py`
- Modify: `src/circuitry/recorder/report.py` (replace local `_group`/`_stats` with imports)
- Create: `tests/recorder/test_metrics.py`

### New module: `src/circuitry/recorder/_metrics.py`

```python
"""Shared grouping/stats helpers for report.py and compare.py (private).

Do not import this module from outside recorder/. Not part of the public API.
"""
from __future__ import annotations

import json
import pathlib
from collections import defaultdict


def group(rows: list[dict]) -> dict[str, list[tuple[int, float]]]:
    """Group JSONL scalar rows by tag; sort each series by step."""
    by_tag: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for r in rows:
        if r.get("kind") != "scalar":
            continue
        by_tag[r["tag"]].append((int(r["step"]), float(r["value"])))
    for v in by_tag.values():
        v.sort()
    return by_tag


def stats(series: list[tuple[int, float]]) -> tuple[float, float, float, float, float]:
    """Return (first, last, vmin, vmax, delta) over a sorted time series."""
    vals = [v for _, v in series]
    vmin, vmax = min(vals), max(vals)
    return vals[0], vals[-1], vmin, vmax, vmax - vmin


def load_rows(run_dir: pathlib.Path) -> list[dict]:
    """Load JSONL rows from run_dir/metrics.jsonl; return [] if absent."""
    p = run_dir / "metrics.jsonl"
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
```

### `report.py` change

Replace lines 40-70 (the local `_group` and `_stats` definitions) with:

```python
from circuitry.recorder._metrics import group as _group, stats as _stats
```

Every internal call site (`_group(rows)`, `_stats(series)`) is unchanged in name.

### Steps

- [ ] **Step 1: Write the failing unit tests**

```python
# tests/recorder/test_metrics.py
from circuitry.recorder._metrics import group, stats


def test_group_sorts_by_step():
    rows = [
        {"tag": "a/b", "value": 2.0, "step": 1, "kind": "scalar"},
        {"tag": "a/b", "value": 1.0, "step": 0, "kind": "scalar"},
    ]
    g = group(rows)
    assert g["a/b"] == [(0, 1.0), (1, 2.0)]


def test_group_ignores_non_scalar():
    rows = [
        {"tag": "a/b", "value": 1.0, "step": 0, "kind": "scalar"},
        {"tag": "a/b", "step": 1, "kind": "histogram"},
    ]
    g = group(rows)
    assert len(g["a/b"]) == 1


def test_stats_single_point():
    first, last, vmin, vmax, delta = stats([(0, 5.0)])
    assert first == last == vmin == vmax == 5.0
    assert delta == 0.0


def test_stats_two_points():
    first, last, vmin, vmax, delta = stats([(0, 2.0), (1, 5.0)])
    assert first == 2.0
    assert last == 5.0
    assert delta == 3.0
```

- [ ] **Step 2: Verify tests fail** (module doesn't exist yet)

```bash
venv/bin/pytest tests/recorder/test_metrics.py -v
```

- [ ] **Step 3: Create `_metrics.py`** (exact body above)

- [ ] **Step 4: Update `report.py`** — remove `_group` (lines 40-48) and `_stats` (lines 66-70) and add import line.

- [ ] **Step 5: Run all recorder tests**

```bash
venv/bin/pytest tests/recorder/ -q
```
Expected: all existing `test_report.py` tests + new `test_metrics.py` tests PASS.

- [ ] **Step 6: Run full suite + layering**

```bash
venv/bin/pytest -q
venv/bin/pytest tests/test_layering.py -v
```
Expected: all green.

- [ ] **Step 7: Commit**

```bash
git add src/circuitry/recorder/_metrics.py src/circuitry/recorder/report.py \
        tests/recorder/test_metrics.py
git commit -m "$(cat <<'EOF'
refactor(recorder): extract _group/_stats into shared recorder/_metrics.py

Mechanical refactor — no behavior change. Prereq for compare.py (T4) which
needs the same grouping/stats helpers without duplicating them.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

**Acceptance gate:** `venv/bin/pytest tests/recorder/ -q` green; `test_report.py` assertions byte-identical.

---

## Task 3 — Verdict/flag block + compact mode + `--compact` CLI flag + tests

**Prereq: T2 complete.** Must run AFTER T2; must NOT overlap with T4 (both touch report.py + cli/main.py).

**Files:**
- Modify: `src/circuitry/recorder/report.py`
- Modify: `src/circuitry/cli/main.py`
- Create: `tests/recorder/test_report_flags.py`

### New symbols in `report.py`

```python
# Module-level constant (place after HERO_SECTIONS, before GRAD_PER_PARAM_TOP_K)
FLAG_RULES: list[tuple[str, str, ...]] = [
    ("activation/dead_fraction",  "dead_rising",
     lambda last, d: d > 0 and last > 0.05,
     "dead_fraction rising (last={last:.3f}, Δ={d:+.4g})"),
    ("weight/effective_rank",     "rank_collapsing",
     lambda last, d: d < 0 and last < 10.0,
     "effective_rank collapsing (last={last:.2f}, Δ={d:+.4g})"),
    ("grad/global",               "grad_norm_spiking",
     lambda last, d: d > 0 and last > 10.0,
     "grad_norm spiking (last={last:.4g}, Δ={d:+.4g})"),
    ("weight/attention_head_rank","attn_rank_low",
     lambda last, d: last < 2.0,
     "attention_head_rank critically low (last={last:.2f})"),
]

def _build_flags(grouped: dict, step_count: int) -> list[str]:
    """Return markdown lines for the ## Flags block, or [] if step_count <= 1."""
```

### `build_report` signature change

```python
def build_report(
    run_dir: str | pathlib.Path,
    out_path: str | pathlib.Path | None = None,
    *,
    compact: bool = False,
) -> pathlib.Path:
```

Internal logic:
1. After the `## Summary` block is appended (current line 190-203 area), call
   `_build_flags(grouped, step_count)` and append the result.
2. When `compact=True`, after appending flags, write the file immediately (skip
   matched_modules, attach_summary, section tables, advanced details).

### `cli/main.py` change

```python
# In _cmd_report:
out = build_report(run_dir=args.run, out_path=args.out, compact=args.compact)

# In main(), subparser for 'report':
p_report.add_argument("--compact", action="store_true", default=False,
                       help="emit only Summary + Flags, suppress per-tag tables")
```

### Steps

- [ ] **Step 1: Write the failing tests**

```python
# tests/recorder/test_report_flags.py
from __future__ import annotations
import json
import pathlib
import pytest
from circuitry.recorder.report import build_report


def _write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows))


def test_flags_suppressed_single_step(tmp_path):
    _write(tmp_path / "metrics.jsonl", [
        {"tag": "activation/dead_fraction/mod", "value": 0.2, "step": 0, "kind": "scalar"},
    ])
    out = tmp_path / "report.md"
    build_report(tmp_path, out)
    assert "## Flags" not in out.read_text()


def test_flags_fires_dead_fraction(tmp_path):
    _write(tmp_path / "metrics.jsonl", [
        {"tag": "activation/dead_fraction/mod", "value": 0.01, "step": 0, "kind": "scalar"},
        {"tag": "activation/dead_fraction/mod", "value": 0.15, "step": 1, "kind": "scalar"},
    ])
    out = tmp_path / "report.md"
    build_report(tmp_path, out)
    md = out.read_text()
    assert "## Flags" in md
    assert "dead_rising" in md


def test_no_false_flags_flat(tmp_path):
    _write(tmp_path / "metrics.jsonl", [
        {"tag": "activation/dead_fraction/mod", "value": 0.10, "step": 0, "kind": "scalar"},
        {"tag": "activation/dead_fraction/mod", "value": 0.10, "step": 1, "kind": "scalar"},
    ])
    out = tmp_path / "report.md"
    build_report(tmp_path, out)
    md = out.read_text()
    assert "## Flags" in md
    assert "dead_rising" not in md
    assert "no flags" in md.lower()


def test_compact_omits_tables(tmp_path):
    _write(tmp_path / "metrics.jsonl", [
        {"tag": "weight/effective_rank/mod", "value": 8.0, "step": 0, "kind": "scalar"},
        {"tag": "weight/effective_rank/mod", "value": 7.0, "step": 1, "kind": "scalar"},
    ])
    out = tmp_path / "report.md"
    build_report(tmp_path, out, compact=True)
    md = out.read_text()
    assert "## Summary" in md
    assert "## weight/effective_rank" not in md


def test_compact_includes_flags(tmp_path):
    _write(tmp_path / "metrics.jsonl", [
        {"tag": "activation/dead_fraction/mod", "value": 0.01, "step": 0, "kind": "scalar"},
        {"tag": "activation/dead_fraction/mod", "value": 0.20, "step": 1, "kind": "scalar"},
    ])
    out = tmp_path / "report.md"
    build_report(tmp_path, out, compact=True)
    md = out.read_text()
    assert "## Summary" in md
    assert "## Flags" in md
    assert "dead_rising" in md
```

- [ ] **Step 2: Verify tests fail**

```bash
venv/bin/pytest tests/recorder/test_report_flags.py -v
```

- [ ] **Step 3: Implement `FLAG_RULES` + `_build_flags` in `report.py`**

- [ ] **Step 4: Implement `compact` param in `build_report`**

- [ ] **Step 5: Update `cli/main.py`** — add `--compact` arg + pass to `build_report`.

- [ ] **Step 6: Run all recorder + CLI tests**

```bash
venv/bin/pytest tests/recorder/ tests/test_cli.py -q
venv/bin/pytest tests/recorder/test_report.py -v   # must still pass (regression check)
```

- [ ] **Step 7: Commit**

```bash
git add src/circuitry/recorder/report.py src/circuitry/cli/main.py \
        tests/recorder/test_report_flags.py
git commit -m "$(cat <<'EOF'
feat(recorder): per-family verdict/flag block + compact mode in build_report

FLAG_RULES declarative table; ## Flags block gated on step_count > 1.
compact=True suppresses per-tag tables. --compact CLI flag added to report subcommand.
Default output retains all existing sections unchanged.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

**Acceptance gate:** `venv/bin/pytest tests/recorder/ tests/test_cli.py -q` all green.

---

## Task 4 — `compare.py` module + `circuitry compare` CLI subcommand + tests

**Prereq: T2 complete.** Must run AFTER T3 (also touches `cli/main.py`). Do NOT start while T3 is in progress.

**Files:**
- Create: `src/circuitry/recorder/compare.py`
- Modify: `src/circuitry/cli/main.py`
- Create: `tests/recorder/test_compare.py`

### New symbols in `recorder/compare.py`

```python
# src/circuitry/recorder/compare.py
from __future__ import annotations
import dataclasses
import json
import math
import pathlib
from circuitry.recorder._metrics import group, stats, load_rows


@dataclasses.dataclass
class FamilyDelta:
    section: str          # "weight/effective_rank"
    last_a: float         # mean last-value across all tags in this section, run_a (NaN if absent)
    last_b: float         # mean last-value, run_b
    delta: float          # last_b - last_a
    trend_a: str          # "up" | "down" | "flat"
    trend_b: str
    trend_agrees: bool


def _trend(delta: float) -> str:
    if delta > 0:
        return "up"
    if delta < 0:
        return "down"
    return "flat"


def compare_runs(
    run_a: str | pathlib.Path,
    run_b: str | pathlib.Path,
) -> list[FamilyDelta]:
    """Compare two runs at family/diagnostic (first two tag segments) granularity.

    Reads metrics.jsonl from each run. Returns one FamilyDelta per section
    present in either run. Missing sections produce NaN last values.

    Granularity is section (first two tag segments), NOT per-module.
    Per-module comparison is ill-posed across different architectures.
    """
    ...


def build_compare_report(
    run_a: str | pathlib.Path,
    run_b: str | pathlib.Path,
    out_path: str | pathlib.Path | None = None,
) -> pathlib.Path:
    """Write a markdown compare report. Default out_path: run_a/../compare.md."""
    ...
```

### `_section_and_row` reuse

`compare.py` derives section via `section, _ = _section_and_row(tag)` — import
`_section_and_row` from `report.py` directly (it is already in `report.py` at line 51 and
is a pure function with no side effects). Alternatively, inline the identical one-liner in
`_metrics.py`. The simpler choice: import from `report.py` (already a peer module).

### CLI additions in `cli/main.py`

```python
from circuitry.recorder.compare import build_compare_report


def _cmd_compare(args: argparse.Namespace) -> int:
    out = build_compare_report(run_a=args.run_a, run_b=args.run_b, out_path=args.out)
    print(f"wrote {out}")
    return 0


# In main(), after the 'report' subparser:
p_compare = sub.add_parser("compare", help="compare two runs at family/diagnostic granularity")
p_compare.add_argument("--run-a", required=True, dest="run_a", help="first run directory")
p_compare.add_argument("--run-b", required=True, dest="run_b", help="second run directory")
p_compare.add_argument("--out", default=None, help="output path (default: run_a/../compare.md)")
p_compare.set_defaults(func=_cmd_compare)
```

### Steps

- [ ] **Step 1: Write the failing tests**

```python
# tests/recorder/test_compare.py
from __future__ import annotations
import json
import math
import pathlib
import pytest
from circuitry.recorder.compare import compare_runs, build_compare_report, FamilyDelta


def _write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows))


def test_compare_single_family_delta(tmp_path):
    run_a = tmp_path / "a"
    run_b = tmp_path / "b"
    _write(run_a / "metrics.jsonl", [
        {"tag": "activation/dead_fraction/mod", "value": 0.10, "step": 0, "kind": "scalar"},
        {"tag": "activation/dead_fraction/mod", "value": 0.12, "step": 1, "kind": "scalar"},
    ])
    _write(run_b / "metrics.jsonl", [
        {"tag": "activation/dead_fraction/mod", "value": 0.10, "step": 0, "kind": "scalar"},
        {"tag": "activation/dead_fraction/mod", "value": 0.20, "step": 1, "kind": "scalar"},
    ])
    deltas = compare_runs(run_a, run_b)
    assert len(deltas) == 1
    fd = deltas[0]
    assert fd.section == "activation/dead_fraction"
    assert abs(fd.delta - (0.20 - 0.12)) < 1e-6


def test_compare_missing_family_in_one_run(tmp_path):
    run_a = tmp_path / "a"
    run_b = tmp_path / "b"
    _write(run_a / "metrics.jsonl", [
        {"tag": "weight/effective_rank/mod", "value": 10.0, "step": 0, "kind": "scalar"},
    ])
    _write(run_b / "metrics.jsonl", [])
    deltas = compare_runs(run_a, run_b)
    assert len(deltas) == 1
    assert math.isnan(deltas[0].last_b)


def test_compare_trend_agreement(tmp_path):
    run_a = tmp_path / "a"
    run_b = tmp_path / "b"
    # both rising
    for rdir, v0, v1 in [(run_a, 1.0, 2.0), (run_b, 3.0, 4.0)]:
        _write(rdir / "metrics.jsonl", [
            {"tag": "weight/effective_rank/m", "value": v0, "step": 0, "kind": "scalar"},
            {"tag": "weight/effective_rank/m", "value": v1, "step": 1, "kind": "scalar"},
        ])
    deltas = compare_runs(run_a, run_b)
    assert deltas[0].trend_a == "up"
    assert deltas[0].trend_b == "up"
    assert deltas[0].trend_agrees is True


def test_compare_trend_disagree(tmp_path):
    run_a = tmp_path / "a"
    run_b = tmp_path / "b"
    _write(run_a / "metrics.jsonl", [
        {"tag": "weight/effective_rank/m", "value": 1.0, "step": 0, "kind": "scalar"},
        {"tag": "weight/effective_rank/m", "value": 2.0, "step": 1, "kind": "scalar"},
    ])
    _write(run_b / "metrics.jsonl", [
        {"tag": "weight/effective_rank/m", "value": 4.0, "step": 0, "kind": "scalar"},
        {"tag": "weight/effective_rank/m", "value": 3.0, "step": 1, "kind": "scalar"},
    ])
    deltas = compare_runs(run_a, run_b)
    assert deltas[0].trend_a == "up"
    assert deltas[0].trend_b == "down"
    assert deltas[0].trend_agrees is False


def test_build_compare_report_writes_markdown(tmp_path):
    run_a = tmp_path / "a"
    run_b = tmp_path / "b"
    for rdir in (run_a, run_b):
        _write(rdir / "metrics.jsonl", [
            {"tag": "weight/effective_rank/m", "value": 8.0, "step": 0, "kind": "scalar"},
        ])
    out = tmp_path / "compare.md"
    build_compare_report(run_a, run_b, out_path=out)
    md = out.read_text()
    assert "# circuitry compare" in md
    assert str(run_a) in md
    assert str(run_b) in md


def test_cli_compare_subcommand(tmp_path):
    run_a = tmp_path / "a"
    run_b = tmp_path / "b"
    for rdir in (run_a, run_b):
        _write(rdir / "metrics.jsonl", [
            {"tag": "activation/dead_fraction/m", "value": 0.05, "step": 0, "kind": "scalar"},
        ])
    from circuitry.cli.main import main
    out = tmp_path / "compare.md"
    rc = main(["compare", "--run-a", str(run_a), "--run-b", str(run_b), "--out", str(out)])
    assert rc == 0
    assert out.exists()
```

- [ ] **Step 2: Verify tests fail** (module absent)

```bash
venv/bin/pytest tests/recorder/test_compare.py -v
```

- [ ] **Step 3: Implement `src/circuitry/recorder/compare.py`** (full body per spec §6)

- [ ] **Step 4: Add `compare` subcommand to `cli/main.py`**

- [ ] **Step 5: Run compare + CLI tests**

```bash
venv/bin/pytest tests/recorder/test_compare.py tests/test_cli.py -v
```

- [ ] **Step 6: Run full suite + layering**

```bash
venv/bin/pytest -q
venv/bin/pytest tests/test_layering.py -v
```

- [ ] **Step 7: Commit**

```bash
git add src/circuitry/recorder/compare.py src/circuitry/cli/main.py \
        tests/recorder/test_compare.py
git commit -m "$(cat <<'EOF'
feat(recorder): compare_runs / build_compare_report + circuitry compare CLI

Per-family/diagnostic (not per-module) last-value deltas and trend comparison
between two runs. Reads metrics.jsonl via shared _metrics helpers.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

**Acceptance gate:** `venv/bin/pytest -q` all green; `venv/bin/pytest tests/test_layering.py -v` passes.

---

## Task 5 — Hygiene fold-in: docs + pyproject + TODO + CHANGELOG

**Prereq: T1–T4 complete.** Run last.

**Files:**
- Modify: `docs/design.md`
- Modify: `TODO.md`
- Modify: `pyproject.toml`
- Modify: `CHANGELOG.md`

### `docs/design.md` changes

1. **Header** — bump `**Last updated:** 2026-05-21` → `**Last updated:** 2026-05-29`.

2. **§4.2** — add `compact: bool = False` to `build_report` call block; add
   `compare_runs` / `build_compare_report` to the `from circuitry import ...` block.

3. **§4.3 (CLI block)** — add:
   ```
   circuitry compare --run-a runs/run_a --run-b runs/run_b [--out compare.md]
   circuitry report  --run runs/my_run [--compact]
   ```
   Fix the stale `findings.json` note: change
   `"report accepts either a live metrics.jsonl … or a retrospective findings.json produced by scan"`
   → `"report accepts a metrics.jsonl written by either the live Recorder (via JsonlWriter) or scan_run"`.

4. **§4.4 Recipe code block** — add `enabled: dict[str, bool] = field(default_factory=dict)`
   (currently missing from the example block, though present in source since before v1.0).
   Add two lines of prose after `with_sae`:
   > Use `Recipe.disable(names)` to disable specific diagnostics by name, or
   > `Recipe.only(names)` to run only those named diagnostics. Both return a new `Recipe` via
   > `dataclasses.replace`; both raise `ValueError` if a name is not in the recipe's three
   > diagnostic lists. `custom` `DiagnosticFn` callables are not name-addressable and are
   > unaffected.

### `TODO.md` changes

1. Header line 3: `Released through v0.9.2` → `Released through v1.1.0`.

2. Tick both Ergonomics items (lines 53-60):
   - `[ ] **[feat] No easy way to disable / select a single diagnostic.**` → `[x]`, add `Done (v1.2.0) — Recipe.disable()/only() immutable helpers.`
   - `[ ] **[feat] report is a flat dump with no summary; no A/B compare.**` → `[x]`, add `Done (v1.2.0) — ## Flags verdict block (step_count > 1 gated), compact mode, circuitry compare subcommand.`

### `pyproject.toml` change

Line 15: `"Development Status :: 3 - Alpha"` → `"Development Status :: 4 - Beta"`

### `CHANGELOG.md` entry

Insert at top, after the `# Changelog` header and before `## [1.1.0]`:

```markdown
## [1.2.0] — 2026-05-29

### Added
- `Recipe.disable(names)` — return a new `Recipe` with the named diagnostics disabled
  (raises `ValueError` on unknown names; `custom` callables unaffected).
- `Recipe.only(names)` — return a new `Recipe` running only the named diagnostics.
- `build_report` per-family verdict/flag block (`## Flags`): declarative `FLAG_RULES`
  table produces flags for dead_rising, rank_collapsing, grad_norm_spiking, attn_rank_low.
  Gated on `step_count > 1` to avoid false alarms on static/single-step runs.
- `build_report(compact=True)` — renders only `## Summary` + `## Flags`, suppressing
  per-tag tables. Default output unchanged.
- `circuitry report --compact` CLI flag.
- `circuitry compare --run-a A --run-b B [--out path]` — per-family/diagnostic last-value
  deltas (b−a) and trend comparison between two runs. Reads `metrics.jsonl` from each run.
  Per-module comparison is intentionally excluded (module-name mismatch across architectures).
- `circuitry.recorder.compare` public module (`compare_runs`, `build_compare_report`,
  `FamilyDelta`).
```

### Steps

- [ ] **Step 1: Apply all doc edits** (design.md, TODO.md, pyproject.toml, CHANGELOG.md).

- [ ] **Step 2: Run full suite**

```bash
venv/bin/pytest -q
venv/bin/ruff check src/circuitry/
```
Expected: all green + ruff clean.

- [ ] **Step 3: Commit**

```bash
git add docs/design.md TODO.md pyproject.toml CHANGELOG.md
git commit -m "$(cat <<'EOF'
chore(release): v1.2.0 — Recorder & reporting ergonomics

design.md: §4.2/4.3/4.4 updated for disable/only, compare, compact, flags;
findings.json stale ref fixed; Last updated bumped to 2026-05-29.
TODO.md: ergonomics items ticked; header updated to v1.1.0.
pyproject.toml: classifier 3 - Alpha -> 4 - Beta.
CHANGELOG.md: [1.2.0] entry.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

**Acceptance gate:** `venv/bin/pytest -q` green; `venv/bin/ruff check src/circuitry/` clean; `git log --oneline -1` shows the hygiene commit.

---

## Self-Review checklist

- **Spec coverage:** T1 → spec §3; T2 → spec §6 (shared module); T3 → spec §4 + §5; T4 → spec §6 (compare); T5 → spec §7.
- **Sequencing:** T1 is independent. T2 is prereq for T3 and T4. T3 must complete before T4 (shared `cli/main.py`). T5 is last.
- **No regression:** every task ends with `venv/bin/pytest -q`; T3/T4 also run `test_report.py` explicitly.
- **Layering:** no new root packages; `compare.py` imports only `recorder/_metrics` and stdlib; `cli/main.py` imports `recorder/compare` (same pattern as existing `recorder/report` import).
- **Type/name consistency:** `FamilyDelta` in `compare.py`; `FLAG_RULES`/`_build_flags` in `report.py`; `group`/`stats`/`load_rows` in `_metrics.py`; `disable`/`only` on `Recipe`.
