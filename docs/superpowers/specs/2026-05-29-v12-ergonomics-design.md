# v1.2.0 — Recorder & reporting ergonomics (design spec)

**Status:** approved (2026-05-29).
All claims are grounded against source files listed under "Evidence" per component.

---

## 1. Motivation

v1.1 shipped the patching pillar. The Recorder/reporting surface has two pain points
surfaced by the 2026-05-23 field report (tracked in `TODO.md` Ergonomics items):

1. Dropping or isolating a single diagnostic requires a raw `dataclasses.replace` on the
   recipe — there is no named helper.
2. `build_report` is a flat dump with no verdict, no compact view, and no way to diff
   two runs. Users re-write their own jsonl parsers for every A/B comparison.

v1.2 closes both items with four tightly-scoped additions.

---

## 2. Scope

| # | Component | Touches |
| --- | --- | --- |
| 1 | `Recipe.disable(names)` / `Recipe.only(names)` helpers | `recipes/__init__.py` |
| 2 | `build_report` per-family verdict/flag block | `recorder/report.py` |
| 3 | `build_report` compact mode + `--compact` CLI flag | `recorder/report.py`, `cli/main.py` |
| 4 | `circuitry compare run_a run_b` CLI subcommand | `recorder/compare.py` (new), `recorder/_metrics.py` (new), `recorder/report.py`, `cli/main.py` |
| 5 | Hygiene fold-in | `docs/design.md`, `TODO.md`, `pyproject.toml`, `CHANGELOG.md` |

**Non-goals:**
- Per-module compare (cross-architecture module-name mismatch makes it ill-posed; §5 below).
- Multi-seed aggregation in `compare` (possibly deferred to v1.3; §5 below).
- Any change to `Recorder`, `scan_run`, `MetricWriter`, or `core/`.

---

## 3. Component 1 — `Recipe.disable(names)` / `Recipe.only(names)`

### Grounding

- `src/circuitry/recipes/__init__.py:23` — `enabled: dict[str, bool] = field(default_factory=dict)` exists.
- `recipes/__init__.py:29-47` — `with_prefix` uses `dataclasses.replace`; `with_sae` (~line 49-70) same pattern. Both methods are the exact idiom to mirror.
- `recorder/live.py:589-590` — `_enabled(name)` checks `self.recipe.enabled.get(name, True)`.
  The toggle already gates weight (live.py:610), activation (live.py:662 area), and gradient
  (live.py:864) diagnostic loops.
- `recorder/live.py:880` — the `custom` dispatch is **not** gated by `_enabled`. Custom
  `DiagnosticFn` callables run unconditionally.

### API

```python
# src/circuitry/recipes/__init__.py  (add after `with_sae`)

def disable(self, names: list[str]) -> Recipe:
    """Return a new Recipe with each name in *names* disabled.

    ``names`` must be a subset of the recipe's own
    ``weight_diagnostics + activation_diagnostics + gradient_diagnostics``.
    Raises ``ValueError`` for any name not present in those lists.

    Custom ``DiagnosticFn`` callables in ``self.custom`` are not
    name-addressable and are unaffected by this helper.
    """
    _all = set(self.weight_diagnostics + self.activation_diagnostics + self.gradient_diagnostics)
    unknown = set(names) - _all
    if unknown:
        raise ValueError(
            f"Recipe {self.name!r}: unknown diagnostic name(s) {sorted(unknown)}. "
            f"Available: {sorted(_all)}"
        )
    new_enabled = {**self.enabled, **{n: False for n in names}}
    return dataclasses.replace(self, enabled=new_enabled)

def only(self, names: list[str]) -> Recipe:
    """Return a new Recipe running *only* the diagnostics in *names*.

    The complement (everything in
    ``weight_diagnostics + activation_diagnostics + gradient_diagnostics``
    not in *names*) is disabled. Raises ``ValueError`` for any name not
    present in those lists.

    Custom ``DiagnosticFn`` callables are unaffected.
    """
    _all = set(self.weight_diagnostics + self.activation_diagnostics + self.gradient_diagnostics)
    unknown = set(names) - _all
    if unknown:
        raise ValueError(
            f"Recipe {self.name!r}: unknown diagnostic name(s) {sorted(unknown)}. "
            f"Available: {sorted(_all)}"
        )
    new_enabled = {n: (n in names) for n in _all}
    return dataclasses.replace(self, enabled={**self.enabled, **new_enabled})
```

**Key decisions:**
- Both are **methods on `Recipe`**, not kwargs on `Recorder`. The field `enabled` already
  lives on `Recipe`; keeping selection logic there is consistent with `with_prefix`/`with_sae`.
- The "universe" for `only()` is derived from the recipe's own three lists at call-time, not
  a hardcoded set — forward-compatible as new diagnostics are added to stock recipes.
- `enabled` from an earlier `.disable()` / `.only()` call is merged (latest-wins) so chains
  like `r.disable(["heavy_tail_alpha"]).only(["effective_rank", ...])` compose correctly.
- Names absent from `enabled` default to True via `_enabled` at line 590 — no breakage for
  recipes that never call these helpers.
- `custom` list is **not** addressable. Documented explicitly in both docstrings.
- Fail-fast: unknown name → `ValueError` at construction, not silently at step time.

---

## 4. Component 2 — `build_report` per-family verdict/flag block

### Grounding

- `recorder/report.py:137` — `def build_report(run_dir, out_path=None)` signature.
- `report.py:40-48` — `_group(rows)` aggregates scalar rows into `dict[tag, list[(step, value)]]`.
- `report.py:66-70` — `_stats(series)` returns `(first, last, vmin, vmax, delta)`.
- `report.py:154` — `step_count` computed from grouped values.
- `report.py:190` — `lines.append("## Summary")` — the existing summary block. The new
  verdict block is **additive**, placed after `## Summary`, before the per-tag table sections.

### Declarative FLAG_RULES table

```python
# inside report.py, module-level constant

FLAG_RULES: list[tuple[str, str, Callable[[float, float], bool], str]] = [
    # (section_prefix, flag_label, predicate(last_val, delta), message)
    ("activation/dead_fraction",  "dead_rising",       lambda last, d: d > 0 and last > 0.05,
     "dead_fraction rising (last={last:.3f}, Δ={d:+.4g})"),
    ("weight/effective_rank",     "rank_collapsing",   lambda last, d: d < 0 and last < 10.0,
     "effective_rank collapsing (last={last:.2f}, Δ={d:+.4g})"),
    ("grad/global",               "grad_norm_spiking", lambda last, d: d > 0 and last > 10.0,
     "grad_norm spiking (last={last:.4g}, Δ={d:+.4g})"),
    ("weight/attention_head_rank","attn_rank_low",     lambda last, d: last < 2.0,
     "attention_head_rank critically low (last={last:.2f})"),
]
```

Each rule matches all tags whose section equals the prefix (first two `/`-delimited parts).
For each matching tag the predicate is evaluated using the tag's `last` value and `delta`
(from `_stats`). A flag fires if **any** tag in the family triggers the predicate.

**Step-count gate:** the entire verdict block is suppressed when `step_count <= 1`.
This prevents false alarms on single-step or static runs (a brand-new training run that
has only emitted once will have `delta == 0` for all tags, so the `d > 0` / `d < 0`
predicates are inert anyway, but the gate makes the intent explicit and future-proofs
against numeric noise from a single float).

### Rendered block (markdown)

```markdown
## Flags

| family | flag | detail |
| --- | --- | --- |
| activation/dead_fraction | dead_rising | dead_fraction rising (last=0.123, Δ=+0.045) |
```

If no flags fire the block still renders with a single row:
`| — | — | no flags |`

The block is inserted **after `## Summary`** (report.py:190 area) and **before** the
per-tag table sections (report.py:233 onward).

---

## 5. Component 3 — compact mode

### Grounding

- `report.py:137` — `build_report` currently has two params: `run_dir`, `out_path`.
- `cli/main.py:19-21` — `_cmd_report` calls `build_report(run_dir=args.run, out_path=args.out)`.

### API change

```python
def build_report(
    run_dir: str | pathlib.Path,
    out_path: str | pathlib.Path | None = None,
    *,
    compact: bool = False,
) -> pathlib.Path:
```

When `compact=True`:
- Render `# circuitry report` header + Source run line.
- Render `## Summary` block verbatim.
- Render `## Flags` verdict block (if `step_count > 1`).
- **Suppress** `## Matched modules`, `## Attach summary`, all per-tag `## …` section tables,
  and the `<details>` advanced-metrics block.

Default `compact=False` output is **byte-identical** to the current output (the verdict block
is a new addition, so "byte-identical to the old default" means after T3 lands the default
output gains the flags block; the tables are unchanged).

CLI:

```bash
circuitry report --run runs/my_run [--compact]
```

`cli/main.py` `_cmd_report` gains:
```python
out = build_report(run_dir=args.run, out_path=args.out, compact=args.compact)
```
and the subparser gains `p_report.add_argument("--compact", action="store_true", default=False)`.

---

## 6. Component 4 — `circuitry compare run_a run_b`

### Grounding / background

- `recorder/report.py:40-48` — `_group` (private): groups JSONL rows by tag.
- `recorder/report.py:66-70` — `_stats` (private): `(first, last, vmin, vmax, delta)`.
  Both helpers are currently private inside `report.py`; to use them from `compare.py`
  without duplication, they must be extracted to a shared private module.
- `design.md §4.3` references `findings.json` — that path is stale documentation.
  The **real** metrics format is `metrics.jsonl` (written by `JsonlWriter`, consumed by
  `build_report`). All `compare` docs and code reference `metrics.jsonl` exclusively.
- `recorder/compare.py` — does not yet exist (`grep -r compare src/` finds nothing in recorder/).

### Shared private module — `recorder/_metrics.py`

A mechanical refactor with zero behavior change:

```python
# src/circuitry/recorder/_metrics.py
"""Shared grouping/stats helpers for report.py and compare.py (private)."""

from __future__ import annotations
from collections import defaultdict


def group(rows: list[dict]) -> dict[str, list[tuple[int, float]]]:
    """Group scalar JSONL rows by tag; sort each series by step."""
    ...  # exact body of current report.py:_group

def stats(series: list[tuple[int, float]]) -> tuple[float, float, float, float, float]:
    """Return (first, last, vmin, vmax, delta) over a sorted time series."""
    ...  # exact body of current report.py:_stats
```

`report.py` then replaces:
```python
from circuitry.recorder._metrics import group as _group, stats as _stats
```
and removes the local definitions. Existing `report.py` tests import `build_report`
directly — they do not reference `_group`/`_stats` by name, so they stay green automatically.

### `compare.py` API

```python
# src/circuitry/recorder/compare.py

@dataclass
class FamilyDelta:
    section: str            # e.g. "weight/effective_rank"
    diagnostic: str         # e.g. "effective_rank"
    last_a: float           # mean of last-value across all tags in family, run_a
    last_b: float           # mean of last-value across all tags in family, run_b
    delta: float            # last_b - last_a
    trend_a: str            # "up" | "down" | "flat"
    trend_b: str
    trend_agrees: bool      # True if both trends have same sign (or both flat)


def compare_runs(
    run_a: str | pathlib.Path,
    run_b: str | pathlib.Path,
) -> list[FamilyDelta]:
    """Compare two runs at family/diagnostic granularity.

    Reads ``metrics.jsonl`` from each run. Returns one ``FamilyDelta``
    per (section, diagnostic) present in either run; missing families
    produce NaN last values with a note.

    Granularity is family/diagnostic (first two tag segments), NOT
    per-module. Per-module comparison is ill-posed across architectures
    with different module names.
    """
    ...


def build_compare_report(
    run_a: str | pathlib.Path,
    run_b: str | pathlib.Path,
    out_path: str | pathlib.Path | None = None,
) -> pathlib.Path:
    """Write a markdown compare report to out_path (default: run_a/../compare.md)."""
    ...
```

**Trend sign computation:**
- `delta > 0` → `"up"`; `delta < 0` → `"down"`; `delta == 0` → `"flat"`.
  `delta` here is the intra-run delta (last − first), computed via `_stats`.
- `trend_agrees = (trend_a == trend_b)` (or both flat).

**Output format** (markdown table, one row per `FamilyDelta`):

```markdown
# circuitry compare

A: `runs/run_a`
B: `runs/run_b`

| family/diagnostic | last_a | last_b | Δ (b−a) | trend_a | trend_b | agrees |
| --- | ---: | ---: | ---: | --- | --- | --- |
| weight/effective_rank | 12.3 | 11.1 | −1.2 | down | down | yes |
```

### Multi-seed aggregation

Mean/std of last-value per side across a list of run directories would be the natural
extension: `compare_runs(runs_a=[...], runs_b=[...])`. This is flagged as a **possible
v1.3 deferral**: the per-side scalar-aggregation story is clear, but there is no confirmed
consumer in the current user base. The v1.2 spec ships single-run-per-side; the function
signature uses positional `run_a/run_b` (not lists) to keep the future upgrade non-breaking
(a `runs_a/runs_b` overload can be added additively).

---

## 7. Design-contract fit (`docs/design.md` amendments required)

| Section | Amendment |
| --- | --- |
| §4.2 | `build_report` signature: add `compact: bool = False` |
| §4.2 | New `compare_runs` / `build_compare_report` in the public `from circuitry import ...` block |
| §4.3 (CLI) | Add `circuitry compare run_a run_b [--out path]`; add `--compact` to `report` |
| §4.3 | Fix stale `findings.json` reference → `metrics.jsonl` |
| §4.4 (Recipe) | Add `disable(names)` / `only(names)` to the code block; add `enabled: dict[str, bool]` field (currently absent from the §4.4 example block, though present in source at line 23) |
| §4.4 (Recipe) | Add prose: both helpers raise `ValueError` on unknown names; custom callables unaffected |
| Header | Bump "Last updated" from 2026-05-21 → 2026-05-29 |

---

## 8. Layering invariants

- `recorder/_metrics.py` imports only stdlib + no `circuitry.*` — within `recorder/`, zero
  layering impact.
- `recorder/compare.py` imports `recorder/_metrics` and `pathlib`/`json`/`dataclasses` — no
  `cli/` import; no `core/` side-effects; consistent with layering rules (§3 design.md).
- `cli/main.py` gains a `compare` subcommand importing `recorder/compare.py` — `cli` already
  imports `recorder/report.py`; same pattern.
- `test_layering.py` allowlist unchanged (no new root packages).

---

## 9. Testing strategy

All tests run on CPU with no real model downloads.

### T1 — `Recipe.disable` / `Recipe.only`

- `tests/recipes/test_recipe_helpers.py` (new).
- Assert `disable(["effective_rank"])` sets `enabled["effective_rank"] = False`; other names
  unaffected (still absent from `enabled`, i.e., default-True).
- Assert `only(["effective_rank"])` disables every name except `effective_rank`.
- Assert both raise `ValueError` with unknown name; error message contains the unknown name.
- Assert `custom` functions on a recipe with `disable(["effective_rank"])` still fire: attach
  a `Recorder` with a one-step run, assert the custom tag appears in writer output.
- Assert chains compose: `r.disable(["heavy_tail_alpha"]).only(["effective_rank"])` produces
  `enabled` where both `heavy_tail_alpha` and everything except `effective_rank` are False.
- Use the stock `"llm"` recipe (from `recipes/llm.py`) for all tests — no fixture model needed.

### T2 — `_metrics.py` refactor

- No new tests beyond verifying existing `tests/recorder/test_report.py` stays green after the
  refactor (all 6 tests in that file import `build_report` only, not `_group`/`_stats`).
- Add two unit tests in `tests/recorder/test_metrics.py` (new):
  - `test_group_sorts_by_step`: out-of-order JSONL rows sort correctly.
  - `test_stats_single_point`: `delta == 0` for a one-step series.

### T3 — verdict/flags + compact mode

- `tests/recorder/test_report_flags.py` (new).
- `test_flags_suppressed_single_step`: `step_count == 1` → no `## Flags` block.
- `test_flags_fires_dead_fraction`: write JSONL with `activation/dead_fraction/mod` rising
  from 0.01 → 0.15 (2 steps) → assert `## Flags` block present, contains `dead_rising`.
- `test_no_false_flags_flat`: flat series → no flags even for `step_count > 1`.
- `test_compact_omits_tables`: `compact=True` → `## Summary` present, `## weight/effective_rank`
  absent.
- `test_compact_includes_flags`: `compact=True` with rising dead_fraction → `## Flags` present.
- `test_default_output_unchanged_structure`: `compact=False` → all existing test_report.py
  assertions still hold (run the full suite; no regression).

### T4 — `compare`

- `tests/recorder/test_compare.py` (new).
- `test_compare_single_family_delta`: two `tmp_path` run dirs, each with a two-step
  `activation/dead_fraction/mod` series; assert `FamilyDelta.delta == last_b - last_a`.
- `test_compare_missing_family_in_one_run`: family present in run_a only → row with NaN
  `last_b` (or sentinel) in result.
- `test_compare_trend_agreement`: run_a rising, run_b rising → `trend_agrees=True`;
  run_a rising, run_b falling → `trend_agrees=False`.
- `test_build_compare_report_writes_markdown`: calls `build_compare_report`, asserts the
  output file contains `# circuitry compare` and the paths of both runs.
- `test_cli_compare_subcommand`: invoke `main(["compare", "--run-a", str(tmp_a),
  "--run-b", str(tmp_b)])` → exit code 0, writes a file.

### T5 — hygiene

- Manual / diff-review acceptance: `docs/design.md` diffs match §7 amendments above;
  `TODO.md` Ergonomics items ticked; `pyproject.toml` classifier updated; `CHANGELOG.md`
  entry present for `[1.2.0]`.
- Full suite green: `venv/bin/pytest -q` after all changes.

---

## 10. Edge cases

| Condition | Behavior |
| --- | --- |
| `disable([])` | No-op; returns `dataclasses.replace(self, enabled=self.enabled)` |
| `only([])` | Disables everything; valid (a recipe that emits nothing) |
| Name in `enabled` already False, passed to `disable` | Idempotent; still False |
| `step_count == 1` in `build_report` | No `## Flags` block rendered |
| `compare_runs` with identical runs | All `delta == 0`, all `trend_agrees = True` |
| `compare_runs` with a run that has no `metrics.jsonl` | `FileNotFoundError` with run path in message |
| `--compact` and no `metrics.jsonl` | Compact output still renders; `## Summary` shows "no metrics found" |

---

## 11. Evidence (verified before writing)

- `recipes/__init__.py:23` — `enabled` field confirmed present.
- `recipes/__init__.py:29-47` — `with_prefix` body is the exact `dataclasses.replace` pattern mirrored by `disable`/`only`.
- `recorder/live.py:589-590` — `_enabled` reads `recipe.enabled.get(name, True)`.
- `recorder/live.py:609,661,863` — three diagnostic loops each call `_enabled`; custom loop at line 880 does not.
- `recorder/report.py:40-70` — `_group` (line 40), `_stats` (line 66) are the two private helpers to extract.
- `recorder/report.py:137` — current `build_report` signature (two params); `step_count` at line 154.
- `recorder/report.py:190` — `## Summary` appended here; verdict block inserted after.
- `cli/main.py:19-21` — `_cmd_report` calls `build_report`; `compare` subcommand entirely absent.
- `design.md §4.3` — stale `findings.json` reference confirmed; amendment required.
- `design.md §4.4` — `enabled` field absent from the code block; `disable`/`only` prose absent.
- `pyproject.toml:15` — `Development Status :: 3 - Alpha` confirmed; flip to `4 - Beta`.
- `TODO.md:53-60` — two Ergonomics items are unticked; both closed by v1.2.
- `TODO.md:3` — header reads "Released through v0.9.2"; must be updated to v1.1.0.
