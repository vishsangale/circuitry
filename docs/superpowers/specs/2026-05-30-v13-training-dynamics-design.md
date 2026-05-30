# v1.3.0 — Training-dynamics diagnostics (design spec)

**Status:** approved (2026-05-30).
All claims are grounded against source files listed under "Evidence" per component.

---

## 1. Motivation — the niche circuitry owns

The 2026 ecosystem survey (design.md §1) identifies one unsatisfied lane: live, low-overhead
interpretability **inside the training loop** — diagnosing how a model *changes* over training
rather than analyzing its current-step state. TransformerLens / nnsight / SAELens / pyvene all
operate post-hoc; none has a live training-time hook path with a ≤10% wall-clock budget.

v1.3 lands that niche's first real capability: **formation, collapse, and drift diagnostics**
that track weight dynamics across emit steps. The key insight from the roadmap: ~70% of this
work is wiring three primitives that already shipped in `core/` but are referenced **nowhere**
in `recorder/` or `recipes/`:

| Primitive | Location | Status |
| --- | --- | --- |
| `update_delta(sd_now, sd_prev)` | `core/weight.py:118` | shipped, unwired |
| `direction_cosine(sd_now, sd_prev, sd_prev_prev)` | `core/weight.py:136` | shipped, unwired |
| `rank_trajectory(state_dicts)` | `core/spectral.py:41` | shipped, unwired |

These three are also **absent from design.md §4.1** (the public primitive catalog) despite
being in `core/`. v1.3 adds them to the catalog and wires them live.

---

## 2. Primitive signatures (exact, from source)

### `weight.update_delta` (`core/weight.py:118`)

```python
def update_delta(
    sd_now: Mapping[str, torch.Tensor],
    sd_prev: Mapping[str, torch.Tensor],
) -> dict[str, float]:
    """L2 norm of the delta between two state-dict snapshots, per parameter.

    Returns {name: ||sd_now[name] - sd_prev[name]||_2} for every name
    present in both. Names missing from either side are skipped.
    """
```

### `weight.direction_cosine` (`core/weight.py:136`)

```python
def direction_cosine(
    sd_now: Mapping[str, torch.Tensor],
    sd_prev: Mapping[str, torch.Tensor],
    sd_prev_prev: Mapping[str, torch.Tensor],
) -> dict[str, float]:
    """Cosine similarity between two consecutive parameter updates.

    Update_1 = sd_prev - sd_prev_prev
    Update_2 = sd_now  - sd_prev

    Returns {name: cos(Update_1, Update_2)}. Zero-norm updates return 0.0.
    """
```

`direction_cosine` needs **three** snapshots: `sd_now`, `sd_prev`, and `sd_prev_prev`. This
is the only primitive that requires two prior emit-step weights rather than one.

### `spectral.rank_trajectory` (`core/spectral.py:41`)

```python
def rank_trajectory(
    state_dicts: Sequence[Mapping[str, torch.Tensor]],
) -> dict[str, list[float]]:
    """Effective rank per 2D parameter across an ordered sequence of state dicts.

    Non-2D tensors (biases, layer norms) are skipped.
    """
```

`rank_trajectory` accepts a **sequence** of state dicts and returns a parallel list of
effective-rank values per parameter. In the live Recorder path we call it with a length-2
or length-3 window (current + prior snapshot(s)) and read off the last element as the
per-step `rank_trajectory` scalar — consistent with how the scan path would call it
over a long sequence of checkpoints.

---

## 3. The core design problem — cross-step snapshot holder

### Problem statement

`update_delta` and `direction_cosine` are **cross-step**: they compare the current emit-step
weights against one or two prior emit-step weights. But the Recorder (`recorder/live.py`) is
currently stateless per-step: `step()` builds a fresh `StepContext` from current tensors,
runs diagnostics, then discards the context. There is nowhere to store the previous snapshot.

Concretely, `live.py:556-583`: weights are assembled fresh from `model.named_parameters()`
on every emit step into a local `weights: dict[str, Tensor]` and stuffed into a `StepContext`
dataclass that goes out of scope at the end of `_run_diagnostics`. The previous emit-step
weight tensor is gone.

### Solution — `_prev_weights` and `_prev_prev_weights` on `Recorder`

Add two instance attributes to `Recorder.__init__`:

```python
# recorder/live.py  (add to __init__, after _current_step)
self._prev_weights: dict[str, torch.Tensor] = {}        # prior emit-step snapshot (CPU)
self._prev_prev_weights: dict[str, torch.Tensor] = {}   # two steps back (CPU)
```

#### Lifecycle

1. **`attach()`** — no change. Snapshots start empty.

2. **`step()` — post-diagnostics roll**

   After `_run_diagnostics(ctx)` returns (current line `live.py:585`), and after
   `_captured_activations.clear()` (line `live.py:587`), roll the snapshots forward:

   ```python
   # Roll snapshots: prev_prev ← prev, prev ← current (CPU-detached, matched-modules only)
   self._prev_prev_weights = self._prev_weights
   self._prev_weights = {
       name: t.detach().cpu()
       for name, t in ctx.weights.items()
   }
   ```

   This runs only on emit steps (already inside the `if not self._should_capture(...)` guard
   at line `live.py:548`).

3. **`detach()`** — clear both snapshots to release memory:

   ```python
   # recorder/live.py  detach()  (after hook removal)
   self._prev_weights.clear()
   self._prev_prev_weights.clear()
   ```

#### Design properties

| Property | Mechanism |
| --- | --- |
| **Detached** | `.detach().cpu()` — no gradient tape attached; no CUDA memory held |
| **CPU-resident** | `.cpu()` — snapshot sits in RAM; GPU VRAM is untouched |
| **Matched-modules only** | built from `ctx.weights` which is already filtered to recipe hook-point matches |
| **First-step guard** | `_prev_weights` is empty on the first emit; `update_delta` skips names absent in either dict; the diagnostic loop checks `if not self._prev_weights: continue` before calling |
| **`direction_cosine` guard** | requires `_prev_prev_weights` non-empty too; checked separately |
| **Size** | for the stock LLM recipe with ~18 weight matrices on a 7B model at bf16, each snapshot is ~18 × (layer_dim × head_dim × 2 bytes) ≈ tens of MB — acceptable RAM cost |

#### §4.4 StepContext is unchanged

The `StepContext` dataclass shape (`live.py:575` area) is **not modified**. The snapshot
holder is internal recorder state, not part of the context passed to custom `DiagnosticFn`
callables. Users who want cross-step delta in a custom diagnostic can access `ctx.weights`
and maintain their own prior snapshot via `ctx.user` pass-through.

---

## 4. Weight-diagnostic dispatch — wiring the three primitives

### Tag namespace

| Primitive | Tag pattern |
| --- | --- |
| `update_delta` | `weight/update_delta/<module_name>` |
| `direction_cosine` | `weight/direction_cosine/<module_name>` |
| `rank_trajectory` | `weight/rank_trajectory/<module_name>` |

These follow the existing convention `weight/<diagnostic_name>/<module_name>` used by
`effective_rank`, `stable_rank`, etc. (`live.py:659`).

### `_WEIGHT_DIAGS` table (not used for cross-step primitives)

The current `_WEIGHT_DIAGS` dict (`live.py:32`) maps names → single-tensor functions that
operate on singular values. `update_delta` and `direction_cosine` take two or three full
state-dict snapshots, not singular values — they cannot use the shared SVD cache path.
`rank_trajectory` also takes full tensors (internally calls `weight.effective_rank` per
param). All three are dispatched **outside** `_WEIGHT_DIAGS` as named special cases in
`_run_diagnostics`, parallel to the `attention_head_rank` and `sv_histogram` special-casing
already present (`live.py:612-659`).

### Dispatch logic in `_run_diagnostics`

Insert after the `sv_histogram` branch (current `live.py:647-659` area), still inside the
`for name in self.recipe.weight_diagnostics:` loop:

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
        continue  # need at least two prior snapshots
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
    traj = _spectral.rank_trajectory([self._prev_weights, ctx.weights])
    for mod_name, vals in traj.items():
        # vals is length-2; vals[-1] is the current-step rank.
        self._writer.add_scalar(
            f"weight/rank_trajectory/{mod_name}", vals[-1], ctx.step
        )
```

The `_spectral` import is added at the top of `live.py`:

```python
from circuitry.core import spectral as _spectral
```

### State-dict key alignment

`update_delta` and `direction_cosine` take `Mapping[str, Tensor]` keyed by **parameter
name** (as returned by `model.named_parameters()`). The `ctx.weights` dict is keyed by
**module name** (as resolved by the recipe hook point + inventory). These are different
namespaces. The current weight dispatch path in `live.py:568-573` populates `ctx.weights`
with module-name keys and the tensors from `self._param_for_module`. The snapshot
`_prev_weights` is built from `ctx.weights` — so both `ctx.weights` and `_prev_weights`
share module-name keys. The primitives receive module-name-keyed dicts; they will correctly
compute deltas for matching keys. No alignment problem.

---

## 5. `llm` recipe additions

Add the three new diagnostic names to `recipes/llm.py:RECIPE.weight_diagnostics`:

```python
weight_diagnostics=[
    "effective_rank", "attention_head_rank", "stable_rank",
    "heavy_tail_alpha", "sv_histogram",
    # v1.3 training-dynamics additions:
    "update_delta", "rank_trajectory", "direction_cosine",
],
```

`direction_cosine` requires two prior steps (emits nothing on the first two emit steps).
Adding it to the stock recipe is correct — the first-step and second-step guards in
`_run_diagnostics` handle the skip transparently.

No new `HookPoint` entries are needed: the existing WEIGHT hook points already capture
the full set of matched weight tensors per emit step.

---

## 6. §4.1 catalog additions (design.md amendment)

The following two entries are absent from design.md §4.1 despite being in `core/` since
before v1.0 (confirmed by `grep -rn "update_delta\|direction_cosine" src/circuitry/recorder/`
returning nothing). Add under the `# weight-space` block:

```python
# weight-space (training dynamics — new in v1.3)
weight.update_delta(sd_now: Mapping[str, Tensor], sd_prev: Mapping[str, Tensor]) -> dict[str, float]
weight.direction_cosine(sd_now: Mapping[str, Tensor], sd_prev: Mapping[str, Tensor],
                        sd_prev_prev: Mapping[str, Tensor]) -> dict[str, float]

# spectral (training dynamics)
spectral.rank_trajectory(state_dicts: Sequence[Mapping[str, Tensor]]) -> dict[str, list[float]]
```

---

## 7. Report and compare surfacing

### `build_report` additions (`recorder/report.py`)

**New `HERO_SECTIONS` entries:**

```python
HERO_SECTIONS = frozenset({
    ...existing entries...,
    # v1.3 training-dynamics:
    "weight/update_delta",
    "weight/rank_trajectory",
    "weight/direction_cosine",
})
```

Rationale: these are formation/collapse signals — they belong in the top ("hero") sections
that a user sees first, not in the `<details>` advanced block.

**New `FLAG_RULES` entries:**

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

The `signed` parameter in `FLAG_RULES` predicates is `last - first` (the intra-run trend),
exactly as the existing rules use it (`report.py:47`, note comment about `signed = last - first`).

**Summary block note:** no structural change to `build_report`; the new tags appear
automatically in the per-family sections because `_section_and_row` splits any tag of the
form `weight/update_delta/module.name` into section=`weight/update_delta`, row=`module.name`.

### `compare_runs` / `build_compare_report`

No code change required: `compare_runs` operates at family/diagnostic granularity (first two
tag segments). The new `weight/update_delta`, `weight/rank_trajectory`, and
`weight/direction_cosine` families appear automatically as new rows in the compare table for
any run pair where at least one run used v1.3 diagnostics.

---

## 8. Scoping fork — Option A vs Option B

### Option A — weight-dynamics only (this release)

Wire `update_delta`, `direction_cosine`, `rank_trajectory` into the weight-diagnostic dispatch
and the `llm` recipe. Add the cross-step weight snapshot. Surface formation/collapse trends in
`build_report`. No extra forward pass; fits the §10 ≤10% wall-clock budget; fully CPU-testable.

**Effort:** ~3 targeted files modified (`live.py`, `recipes/llm.py`, `recorder/report.py`) +
`docs/design.md`. No new primitives. No new optional deps. Conservative scope.

**Risk:** very low. The primitives are already correct and tested. The snapshot holder is
small and RAII-clean. The dispatch is parallel to existing special-case paths already in
`_run_diagnostics`.

**Value:** lands the differentiated niche claim. Live rank-trajectory and update-delta
diagnostics have no equivalent in any current tool. The v1.2 `compare` subcommand
immediately gains training-dynamics families — users get trend plots and A/B collapse
detection for free.

### Option B — also add representational-drift primitive (deferred)

A new pure `core` primitive measuring representational drift: cosine or CKA similarity
between activations on a fixed probe batch at step T vs a stored reference at step 0 (or
any anchor step). Requires:

1. **A stored probe batch** — a small fixed batch (e.g., 16 examples) loaded at `attach()`
   time from a file or provided as a `Recipe` kwarg.
2. **Reference activations** — captured on the first emit step (or at `attach()` time via
   a single forward pass) and stored on CPU.
3. **A second forward pass each emit step** — the probe batch is passed through the model
   to collect current activations, which are compared against the reference. This adds
   ~(probe_batch / train_batch) × forward_pass_time per emit step.

**Performance impact:** the §10 GPU wall-clock budget is already at +12.66% overage (noted
in the roadmap analysis). Adding a mandatory second forward pass even at `every_n_steps=200`
would push this further over budget. Option B **must** be opt-in (off by default via the
`Recipe.enabled` toggle or a dedicated `probe_batch` flag on `Recorder`).

**Design complexity:** probe storage (file path or in-memory tensor), reference capture
lifecycle (captured once, held in CPU RAM, updated on explicit `reset_reference()` call),
CKA computation (quadratic in token count — needs a `max_tokens` cap), new `Recipe` field
for probe batch spec. Bigger surface than Option A.

**Effort:** ~2× Option A. New primitive in `core/activation.py` or `core/drift.py`, new
`Recorder` probe path, new `Recipe` field, new tests with a real forward pass (cannot be
purely CPU-synthetic without a model).

### Recommendation

**Ship Option A as v1.3.0. Defer Option B to v1.3.1 or v1.4.**

Reasoning: Option A is the "70% is unwiring" path — it delivers the formation/collapse niche
claim with near-zero implementation risk and zero perf cost. Option B's probe-based drift
measurement is valuable but orthogonal, has real perf cost (second forward), and needs a
richer design for probe lifecycle and the already-over-budget §10 GPU constraint. Shipping
Option A first gives users something immediately useful and establishes the cross-step
snapshot infrastructure that Option B can reuse.

**I agree with the stated leaning.** Option A is the right call for v1.3.0.

---

## 9. Performance story

### Option A (this release)

- **No extra forward pass.** `update_delta` and `direction_cosine` operate on detached CPU
  tensors already captured by the existing WEIGHT hook path.
- **CPU-only.** The snapshot roll and L2 norm are pure CPU ops on float32 copies of
  bf16/fp16 weight matrices. On a 7B model the snapshot of matched modules is tens of MB;
  the L2 norm per tensor is microseconds.
- **SVD not re-run.** `rank_trajectory` calls `weight.effective_rank` internally, which calls
  `singular_values` — this does run SVD per weight matrix per emit step. However, `rank_trajectory`
  in the live path only computes `effective_rank(current)` (the prior snapshot's rank was
  computed on the prior emit step). The SVD for the current weights is **already computed**
  by the existing `_sv_cache` in `_run_diagnostics` (`live.py:601-607`). We can avoid a
  double-SVD by computing the current effective rank via `_sv(w)` (the cached singular values)
  rather than calling `rank_trajectory` in isolation. The dispatch branch should call
  `_w._effective_rank_from_sv(_sv(w))` and emit as `weight/rank_trajectory/<module>` — same
  semantics, zero extra SVD cost. The standalone `rank_trajectory` primitive is unchanged
  and used as-is by `scan_run`.
- **Wall-clock budget (§10):** Option A adds zero forward passes and reuses the cached SVD.
  Overhead is dominated by the snapshot copy (`.detach().cpu()` on ~18 tensors), which
  is a memcpy from GPU to CPU on emit steps only. At `every_n_steps=200` this is
  negligible. Option A is within the ≤10% budget.

### Option B (deferred)

- Requires a second forward pass per emit step on a probe batch.
- Currently estimated to push overhead from +12.66% to unacceptable levels at default settings.
- Must be opt-in: `recipe.enabled["drift_probe"] = False` by default, with a clear doc
  warning that enabling it at small `every_n_steps` will breach the §10 budget.

---

## 10. CPU-only testing strategy

All v1.3 tests run on CPU with no model downloads, consistent with §6 of design.md.

| Component | Test location | Strategy |
| --- | --- | --- |
| Snapshot roll | `tests/recorder/test_live_snapshot.py` (new) | Tiny 2-layer `nn.Linear` MLP; attach Recorder; run 3 emit steps; after step 0 assert `_prev_weights` non-empty and values match current; after step 1 assert `_prev_prev_weights` non-empty; detach and assert both dicts empty. |
| `update_delta` dispatch | `tests/recorder/test_live_snapshot.py` | RecordingWriter; after 2 emit steps assert `weight/update_delta/<module>` tag present with a positive float. |
| `direction_cosine` dispatch | `tests/recorder/test_live_snapshot.py` | After 3 emit steps (needs prev and prev_prev) assert `weight/direction_cosine/<module>` tag present; first-step and second-step suppression asserted. |
| `rank_trajectory` dispatch | `tests/recorder/test_live_snapshot.py` | After 2 emit steps assert `weight/rank_trajectory/<module>` tag present; value equals `effective_rank` of current weights (within float tolerance). |
| First-step guard | `tests/recorder/test_live_snapshot.py` | After exactly 1 emit step assert no `update_delta` / `direction_cosine` / `rank_trajectory` tags emitted. |
| `build_report` flags | `tests/recorder/test_report_flags.py` (extend existing) | Inject synthetic JSONL with declining `weight/rank_trajectory` → assert `rank_collapse_trend` flag fires. |
| `detach()` cleanup | `tests/recorder/test_live_snapshot.py` | After `detach()`, assert `recorder._prev_weights == {}` and `recorder._prev_prev_weights == {}`. |
| `llm` recipe wiring | `tests/recipes/test_llm_dynamics.py` (new) | Tiny toy transformer; attach with llm recipe; 3 emit steps; assert all three new diagnostic families appear in writer output. |

**No GPU required.** The snapshot copy is `.cpu()` — on a CPU-only test environment this
is a no-op (already CPU). The SVD reuse path uses `_sv_cache` which is populated by the
existing `effective_rank` dispatch already in the test suite.

---

## 11. New optional dependencies

**None for Option A.** All three primitives use only `torch` and `numpy` already in the
dependency set. No new imports at module level; no new entries in `pyproject.toml`.

---

## 12. Layering invariants

No new layering concerns:

- `core/weight.py` and `core/spectral.py` are unchanged (primitives already exist).
- `recorder/live.py` imports `from circuitry.core import spectral as _spectral` — `core/`
  is already imported here (`live.py:17-22`); adding `spectral` follows the existing pattern.
- `recipes/llm.py` only adds strings to `weight_diagnostics` — no new imports.
- `recorder/report.py` only adds entries to `HERO_SECTIONS` and `FLAG_RULES` — no new imports.
- `test_layering.py` allowlist is unchanged.

---

## 13. Design-contract amendments required (design.md)

| Section | Amendment |
| --- | --- |
| Header | Bump "Last updated" to 2026-05-30 |
| §4.1 | Add `update_delta`, `direction_cosine` under `# weight-space`; add `rank_trajectory` is already listed under `# spectral` but was undocumented as "shipped" — add note that it is now wired live |
| §4.4 | Add prose: "The Recorder maintains `_prev_weights` and `_prev_prev_weights` (detached CPU snapshots of matched-module weights from prior emit steps) to support cross-step primitives (`update_delta`, `direction_cosine`, `rank_trajectory`). Both are empty at `attach()`, populated after each emit step, and cleared by `detach()`." |
| §5 (Recipe internals) | Update `weight_diagnostics` list for the `llm` recipe example to include the three new names |

---

## 14. Evidence (verified before writing)

- `core/weight.py:118` — `update_delta` signature and body confirmed; takes two `Mapping[str, Tensor]` dicts.
- `core/weight.py:136` — `direction_cosine` signature confirmed; takes three `Mapping[str, Tensor]` dicts.
- `core/spectral.py:41` — `rank_trajectory` signature confirmed; takes `Sequence[Mapping[str, Tensor]]`.
- `grep -rn "update_delta|direction_cosine|rank_trajectory" src/circuitry/recorder/ src/circuitry/recipes/` — returns nothing; confirmed unwired.
- `recorder/live.py:32-37` — `_WEIGHT_DIAGS` confirmed; `update_delta`/`direction_cosine`/`rank_trajectory` absent.
- `recorder/live.py:556-583` — weight assembly into local dict, no prior-step snapshot stored.
- `recorder/live.py:585-587` — `_run_diagnostics(ctx)` call then `_captured_activations.clear()` — insertion point for snapshot roll.
- `recorder/live.py:522-530` — `detach()` body; insertion point for snapshot clear.
- `recorder/live.py:601-607` — `_sv_cache` per-step SVD cache; reuse path for `rank_trajectory`.
- `recorder/live.py:612-659` — `attention_head_rank` and `sv_histogram` special-case pattern to mirror.
- `recorder/report.py:28-39` — `HERO_SECTIONS` frozenset; insertion point.
- `recorder/report.py:43-72` — `FLAG_RULES` and `_build_flags`; insertion point for new rules.
- `recipes/llm.py:36-38` — `weight_diagnostics` list; insertion point.
- `design.md §4.1` — `update_delta`/`direction_cosine` absent from catalog; `rank_trajectory` listed at line 137 (`spectral.rank_trajectory`) but never documented as wired.
- `design.md §10` — +12.66% GPU overhead noted; Option B opt-in requirement grounded here.
