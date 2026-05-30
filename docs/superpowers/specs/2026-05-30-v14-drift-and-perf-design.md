# v1.4.0 Design Spec — Representational-Drift Probe + Perf/Determinism

**Status:** DRAFT for user review. This is a design pass — no code has been written.
**Date:** 2026-05-30
**Target release:** v1.4.0 (minor)
**Author:** design subagent (orchestrated)

> This spec combines two workstreams into one minor release per the user's decision:
> **(A)** perf/determinism fixes, and **(B)** the representational-drift probe (Option B feature).
> Sequencing is **PERF-FIRST** so the drift probe is benchmarked against an honest budget.
>
> Section 4 (OPEN FORKS) contains decisions that are **NOT finalized here** — they are for the
> user to confirm. Do not treat the recommended defaults as committed.

---

## 1. Scope & SemVer rationale

### 1.1 Why one combined minor release (1.4.0)

Both workstreams are **purely additive at the public-API level**:

- Workstream A (perf/determinism) adds *optional* kwargs with back-compat defaults
  (`singular_values(..., seed=None, use_gram=...)`, ACDC `run(..., ablation_mode="corrupted",
  eap_skip_threshold=None)`) and changes *internal* behavior (an opt-in Gram fast path, a
  documentation correction). No existing signature changes, no removals.
- Workstream B (drift probe) adds a new pure primitive in `core/`, a new optional `Recipe`
  field (`probe_batch`), a new opt-in diagnostic string key (`"drift_probe"`), and new internal
  `Recorder` state. All default-OFF; no existing run changes behavior.

Additive feature + additive perf work + no breaking change = **MINOR** under SemVer. There is no
need to split into two releases; bundling lets the drift probe ship measured against the
post-perf-fix budget rather than the current (over-budget) one.

### 1.2 Why perf-first ordering

The §10 budget is **already breached** on the only numbers we have. The README (README.md:65-70)
reports v0.2.0a0 CPU runs of **+14.9%** and **+14.7%** overhead on an 88M-param model at
`every_n_steps=200` — both over the ≤10% budget (design.md:439).

> **Grounding correction (must propagate into the spec discussion and the eventual changelog):**
> The task brief referenced a "+12.66%" measurement. **That figure does not exist anywhere in the
> repo** (grep of README, CHANGELOG, scripts: zero hits). The real on-record numbers are +14.9% /
> +14.7%. This spec uses the real numbers. The "+12.66%" should be considered a brief artifact, not
> a measurement to reconcile against.

The drift probe adds a **second forward pass per emit step** — the single largest cost any
diagnostic has introduced. Benchmarking the probe on top of an *already-broken* budget would
produce a meaningless number. Therefore:

1. Land the determinism fix (correctness; no perf cost).
2. Land the Gram SVD fast path (reduces baseline diagnostic cost).
3. Re-validate §10 on GPU (the rtx host) and correct README/design honestly.
4. *Then* design/benchmark the drift probe against the corrected, honest budget.

### 1.3 Why drift is opt-in / default-OFF

The drift probe:

- runs a **full second forward pass** on a stored probe batch every emit step (the dominant cost,
  explicitly deferred as opt-in in the v1.3.0 changelog);
- stores **reference activations on CPU** (potentially hundreds of MB for a large LLM across all
  layers — see §3.5 storage math);
- is only meaningful when the user supplies a fixed probe batch.

It is gated OFF by default via `enabled={"drift_probe": False}` in the stock recipe, mirroring the
existing philosophy that expensive diagnostics are off by default (design.md:440). Users opt in via
`recipe.only(["drift_probe"])` or by clearing the disable and supplying a `probe_batch`.

---

## 2. Workstream A — perf / determinism

Four items. Each gives exact location, the change, and whether it is GPU-gated (needs the rtx CUDA
host) or CPU-validatable here on macbook.

### A1. Unseeded `randperm` determinism fix — **CPU-testable here**

**Location:** `src/circuitry/core/weight.py:42`, inside `singular_values(...)`:

```python
idx = torch.randperm(n, device=M.device)[:max_dim]   # no generator=, no seed guard
```

**Problem (confirmed):** This path fires whenever `min(M.shape) > max_dim` (default 512). It draws
an *unseeded* random column subsample, so repeated calls on the same matrix return different
singular values. This directly violates the §4.1 invariant "Deterministic on CPU" (design.md:167)
and the module docstring (weight.py:1). The v0.9.1 SVD-sharing fix made all per-matrix diagnostics
*share one draw within a step*, which shrinks the blast radius but does not remove cross-step /
cross-call nondeterminism.

**Change:** Add a `seed: int | None = None` kwarg to `singular_values`. When `seed is not None`,
build a device-correct generator and pass it to `randperm`:

```python
def singular_values(W, k=None, max_dim=512, *, seed=None, use_gram="auto"):
    ...
    if max_dim is not None and min(M.shape) > max_dim:
        axis = 1 if M.shape[1] > M.shape[0] else 0
        n = M.shape[axis]
        if seed is not None:
            g = torch.Generator(device=M.device).manual_seed(seed)
            idx = torch.randperm(n, device=M.device, generator=g)[:max_dim]
        else:
            idx = torch.randperm(n, device=M.device)[:max_dim]
        M = M.index_select(axis, idx)
```

**Statistical-behavior preservation:** Default `seed=None` keeps today's exact behavior (one
unseeded draw) for callers who do not opt in — no silent change to the subsample distribution. The
subsample is still uniform-without-replacement over the long axis; only the RNG *source* changes
when a seed is given. The Recorder's `_run_diagnostics` SVD cache (`_sv` at live.py:619-624) already
shares one draw per matrix per step; passing a fixed seed makes that draw reproducible across steps
and across runs.

**Recorder wiring (decision point — see Fork A1 in §4):** the Recorder can either (a) pass a fixed
`seed` (e.g. the step index, or a constant) into the shared `_sv` helper so cross-step rank
trajectories are reproducible, or (b) leave the primitive seed-less and document that callers must
set the global torch seed. Recommended: the Recorder passes a fixed constant seed so
`update_delta`/`rank_trajectory` comparisons across steps are not polluted by subsample noise.

**GPU-gated?** No. The fix is device-agnostic (`device=M.device` is already correct; the CUDA
device-propagation fix in the changelog handled the device mismatch). Determinism is fully testable
on CPU.

### A2. `eigvalsh(Gram)` SVD fast path — **CPU-testable here (correctness + tolerance); GPU for perf claim**

**Location:** `src/circuitry/core/weight.py:44`, the sole SVD call:

```python
s = torch.linalg.svdvals(M)   # unconditional full SVD after subsampling
```

The fast path slots in **between line 43 (post-subsampling `M`) and line 44**.

**Opportunity (confirmed):** After subsampling, `M` is `(m, max_dim)` or `(max_dim, n)`. When the
*smaller* dimension is much smaller than the larger (high aspect ratio), forming the smaller Gram
matrix and taking its eigenvalues is cheaper than a full SVD:

- if `m < n`: `G = M @ M.T` is `(m, m)`; `sigma = sqrt(clamp_min(eigvalsh(G), 0))`
- if `n < m`: `G = M.T @ M` is `(n, n)`; same.

Cost: `O(min^2 · max + min^3)` for the Gram path vs `O(min · max^2)` for full `svdvals`. Breakeven
is roughly when the smaller dimension is `< max_dim / 3`.

**Numerical caveat (must be documented and tested):** Gram eigenvalues are `sigma^2`, so the
condition number squares and small singular values lose ~half their bits. On float32, singular
values below `~sqrt(eps) · sigma_max ≈ 1e-4 · sigma_max` are unreliable. Consequences for the
consuming diagnostics:

- `condition_number` (`sigma_max/sigma_min`) is the most sensitive — `sigma_min` is exactly where
  Gram precision is worst. The Gram path **must not** be used when `condition_number` is the
  consumer, OR must fall back to `svdvals` when the smallest computed eigenvalue is in the ambiguous
  band.
- `effective_rank`, `stable_rank`, `heavy_tail_alpha` are dominated by the *large* singular values
  and tolerate the Gram path well.

**Change (proposed, see Fork A2 in §4 for the gating policy):** Add a `use_gram` parameter to
`singular_values`. Options: `"auto"` (use Gram when aspect ratio passes the breakeven threshold AND
not in float32-degenerate territory; else full SVD), `True` (force), `False` (never — today's
behavior). Default `"auto"` is back-compat *numerically only if* tolerances are accepted; the safe
conservative default is `False` (no behavior change) with `"auto"` opt-in. **Recommend `"auto"`
with a hard fallback to `svdvals` when (a) no subsampling occurred (`all(M.shape <= max_dim)` — Gram
buys nothing), or (b) the matrix is float32 and the consumer needs small singular values.**

Implementation note: dtype-promote the Gram to float64 before `eigvalsh` to recover precision
cheaply (the Gram is small — `min × min`), then cast back. This is the same float64-accumulation
trick the drift primitive uses (§3.1).

**GPU-gated?** Correctness and numerical-tolerance tests are CPU-only (compare Gram-path vs
`svdvals` on fixed matrices, assert relative error within tolerance for the large singular values).
The *perf claim* ("Gram path is faster for wide LLM matrices") needs GPU validation on rtx, because
the breakeven and the absolute win depend on the BLAS backend and device.

### A3. §10 budget re-validation — **GPU-GATED (needs rtx CUDA host)**

**Location:** `scripts/bench_50m.py` (harness exists, lines 1-148). Budget claims at
`docs/design.md:439` and `README.md:63`. Stale numbers at `README.md:65-70`.

**Two confirmed problems with the on-record story:**

1. **Over budget on the only numbers we have.** +14.9% / +14.7% CPU at `every_n_steps=200` > 10%.
2. **Harness default cadence is more pessimistic than the budget scenario.** `bench_50m.py:123`
   defaults `--every-n-steps 25` (4 emissions / 100 steps), while the budget and the README runs are
   stated at `every_n_steps=200` (0.5 emissions / 100 steps). Anyone running the harness with
   defaults measures a *harder* scenario than the budget claims — a footgun. The README's quoted
   runs used the explicit flags `--n-layers 8 --d-model 768` at `every_n_steps=200`, so they are not
   the harness default.

**Plan:**

- **Run the bench on rtx (CUDA), not macbook.** Submit via the `lab-control` Ray MCP
  (`ray_submit`), targeting the rtx RTX 5080 head. CPU numbers inflate the ratio (README.md:72
  already flags this; pure-Python/CPU diagnostic overhead is a larger fraction of a slow CPU step).
  The v0.9.1 SVD-sharing fix reportedly dropped per-emission GPU cost ~4×, so a post-v0.9.1 GPU run
  is the first honest measurement of the current code.
- **Measure at BOTH cadences** (`every_n_steps=200` for the budget claim; `every_n_steps=25` for the
  harness default) and at the spec'd model size (50M-param, design.md:445) plus the README's 88M
  config, so we report apples-to-apples against the budget and explain the harness default.
- **Re-run after A2** so the budget reflects the Gram fast path.

**Honest correction either way (this is required regardless of the GPU result):**

- *If GPU is ≤10% at `every_n_steps=200`:* update README.md:65-70 to report the GPU numbers as the
  primary budget evidence, keep the CPU numbers labeled "CPU, inflated ratio, informational," and
  state the device explicitly. Update the harness default to `--every-n-steps 200` (or document the
  default mismatch prominently) so `bench` defaults match the budget scenario.
- *If GPU still exceeds 10%:* do **not** quietly leave the ≤10% claim. Either (a) tighten the
  default recipe (move the heaviest diagnostics — `heavy_tail_alpha`, `singular_values` on the
  widest matrices — to default-OFF, then re-measure), or (b) amend design.md:439 and README.md:63 to
  state the *measured* budget with the device and recipe it was measured under. The §10 contract is
  load-bearing; a false ≤10% claim is worse than an honest "≤X% on GPU, full recipe."

**GPU-gated?** Yes, fully. The deliverable is a measured number on rtx. macbook can only produce the
(inflated) CPU number, which we already have.

### A4. ACDC follow-ons (additive kwargs) — **CPU-testable here**

**Location:** `src/circuitry/patching/acdc.py:363-403`, `ACDC.run(...)`. Confirmed extension points;
both are documented as explicit follow-ons in `docs/superpowers/specs/2026-05-25-acdc-design.md`
(zero/mean ablation at line 54; `eap_skip_threshold` at lines 198-202).

**A4a — `ablation_mode: str = "corrupted"`.** Today only corrupted-resample ablation exists
(`corr_act[u] - live[u]` delta, acdc.py:139/276). The cleanest insertion point is
`_cache_corrupted_acts()` (acdc.py:77-91), which returns the `corr_act` dict:

- `"corrupted"` (default, back-compat): today's behavior.
- `"zero"`: substitute zeros for the cached corrupted activations.
- `"mean"`: substitute the per-writer mean activation.

No existing caller breaks (default preserves current behavior).

**A4b — `eap_skip_threshold: float | None = None`.** Short-circuits the inner loop at
acdc.py:389-400: before `removed.add(edge)`, skip (treat as kept) any edge whose `|EAP score|`
exceeds the threshold:

```python
if eap_skip_threshold is not None and eap_scores is not None \
        and abs(eap_scores.get(edge, 0.0)) > eap_skip_threshold:
    continue   # assume kept; do not test
```

This is the 5-10× ecosystem speedup, opt-in (default `None` = test every edge = today's behavior).

**GPU-gated?** No. Both are pure CPU-testable additive kwargs on small toy graphs.

---

## 3. Workstream B — representational-drift probe (Option B)

The probe answers "how much has each layer's representation drifted from an anchor?" during
training, via a per-emit-step second forward pass on a fixed probe batch and a representational-
similarity metric against stored reference activations.

The layering split is strict (design.md:97-109):

- **`core/`**: the pure similarity primitive (tensor in → float out, no I/O, no `.cuda()`).
- **`recorder/live.py`**: probe wiring (second forward pass, temporary hooks, reference-snapshot
  lifecycle).
- **`recipes/`**: one new dataclass field + one diagnostic string key. No `cli/` import.

### 3.1 Core primitive (proposed signature)

New function in `src/circuitry/core/activation.py`, alongside `token_similarity` (activation.py:81):

```python
def repr_drift_cka(
    ref: torch.Tensor,        # reference activations, (n, d) or (batch, seq, d)
    cur: torch.Tensor,        # current activations, same shape semantics
    *,
    max_samples: int = 256,   # subsample rows before the Gram (CKA is O(n^2 d))
    eps: float = 1e-10,       # denominator guard for degenerate layers
    seed: int = 0,            # seeded subsample → CPU-deterministic
) -> float:
    """1 - linear_CKA(ref, cur). 0.0 == identical representation; ->1.0 == fully drifted."""
```

(Final metric choice is **Fork B-i** in §4. The signature above assumes linear CKA, the recommended
default.) Properties required by the `core/` contract:

- **Pure.** Two activation matrices in, one float out. No hooks, no logging, no side effects.
- **CPU-deterministic.** The only nondeterministic step is the `max_samples` subsample, which uses a
  **seeded** `torch.Generator` (same fix pattern as A1 — this is why A1 lands first conceptually).
- **No `.cuda()`.** Device taken from inputs.
- **float64 internally.** Center each matrix along axis 0 (column-centering — algebraically
  equivalent to double-centering the Gram for the linear kernel), compute Gram products in float64
  to avoid CKA values drifting slightly outside `[0, 1]`, add `eps` to the denominator, and return
  `0.0` (full drift) rather than NaN for a degenerate (all-zero / constant) layer.

Linear CKA is `||X_c^T Y_c||_F^2 / (||X_c^T X_c||_F · ||Y_c^T Y_c||_F)` (HSIC normalization). For a
small probe batch (`n ≤ 256`) the cost is two `(n, n)` Gram products — well within budget for an
opt-in per-emit diagnostic. `token_similarity` (activation.py:81-99) already builds a cosine Gram
matrix and is a structural reference for the implementation.

**Why not reuse `token_similarity`?** It computes off-diagonal cosine of token states *within one
tensor* — not a cross-checkpoint comparison. Drift needs a two-argument similarity. No CKA / drift /
representational-similarity helper exists in `core/` or `sae/` today (confirmed by grep).

### 3.2 Recorder probe path (second forward pass, default OFF)

The induction-score block (live.py:798-861) is the **exact, verbatim-reusable template**: it
registers temporary per-module forward hooks into a *local* dict, runs `self.model(probe)` under
`torch.inference_mode()` in a `try/finally` that removes the handles, then processes the captured
tensors. The drift probe mirrors this in `_run_diagnostics` under a new `if name == "drift_probe":`
branch.

Key facts that constrain the wiring:

- The main-pass `_captured_activations` dict is **cleared** after `_run_diagnostics` (live.py:595),
  so the probe pass needs its **own** local capture dict and its own temporary hooks — it cannot
  reuse the main-pass captures.
- `attach()` runs **no forward pass** (live.py:487 comment), so reference activations **cannot** be
  captured at attach time. They must be captured on the **first emit step** (the only viable slot;
  see §3.4).
- The probe pass runs inside `step()`, which is already rank-0-only (the no-op guard at the top,
  live.py:543). No additional distributed handling is needed (design.md §11 inherited).

Probe device: `next(self.model.parameters()).device`, then `probe.to(device)` — same as
induction-score (live.py:835-836).

### 3.3 Recipe field + enabled toggle

In `src/circuitry/recipes/__init__.py` (dataclass at lines 14-27):

- Add a plain dataclass field `probe_batch: torch.Tensor | None = None`. This is *configuration*,
  not a diagnostic name — it parallels `induction_probe_seq_len: int = 25` (line 26), which is also a
  plain config field rather than a diagnostic string. It is **not** part of
  `weight_diagnostics`/`activation_diagnostics`/`gradient_diagnostics`.
- Add the string `"drift_probe"` to the stock recipe's `activation_diagnostics`, gated OFF by
  default via `enabled={"drift_probe": False, ...}`. Because `_enabled(name)` defaults absent keys to
  `True` (live.py:606-607), the explicit `False` is what makes it opt-in.
- Possibly add `drift_max_samples: int = 256` and `drift_max_tokens: int | None` config fields
  (Fork B-vi). These mirror `lens_max_tokens` (recipes/__init__.py:27).

The `disable`/`only` helpers (recipes/__init__.py:72-114) already accept any string in the
diagnostic lists, so `recipe.only(["drift_probe"])` works with no helper changes.

> `recipes/` imports only `circuitry.recorder.hooks` (HookPoint, StepContext) today — adding a
> tensor field and a string introduces **no** new import and does **not** touch `cli/`. Layering
> holds.

### 3.4 Reference-capture lifecycle (with the v1.3 forced-copy lesson)

New `Recorder` state, declared in `__init__` alongside `_prev_weights` (live.py:158-159):

```python
self._ref_probe_activations: dict[str, torch.Tensor] | None = None  # anchor capture
self._drift_anchor_step: int | None = None
```

Lifecycle:

1. **Capture (first emit, inside `step()`):** when the `drift_probe` branch runs and
   `self._ref_probe_activations is None` and `recipe.probe_batch is not None`, run the probe pass,
   capture per-layer activations, and store them as the anchor. **No drift is emitted on the anchor
   step** (CKA against itself is trivially 1.0 / drift 0.0) — same "skip first emit" pattern as
   `update_delta`/`direction_cosine` (design.md:244-247).
2. **Compare (subsequent emits):** run the probe pass again, compute `repr_drift_cka(ref, cur)`
   per layer, emit one scalar per layer.
3. **Reset:** clear `_ref_probe_activations` in `detach()` (alongside `_prev_weights.clear()` at
   live.py:531-533), and expose a public `reset_drift_reference()` method for checkpoint-resume
   (mirrors the `_prev_weights` clear-in-detach pattern).

**The v1.3 forced-copy lesson is load-bearing here (live.py:597-603).** Reference activation tensors
captured from the probe pass **MUST** be stored with `.detach().to("cpu", copy=True)`. On a CPU
model `.to("cpu")` is a no-op, so without `copy=True` the "reference" aliases the probe-pass tensor,
which the next probe pass may overwrite — silently making CKA ≈ 1.0 forever. This is the exact
failure mode that made `update_delta` identically zero in v1.3 before the `copy=True` fix.

### 3.5 Sampling / max_tokens cap & storage budget

CKA is **O(n² d)**, quadratic in the row count `n`. The probe batch must be small and the row count
capped:

- **Probe batch:** small and fixed — e.g. 1-4 sequences. Documented in the `probe_batch` field
  docstring as the dominant cost.
- **`max_samples` cap (default 256):** enforced *inside the primitive* before the Gram, via a seeded
  subsample (A1 pattern). Doubling `n` quadruples Gram cost and memory.
- **`max_tokens` cap (optional):** truncate each sequence's token dimension before flattening to
  `(n, d)`, mirroring `lens_max_tokens`. Bounds `n = batch · tokens`.

**Storage math (why this must be bounded / opt-in):** reference activations are
`n_layers × n × d × 8 bytes` (float64) on CPU. For 256 rows × 4096 `d` × ~32 layers ≈ **~270 MB**.
Capping `max_tokens` and `max_samples`, and storing only the matched layers, keeps this bounded. The
field docstring must state the storage cost explicitly.

### 3.6 Tag namespace

Per-layer scalars, keyed by the same module-name convention as existing per-module diagnostics
(`weight.update_delta`, `induction_score/{mn}/head_{i}` at live.py:858):

```
activation/repr_drift_cka/<module_name>      # e.g. activation/repr_drift_cka/model.layers.5.mlp
```

Consumed naturally by the existing `MetricWriter.add_scalar` loop. The report `FLAG_RULES` can flag
any layer whose drift exceeds a configurable threshold (CKA below `cka_drift_threshold`, default
0.7; severe below 0.5 — see Fork B-iv on whether to emit per-layer or a single scalar).

---

## 4. OPEN FORKS (for user decision — NOT finalized)

Each fork lists 2-3 options, a recommended default, and rationale. The recommendations are
**proposals**; the user confirms before implementation.

### Workstream A forks

**Fork A1 — Recorder seed policy for `singular_values`.**
- (a) Recorder passes a **fixed constant seed** into the shared `_sv` helper so cross-step rank
  trajectories use the *same* subsample every step.
- (b) Recorder passes `seed=step` (subsample varies per step but is reproducible per run).
- (c) Primitive stays seed-less; document "set global torch seed before calling."
- **Recommend (a).** A fixed subsample makes `update_delta`/`rank_trajectory`/`direction_cosine`
  comparisons across steps reflect *weight* changes, not subsample churn. (b) adds noise to exactly
  the cross-step diagnostics; (c) is fragile (relies on the user).

**Fork A2 — Gram fast-path default.**
- (a) `use_gram="auto"` default (Gram when aspect ratio + dtype safe; fallback to `svdvals`
  otherwise).
- (b) `use_gram=False` default (opt-in only; zero behavior change).
- (c) `use_gram=True` default (always Gram on subsampled matrices).
- **Recommend (a)** with hard `svdvals` fallback when no subsampling occurred or when float32 +
  small-singular-value consumer (`condition_number`). Gains the perf win where it is safe; never
  silently degrades `condition_number`. If the user is conservative about any numerical change,
  fall back to (b).

### Workstream B forks (the drift-probe decisions)

**Fork B-i — Drift metric.**
- linear CKA / mean cosine similarity / RBF-kernel CKA.
- **Recommend linear CKA.** Parameter-free, pure-function, CPU-deterministic in float64, invariant
  to orthogonal rotation and isotropic rescaling (so LayerNorm-scale growth and LR-warmup uniform
  scaling don't masquerade as drift). Mean cosine is cheaper (O(n d)) but is fooled by isotropic
  rescaling and mean shifts and conflates rotation with magnitude. RBF-CKA needs a bandwidth
  `sigma` — a hidden hyperparameter that violates the pure-function spirit and whose mis-setting is
  a known reliability failure (Nguyen et al. 2022). SVCCA/Procrustes need `k`/regularization and are
  numerically unstable when `n < d` (the typical probe-vs-wide-layer case).

**Fork B-ii — Probe-batch source.**
- (a) `Recipe.probe_batch` kwarg tensor (user passes a fixed tensor).
- (b) file path (recipe loads a saved batch).
- (c) capture the first training batch automatically.
- **Recommend (a).** A plain `Tensor` field is the most explicit, testable, and layering-clean
  option (no I/O in `recipes/`), and parallels `induction_probe_seq_len`. (b) introduces file I/O
  into the recipe layer. (c) is convenient but couples the anchor to whatever the first batch
  happened to be (non-reproducible, and the "fixed" probe is no longer fixed across resumes).

**Fork B-iii — Reference anchor.**
- (a) Anchor at **step 0 / first emit**, fixed for the run.
- (b) **Resettable** anchor via `reset_drift_reference()` (re-anchors at the next emit).
- (c) User-specified anchor step.
- **Recommend (a) as the default behavior + (b) as an exposed method.** First-emit capture is the
  only viable slot (attach runs no forward, live.py:487). Exposing `reset_drift_reference()` covers
  checkpoint-resume and "re-anchor after warmup" without complicating the default. (c) adds config
  surface for a rare need; defer.

**Fork B-iv — Per-layer dict vs single scalar.**
- (a) Per-layer dict of scalars (`activation/repr_drift_cka/<module>`).
- (b) Single global scalar (e.g. min or mean CKA across layers).
- **Recommend (a).** Per-layer resolution is what makes drift actionable — it shows *where* the
  representation is moving (early vs late, attn vs MLP). A single scalar loses that. The report's
  `FLAG_RULES` can derive a global flag (e.g. `min_cka < 0.7`) from the per-layer scalars, so (a)
  subsumes (b).

**Fork B-v — Default-OFF mechanism.**
- (a) `"drift_probe"` in `activation_diagnostics` with `enabled={"drift_probe": False}` in the stock
  recipe (user opts in via `only`/clearing the disable).
- (b) Not in the stock recipe at all; user must add the string + a `probe_batch`.
- (c) Off purely by `probe_batch is None` (string present, enabled, but no-op without a batch).
- **Recommend (a).** Consistent with the existing `enabled`-gate philosophy for expensive
  diagnostics, discoverable (it shows up in `list-recipes`/matched output), and impossible to
  trigger accidentally because it *also* needs a `probe_batch`. (c) alone is too implicit (a user
  who sets a probe batch for some other reason would silently start paying the cost).

**Fork B-vi — Sampling cap location & defaults.**
- (a) Cap rows inside the **primitive** (`max_samples=256`); cap tokens in the **recorder** via a
  `drift_max_tokens` recipe field.
- (b) Cap everything in the recorder before calling the primitive.
- **Recommend (a).** The row cap is a numerical property of CKA (quadratic), so it belongs in the
  primitive where it is unit-testable in isolation; the token cap is a capture-time concern that
  belongs with the probe wiring. Defaults: `max_samples=256`, `drift_max_tokens` configurable
  (default `None` = use full sequence, with the storage warning in the docstring).

---

## 5. CPU-only test strategy (everything not GPU-gated)

All of the following run on macbook via `.venv/bin/pytest`. **Never bare `python`/`pytest`.**

**A1 — randperm determinism (`tests/core/test_weight.py`):**
- Build a matrix with `min(shape) > 512` so the subsample path fires. Assert
  `singular_values(W, seed=0) == singular_values(W, seed=0)` (bitwise / `allclose` exact).
- Assert `seed=None` preserves today's behavior (still draws; two unseeded calls may differ — assert
  the *shape* and that the values are a valid subsample, not equality).
- Recorder-level: two `Recorder` runs with a fixed seed produce identical `rank_trajectory` /
  `update_delta` series on a CPU toy model.

**A2 — Gram fast path tolerance (`tests/core/test_weight.py`):**
- For a battery of fixed wide matrices (varied aspect ratios, both float32 and float64), compare
  `singular_values(W, use_gram=True)` against `use_gram=False` (`svdvals`). Assert relative error on
  the **top-k** singular values within tolerance (e.g. `1e-5` float64, `1e-3` float32 large values).
- Assert the **auto** policy falls back to `svdvals` when (a) no subsampling occurs, and (b) float32
  + a `condition_number` consumer; verify `condition_number` is unchanged vs the `svdvals` path.
- Degenerate inputs (rank-deficient, all-zero column block): assert no NaN, finite results.

**A4 — ACDC kwargs (`tests/patching/test_acdc.py`):**
- `ablation_mode="zero"` / `"mean"` / `"corrupted"` on a tiny toy graph: assert the surviving
  circuit and recovery metric are computed without error and that `"corrupted"` reproduces the
  pre-change result (back-compat regression).
- `eap_skip_threshold`: with a threshold below all `|EAP|` scores, assert every edge is kept (no
  edge tested); with `None`, assert today's behavior; mid-threshold, assert only sub-threshold edges
  are pruned.

**B (drift) — primitive (`tests/core/test_activation.py`):**
- `repr_drift_cka(X, X) == 0.0` (identical → no drift), in `[0, 1]` for random pairs.
- Invariances: assert CKA-drift is unchanged under (i) orthogonal rotation of `cur`, (ii) isotropic
  rescaling `cur *= 2`. (These are the properties that justify CKA over cosine — they must be tests,
  not just claims.)
- Determinism: `seed=0` subsample gives identical output across calls when `n > max_samples`.
- Degenerate layer (all-zero `cur`): returns `0.0` or `1.0` deterministically (per chosen
  convention), never NaN; float64 path keeps output in `[0, 1]` (no `1.0 + 1e-7` overflow).

**B (drift) — recorder wiring (`tests/recorder/test_live.py`), CPU toy model:**
- First emit captures the anchor and emits **no** `repr_drift_cka/*` scalar; second emit emits one
  scalar per matched layer.
- **Aliasing regression (the v1.3 lesson):** mutate the model weights between emits and assert drift
  is **nonzero** — this fails if `copy=True` was dropped (reference aliases live storage → CKA stays
  1.0).
- Probe pass uses temporary hooks and `inference_mode`, and **removes all handles** in `finally`
  (assert no leftover handles, mirroring the induction-score test).
- `detach()` clears `_ref_probe_activations`; `reset_drift_reference()` re-anchors on the next emit.
- Default-OFF: a stock-recipe run with a `probe_batch` set but `drift_probe` disabled emits nothing.
- Layering test (`tests/test_layering.py`) stays green: `core/` gains no new imports; `recipes/`
  gains a tensor field + string, no `cli/` import.

**Bench (CPU smoke only):** run `scripts/bench_50m.py --device cpu` as a smoke test that the
instrumented loop (with drift OFF, then drift ON on a tiny probe) runs end-to-end. The **budget
number** is NOT validated on CPU — that is GPU-gated (§A3 / §6).

---

## 6. Layering compliance & design.md amendments

### 6.1 Layering (CI-enforced; design.md:97-109)

- **`core/repr_drift_cka` stays pure:** tensors in → float out, no hooks, no logging, no I/O. Imports
  only `torch`/`numpy`/`math` (and possibly `circuitry.core.*`, same layer). **No `.cuda()`** —
  device from inputs. ✓
- **`core/weight.py` A1/A2:** still imports only `math`/`numpy`/`torch`; `randperm`/`eigvalsh` use
  `device=M.device`, no hardcoded `"cuda"`. ✓
- **`recipes/`:** adds a tensor field + a diagnostic string + optional int config fields; imports
  unchanged (`circuitry.recorder.hooks` only); **no `cli/` import.** ✓
- **`patching/acdc.py` A4:** additive kwargs only; `patching/` may import `core/`/`recipes/`, must
  not import `cli/` — unchanged. ✓
- **`recorder/live.py`:** probe wiring imports `core/` primitives (allowed). ✓

### 6.2 design.md amendments required (same commit as code per project convention)

- **§4.1 catalog (design.md:107-163):** add `activation.repr_drift_cka(ref, cur, *, max_samples=256,
  eps=1e-10, seed=0) -> float` under the activation-space block. Add the `seed`/`use_gram` kwargs to
  the `weight.singular_values` listing (currently line 118).
- **§4.4 Recipe (design.md:221-247):** add `probe_batch: torch.Tensor | None = None` (and any
  `drift_max_samples`/`drift_max_tokens` fields) to the dataclass listing; note `"drift_probe"` is a
  default-OFF activation diagnostic; document the new `_ref_probe_activations` internal state and
  `reset_drift_reference()` alongside the existing `_prev_weights` note (the v1.3 forced-copy lesson
  applies identically).
- **§10 budget (design.md:439-445):** correct the budget evidence honestly after the GPU re-run
  (Fork-independent — see §A3). Either report GPU numbers as primary evidence with the device named,
  or amend the ≤10% claim to the measured value. Note that the drift probe adds a second forward
  pass and is **excluded from the default budget** (it is default-OFF); document its incremental cost
  separately. Fix the harness-default-cadence footgun (bench defaults to `every_n_steps=25`, budget
  is stated at `200`).
- **§4.1 invariant note (design.md:166-167):** once A1 lands, the "Deterministic on CPU" invariant is
  *actually* honored for the subsample path when a seed is supplied — note the `seed` requirement.
- **README.md:63-72:** update the budget statement and the measured-overhead table to the corrected
  GPU numbers (or the honest amended claim), and remove/relabel the stale CPU-only v0.2.0a0 figures.

---

## 7. Proposed task breakdown (perf first, then drift)

Ordered. CPU-testable on macbook unless marked **[rtx]**.

**Phase A — perf / determinism (must land first):**

1. **A1 — randperm seed fix** in `core/weight.py` + Recorder seed wiring (Fork A1). Tests:
   determinism on CPU. *CPU here.* (no perf cost; correctness gate.)
2. **A2 — Gram SVD fast path** in `core/weight.py` (Fork A2). Tests: numerical tolerance + auto
   fallback on CPU. *Correctness CPU here; perf claim deferred to task 4.* **[perf validation: rtx]**
3. **A4 — ACDC `ablation_mode` + `eap_skip_threshold`** in `patching/acdc.py`. Tests: toy-graph
   back-compat + new modes. *CPU here.*
4. **A3 — §10 budget re-validation on GPU** via `lab-control` Ray MCP (`ray_submit` to rtx). Run
   `bench_50m.py` at both cadences and both model sizes, post-A2. Then correct README/design
   honestly (required either way). **[rtx — GPU-GATED; cannot be done on macbook.]**

**Phase B — drift probe (after the budget is honest):**

5. **B-core — `repr_drift_cka` primitive** in `core/activation.py` (Fork B-i metric, B-vi cap).
   Tests: identity, invariances, determinism, degenerate. *CPU here.*
6. **B-recipe — `probe_batch` field + `"drift_probe"` default-OFF key** in `recipes/__init__.py`
   (Forks B-ii, B-v, B-vi). Tests: recipe construction, `only`/`disable`, layering. *CPU here.*
7. **B-recorder — probe pass + reference lifecycle** in `recorder/live.py` (Forks B-iii, B-iv).
   Reuse the induction-score template; apply the `copy=True` lesson; add `reset_drift_reference()`.
   Tests: anchor/compare/reset, aliasing regression, handle cleanup, default-OFF. *CPU here.*
8. **B-bench — incremental cost of drift ON** via `bench_50m.py` with a tiny probe. CPU smoke here;
   **[rtx]** for the reported incremental-overhead number (so the §10 amendment can state drift's
   cost honestly).
9. **docs — design.md §4.1/§4.4/§10 + README + CHANGELOG** amendments (§6.2). Same commit(s) as the
   code they describe, per project convention. *CPU here.*

**GPU-gated subset (rtx only):** tasks 4, the perf-claim half of task 2, and the reported-number
half of task 8. Everything else is CPU-testable on macbook now.

---

## 8. Open risks

- **GPU budget may still exceed 10% even on rtx.** Mitigation: recipe tightening (move heavy
  diagnostics default-OFF) or an honest amendment of the §10 number. Either way the claim must match
  the measurement.
- **Gram fast path numerics on float32 `condition_number`.** Mitigation: hard `svdvals` fallback for
  the small-singular-value consumer; tolerance tests gate the merge.
- **Drift reference storage blow-up** on large LLMs (~270 MB at 256×4096×32). Mitigation:
  `max_samples` + `max_tokens` caps, store only matched layers, default-OFF, documented cost.
- **The "+12.66%" brief artifact** could re-enter the changelog if not caught. Mitigation: the
  changelog must cite the real +14.9%/+14.7% (or the new GPU numbers), never +12.66%.
- **CKA degenerate-layer convention** (dead ReLU layer → all-zero `cur`) must be agreed: return 0.0
  (no drift) or 1.0 (full drift)? Recommend 0.0-with-a-note OR a sentinel — flagged for the user in
  the primitive docstring; tests pin whatever is chosen.
- **Probe-batch device / dtype mismatch** with the model (e.g. bf16 model, fp32 probe). Mitigation:
  cast probe to the model's param dtype/device in the recorder, like induction-score.
