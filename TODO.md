# circuitry — TODO / open items

Tracking doc for open work and future improvements. Released through **v1.8.0** (2026-06-01);
tags + GitHub Releases v0.1.0 → v1.8.0 are published. An Unreleased cycle on
`feat/recsys-and-attach-fix` carries: version single-sourcing; a dense-model strict-attach fix
(`HookPoint.optional`); the sequential-recsys recipe; and the real-model-eval follow-ups —
NaN-aware/`valid_mask` attention entropy (recsys B), broadened `attention_head_rank` metadata
resolution (fingerprint #1), `sae-lens`/`tensorboard` as optional extras + `writer="auto"`
(fingerprint #2), `Recipe.forward_fn` for non-HF probe passes (recsys C/D), and flexible
`scan_run(checkpoints=...)` discovery (fingerprint #3). The design contract is `docs/design.md`
— any change to a CI-enforced invariant must amend it first.

Legend: **[bug]** correctness · **[debt]** tech debt / cleanup · **[feat]** new capability ·
**[val]** validation / benchmarking · **[docs]** documentation · **[hygiene]** repo housekeeping.

---

## v1.4.x follow-ups / pending validation

- [x] **[val] §10 GPU budget re-validation (A3).** Done (2026-05-30, RTX 5080, 88M decoder,
  full llm recipe, `every_n_steps=200`): **+5.3% at a realistic step (batch 16 × seq 512)** —
  within the ≤10% budget. The tiny default batch (4 × 64) shows +45% because the baseline step
  is ~10× cheaper and the fixed per-emit diagnostic cost dominates the ratio. §10 + README
  updated. (Also surfaced + fixed a GPU device-mismatch crash in update_delta/direction_cosine,
  shipped as v1.4.1.)
- [ ] **[val] Drift probe forward-pass cost on GPU.** Not yet characterised — the drift probe
  is off by default so it wasn't in the A3 run. Benchmark the opt-in second forward pass
  (`Recipe.probe_batch` set, `"drift_probe"` enabled) at representative probe batch sizes when
  someone enables it in production.

## v1.3.x follow-ups

- **[debt] `update_delta_vanishing` flag uses an absolute L2-norm threshold** (`1e-6` in
  `recorder/report.py` `FLAG_RULES`). This is scale-dependent: a healthy step on a large
  matrix can exceed it while a healthy step on a tiny one can fall below it. Consider
  normalizing by parameter element count or by `‖W‖` before thresholding. Non-blocking;
  flagged in the v1.3 adversarial review.

---

## External feedback — real-model evaluations (2026-06-01 / 06-02)

Two field reports run against the freshly-released v1.8.0:
`docs/observations/2026-06-01-leaderboard-fingerprint-feedback.md` (67 custom dense 1M-param
LMs, retrospective weight fingerprinting) and
`docs/observations/2026-06-01-recsys-sasrec-evaluation.md` (trained SASRec D=64). Two items were
actioned this cycle (CHANGELOG Unreleased); the rest are triaged below with severity.

**Already shipped (Unreleased):**
- [x] **[bug] Dense-model strict attach (fingerprint #7, HIGH).** MoE-only `llm`-recipe HookPoints
  hard-failed `strict=True` attach on every dense model. Fixed via `HookPoint.optional`
  (`tests/recorder/test_optional_hookpoints.py`); also resolves the F37 warning-noise follow-up.
- [x] **[feat] Sequential-recsys recipe (recsys finding A).** Shipped `recipes/recsys.py`
  (SASRec / BERT4Rec / GRU4Rec), complementary to `two_tower`.

**Correctness:**
- [x] **[bug] `attention_pattern_entropy` returns NaN for left-padded models (recsys B).** Done
  (Unreleased) — the per-head mean is now NaN-aware (drops fully-`-inf`-masked PAD rows
  automatically; identical on unpadded patterns) and accepts an optional `valid_mask`
  (`True`=valid query row, broadcastable to `(B, H, T)`, 2-D `(B, T)` auto-expanded across heads)
  in `core/attention.py::attention_pattern_entropy`. NOTE-C updated. Test:
  `tests/core/test_attention.py`.

**Usability / API (MEDIUM):**
- [x] **[feat] `attention_head_rank` head-metadata resolution (fingerprint #1 + recsys D-variant).**
  Done (Unreleased) — resolution order is now (a) explicit `Recipe.attn_head_meta`
  (`n_heads`/`n_kv_heads`/`head_dim`/`hidden_size`); (b) any submodule exposing a `.config` with
  `num_attention_heads` (covers `model.config`, `text_config`, HF-wrapped `model.model.config`);
  (c) `num_heads`/`head_dim` attrs read off a `self_attn`/`attn`/`attention` submodule. The "no
  usable metadata" warning moved to `attach()` and names what was searched. Test:
  `tests/recorder/test_head_meta_resolution.py`.
- [x] **[feat] Custom forward entry point — `Recipe.forward_fn` (recsys C/D).** Done (Unreleased) —
  added `Recipe.forward_fn(model, batch) -> output`; the recorder's internal probe passes
  (`induction_score`, `drift_probe`) call it when set, else fall back to `model(probe,
  output_attentions=True)` with a `TypeError` fallback. A non-HF model with no resolvable
  `config` now warns once at `attach()` pointing at `forward_fn`. (Recipe-level `attn_kwargs` for
  `need_weights=True` injection remains a possible future refinement.) Test:
  `tests/recorder/test_forward_fn_and_scan_discovery.py`.
- [x] **[feat] `scan_run` checkpoint discovery is too rigid (fingerprint #3).** Done (Unreleased)
  — `scan_run(checkpoints=...)` accepts a single file, a glob string, a list of paths, or a list
  of explicit `(step, path)` pairs; `step*.pt` stays the default. Enables single-snapshot +
  arbitrary-named retrospective scans. Test: `tests/recorder/test_forward_fn_and_scan_discovery.py`.
- [x] **[debt] `sae-lens` / `tensorboard` are hard deps but only lazy-imported (fingerprint #2).**
  Done (Unreleased) — moved to extras (`circuitry[sae]`, `circuitry[tensorboard]`, `circuitry[all]`);
  core install is `torch` + `numpy` only. Default `writer="auto"` (tensorboard if installed, else
  jsonl with a one-time warning); explicit `writer="tensorboard"` and the SAE loader raise a
  friendly install-pointing `ImportError`. `import circuitry` never pulls either. Test:
  `tests/test_optional_deps.py`.

**DX / docs (LOW):**
- [x] **[feat] `.only()` / `.disable()` are invisible on the dataclass lists (fingerprint #4).**
  Done (Unreleased) — added `Recipe.effective_diagnostics()` (declared lists minus disabled,
  grouped by family) and `Recipe.active_diagnostics` (flat list); `.only()` / `.disable()`
  docstrings now point at them. Test: `tests/recipes/test_recipe_helpers.py`.
- [x] **[docs] Static vs trajectory diagnostics on single-snapshot scans (fingerprint #5).** Done
  (Unreleased) — `scan_run` warns once when trajectory diagnostics (`update_delta` /
  `rank_trajectory` / `direction_cosine`) are requested with <2 checkpoints (they compare
  consecutive snapshots), and the split is documented in the `scan_run` docstring + design §4.2.
  Test: `tests/recorder/test_scan.py`.
- [x] **[docs] `sv_histogram` emits artifacts, not scalars (fingerprint #6).** Done (Unreleased)
  — `sv_histogram` now also emits `spectral/per_param/<m>/sv_max`, `sv_min`, and `spectral_entropy`
  scalars (new `core.weight.spectral_entropy` primitive) so the spectrum shows up in tabular/CSV
  exports. Tests: `tests/core/test_weight.py`, `tests/recorder/test_step_extensions.py`.
- [x] **[feat] `in_proj_weight` unreachable by the recipe DSL (recsys follow-up #3).** Done
  (Unreleased) — added `TensorSource.NAMED_PARAM`, whose `pattern` matches parameter names and
  feeds the matched ≥2-D parameter to the weight diagnostics. Reaches the fused
  `nn.MultiheadAttention.in_proj_weight` (and any other named param) that the WEIGHT
  primary-weight resolver can't. recsys NOTE-A + design §4.4 updated. Test:
  `tests/recorder/test_named_param_source.py`.
- [x] **[bug] Gradient diagnostics emit nothing on SASRec (recsys follow-up #5).** Done
  (Unreleased) — root cause: the recorder populates `ctx.gradients` ONLY from `TensorSource.GRAD`
  hook points, but the `recsys` recipe (and, latently, `two_tower` and `vision`) declared a
  gradient diagnostic with no GRAD hook point. Each recipe now ships GRAD hooks mirroring its
  WEIGHT patterns; a guard test (`tests/recipes/test_gradient_hookpoints.py`) asserts every stock
  recipe with a gradient diagnostic has a GRAD hook point. (Only `llm` had wired it correctly.)

---

## External feedback — softmax-vs-sigmoid attention study (2026-05-23)

Field report from a user running circuitry to compare softmax vs sigmoid attention
(HF-Llama, 1M & 10M params, 16 GB GPU, live `Recorder` + jsonl). Verified against current
`main` (HEAD `e1bfbd5`); line references and staleness notes are mine. Suggested release
grouping is a proposal, not yet decided.

**Correctness / robustness (✅ SHIPPED in v0.9.2):**

- [x] **[bug] `attention_pattern_entropy` is not normalization-invariant.** Done (v0.9.2) — rows are
  normalized (eps-clamped) before entropy; Recorder warns once on non-normalized rows. Original detail:
  `core/attention.py:56-62`
  computes raw `-Σ xlogy(p,p)` over the key axis. Valid for softmax (rows sum to 1), but for
  attention whose weights don't sum to 1 (sigmoid / some linear attention) it conflates
  *concentration* with *total attention mass* — the number isn't a true entropy and
  cross-architecture comparison is confounded. **This blocked the reporter's core experiment**
  (their highest-value item). Fix: normalize rows by their sum before entropy, or emit a separate
  normalization-invariant concentration metric; warn when row-sums deviate from 1. Confirmed
  accurate against current code.
- [x] **[bug] `logit_lens_kl` OOMs at modest scale and takes down the whole run.** Done (v0.9.2) —
  token-chunked (`chunk_size=256`); per-layer OOM warns + skips + keeps the run alive; new
  `Recipe.lens_max_tokens` cost lever. (`lens_layers` intentionally deferred.) Original detail: `core/lens.py:72`
  materializes a full `(batch, seq, vocab)` logits tensor, upcast to float32 (lines 66/69 — doubles
  the footprint vs bf16); the Recorder calls it once per layer, so cost accumulates across layers
  (~1.5 GB/layer at seq 512, vocab 47k → OOM on a 16 GB GPU at 10M params). Fix: chunk over
  layers/seq and free between; subsample positions/layers; run in model dtype; degrade gracefully
  (skip + warn) on OOM instead of crashing the run. Expose `lens_layers` / `lens_max_tokens` knobs.
  Confirmed accurate.
- [x] **[bug] `output_attentions=True` kwarg injection breaks on wrapped models.** Done (v0.9.2) —
  enabled via `config.output_attentions` (set as the final `attach()` step, restored on `detach()`)
  instead of forward-kwarg injection; no more `TypeError` on `**kwargs`-less wrappers. Original detail:
  `recorder/live.py:427-435`
  installs a forward-pre-hook on the passed model that injects `output_attentions=True` into kwargs;
  a thin wrapper whose `forward(input_ids, labels)` lacks `**kwargs` raises `TypeError` (the reporter
  had to hand it the inner `LlamaForCausalLM`). Fix: prefer `config._attn_implementation="eager"` /
  `config.output_attentions=True` over kwarg injection (aligns with the v0.9.1 SDPA-warn path at
  `live.py:256`), or catch `TypeError` and fall back, or accept a `target_module` arg; at minimum
  document "attach to the HF model, not a wrapper." Confirmed accurate.

**Ergonomics (informs the v1.0 surface):**

- [x] **[feat] No easy way to disable / select a single diagnostic.** Dropping `logit_lens_kl`
  required `dataclasses.replace(get_recipe("llm"), activation_diagnostics=[…])`. Fix: a
  `disable=[...]` / `only=[...]` argument on `Recorder` / recipe. — Done (v1.2.0): `Recipe.disable(names)` and `Recipe.only(names)` methods added (`recipes/__init__.py`); immutable helpers over the existing `enabled` dict, fail-fast on unknown names, custom callables unaffected.
- [x] **[feat] `report` is a flat dump with no summary; no A/B compare.** A live run renders 80 KB+
  enumerating every module × diagnostic × step — fine for one model, unusable for comparison. Fix:
  a top-of-report summary/verdict (per-family final value + trend + flags), a compact mode, and a
  `circuitry compare run_a run_b` (per-family deltas, seed aggregation) — otherwise every comparison
  user re-writes a jsonl parser, as the reporter did. — Done (v1.2.0): `build_report` gains a `## Flags` verdict block (declarative rules, gated on >1 step) + `compact=True` / `--compact` mode; `circuitry compare run_a run_b` subcommand added (`recorder/compare.py`: `compare_runs` / `build_compare_report`), family/diagnostic-granular deltas + trend agreement.
- [ ] **[bug] `build_report` per-tag `Δ` column shows the unsigned *range* (`vmax − vmin`), not the
  signed `last − first`.** Surfaced during the v1.2.0 review: a monotonically *decreasing* metric
  (e.g. `effective_rank: 15 → 5`) renders `Δ = 10` in the table, reading like an increase. Pre-existing
  (`_stats` in `recorder/_metrics.py`); the new v1.2 `## Flags` block and `compare` correctly use a signed
  `last − first`, so the table is now the only place with range semantics. Deferred: fixing it changes
  existing report output and the `test_report` golden expectations — schedule as a v1.2.x point-fix with a
  test update, not folded into the feature release.

**Trivial / docs (✅ SHIPPED in v0.9.2):**

- [x] **[feat] No `circuitry --version`.** Done (v0.9.2) — added to the top-level parser.
- [x] **[docs] Clarify `scan` vs. live `report`.** Done (v0.9.2) — README + design.md now document both
  the live `metrics.jsonl` path and the checkpoint-scan →
  `findings.json` → `report` flow.

**Already addressed in current `main` (no action — reporter's item #4):**

- ✅ **MLP weights ARE hooked.** The reporter saw only the `.*\.(q|k|v|o)_proj$` weight pattern, but
  `recipes/llm.py:13-14` also hooks `w1|w2|w3|gate_proj|up_proj|down_proj` as `WEIGHT` (since the
  original recipe, commit `1a9b9bf`), so `effective_rank` / `stable_rank` / `heavy_tail_alpha` /
  `sv_histogram` already cover gate/up/down_proj. Likely an older snapshot or a partial read of the
  recipe — no rank-analysis blind spot on current main.

## v0.9.1 — near-term (tech debt surfaced during v0.9 validation)

- [x] **[bug] Lens dispatcher over-iterates activation sources.** Done — the `logit_lens_kl`
  dispatcher now keeps only activations whose module name ends in `.layers.N` (residual-stream
  block boundary), not every d_model-shaped capture (175 → 35 on Gemma 4). d_model retained as
  a secondary guard; +1 regression test (198 total).
- [x] **[debt] Test-filter DRY violation.** Done — extracted to a shared
  `llm_recipe_no_hf_diagnostics` fixture + `HF_ONLY_ACTIVATION_DIAGNOSTICS` constant in
  new `tests/conftest.py`; e2e + perf tests consume the fixture. 194 tests pass, ruff clean.
- [x] **[bug] SDPA silently emits zero induction/entropy tags.** Done — `Recorder.attach()`
  now WARNs when `induction_score`/`attention_pattern_entropy` is requested but the resolved
  `config._attn_implementation` (or `text_config`) is non-eager, pointing at the
  `attn_implementation="eager"` workaround. Only fires on positive detection; non-HF models
  stay quiet. +3 tests (197 total).
- [x] **[debt] `layer_norm` gradient diagnostic naming is misleading.** Done — renamed to
  `grad_norm_per_module` (core primitive + diagnostic key + `gradient/grad_norm_per_module/<m>`
  tag); `vision` / `two_tower` recipes updated; design.md + CHANGELOG updated. Breaking
  tag-namespace change. 194 tests pass.
- [x] **[debt] Promote `norms_per_param`'s per-module compute into a core primitive.** Done —
  added `core.gradient.total_grad_norm(per_module_norms)`; `norms_per_param` now composes
  `grad_norm_per_module` + `total_grad_norm` instead of inlining the loop. `grad_norm_per_module`
  also gained float32 precision. Emitted tags unchanged. +5 tests (203 total).

## v0.9.x — validation / benchmarking debt

- [x] **[val] Capture induction_score + attention_pattern_entropy on a real model.** Done —
  `smoke_hf_model.py` gained `--attn-impl`/`--device`; Qwen2.5-0.5B under eager emits real
  per-head values (induction_score up to 0.986 → a strong induction head; entropy 0–2.88 nats),
  SDPA emits zero + fires the item-6 WARN. See observations doc.
- [x] **[val] Re-validate SAE on a representative sequence.** Done — 189-token prose on Gemma 2 2B
  + gemma-scope layer-8 SAE: recon_mse 3130→83.5, l0 1293→106.5 (near the ~71 design point;
  residual gap plausibly bf16). The 4-token v0.9.0 numbers were non-representative as suspected.
- [x] **[val] GPU re-measurement of the performance budget.** Done — **negative result**: at
  aggressive cadence the v0.9 stock recipe blows the ≤10% §10 budget on GPU (+1202% at every-25;
  +257–306% at lower cadences). Per-emission cost ~4.1 s (≈290 training steps) is dominated by
  SVD weight diagnostics, which don't shrink on GPU while the training step does. CPU budget
  holds. Bring-up also surfaced + fixed 3 GPU device bugs in `core/`. See observations doc + the
  follow-up below.

- [x] **[perf] GPU cost of SVD-based weight diagnostics dominates emission time.** Done (primary
  win) — the SVD was computed 4× redundantly (each of effective_rank / stable_rank /
  heavy_tail_alpha / sv_histogram called `singular_values` independently). Recorder now computes
  it once per matrix per step and shares it: ~4× fewer SVDs, per-emission ~4.1 s → ~1.08 s on the
  88M GPU bench (+1202% → +306% at every-25). Helps CPU equally. 206 tests pass.
- [ ] **[perf] Further SVD cost reduction (optional).** After sharing, the irreducible cost is one
  SVD per matrix (~1 s for 58 matrices on GPU). Headroom: lower `max_dim` default (accuracy
  trade-off), or compute singular values via eigvalsh on the smaller Gram matrix (W^T W). Only
  needed if the ≤10% §10 budget must hold at aggressive cadences on fast GPUs.

## v1.0 — major: causal / activation-attribution patching pillar

Release banner spanning sequential sub-spec cycles (each its own spec→plan→implement).
Specs + plans live under `docs/superpowers/`.

- [x] **Sub-spec 1 — core intervention primitive.** Done (2026-05-24, on `main`, unreleased).
  `circuitry.patching` (`Site` / `patch_site` / `PatchRunner` / `HFSiteResolver` + `TLSiteResolver`)
  + `core/patching.py` metrics (`logit_diff` / `kl_divergence` / `ce_loss`) + `docs/design.md` §4.6
  intervention-mode contract amendment. 276 tests, Gemini-reviewed.
- [x] **Sub-spec 2 — EAP** (edge attribution patching). Done (2026-05-24, on `main`, unreleased).
  `circuitry.patching.graph` (Node/Edge/build_graph) + `EAPRunner`/`EAPResult` (2-fwd+1-bwd analytic
  scoring) + vanilla & activation-path EAP-IG (`ig_steps`) + TL and HF (RMSNorm/GQA, Llama-family)
  backends + `core/patching` `_t` differentiable metrics. Exact per-edge cross-check vs `patch_site`
  on linear toys; rank-correlation on real HF Llama. 300 tests, Gemini-reviewed clean.
- [x] **Sub-spec 3 — AtP\*** (attribution patching + corrections). Done (2026-05-25, on `main`,
  unreleased). `circuitry.patching.atp` (`AtPRunner`/`AtPResult`/`AtPNode`): vanilla AtP +
  QK fix (attn-pattern recomputation, RoPE-aware) + GradDrop + neuron-level nodes + `verify_top_k`
  (real `patch_site` calibration); HF (eager, Llama, GQA) + TL backends. Exact vanilla/neuron gates
  vs brute-force on linear toys; QK fix beats vanilla on real Llama. 330 tests, Gemini-reviewed
  (fixed GQA v-slot crash + GradDrop spec sync). `transformers` added as approved lazy optional dep.
- [x] **Sub-spec 4 — ACDC** (iterative circuit pruning). Done (2026-05-25, on `main`, unreleased).
  `circuitry.patching.acdc` (`ACDCRunner`/`ACDCResult`): greedy reverse-topological edge pruning with
  corrupted-resample set ablation (corrupted−live deltas, pre-LN per-head injection), last-token
  KL-to-clean recovery (configurable position + custom metric), `tau` threshold + `sweep` Pareto helper,
  `topo`/`eap` ordering, HF (eager, Llama, GQA kv-group) + TL backends. Exact empty-/full-ablation
  anchors. 128 new tests, Gemini-reviewed. v1.0 patching pillar (4 of 4 sub-specs) complete.
  **Follow-ons:** mean / zero ablation modes; the EAP-score skip-speedup (`eap_skip_threshold`);
  per-query-head k/v under GQA; SAE-feature circuits; report/compare integration.
- [ ] **later — SAE-feature circuits** (intervention site = SAE feature; builds on the SAE primitive).

## Future research / features (from `docs/v0.9-research` ledger)

- [ ] **[feat] Training-dynamics diagnostics** (Q5): grokking signals, induction-head
  formation curves, representational drift over a run.
- [ ] **[feat] Tooling-landscape gap analysis** (Q6): position vs TransformerLens / nnsight /
  pyvene / sae_lens; identify the differentiated surface.
- [ ] **[feat] Tuned lens** (extends Q3 logit lens) and **copy-suppression heads** (extends Q4).

## Multi-process (design §11 — additive future-release path)

- [ ] **[feat] DDP / FSDP-aware reductions.** Current releases are single-process; non-zero
  ranks no-op and FSDP-sharded params give wrong rank-0 diagnostics. The upgrade is additive
  (new `TensorSource.WEIGHT_FULL` / `ACTIVATION_FULL` + `core/distributed.py`); no API rewrite.
  See design §11.

## Repo hygiene

- [x] **[hygiene] Add `.claude/` to `.gitignore`.** Done — `.claude/` ignored, no longer in `git status`.
- [x] **[hygiene] Resolve stale working-tree plans.** Done — deleted `docs/plan-v0.8.md`
  (shipped v0.8 scaffolding) and `docs/v0.9-research/.plan-mech-interp-tooling-2024-2026.md`
  (stale ledger). Both were untracked. The cited `lit-review.md` / `lit-review-provenance.md`
  artifacts in `docs/v0.9-research/` are retained.
- [ ] **[hygiene] Decide fate of `latent-superpowers-inspect` archival branch.** Lives on
  `feat/inspect-checkpoint-skill`, not main; `pre-circuitry-extraction` tag preserves rollback.
  **Deferred by decision (2026-05-23): leave as-is, revisit later** — out of scope for circuitry
  repo work; the tag already preserves rollback.
