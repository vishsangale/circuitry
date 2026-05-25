# circuitry — TODO / open items

Tracking doc for open work and future improvements. Released through v0.9.2 (2026-05-23);
all tags + GitHub Releases v0.1.0 → v0.9.2 are published. The design contract is
`docs/design.md` — any change to a CI-enforced invariant must amend it first.

Legend: **[bug]** correctness · **[debt]** tech debt / cleanup · **[feat]** new capability ·
**[val]** validation / benchmarking · **[docs]** documentation · **[hygiene]** repo housekeeping.

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

- [ ] **[feat] No easy way to disable / select a single diagnostic.** Dropping `logit_lens_kl`
  required `dataclasses.replace(get_recipe("llm"), activation_diagnostics=[…])`. Fix: a
  `disable=[...]` / `only=[...]` argument on `Recorder` / recipe.
- [ ] **[feat] `report` is a flat dump with no summary; no A/B compare.** A live run renders 80 KB+
  enumerating every module × diagnostic × step — fine for one model, unusable for comparison. Fix:
  a top-of-report summary/verdict (per-family final value + trend + flags), a compact mode, and a
  `circuitry compare run_a run_b` (per-family deltas, seed aggregation) — otherwise every comparison
  user re-writes a jsonl parser, as the reporter did.

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
- [ ] **Sub-spec 4 — ACDC** (iterative circuit pruning). Reuses EAP's edge graph.
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
