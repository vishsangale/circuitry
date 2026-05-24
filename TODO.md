# circuitry — TODO / open items

Tracking doc for open work and future improvements. Released through v0.9.0 (2026-05-23);
all tags + GitHub Releases v0.1.0 → v0.9.0 are published. The design contract is
`docs/design.md` — any change to a CI-enforced invariant must amend it first.

Legend: **[bug]** correctness · **[debt]** tech debt / cleanup · **[feat]** new capability ·
**[val]** validation / benchmarking · **[hygiene]** repo housekeeping.

---

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

- [ ] **[perf] GPU cost of SVD-based weight diagnostics dominates emission time.** ~4.1 s/emission
  on an 88M model (effective_rank / stable_rank / heavy_tail_alpha / sv_histogram, all via
  `svdvals` over ~80 matrices). Profile and consider GPU-aware mitigations: lower `max_dim`
  default, batch/skip some SVDs, a lighter default recipe on GPU, or documenting a larger default
  `every_n_steps` on GPU. (Surfaced by the item-10 GPU bench.)

## v1.0 — major

- [ ] **[feat] Causal / activation-attribution patching pillar.** (Research Q2: IOI, ACDC,
  AtP*, EAP, sparse-feature-circuits.) Deferred from v0.9 because patching crosses
  `docs/design.md`'s observation-only contract — requires a design amendment before work.

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
