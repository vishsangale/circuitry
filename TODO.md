# circuitry — TODO / open items

Tracking doc for open work and future improvements. Released through v0.9.0 (2026-05-23);
all tags + GitHub Releases v0.1.0 → v0.9.0 are published. The design contract is
`docs/design.md` — any change to a CI-enforced invariant must amend it first.

Legend: **[bug]** correctness · **[debt]** tech debt / cleanup · **[feat]** new capability ·
**[val]** validation / benchmarking · **[hygiene]** repo housekeeping.

---

## v0.9.1 — near-term (tech debt surfaced during v0.9 validation)

- [ ] **[bug] Lens dispatcher over-iterates activation sources.** `logit_lens_kl` iterates
  ALL d_model-shaped activations (175 entries on Gemma 4 = 35 layers × 5 sources:
  self_attn / mlp / layernorm outputs, not just block outputs). Should filter by
  HookPoint `source` so only block (residual-stream) outputs feed the lens.
  Today the d_model filter masks crashes but still does wasted work and may mix sources.
- [x] **[debt] Test-filter DRY violation.** Done — extracted to a shared
  `llm_recipe_no_hf_diagnostics` fixture + `HF_ONLY_ACTIVATION_DIAGNOSTICS` constant in
  new `tests/conftest.py`; e2e + perf tests consume the fixture. 194 tests pass, ruff clean.
- [ ] **[bug] SDPA silently emits zero induction/entropy tags.** On SDPA-default HF models
  (Gemma 2/4 and most production models) `induction_score` and `attention_pattern_entropy`
  emit zero tags because SDPA doesn't return per-head weights. Workaround is
  `attn_implementation="eager"`. Add a WARN at attach time when these diagnostics are
  requested but the model uses SDPA, pointing at the eager workaround.
- [ ] **[debt] `layer_norm` gradient diagnostic naming is misleading.** Produces identical
  numbers to `norms_per_param` under a different tag prefix (noted in v0.4.0 CHANGELOG).
  Still used by `vision` / `two_tower` recipes — candidate for rename or removal.

## v0.9.x — validation / benchmarking debt

- [ ] **[val] Capture induction_score + attention_pattern_entropy on a real model.** Both were
  never actually captured (SDPA blocked them). Re-run a small model with
  `attn_implementation="eager"` and record real numbers.
- [ ] **[val] Re-validate SAE on a representative sequence.** v0.9 SAE numbers came from a
  non-representative 4-token input (recon_mse 3130, l0 1293 vs expected ~71 at scale).
  Re-run on a realistic sequence length.
- [ ] **[val] GPU re-measurement of the performance budget.** Bench numbers are CPU-only
  (~15% overhead claim, GPU TBD since M2). Confirm the ≤10% wall-clock budget (design §10)
  holds on GPU, including the v0.9 lens + induction probe pass (+134% Phase-4 on CPU).

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
