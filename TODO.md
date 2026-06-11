# circuitry — TODO / open items

Open work and known debt as of **v1.22.0** (2026-06-08). Completed items live in `CHANGELOG.md`
and git history. Design contract: `docs/design.md` — amend it before changing any
CI-enforced invariant.

**Effort:** [XS] < ½ day · [S] ½–1 day · [M] 1–3 days · [L] 1–2 weeks · [XL] 2+ weeks

---

## Pending validation

- [ ] **[S][val] Drift probe GPU forward-pass cost.** The opt-in `"drift_probe"` diagnostic
  requires a second forward pass per emit step and has never been benchmarked on GPU — it was
  off-by-default during the v1.4 A3 run. Benchmark at representative probe batch sizes and
  update `docs/design.md §10`.

---

## Performance

- [x] **[M][perf] SVD cost reduction — resolved by existing `use_gram='auto'`.** Benchmarked
  2026-06-08 on CPU (LLM-representative matrix shapes). Findings:
  - Square matrices (4k×4k, dominant in LLMs): eigvalsh is **26% slower** than svdvals on CPU.
    No gain to be had for square weight matrices.
  - Rectangular matrices (4:1 aspect): eigvalsh is **1.3–5× faster** (speedup grows with
    aspect ratio). `use_gram='auto'` already engages the Gram path at ≥3:1 — this is correct.
  - Accuracy: `rel_tail_err ≈ 1e-5`, `Δheavy_tail_alpha ≈ 0.0001` across all shapes at
    float64 promotion. The "degrades spectral tail" concern from the v1.10 prototype was
    pre-float64-promotion (v1.8 added float64 internally); no longer applies.
  - **Decision: no further action needed.** The current `use_gram='auto'` threshold (≥3:1)
    is already optimal. Lowering `max_dim` remains an opt-in hatch if GPU SVD cost ever
    becomes the budget bottleneck at aggressive cadences.

---

## Patching & attribution

- [x] **SAE-feature circuits — all follow-ons shipped (v1.5–1.22).** Node attribution, feature→feature edges (attrib + IG), FeatureACDC, per-position scores, `arch='parallel'` flag (v1.20), `TranscoderWrapper` (v1.21), `SAEFeatureTemporalRunner` (v1.22). True recurrent-SAE attribution (activations that depend on prior-step hidden state) remains a known limitation documented in the v1.22 CHANGELOG.

---

## Multi-process (design §11 — additive future-release path)

- [ ] **[XL][feat] DDP / FSDP-aware reductions.** Current releases are single-process:
  non-zero ranks no-op and FSDP-sharded params give wrong rank-0 diagnostics. The upgrade
  path is additive (`TensorSource.WEIGHT_FULL` / `ACTIVATION_FULL` + `core/distributed.py`
  reduce helpers) with no existing-API rewrite. See design §11 for the contract.
  **Progress (v1.45.0):** `core/distributed.py` shipped (`all_gather_concat`,
  DTensor-aware `full_tensor`, rank introspection; 2-process gloo tests).
  **Progress (v1.46.0):** `TensorSource.WEIGHT_FULL` / `ACTIVATION_FULL` shipped with
  participant mode on non-zero ranks (all-rank collectives, rank-0-only writes;
  2-process gloo recorder tests). Remaining: FSDP1 `summon_full_params` gathering,
  `DDPMetricWriter` cross-rank histogram aggregation, and a 2-process CI job.
  See `docs/plan-sota-3.md` v1.45.

---

## Repo hygiene

- [x] **[XS][hygiene] `latent-superpowers-inspect` archival branch** — branch
  `feat/inspect-checkpoint-skill` no longer exists on the remote (already cleaned up).
