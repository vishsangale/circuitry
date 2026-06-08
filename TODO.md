# circuitry — TODO / open items

Open work and known debt as of **v1.19.0** (2026-06-08). Completed items live in `CHANGELOG.md`
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

- [ ] **[M][perf] Further SVD cost reduction.** After the v0.9 sharing fix the irreducible cost
  is one full SVD per matrix (~1 s for 58 matrices on GPU). Two options investigated:
  (a) lower the `max_dim` default (accuracy trade-off), (b) eigvalsh on the Gram matrix W^T W
  (1.8× speedup in v1.10 prototype but degrades the spectral tail — deferred). Only needed if
  the ≤10% §10 budget must hold at aggressive cadences on fast GPUs.

---

## Patching & attribution

- [ ] **[L][feat] SAE-feature circuits — remaining follow-ons.** Shipped in v1.5–1.7: node
  attribution, feature→feature edges (attrib + IG), FeatureACDC. Remaining items noted in
  the ACDC plan:
  - Transcoder / matching-pursuit SAEs as intervention sites.
  - ~~Per-position feature edges~~ — shipped in v1.19 (`per_position=True` flag + `position_scores`).
  - Temporal / recurrent SAEs (require storing per-step feature activations across forward
    passes — different lifecycle than the current single-forward circuit).

---

## Multi-process (design §11 — additive future-release path)

- [ ] **[XL][feat] DDP / FSDP-aware reductions.** Current releases are single-process:
  non-zero ranks no-op and FSDP-sharded params give wrong rank-0 diagnostics. The upgrade
  path is additive (`TensorSource.WEIGHT_FULL` / `ACTIVATION_FULL` + `core/distributed.py`
  reduce helpers) with no existing-API rewrite. See design §11 for the contract.

---

## Repo hygiene

- [ ] **[XS][hygiene] Decide fate of `latent-superpowers-inspect` archival branch.** Lives
  on `feat/inspect-checkpoint-skill`; `pre-circuitry-extraction` tag preserves rollback.
  Decision deferred 2026-05-23 — revisit when there's a concrete reason to act.
