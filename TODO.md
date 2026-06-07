# circuitry — TODO / open items

Open work and known debt as of **v1.16.0** (2026-06-07). Completed items live in `CHANGELOG.md`
and git history. Design contract: `docs/design.md` — amend it before changing any
CI-enforced invariant.

**Effort:** [XS] < ½ day · [S] ½–1 day · [M] 1–3 days · [L] 1–2 weeks · [XL] 2+ weeks

---

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

---

## Recorder & report

- [ ] **[S][feat] Per-hook-family module counts from `matched_modules.txt`.** The v1.15
  summary tag-count line counts emitted tags per family; a complementary count from
  `circuitry/matched_modules.txt` would show *matched* modules per family (detecting
  recipes that matched 0 modules). Current tag-count is a good proxy but not identical.

---

## Patching & attribution

- [ ] **[M][feat] Circuit result rendering (`to_markdown` / compare).** `EAPResult`,
  `AtPResult`, and `ACDCResult` have no human-readable output. Add a `.to_markdown()` method
  to each (top-K edge table + circuit graph stats) and a `circuitry compare circuit_a.json
  circuit_b.json` CLI subcommand for edge-set deltas between two circuits.

- [ ] **[L][feat] SAE-feature circuits — remaining follow-ons.** Shipped in v1.5–1.7: node
  attribution, feature→feature edges (attrib + IG), FeatureACDC. Remaining items noted in
  the ACDC plan:
  - Transcoder / matching-pursuit SAEs as intervention sites.
  - Per-position feature edges (current implementation is circuit-level, not position-level).
  - Temporal / recurrent SAEs (require storing per-step feature activations across forward
    passes — different lifecycle than the current single-forward circuit).

---

## CLI

- [ ] **[M][feat] `circuitry scan --model-factory` completion.** The `scan` subcommand
  currently exits `rc=2` with a "not yet exposed via CLI" message pointing to the programmatic
  API. Implement `--model-factory dotted.path:callable` (e.g.
  `my_project.models:build_model`) using the existing `_load_entrypoint` helper already used
  by `fit-tuned-lens`, and wire it through to `scan_run`.

---

## Future research directions


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
