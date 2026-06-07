# circuitry — TODO / open items

Open work and known debt as of **v1.14.0** (2026-06-07). Completed items live in `CHANGELOG.md`
and git history. Design contract: `docs/design.md` — amend it before changing any
CI-enforced invariant.

**Effort:** [XS] < ½ day · [S] ½–1 day · [M] 1–3 days · [L] 1–2 weeks · [XL] 2+ weeks

---

## Correctness & docs accuracy

- [ ] **[XS][docs] `gate_stats` return type mismatch.** `design.md §4.1` documents the return
  as `GateStats` (named dataclass); the implementation in `core/activation.py` returns a plain
  `dict[str, float]`. Fix: update the spec signature to `-> dict[str, float]`.

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

## Core primitives

- [ ] **[S][feat] `dynamics.grokking_step` — named helper for loss-curve segmentation.**
  `phase_transition_steps` is general-purpose; add a thin named wrapper
  `grokking_step(series, *, z_threshold=2.5) -> int | None` that returns the single largest
  transition step in a loss/accuracy series (or `None` if none detected). Surface it in the
  `## Training Dynamics` report section for loss-family tags.

---

## Recorder & report

- [ ] **[M][feat] Training dynamics — representational drift trending.** Wire
  `core.activation.repr_drift` into the `## Training Dynamics` section: accumulate a reference
  snapshot at the first emit step, then show per-layer drift (CKA) at each subsequent step.
  Currently `repr_drift` is emitted per-step as a raw scalar but never summarised as a
  trajectory trend in the report.

- [ ] **[S][feat] Per-hook-family module coverage in `build_report`.** Silent recipe
  under-coverage is invisible — a recipe matching 0 modules emits no tags and no warning in
  the report. Add a line per hook family to the `## Summary` block (e.g. "weight: 42 modules
  matched"), drawn from `circuitry/matched_modules.txt`. Complements the existing attach-summary
  table (which shows hook-point–level counts, not per-family).

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

- [ ] **[M][feat] Training dynamics remaining Q5 items.** v1.14 shipped
  `phase_transition_steps` / `head_formation_step` and the `## Training Dynamics` report
  section. Still open: `grokking_step` (see Core primitives above), weight-norm inflection
  curves (apply `phase_transition_steps` to `weight/update_delta_rel` time series and label
  the inflection "norm growth" or "norm collapse"), and representational drift trending
  (see Recorder section above).

- [ ] **[S][feat] Tooling-landscape positioning doc (Q6).** Write `docs/positioning.md`
  comparing circuitry's surface to TransformerLens, nnsight, pyvene, sae_lens — where
  circuitry is differentiated (training-live, modality-agnostic, SAE-circuit patching),
  where it defers. No code; outcome is a doc useful for README and contributor onboarding.

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
