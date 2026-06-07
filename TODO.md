# circuitry — TODO / open items

Open work and known debt as of **v1.14.0** (2026-06-07). History of completed items lives in
`CHANGELOG.md` and git. The design contract is `docs/design.md` — any change to a CI-enforced
invariant must amend it first.

Legend: **[bug]** correctness · **[debt]** tech debt / cleanup · **[feat]** new capability ·
**[val]** validation / benchmarking · **[docs]** documentation · **[hygiene]** repo housekeeping.

---

## Pending validation

- [ ] **[val] Drift probe forward-pass cost on GPU.** Not yet characterised — the drift probe
  is off by default so it wasn't in the v1.4 A3 GPU run. Benchmark the opt-in second forward pass
  (`Recipe.probe_batch` set, `"drift_probe"` enabled) at representative probe batch sizes when
  someone enables it in production.

## Open tech debt

- [ ] **[perf] Further SVD cost reduction (optional).** After the v0.9.x sharing fix, the
  irreducible cost is one SVD per matrix (~1 s for 58 matrices on GPU). Headroom: lower
  `max_dim` default (accuracy trade-off), or compute singular values via eigvalsh on the smaller
  Gram matrix (W^T W). Only needed if the ≤10% §10 budget must hold at aggressive cadences on
  fast GPUs.

## Future features

- [ ] **[feat] SAE-feature circuits** (intervention site = SAE feature; builds on the SAE
  primitive and the v1.0 patching pillar).

- [ ] **[feat] Training-dynamics — remaining Q5 items.** `core/dynamics.py`
  (`phase_transition_steps`, `head_formation_step`) and the `## Training Dynamics` report section
  shipped in v1.14. Remaining: full grokking-signal suite (loss-curve segmentation, weight-norm
  inflection detection), representational drift trending over a run.

- [ ] **[feat] Tooling-landscape gap analysis** (Q6): position vs TransformerLens / nnsight /
  pyvene / sae_lens; identify the differentiated surface.

## Multi-process (design §11 — additive future-release path)

- [ ] **[feat] DDP / FSDP-aware reductions.** Current releases are single-process; non-zero
  ranks no-op and FSDP-sharded params give wrong rank-0 diagnostics. The upgrade is additive
  (new `TensorSource.WEIGHT_FULL` / `ACTIVATION_FULL` + `core/distributed.py`); no API rewrite.
  See design §11.

## Repo hygiene

- [ ] **[hygiene] Decide fate of `latent-superpowers-inspect` archival branch.** Lives on
  `feat/inspect-checkpoint-skill`, not main; `pre-circuitry-extraction` tag preserves rollback.
  Deferred by decision (2026-05-23): leave as-is, revisit later.
