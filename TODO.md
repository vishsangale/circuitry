# circuitry — TODO / open items

Open work and known debt as of **v1.48.0** (2026-06-12). Completed items live in `CHANGELOG.md`
and git history. Design contract: `docs/design.md` — amend it before changing any
CI-enforced invariant.

**Effort:** [XS] < ½ day · [S] ½–1 day · [M] 1–3 days · [L] 1–2 weeks · [XL] 2+ weeks

---

## Pending validation (needs GPU / distributed hardware)

- [ ] **[S][val] Drift probe GPU forward-pass cost.** The opt-in `"drift_probe"` diagnostic
  requires a second forward pass per emit step and has never been benchmarked on GPU — it was
  off-by-default during the v1.4 A3 run. Benchmark at representative probe batch sizes and
  update `docs/design.md §10`.
- [ ] **[M][val] Multi-GPU validation of `*_FULL` sources.** The v1.45/v1.46 distributed path
  is tested with 2-process CPU gloo only (`tests/core/test_distributed.py`,
  `tests/recorder/test_full_sources.py`). Validate on a real multi-GPU NCCL run (DTensor /
  FSDP2 sharding) and record findings under `docs/observations/`.

---

## Multi-process (design §11 — remaining increments)

- [ ] **[L][feat] FSDP1 flat-param gathering.** `core/distributed.py::full_tensor` handles
  DTensor (FSDP2); FSDP1 flat-param sharded modules still produce incorrect rank-0
  diagnostics. Requires `summon_full_params`-style gathering at emit steps behind an
  explicit opt-in flag with wall-clock budget enforcement (§10 still applies).
- [ ] **[M][feat] `DDPMetricWriter`.** Cross-rank histogram aggregation before writing
  (design §11: `MetricWriter` gains an optional `rank` kwarg; default TB adapter stays
  rank-0-only).

---

## Distribution (not library code)

- [ ] **[M][docs] mkdocs-material docs site** built from existing `docs/*.md` + API reference.
- [ ] **[M][docs] Tutorial notebooks** (3–4): IOI circuit end-to-end; live training
  monitoring; SAE feature circuits; graph export → Neuronpedia.
- [ ] **[M][bench] Published MIB + SAEBench numbers in README** (using `benchmarks/`;
  needs model downloads).
- [ ] **[L][docs] Short library paper** (JOSS or arXiv) for citability.
