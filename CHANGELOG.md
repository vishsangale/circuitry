# Changelog

All notable changes to this project will be documented in this file. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] — 2026-05-21

### Removed
- `circuitry.writers.wandb` and the `[wandb]` extras-gated dependency. The wandb adapter had no in-house consumers after the mendu cutover. `MetricWriter` protocol keeps any third-party logger a ~50-LOC subclass — re-addable if demand surfaces.

### Changed
- `Recorder(writer="wandb")` now raises `ValueError`. Allowed string values: `"tensorboard"`, `"jsonl"`, `"null"`. Pass a `MetricWriter` instance for anything else.

## [0.2.0] — 2026-05-21

### Added
- `circuitry.core.activation.token_similarity(h)` — ported from `mendu/paper2/.../spectral_diagnostics.py`.
- `circuitry.core.weight.update_delta(sd1, sd0)` and `direction_cosine(sd2, sd1, sd0)` — weight-dynamics primitives.
- `circuitry.recipes._discovery.discover(model)` re-exported as `circuitry.discover` — LLaMA-family arch discovery.
- `Recorder.step(loss=..., loss_components=...)` — emits `train/<key>` scalars.
- Built-in `"norms_per_param"` gradient diagnostic and `"sv_histogram"` weight diagnostic.
- LLM recipe wires a GRAD `HookPoint` and both new diagnostics by default.
- `scripts/parity_check.py` filled out — dual-pipeline canonical run vs. mendu's pre-cutover recorder, scalar extractor, tolerance comparator.
- `scripts/bench_50m.py` wall-clock measurements recorded in README (CPU ~15% overhead at the v0.2.0a0 snapshot).

### Changed
- `loss` tag renamed to `train/loss`; paper2-specific scalars get `optim/per_param/...` prefixes. Pre-cutover mendu TB runs are not directly comparable to post-cutover runs (accepted trade-off).
- `docs/design.md` §7 rewritten to reflect actual M1+M2 execution (hybrid strategy: paper2 specifics stay in mendu via custom Recipe; circuitry grows only for universals).

### Removed (downstream)
- `mendu/tools/inspect_checkpoint/` and the legacy `paper2/tests/inspect_checkpoint/` suite — replaced by `mendu/paper2/circuitry_recipe.py` (custom Recipe with 5 paper2-specific diagnostics) and direct `circuitry.Recorder` use in the 3 training drivers.
- `latent_inspect_checkpoint` pip package uninstalled from mendu venv.
- `latent-superpowers-inspect` repo: `core/inspect-checkpoint/`, its tests, and 4 `adapters/*/inspect-checkpoint/` directories deleted; tag `pre-circuitry-extraction` preserves rollback. 13 unrelated subsystems in that repo are untouched.

## [0.1.0] — 2026-05-20

### Added
- `circuitry.core.weight` — `effective_rank`, `stable_rank`, `condition_number`, `singular_values`, `heavy_tail_alpha`.
- `circuitry.core.activation` — `dead_fraction`, `kurtosis`, `participation_ratio`, `norm_stats`.
- `circuitry.core.gradient` — `layer_norm`, `signal_propagation_depth`.
- `circuitry.core.spectral` — `esd`, `rank_trajectory`.
- `circuitry.Recorder` — training-time hooks per recipe; matched-modules artifact; `strict`/`expected_min_matches` invariants; rank-0 no-op on multi-rank runs.
- `circuitry.scan_run` and `circuitry.build_report`.
- `circuitry.Recipe` + stock LLM / vision / two_tower recipes; `register_recipe` for custom recipes.
- `circuitry.MetricWriter` protocol with TensorBoard (default, async option), JSONL, null, and optional wandb adapters.
- `circuitry` CLI: `list-recipes`, `report`. (`scan` exposed but requires a model factory — programmatic use only in v0.1.0.)
- Benchmark harness (`scripts/bench_50m.py`) and parity-check stub (`scripts/parity_check.py`) for the M2 mendu cutover.

### Known limits
- Single-process only. Non-zero ranks no-op; FSDP-sharded parameters produce incorrect diagnostics on rank 0. Multi-process design is in `docs/design.md` §11.
- README benchmark numbers are not filled in — run `scripts/bench_50m.py` yourself.
