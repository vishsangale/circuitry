# Changelog

All notable changes to this project will be documented in this file. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] — 2026-05-21

### Removed
- `circuitry.writers.wandb` and the `[wandb]` extras-gated dependency. No in-house consumers used it. `MetricWriter` protocol keeps any third-party logger a ~50-LOC subclass — re-addable if demand surfaces.

### Changed
- `Recorder(writer="wandb")` now raises `ValueError`. Allowed string values: `"tensorboard"`, `"jsonl"`, `"null"`. Pass a `MetricWriter` instance for anything else.

## [0.2.0] — 2026-05-21

### Added
- `circuitry.core.activation.token_similarity(h)` — pairwise cosine-similarity statistics on residual-stream activations.
- `circuitry.core.weight.update_delta(sd1, sd0)` and `direction_cosine(sd2, sd1, sd0)` — weight-dynamics primitives.
- `circuitry.recipes._discovery.discover(model)` re-exported as `circuitry.discover` — LLaMA-family arch discovery.
- `Recorder.step(loss=..., loss_components=...)` — emits `train/<key>` scalars.
- Built-in `"norms_per_param"` gradient diagnostic and `"sv_histogram"` weight diagnostic.
- LLM recipe wires a GRAD `HookPoint` and both new diagnostics by default.
- `scripts/bench_50m.py` wall-clock measurements recorded in README (CPU ~15% overhead at the v0.2.0a0 snapshot).

### Changed
- `loss` tag renamed to `train/loss`; per-parameter optimizer scalars get `optim/per_param/...` prefixes. Pre-v0.2.0 TB runs are not directly comparable to post-v0.2.0 runs (accepted trade-off).

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
- Benchmark harness (`scripts/bench_50m.py`).

### Known limits
- Single-process only. Non-zero ranks no-op; FSDP-sharded parameters produce incorrect diagnostics on rank 0. Multi-process design is in `docs/design.md` §11.
