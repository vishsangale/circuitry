# Changelog

All notable changes to this project will be documented in this file. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
