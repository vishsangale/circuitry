# Changelog

All notable changes to this project will be documented in this file. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.0] — 2026-05-21

### Changed (breaking)
- **`build_report` markdown layout overhauled.** Sections now group by `family/diagnostic` (e.g. `## weight/effective_rank`, `## activation/dead_fraction`) instead of just `family` (`## weight` collapsed every weight diagnostic into one giant table in prior releases). Row identifiers are the tag tail (typically a module name) rather than the full tag. Closes L3 from the v0.4.0 HF-smoke observations doc.

### Added
- **Δ column + movement-aware sort** in `build_report` tables. Each row shows `max - min` over the emit window; rows where Δ > 0 sort to the top of their section. Static rows render Δ as `—`. Top-of-report `## Summary` block reports total tags / moving / static / emit-step counts. Closes M2 from the observations doc.
- **`scan_run(writer=...)`**: optional `writer: MetricWriter | str` parameter (default `"tensorboard"` for back-compat). With `writer="jsonl"`, scan output is `build_report`-compatible, closing the live/scan workflow asymmetry. Closes M3-followup from the observations doc.
- `scripts/smoke_hf_model.py --scan` now uses `writer="jsonl"` and invokes `build_report` on the scan output too, generating a second markdown report from the post-hoc workflow.

## [0.4.1] — 2026-05-21

### Fixed
- `matched_modules.txt` and the `## Matched modules` report section now label each `HookPoint` with its actual regex (or `<modules>` / `<selector>` as appropriate). The previous label logic in `Recorder.attach()` had a mis-parenthesized ternary that always fell through to `<selector>` for pattern-based hooks. From the v0.4.0 HF smoke observations doc (M1).

### Changed
- `Recorder.step(loss=..., loss_components=...)` now accepts `torch.Tensor` in addition to `float`. The Tensor is detached and `.item()`-ified internally, so a `requires_grad=True` training loss can be passed directly without triggering the PyTorch UserWarning about Tensor→scalar conversion. Non-scalar (multi-element) Tensors raise `ValueError` with a clear message. From the v0.4.0 HF smoke observations (L1).
- `scripts/smoke_hf_model.py` uses `transformers`' new `dtype=` kwarg (the `torch_dtype=` form was deprecated in transformers ≥4.45). (L2.)

## [0.4.0] — 2026-05-21

### Added
- `llm` recipe's attention and per-block layernorm `HookPoint`s now cover three naming conventions out of the box: HuggingFace transformers (`self_attn`, `input_layernorm`, `post_attention_layernorm`), HF GPT-2-family (`attn`, `ln_1`, `ln_2`), and canonical LLaMA reference (`attention`, `attention_norm`, `ffn_norm`). The stock recipe now attaches cleanly to vanilla HF decoder LMs (Qwen, Gemma, Llama-family, etc.) without requiring a custom Recipe.

### Changed (breaking)
- `Recorder(strict=False)` now applies to **all** match failures, including 0-match HookPoints. The contract: `strict=True` (default) raises on any HookPoint that matches 0 modules or fewer than `expected_min_matches`; `strict=False` warns and skips the unmatched HookPoint, letting the Recorder attach with the remaining hooks. Prior behavior was "0-match always raises regardless of strict", which made dropping circuitry into an existing training script difficult without authoring a perfect Recipe first.
- `llm` recipe `gradient_diagnostics` no longer includes `"layer_norm"` — it produced identical numbers to `"norms_per_param"` under a different tag prefix. The duplicate emission is gone; only `grad/per_param/<name>/norm` and `grad/global/total_norm` are written. The `layer_norm` diagnostic itself remains available (`vision` and `two_tower` recipes still use it) but the naming is misleading and is a candidate for renaming or removal in a future release.

### Investigation
Driven by the v0.3.0 HuggingFace smoke test (`scripts/smoke_hf_model.py`); findings recorded in `docs/observations/2026-05-21-hf-qwen-smoke.md`.

## [0.3.0] — 2026-05-21

### Removed
- `circuitry.writers.wandb` and the `[wandb]` extras-gated dependency. The adapter had no known consumers. `MetricWriter` protocol keeps any third-party logger a ~50-LOC subclass — re-addable if demand surfaces.

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
