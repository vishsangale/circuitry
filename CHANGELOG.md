# Changelog

All notable changes to this project will be documented in this file. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `Recorder.attach()` now logs a WARN when `induction_score` or `attention_pattern_entropy` is requested but the model's resolved attention implementation (`config._attn_implementation`, or `text_config._attn_implementation` for multimodal) is not `"eager"`. SDPA / flash-attention silently drop per-head attention weights even with `output_attentions=True`, so these diagnostics would otherwise emit zero tags with no explanation; the warning points at the `attn_implementation="eager"` workaround. Only fires when a non-eager implementation is positively detected — non-HF models (no `.config`) stay quiet.

### Changed
- **Renamed the `layer_norm` gradient diagnostic to `grad_norm_per_module`.** The old name was misleading — it computes a per-module gradient Frobenius/L2 norm, with no relation to the `LayerNorm` module. The core primitive `circuitry.core.gradient.layer_norm` is now `grad_norm_per_module`, the recipe diagnostic key `"layer_norm"` is now `"grad_norm_per_module"`, and the emitted tag prefix `gradient/layer_norm/<module>` is now `gradient/grad_norm_per_module/<module>`. **Breaking** for anyone selecting `"layer_norm"` in `gradient_diagnostics` or reading the old tag namespace (the stock `vision` and `two_tower` recipes are updated; `llm` was already on `norms_per_param`).

## [0.9.0] — 2026-05-23

### Added
- `circuitry.core.lens.logit_lens_kl(residual, unembed, final_logits, *, layer_norm=None)` — per-layer KL between the logit-lens projection of the residual stream and the model's final logits. Nostalgebraist 2020 logit lens.
- `circuitry.core.attention.induction_score(attn_pattern, *, seq_len_repeat)` — Olsson et al. 2022 prefix-matching probability on a repeated-random-token probe, per head.
- `circuitry.core.attention.attention_pattern_entropy(attn_pattern)` — per-head Shannon entropy (nats) of the attention distribution over keys.
- `circuitry.sae` submodule: `load_sae(release, sae_id, device)` (thin wrapper over `sae_lens.SAE.from_pretrained`) + `sae_reconstruction_error(x, sae)` returning `{recon_mse, l0, l1, frac_alive, ce_recovered_proxy}`.
- `Recipe.sae_checkpoints: dict[str, tuple[str, str]] | None` field, `Recipe.induction_probe_seq_len: int = 25` field, `Recipe.with_sae(mapping)` builder method.
- Recorder dispatchers for `logit_lens_kl`, `induction_score`, `attention_pattern_entropy`, `sae_reconstruction`. `attention_pattern_entropy` sources from the user's main forward pass via `output_attentions=True` injection; `induction_score` uses a synthetic probe pass that runs exactly once per step shared across all hooked self_attn modules.
- Report-renderer HERO_SECTIONS: `activation/logit_lens_kl`, `activation/induction_score`, `activation/attention_pattern_entropy`, `activation/sae`.
- `scan_run` gains a `strict: bool = True` parameter (backwards-compatible) for consistency with `Recorder`.

### Changed
- Stock `llm` recipe grows from 9 to 10 HookPoints (adds block-output) and from 4 to 7 default activation diagnostics (adds logit_lens_kl, induction_score, attention_pattern_entropy). SAE remains opt-in: `Recipe.with_sae(...)` loads checkpoints but does NOT auto-append `"sae_reconstruction"` — user adds it to `activation_diagnostics` explicitly to incur per-step encode+decode cost.
- New hard dependency: `sae-lens >= 4.0.0`.

### Validation
- Gemma 4 E2B on CPU with `--prefix model.language_model` (1 step): **92.74 s** Phase-4 wall-clock (vs v0.8's 39.6 s; +134% due to `logit_lens_kl` + `induction_score` probe pass). Layer 0 logit_lens_kl = 0.299 nats; layer 34 = 3.894 nats; peak at layer 29 = 8.813 nats. `induction_score` and `attention_pattern_entropy` not captured: Gemma 4 uses SDPA attention which does not return per-head attention weights. Two fixes landed: (1) `logit_lens_kl` dispatcher filters activations by d_model to handle multimodal architectures (Gemma 4's gate tensors are wider than the residual stream); (2) layer sort order changed from lexicographic to numeric to ensure the reference distribution comes from the true final layer. Full findings in `docs/observations/2026-05-23-v0.9-gemma-validation.md`.
- Gemma 2 2B with `gemma-scope-2b-pt-res` width-16k SAE at layer 8 (SAE opt-in): **71.86 s** vs **72.35 s** without SAE — overhead ≈ 0% (within noise) for one step on a 2B model. SAE recon_mse = 3130 on a 4-token sequence (non-representative input; l0 = 1293 vs expected ~71 at scale).
- Test count: 157 (v0.8) → 194 (v0.9).

## [0.8.0] — 2026-05-22

### Added
- **`circuitry.core.weight.attention_head_rank(W, n_heads, head_dim, axis)`** — per-head `effective_rank` of an attention projection. Caller supplies the head count (GQA-friendly: pass `num_key_value_heads` for k/v_proj).
- **`circuitry.core.activation.gate_stats(x, eps=1e-6)`** — `frac_active` / `mean_abs` / `std` on a post-gate MLP activation tensor. Captured by hooking `down_proj` INPUT.
- Stock `llm` recipe now emits `weight/attention_head_rank/<module>/head_<i>` per q/k/v/o_proj and `activation/gate_stats/<down_proj>/{frac_active,mean_abs,std}` by default. The recipe gains one new INPUT HookPoint for `down_proj`.
- `Recorder.attach()` discovers attention head metadata from `model.config` (or `model.config.text_config` for multimodal HF) when `attention_head_rank` is requested. Missing config → WARN + skip (no crash).

### Changed
- **`build_report` markdown layout**: hero diagnostics (`weight/effective_rank`, `weight/attention_head_rank`, `activation/dead_fraction`, `activation/gate_stats`, `grad/global/total_norm`) render inline. Everything else (`stable_rank`, `kurtosis`, `participation_ratio`, `heavy_tail_alpha`, `layer_norm`, full per-param gradient tables) is wrapped in `<details><summary>Advanced metrics</summary>`. TensorBoard / JSONL emission is unchanged — only the markdown rendering is opinionated.
- `grad/per_param/*` tables with more than 20 rows trim to top-10 + bottom-10 by max-magnitude, with an elision row between.
- Report header subtitled `static (1 step)` or `dynamic (N steps)`; single-step runs include a `## Summary` note that Δ is uniformly zero (closes the framing concern raised in the Gemini-pro review of v0.7.0).

### Validation
Re-ran the v0.7.0 Gemma 4 E2B smoke with the v0.8.0 stock recipe (`google/gemma-4-E2B`, bf16, 1 step, `--prefix model.language_model`, 39.6 s on CPU). Layer 0 q_proj per-head ranks span [225.7, 240.6] across 8 heads — narrow at the diagnostic step, expected to widen during training. Layer 0 `down_proj` `gate_stats.frac_active` is 0.9997 while `.mlp` `dead_fraction` is 0.4942 — the large gap surfaces SwiGLU's compute-and-discard pattern, which `dead_fraction` alone cannot distinguish from gated-off neurons. The trimmed v0.8.0 report adds two new hero sections (attention_head_rank + gate_stats) and collapses 819 lines of low-signal diagnostics behind `<details>`. The 7 Gemma 4 "full-attention" layers (4, 9, 14, 19, 24, 29, 34) emit a skip warning from `attention_head_rank` because their doubled projection size does not match the configured `n_heads × head_dim` — this is an architectural detail of Gemma 4's interleaved attention, not a bug; the primitive's caller-supplies-`n_heads` contract is doing what it advertises. Full findings in `docs/observations/2026-05-21-gemma-smoke.md`.

## [0.7.0] — 2026-05-21

### Added
- **`Recipe.module_prefix`** — new field (default `None`) on `Recipe`; holds an optional dotted module-name prefix used to scope hook matching.
- **`Recipe.with_prefix(prefix)`** — returns a new `Recipe` via `dataclasses.replace` with `module_prefix=prefix` and name renamed to `<name>@<prefix>`. Latest-wins semantics: `r.with_prefix("a").with_prefix("b")` yields `module_prefix="b"`. Docstring warns that `expected_min_matches` thresholds calibrated to whole-model counts should be lowered after scoping.
- **`filtered_matches(model, hp, recipe)`** — new helper in `src/circuitry/recorder/hooks.py`. Wraps `match_modules` and, when `recipe.module_prefix` is set, keeps only module names that equal the prefix or start with `prefix + "."`. Used in both `Recorder.attach()` and `scripts/smoke_hf_model.py:_build_safe_recipe`.
- **`<run_dir>/circuitry/attach_summary.json`** — written by `Recorder.attach()` after resolving all hook points. Schema: `{"hook_points": [{"idx", "source", "label", "matched", "resolved", "unresolved"}], "totals": {"matched", "resolved", "unresolved"}}`. For OUTPUT/INPUT hooks `resolved == matched` and `unresolved == 0`; for WEIGHT/GRAD, `resolved` counts modules where `inventory.find_primary_weight()` returned non-None.
- **`## Attach summary` block in `build_report`** — rendered after `## Summary` when `attach_summary.json` is present; silently skipped for older runs. Shows a table of per-hook-point counts plus a totals row.
- **`--prefix <PREFIX>`** flag on `scripts/smoke_hf_model.py` — applies `filtered.with_prefix(args.prefix)` after `_build_safe_recipe`; echoes `prefix: <value>` in the run header.

### Changed
- **`Recorder.attach()` now writes `attach_summary.json`** in addition to `matched_modules.txt` and `inventory.json` — non-breaking (new file, no removed files).
- `Recorder.attach()` now routes module matching through `filtered_matches` (instead of bare `match_modules`) so that a `recipe.module_prefix` constraint is honoured at attach time.

### Validation
Re-ran the Gemma 4 E2B smoke with `--prefix model.language_model`. See `docs/observations/2026-05-21-gemma-smoke.md` for the closed H2 bullet with attach-summary numbers. Closes H2 ("stock recipe isn't modality-aware") from that observations doc.

## [0.6.0] — 2026-05-21

### Added
- **`circuitry.ModelInventory`** — frozen snapshot of every named `torch.nn.Parameter` in a model. Built once at `Recorder.attach()` time and persisted to `<run_dir>/circuitry/inventory.json`. Replaces the old `getattr(module, "weight", None)` heuristic for resolving `WEIGHT` / `GRAD` HookPoints. Catches weights hidden inside wrapper Linear classes (e.g. HF `Gemma4ClippableLinear`) whose `.weight` lives on a child module rather than on themselves.
- **`circuitry.ParameterRecord`** — dataclass describing a single Parameter (name, shape, dtype, owning-module class, leaf attribute). Useful for "why didn't my regex match?" debugging.
- `ModelInventory.with_prefix(prefix)` — modality scoping helper for multimodal models (e.g. `inv.with_prefix("model.language_model")`).
- `ModelInventory.find_primary_weight(module_name)` — deterministic module→Parameter resolution: prefers a direct `.weight`, else returns the single 2-D+ Parameter in the subtree, else `None`.

### Changed
- **`Recorder.attach()` now logs WARN per matched module whose primary weight can't be resolved** (no direct `.weight`, ambiguous multiple 2-D children, or no 2-D Parameter in the subtree). `matched_modules.txt` shows the resolution tail per module (`q_proj → linear.weight (768, 768)`) or `UNRESOLVED (<class>)` so silent drops become visible.
- `Recorder.step()` weight/gradient extraction uses the inventory-derived module→Parameter map instead of `getattr(module, "weight")`. Tag layout is unchanged (still keyed by module name).

### Validation
Re-running the v0.4.0 HF smoke against `google/gemma-4-E2B` (multimodal: language + vision + audio towers): weight-diagnostic emission jumps from **310** to **1080** scalar tags. Vision tower coverage: **1 → 113** modules. Audio tower coverage: **0 → 36** modules. Closes the H1 "silent skip on wrapper Linear" finding from `docs/observations/2026-05-21-gemma-smoke.md`.

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
