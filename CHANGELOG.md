# Changelog

All notable changes to this project will be documented in this file. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.2.0] — 2026-05-30

### Added
- **`Recipe.disable(names)` / `Recipe.only(names)`** selection helpers
  (`recipes/__init__.py`). Both return a new `Recipe` via `dataclasses.replace`,
  writing into the existing `enabled: dict[str, bool]` field. `disable` marks each
  name `False`; `only` disables every name *not* in the list. Fail-fast: unknown
  name → `ValueError` at construction (not silently at step time). Custom
  `DiagnosticFn` callables are not name-addressable and are unaffected.
- **`build_report` per-family `## Flags` verdict block** (`recorder/report.py`).
  Declarative `FLAG_RULES` table checks four families (`activation/dead_fraction`,
  `weight/effective_rank`, `grad/global`, `weight/attention_head_rank`) using
  signed trend (last − first). Block suppressed when `step_count ≤ 1` (no
  false alarms on single-step or static runs). Renders "no flags" row when all
  predicates are clear.
- **`build_report` compact mode** (`recorder/report.py`, `cli/main.py`).
  `build_report(..., compact=True)` renders only the header, `## Summary`, and
  `## Flags` blocks; suppresses `## Matched modules`, per-tag `## …` table
  sections, and the `<details>` advanced-metrics block. CLI: `circuitry report
  --compact`.
- **`circuitry compare run_a run_b`** CLI subcommand + `compare_runs` /
  `build_compare_report` in `circuitry.recorder.compare` (`recorder/compare.py`,
  new; `recorder/_metrics.py` shared helpers). Compares two runs at
  family/diagnostic granularity (first two tag segments); per-module comparison is
  intentionally omitted (cross-architecture module-name mismatch makes it
  ill-posed). Returns one `FamilyDelta` per (section, diagnostic) present in
  either run; families absent from one side get `NaN` last values. Trend direction
  computed from intra-run signed delta (last − first). Robust to a missing,
  empty, or malformed `metrics.jsonl` (clear `ValueError` rather than a silent
  all-`NaN` report or a bare traceback); `trend_agrees` is `False` when a family
  is present on only one side.

### Changed
- **`ruff` pinned to `==0.15.15`** (was `>=0.4`) in the `dev` extra. The unpinned
  range silently drifted newer ruff lint rules into CI; the pin makes `ruff check
  src tests` deterministic. Cleared the pre-existing rot it had surfaced in
  `tests/patching` / `tests/recorder` (import sorting, unused imports, ambiguous
  `l`, a semicolon compound statement; the `pytest.importorskip` E402 pattern via
  `[tool.ruff.lint.per-file-ignores]`).

### Docs
- `docs/design.md`: §4.2 documents `compact` kwarg on `build_report` and the new
  `compare_runs` / `build_compare_report` public surface; §4.3 adds the
  `circuitry compare` CLI line + `--compact` flag and corrects the stale
  `findings.json` reference (scan writes `metrics.jsonl` when `writer="jsonl"`,
  not a separate `findings.json`); §4.4 adds `enabled` field and `disable` /
  `only` to the `Recipe` code block and documents them in prose; §1 ("Naming
  clarity" + non-goals) and §8 reconciled so the shipped activation-patching
  pillar (v1.0, §4.6) and SAE reconstruction metrics are no longer listed as
  out-of-scope. "Last updated" bumped to 2026-05-30.
- `README.md`: status `v0.3.0 (alpha)` → `v1.2.0 (beta)`; scope statement updated
  (activation patching + SAE shipped); CLI gains `circuitry compare` and `report
  --compact`; stale `findings.json` → `metrics.jsonl`; renamed primitive
  `layer_norm` → `grad_norm_per_module`; new patching/attribution + ergonomics
  bullets.
- `pyproject.toml` classifier: `Development Status :: 3 - Alpha` →
  `4 - Beta`.

## [1.1.0] — 2026-05-26

### Fixed
- HF patching backend now honors `config.head_dim`; EAP/AtP*/ACDC run on Gemma-2/3
  (previously crashed when `head_dim != hidden_size / num_attention_heads`).

### Added
- `circuitry.patching.to_hooked_transformer(hf_model, model_name, ...)` — bridge a
  loaded HF model into TransformerLens so non-Llama architectures (GPT-2, …) are
  usable with the TL patching backend.
- Clear `ValueError` (pointing to `to_hooked_transformer`) when the HF backend is
  given an unsupported (non-Llama) layout, replacing a cryptic `AttributeError`.

## [1.0.0] — 2026-05-25

### Added
- **Activation patching — core intervention primitive** (v1.0 patching pillar, sub-spec 1).
  New `circuitry.patching` subsystem, the library's first *interventional* (not
  observation-only) capability:
  - `Site` — node-level intervention points (`resid_pre/post`, `attn_head_out`, `mlp_out`,
    `mlp_neuron`, optional position slice).
  - `patch_site()` — context manager that replaces a site's activation for one forward pass,
    guaranteeing restore-on-exit (hook removed, eval/train mode and param `requires_grad`
    restored, param values untouched) even on exception. Frozen-model, activation-grad-only.
  - `PatchRunner` — clean/corrupted prompt-pair runner (denoise / noise), tensor-or-dict
    model inputs, position-alignment validation.
  - Dual-path site resolution: `HFSiteResolver` (config-declared layout; per-head needs eager
    attention, per-neuron Llama-family-first) and `TLSiteResolver` (TransformerLens hook names,
    lazy `transformer_lens` import).
  - Pure metrics in `circuitry.core.patching`: `logit_diff`, `kl_divergence` (chunked), `ce_loss`.
  - `docs/design.md` §4.6 adds the sanctioned intervention-mode contract; `transformer_lens`
    is an approved optional dependency (lazy import — circuitry runs without it).
  Attribution methods (EAP, AtP\*, ACDC) build on this primitive in follow-on sub-specs.
- **EAP — Edge Attribution Patching** (v1.0 patching pillar, sub-spec 2). Gradient-based
  approximate attribution over the residual-stream computation graph:
  - `circuitry.patching.graph` — `Node` / `Edge` / `Slot` / `build_graph`: the causal
    edge graph (writers = embed + attention heads + MLPs; readers = q/k/v + mlp_in +
    logits_in).
  - `EAPRunner(model, resolver).run(clean_inputs, corrupted_inputs, metric, ig_steps=1)`
    → `EAPResult` (per-edge scores; `top_k` / `threshold` / `ranked` helpers). 2-forward +
    1-backward analytic scoring (`Δact · grad`), no per-edge forward passes.
  - Vanilla EAP (`ig_steps=1`) and activation-path EAP-IG (`ig_steps=N`).
  - Backends: TransformerLens (native per-slot hooks) and HF (eager, Llama-family —
    per-head `z@W_O` writers; q/k/v reader gradients back-mapped to residual space with
    the RMSNorm scale as a stop-gradient constant; GQA-aware).
  - Differentiable metric siblings in `circuitry.core.patching`: `logit_diff_t` /
    `kl_divergence_t` / `ce_loss_t` (no `.detach()`); the float versions are unchanged.
  - Correctness gated by an exact per-edge cross-check against brute-force `patch_site`
    patching on linear toys, plus a rank-correlation check on a real HF Llama.
- **AtP\* — Attribution Patching with corrections** (v1.0 patching pillar, sub-spec 3).
  Node attribution (vs EAP's edges):
  - `circuitry.patching.atp` — `AtPRunner(model, resolver).run(clean_inputs,
    corrupted_inputs, metric, *, neurons=, graddrop=, qk_fix=)` → `AtPResult`
    (`AtPNode`-keyed scores; `ranked`/`top_k`/`threshold`/`verify_top_k`).
  - Vanilla AtP (`Δact · full-grad`) for value / MLP / neuron / embed nodes;
    **neuron-level** nodes feasible (O(nodes)).
  - **QK fix** for query/key nodes: recompute the attention pattern with the patched
    (post-RoPE) q/k, propagate through the clean V, project via `W_O` to `d_model`,
    dot with the attention-output gradient. **GradDrop** (`graddrop=True`):
    sum of |per-position score| to counter sign-cancellation.
  - `verify_top_k` calibrates the top-K against real `patch_site` patching.
  - Backends: HF (eager, Llama-family; GQA-aware; `output_attentions` for the QK
    pattern) and TL (vanilla q/k on the TL path). `transformers` joins
    `transformer_lens` as an approved lazy optional dependency (AtP\* QK-fix RoPE).
  - Correctness gated by exact vanilla/neuron cross-checks vs brute-force `patch_site`
    on linear toys, and a real-Llama check that the QK fix beats vanilla against
    brute-force; frozen-model contract verified (no parameter-gradient leak).
- **ACDC — Automatic Circuit DisCovery** (v1.0 patching pillar, sub-spec 4).
  Greedy reverse-topological edge pruning with corrupted-resample set ablation:
  - `circuitry.patching.acdc` — `ACDCRunner(model, resolver).run(clean_inputs,
    corrupted_inputs, tau, *, ordering=, position=, metric=)` → `ACDCResult`
    (pruned circuit edges; `n_kept()` / `circuit_graph()`).
  - Forward-only edge pruning: ablated edges feed corrupted-run activations (cached
    once), kept edges propagate live (current-circuit) activations; deltas injected
    **pre-LayerNorm** per reader/slot (per-head rebuild on HF eager, native on TL).
  - Recovery metric: last-token KL to clean (default, configurable `position`);
    custom metric callable accepted. Single-threshold `tau` (per-edge tolerance) +
    `sweep(taus)` Pareto helper `[(τ, n_kept, final_kl), …]`.
  - Traversal orderings: `"topo"` (reverse-topological with deterministic tie-break
    key) and `"eap"` (lowest `|EAP score|` first, consumes `EAPResult.scores`).
    v1.0 ships traversal-ordering; the EAP-score skip speedup is a documented
    follow-on.
  - Backends: HF (eager, Llama-family; GQA k/v at kv-group granularity) and
    TransformerLens. Empty- and full-ablation anchors are exact under
    `corrupted − live` + pre-LN injection.
  - Correctness gated by empty-/full-ablation anchors on real HF Llama + toy,
    live-vs-clean delta guard (dead-edge pruning on a constructed toy), and
    determinism across runs (topo + eap orderings both deterministic, Pareto
    monotonicity). Reuses EAP's graph, writer-activation caching, and backends.

## [0.9.2] — 2026-05-23

### Fixed
- **`attention_pattern_entropy` is now normalization-invariant.** Each query row
  is divided by its key-axis sum before the entropy, so the metric is comparable
  across attention variants (softmax / sigmoid / linear). Softmax rows sum to 1,
  so existing values are unchanged within fp tolerance; fully-masked rows yield 0.
  The Recorder warns once when captured rows don't sum to 1.
- **`logit_lens_kl` no longer OOMs the run.** The KL is computed in token chunks
  (`chunk_size`, default 256) so the `(tokens, vocab)` lens-logits transient stays
  bounded; a per-layer OOM now empties the cache, warns, skips that emission, and
  keeps training alive instead of crashing. New `Recipe.lens_max_tokens` caps each
  sequence to its first N positions as a cost lever (`None` = all tokens = exact).
- **`output_attentions` capture no longer breaks wrapped models.** Per-head
  attention weights are enabled via `config.output_attentions` instead of a
  forward-kwarg injection that raised `TypeError` on wrappers whose `forward()`
  lacks `**kwargs`. Set as the final `attach()` step (a failed attach never mutates
  the config) and restored on `detach()`.

### Added
- **`circuitry --version`** prints the installed version and exits.

### Docs
- Clarified that `circuitry report` runs on a live `metrics.jsonl` (no `scan`)
  as well as on a retrospective `findings.json`.

## [0.9.1] — 2026-05-23

### Fixed
- **GPU device-correctness in three `core/` primitives.** `weight.singular_values` (and thus `effective_rank` / `stable_rank` / `sv_histogram` / `heavy_tail_alpha`) built its `max_dim` subsample index with a CPU `torch.randperm` and `index_select`-ed a CUDA matrix — a hard crash on GPU weights wider than `max_dim`. `attention.induction_score` indexed a CUDA attention tensor with CPU `arange` indices. `spectral.esd` returned CPU `linspace` edges alongside CUDA `histc` counts. All fixed by propagating the input tensor's `.device` (no `.cuda()` calls — invariant #4 preserved). Surfaced bringing up the GPU benchmark; CPU paths were unaffected.
- `logit_lens_kl` dispatcher now runs once per residual-stream block output instead of once per d_model-shaped activation. It previously kept every captured activation whose last dim matched the unembed `d_model` — on Gemma 4 that was 175 entries (35 layers × 5 sources: self_attn / mlp / layernorm / block outputs) rather than the intended 35 block outputs. The dispatcher now keeps only activations whose module name ends in `.layers.N` (the residual-stream block boundary); the `d_model` check is retained as a secondary guard.

### Added
- `circuitry.core.gradient.total_grad_norm(per_module_norms) -> float` — global gradient L2 norm (`sqrt(sum of squares)`) over the per-module norms from `grad_norm_per_module`. The `norms_per_param` recorder diagnostic now composes the two `core/` primitives (`grad_norm_per_module` + `total_grad_norm`) instead of re-implementing the per-module norm loop inline, restoring design.md's core-primitive layering. Emitted tags (`grad/per_param/<m>/norm`, `grad/global/total_norm`) are unchanged.
- `Recorder.attach()` now logs a WARN when `induction_score` or `attention_pattern_entropy` is requested but the model's resolved attention implementation (`config._attn_implementation`, or `text_config._attn_implementation` for multimodal) is not `"eager"`. SDPA / flash-attention silently drop per-head attention weights even with `output_attentions=True`, so these diagnostics would otherwise emit zero tags with no explanation; the warning points at the `attn_implementation="eager"` workaround. Only fires when a non-eager implementation is positively detected — non-HF models (no `.config`) stay quiet.

### Changed
- **SVD-derived weight diagnostics now share one SVD per matrix per step** (~4× fewer SVDs). `effective_rank`, `stable_rank`, `heavy_tail_alpha`, `condition_number`, and `sv_histogram` previously each called `singular_values(W)` independently; the recorder now computes the SVD once per matrix per step and feeds it to all of them. On a synthetic 88M model the GPU per-emission cost dropped ~4× (e.g. +1202% → +306% overhead at `every_n_steps=25`); CPU benefits equally. Minor numerical effect: for matrices wider than `max_dim` (512) the four diagnostics now share a single random column subsample instead of each drawing its own, so they are mutually consistent (unchanged for matrices ≤ `max_dim`, which are not subsampled). Public primitive signatures are unchanged.
- `grad_norm_per_module` now computes each norm in float32 (casts the gradient before `vector_norm`), improving precision on bf16/fp16 gradients. float32 gradients are unaffected.
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
