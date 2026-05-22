# 2026-05-21 — HuggingFace smoke test (Qwen/Qwen2.5-0.5B)

Reproducible via `scripts/smoke_hf_model.py`. This is the first time circuitry has been pointed at an arbitrary off-the-shelf HF model (rather than a small canonical LLaMA in the bench harness). The exercise was deliberately un-patched: run the stock `llm` recipe against the model's actual naming and see what shakes loose.

**Repro:**
```bash
venv/bin/pip install transformers accelerate    # dev-only deps
venv/bin/python scripts/smoke_hf_model.py --model Qwen/Qwen2.5-0.5B
```

**Run artifacts (kept on disk for now, not committed):**
- `runs/hf_smoke/findings.json` — structured summary (model, hook-point match counts, attach result, losses).
- `runs/hf_smoke/filtered/inspect/report.md` — 79KB markdown report from the filtered run.
- `runs/hf_smoke/filtered/circuitry/matched_modules.txt` — 301 lines, the per-HookPoint match list.
- `runs/hf_smoke/filtered/metrics.jsonl` — 1894 lines, all scalar emissions.
- `runs/hf_smoke.log` — full stdout.

---

## What worked

1. **Pipeline end-to-end on a real HF model.** Load (14.5s), attach, 4 train steps with synthetic data + emit at every-N=2, build_report — all completed cleanly once the recipe was filtered. Total runtime ~40s on CPU for a 494M-param model. Peak memory ~5GB (fp32 weights + grads + activations on seqlen=32).

2. **6 of 8 stock LLM HookPoints matched Qwen's HF naming out of the box:**
   - `WEIGHT .*\.(q|k|v|o)_proj$` → 96 modules (24 layers × 4 projections) ✓
   - `WEIGHT .*\.(w1|w2|w3|gate_proj|up_proj|down_proj)$` → 72 modules ✓
   - `OUTPUT .*\.mlp$` → 24 modules ✓
   - `WEIGHT embed.*` → `model.embed_tokens` ✓
   - `WEIGHT lm_head$` → `lm_head` ✓
   - `GRAD .*\.(q|k|v|o)_proj$` → 96 modules ✓

3. **All declared diagnostics fired.** Recipe lists 4 weight + 3 activation + 2 gradient diagnostics; jsonl confirms every one was emitted per emit step (946 scalar tags × 2 emit steps + 4 `train/loss` = 1894 events).

4. **Diagnostic numbers show real signal.** The weight effective-rank pattern *clearly identifies GQA*:
   - `q_proj`, `o_proj`: effective_rank ~200–415 (full attention dim)
   - `k_proj`, `v_proj`: effective_rank ~60–130 (GQA-reduced KV heads — Qwen2.5-0.5B has 14 q-heads but 2 kv-heads)

   This is the kind of architecture-detection signal the library is supposed to surface. It worked.

5. **No numerical blow-ups.** No NaN/Inf, no silent failures inside primitives. SVDs on the 896-dim Qwen layers completed in reasonable time (~6s per emit step total for ~170 weight matrices on CPU).

---

## What didn't work

### HIGH

**H1. `Recorder.attach()` is unrecoverable on a 0-match HookPoint, ignoring `strict=False`.**
- Location: `src/circuitry/recorder/live.py:136-139`.
- Behavior: `if len(names) == 0: raise RuntimeError(...)` — fires regardless of `strict`.
- Impact: dropping circuitry into an existing HF training script with the stock `llm` recipe is impossible without either (a) authoring a custom Recipe, or (b) pre-filtering HookPoints as `scripts/smoke_hf_model.py` does in `_build_safe_recipe`. The `strict` constructor flag controls under-match but not zero-match. UX-wise this hard-fails the very onboarding path the README advertises.
- Surfaced by: Phase 3 of the smoke script — `RuntimeError: HookPoint 2 (<selector>) matched 0 modules — refusing to attach`.

**H2. Stock `llm` recipe is GPT-2/canonical-LLaMA flavored, not HF-canonical.**
- Location: `src/circuitry/recipes/llm.py`.
- Two of eight HookPoints miss against HF transformers naming:
  - HookPoint 2: `OUTPUT .*\.attn$` — HF uses `self_attn`. Suggested: `.*\.(self_)?attn$`.
  - HookPoint 4: `OUTPUT .*\.ln_[12]$` — HF uses `input_layernorm` / `post_attention_layernorm`; only GPT-2-family uses `ln_1`/`ln_2`. Suggested: extend regex or split into two HookPoints.
- Impact: combined with H1, the stock recipe is unusable on a vanilla HF model. This is the biggest adoption blocker — HF transformers is the most common naming convention in the wild today.

**H3. `layer_norm` and `norms_per_param` gradient diagnostics are duplicates.**
- Evidence: identical numbers under different tag prefixes:
  ```
  grad/per_param/model.layers.0.self_attn.q_proj/norm  = 2.238
  gradient/layer_norm/model.layers.0.self_attn.q_proj  = 2.238
  ```
- Source: `src/circuitry/recorder/live.py:40` registers `"layer_norm": _grad.layer_norm`; `_grad.layer_norm` apparently computes per-param L2 norms — the same thing the `norms_per_param` diagnostic does.
- Impact: the LLM recipe wires both → every gradient is reported twice. The report has two near-identical sections (`## grad` and `## gradient`). Either rename one (it's misleading — `layer_norm` reads as "the norm of a LayerNorm layer's grad" not "per-param L2 norm in a dict") or repurpose to compute a genuinely different quantity (e.g., layer-summed grad norm).

### MEDIUM

**M1. `matched_modules.txt` mislabels every HookPoint as `<selector>`.**
- Location: `src/circuitry/recorder/live.py:128`.
- Bug:
  ```python
  label = hp.pattern or "<modules>" if hp.modules is not None else "<selector>"
  ```
  Python operator precedence parses this as `(hp.pattern or "<modules>") if hp.modules is not None else "<selector>"`. For pattern-only HookPoints (`modules is None`), label always falls through to `"<selector>"`.
- Impact: the artifact and the report's `## Matched modules` section both show `target=<selector>` for every entry, making it impossible to read off which regex matched what. Fix: explicit `if/elif/else` chain.

**M2. Report's `## activation` section reports zero variance.**
- Every metric is reported with `first == last == min == max`. Reason: lr=0 in the smoke script, so weights and activations don't change between emit steps. This is *expected* given the script — not a library bug — but a real session of training would want the "is this signal moving?" view to be more prominent. Worth noting that the current report format (first/last/min/max columns) is built for a longer-horizon training run.

**M3. `scan_run` post-hoc workflow — exercised in v0.4.2 work, new gap found.**
- `scripts/smoke_hf_model.py --scan` now drives the full post-hoc path: training-time checkpoint saves (`<run_dir>/checkpoints/step{N}.pt`) plus a `scan_run(...)` call with an `AutoConfig.from_pretrained()`-derived `model_factory` (avoids redownloading weights).
- **Works:** scan_run completes on HF checkpoints and emits TB events under `<out_dir>/events.out.tfevents.*` plus the usual `matched_modules.txt`.
- **New gap (M3-followup):** `scan_run` hardcodes `TensorBoardWriter` at `src/circuitry/recorder/scan.py:55`, so the scan output is **incompatible with `build_report`** (which reads `metrics.jsonl` from a `JsonlWriter`). The smoke script's findings.json now records `scan_run.build_report_compatible: false`. To inspect numerically, users must run `tensorboard --logdir <scan_out>` instead. Fix is small (pass a writer through `scan_run`'s signature, default to TB for back-compat) and is a candidate for the report-renderer rewrite in Tier 3.

### LOW

**L1. `rec.step(step=step, loss=float(loss))` warns about Tensor→scalar conversion.**
- PyTorch UserWarning: "Converting a tensor with requires_grad=True to a scalar may lead to unexpected behavior. Consider using tensor.detach() first."
- Cleaner API: `Recorder.step` accepts `loss: float | Tensor` and detach-and-item-s internally.

**L2. `transformers` deprecation: `torch_dtype` → `dtype`.** Cosmetic; one-line script fix.

**L3. Section names from tag-prefix split are confusing.**
- `## grad` (with `grad/global/total_norm` and `grad/per_param/...`) and `## gradient` (with `gradient/layer_norm/...`) differ by one character. Two separate sections, identical content modulo prefix. Root cause is H3 (the duplicate diagnostic); fixing H3 fixes this. Independent of H3, the auto-section-from-prefix logic should probably be replaced with a "group by diagnostic" view driven by the recipe's `weight_diagnostics` / `activation_diagnostics` / `gradient_diagnostics` lists.

---

## Proposed improvement plan

Aim is a **v0.4.0** focused on "make the library Just Work on a stock HuggingFace decoder LM."

**Tier 1 — adoption blockers (must-fix for v0.4.0):**
- Fix `attach()` 0-match semantics: respect `strict=False` to warn-and-skip the HookPoint rather than raise (H1).
- Update the stock `llm` recipe to cover HF transformer naming alongside the existing patterns (H2):
  - HookPoint 2: `r".*\.(self_)?attn$"`
  - HookPoint 4: replace with two: `r".*\.input_layernorm$"` and `r".*\.post_attention_layernorm$"` (or a single combined regex).
- Resolve the `layer_norm` / `norms_per_param` duplication (H3): either rename `layer_norm` → `per_param_norm` (and drop one from the LLM recipe), or repurpose `layer_norm` to compute layer-aggregated grad norms (`sum(|g|^2)` per `model.layers.N`).

**Tier 2 — quality (worth bundling into v0.4.0):**
- Fix the `<selector>` label bug in `attach()` (M1).
- `Recorder.step` accepts `Tensor` for `loss` (L1).
- Migrate `scripts/smoke_hf_model.py` to use `transformers`' new `dtype=` arg (L2).
- Re-run the smoke after Tier 1 fixes and confirm `Recorder(recipe="llm")` attaches cleanly to Qwen2.5-0.5B without filtering. Pin the resulting numbers as a regression artifact.

**Tier 3 — separate work, not blocking v0.4.0:**
- ~~Programmatic + CLI `scan_run` smoke test on an HF model with a saved checkpoint (M3).~~ **Done** in `scripts/smoke_hf_model.py --scan` (post-v0.4.1). Surfaced a new follow-up: `scan_run` is TB-only and incompatible with `build_report`; pluggable writer fix tracked under M3-followup.
- Report renderer: switch from auto-prefix-based section split to recipe-diagnostic-list-driven grouping (L3). Bundle with M3-followup so `scan_run` can write jsonl and `build_report` can consume it.
- "Movement-aware" report view: highlight tags where `min != max` over the emission window (M2). A static-weight report shouldn't drown in numbers that haven't moved.

**Out of scope:**
- GPU re-measurement of `bench_50m.py` (already noted in README; orthogonal).
- Multi-process / FSDP path (deferred per design.md §11).

---

## Reproducibility note

The smoke script depends on `transformers` and `accelerate`, both of which are **dev-only dependencies** — they are intentionally not in `pyproject.toml`'s base or `[dev]` extras. Installing them is a one-liner (`venv/bin/pip install transformers accelerate`) and only needed if you want to run `scripts/smoke_hf_model.py`. The library itself does not import `transformers`.
