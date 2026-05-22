# 2026-05-21 — Gemma 3 1B-it + Gemma 4 E2B smoke

Second round of smoke testing against the v0.5.0 library, this time against Gemma-family HF models. Run via `scripts/smoke_hf_model.py`.

**Setup notes:**
- Gemma 3 family is **gated** (license: `gemma`, terms required on HF Hub). User had access; works with `HF_TOKEN` set or `huggingface-cli login`.
- Gemma 4 family (released ~April 2026) ships under **Apache 2.0** — ungated.
- The smallest Gemma 4 variant is named `E2B` (~"Effective 2B") but **the actual total parameter count is 5.1B** and the file is **10.25 GB** on disk. Caveat: don't confuse the name with size.
- Both runs used the v0.5.0 stock `"llm"` recipe with `strict=False`; bf16 weights on CPU.

---

## Run 1: `google/gemma-3-1b-it`

| Phase | Result |
|---|---|
| Load | 1.00B params, 4.0 GB fp32, 449 modules, 26 transformer layers |
| Stock recipe coverage | **8/8 HookPoints match cleanly** — no MISS |
| `Recorder(recipe="llm").attach()` | **OK** |
| Filtered run (4 steps) | 17.2 s on CPU |
| Report | 970 scalar tags, 11 sections, all static (lr=0 — expected) |

**Architectural signal in the numbers (effective_rank):**

| Module slice | Layer 0 q/o_proj | Layer 0 k/v_proj |
|---|---:|---:|
| Gemma 3 1B-it | ~432 / ~447 | ~235 / ~246 |
| Qwen 2.5 0.5B (v0.4.0 baseline) | ~220 / ~341 | ~63 / ~114 |

Both models use grouped-query attention, but Gemma 3 1B uses a milder kv-head reduction than Qwen 2.5 0.5B. The k/v effective-rank ratio (k or v rank / q rank) is ≈0.55 for Gemma 3 1B vs ≈0.30 for Qwen 0.5B — circuitry surfaces this as a one-line read of the report. Nothing else to flag from this run.

---

## Run 2: `google/gemma-4-E2B` — multimodal

| Phase | Result |
|---|---|
| Load | 5.10B params, 10.21 GB **bf16**, 1596 modules |
| Stock recipe coverage | **8/8 HookPoints match cleanly** — but matches span three towers (see below) |
| `Recorder(recipe="llm").attach()` | OK (technically; produces semantically mixed output) |
| Filtered run (1 step) | 37.8 s on CPU |
| Report | 1006 scalar tags, but only 211 distinct weight modules got diagnostics out of ~700 matched (see H1 below) |

### What's actually in the model (matched_modules.txt subset counts):

| Module-name prefix | Match count |
|---|---:|
| `model.language_model.*` | 447 |
| `model.vision_tower.*` | 242 |
| `model.audio_tower.*` | 84 |
| `model.embed_vision`, `model.embed_audio` | 6 |

Three separate Transformer-style towers under one root: language + vision (SigLIP-style) + audio. The stock `llm` recipe matched modules across all three because they share HF-canonical names (`self_attn.q_proj`, `mlp.gate_proj`, `input_layernorm`, etc.). This is the recipe doing its job — but the user has no way to *constrain* it to the language model, and the report ends up mixing text + vision + audio diagnostics under the same section headings.

### What didn't work

**H1 (HIGH) — Silent weight-extraction skip on wrapper Linear classes.**

Of 200 `q/k/v/o_proj` weight HookPoint matches, only **107** got `weight/effective_rank` emitted (≈100 from `language_model`, only 1 from `vision_tower`, 0 from `audio_tower`). The recipe matched the modules, but the diagnostic loop silently skipped them. No exception, no warning, no log line.

Root cause: `src/circuitry/recorder/live.py` step():
```python
p = getattr(name_to_mod[n], "weight", None)
if isinstance(p, torch.Tensor):
    weights[n] = p.detach()
```

Gemma 4's vision and audio attention projections use `transformers.models.gemma4.modeling_gemma4.Gemma4ClippableLinear` — a wrapper class that holds the actual `nn.Linear` as a child named `.linear`, plus four scalar buffers for clip ranges (`input_min`, `input_max`, `output_min`, `output_max`). The wrapper itself has **no `.weight` attribute** — `getattr(...)` returns `None`, the `isinstance` check fails, the loop just skips the module. The user sees no diagnostic output for those weights but also no signal that anything was skipped.

The same failure mode would hit any model that wraps Linear (LoRA adapters, quantization wrappers, sign-constrained variants like the paper2 `SignConstrainedLinear` we used in the M2 cutover — which was explicitly skipped by name but would otherwise hit this path).

**Suggested fix sketch:**
- When `module.weight` is missing, look one level down: if exactly one direct child has a 2-D+ `.weight` Parameter, use that. Otherwise warn and skip.
- Log a one-line `circuitry: extracted weight from <module>.<child>` at INFO so users see the indirection.
- Strict mode (`Recorder(strict=True)`) raises if any matched WEIGHT HookPoint resolves to zero usable tensors over the run.

**H2 (MEDIUM) — Stock recipe isn't modality-aware.**

Even if H1 is fixed, the user gets text/vision/audio diagnostics mixed into the same `## weight/effective_rank` table. The report has no way to surface "these 100 rows are the language model; these 64 rows are the vision tower". Two reasonable fixes:
- Add a `Recipe(module_prefix=...)` field that prepends the prefix to the HookPoint regex (e.g., `module_prefix="model.language_model"` constrains all hooks to the LM portion).
- Or, ship `llm`, `vision`, `multimodal` recipes that target distinct prefixes and let the user pick (or compose).

The first is more flexible and lets users say `recipe=get_recipe("llm").with_prefix("model.language_model")`.

**M1 (MEDIUM) — vocab_size lookup needs a multimodal fallback.**

`scripts/smoke_hf_model.py` used `model.config.vocab_size`, which doesn't exist on `Gemma4Config` (vocab is nested under `model.config.text_config.vocab_size`). Smoke script now uses `model.get_input_embeddings().num_embeddings` as a robust fallback (already shipped in this commit). Worth noting as a recipe-level expectation: if circuitry ever needs vocab info, it should use the same robust accessor.

### Things that worked despite multimodality

- The new v0.4.0 `Recorder(strict=False)` contract was never tested here — all 8 HookPoints matched something. Good (no regression).
- The v0.5.0 report layout still renders coherently with 1006 tags. The Δ column shows "—" everywhere (single emit step, no movement) which is expected.
- The v0.5.0 `matched_modules.txt` labels show real regexes instead of `<selector>` (closes the M1 from the v0.4.0 observations doc).

---

## Proposed improvement plan

Aim is a **v0.6.0** focused on "make circuitry actually see all the weights in a modern HF model, not just the easy ones."

**Tier 1 — adoption blockers (must-fix for v0.6.0):**
- ~~**H1** Weight extraction handles wrapper Linear classes...~~ **Done in v0.6.0.** Implemented as a `ModelInventory` primitive (`circuitry.core.inventory`) built once at `attach()` time from `model.named_parameters()`. Recorder resolves each WEIGHT/GRAD HookPoint to a Parameter via `inv.find_primary_weight(module_name)` instead of `getattr(module, "weight")`. Re-run against `google/gemma-4-E2B`: weight-diagnostic emission went from 310 → **1080 scalar tags** (vision tower 1 → **113** modules, audio tower 0 → **36** modules). Inventory dumped to `<run_dir>/circuitry/inventory.json`; `matched_modules.txt` now shows `<module> → linear.weight (shape)` or `UNRESOLVED (<class>)` per match.
- ~~**H2** Add `Recipe.with_prefix(prefix: str) -> Recipe` (or `Recipe(module_prefix=...)`) so the stock `llm` recipe can be scoped to `model.language_model.*` for multimodal HF models. Smoke script updates to pass `--prefix model.language_model` by default for multimodal models. **Note:** `ModelInventory.with_prefix()` already exists as the underlying primitive (v0.6.0); only the Recipe-level integration is open.~~ **Closed in v0.7.0.** With `--prefix model.language_model` on `google/gemma-4-E2B`: attach summary reports **447 matched / 447 resolved / 0 unresolved** (was 700+ pre-fix, mixing all three towers). Vision tower and audio tower modules are now cleanly excluded. `lm_head$` matched 0 (it is not under `model.language_model`), silently skipped. The `## Attach summary` block in the report makes these counts visible without opening `matched_modules.txt`.

**Tier 2 — quality:**
- Surface "the recipe matched N modules but emitted diagnostics for only K of them" as an `## Attach summary` block in the report, so silent drops are visible.
- Re-run the Gemma 4 E2B smoke after Tier 1 and confirm 200/200 attention modules get effective_rank.

**Tier 3 — followups (not blocking v0.6.0):**
- Audio-tower vs vision-tower vs language-model diagnostics warrant per-section grouping in the report (something like `## weight/effective_rank — model.language_model.*`). Bigger UX work.
- The `bench_50m` numbers should be re-measured against `gemma-3-1b` and `gemma-4-E2B` once H1 is fixed, since H1 currently makes those runs *artificially fast* by silently skipping modules.

---

## Artifacts

- `scripts/smoke_hf_model.py` (multimodal-robust vocab lookup landed)
- `runs/gemma3_1b/filtered/inspect/report.md` — Gemma 3 1B-it report (gitignored)
- `runs/gemma4_e2b/filtered/inspect/report.md` — Gemma 4 E2B report (gitignored)
- `runs/gemma3_1b.log`, `runs/gemma4_e2b.log` — full stdout (gitignored)
