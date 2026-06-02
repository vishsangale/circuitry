# Real-model evaluation — circuitry v1.7.0

**Date:** 2026-05-31 / 2026-06-01
**Branch:** main @ d713261 (v1.7.0)
**Goal:** Exercise the whole library against *real* models (GPT-2-small + real pretrained
SAEs, freshly-trained small models for training-time metrics) and write down every problem
and improvement opportunity. **Evaluation pass — no `src/` changes were made.**

**Environment:**
- macbook (eval host): torch 2.12 (MPS), transformers 5.9, transformer_lens, sae_lens 6.44.2,
  datasets 4.8.5. Correctness runs on **CPU** (TL warns torch-2.12 MPS may be silently incorrect).
- rtx GPU reachable via Ray (1× RTX 5080) for heavy / perf-sensitive runs.
- Real models/SAEs used: GPT-2-small (HF + TransformerLens); sae_lens `gpt2-small-res-jb`
  (resid, FVU 0.001), `gpt2-small-mlp-out-v5-32k` + `gpt2-small-attn-out-v5-32k`
  (FVU ~0.29/0.32, `normalize_activations='layer_norm'`). Recorder runs on a 2-layer HF
  Llama (656K params), torchvision ResNet-18 / TinyCNN, and a small two-tower model.

**Method:** parallel static code review (2 subagents + Gemini deep-correctness pass) + dynamic
real-model runs (scripts under `scripts/v17_validation/`) + this synthesis. Every finding below
is backed by either measured numbers from a real run or a `file:line` code citation.

Severity legend: 🔴 **bug/correctness** · 🟠 **gap/robustness** · 🟡 **ergonomics/perf** · 🟢 **polish** · ✅ **confirmed-good**

Validation scripts (all under `scripts/v17_validation/`): `warm_up_env.py`, `sae_nodes_real.py`,
`sae_edges_real.py`, `recorder_correctness.py`, `track1_patching_revalidation.py`,
`part1_core_primitives.py`, `part2_vision_recipe.py`, `part3_two_tower_recipe.py`.

---

## 0. Executive summary

The **v1.5–1.7 SAE-circuit math is sound on real models** — this was the part with the most
synthetic-only test coverage, and it held up: splice losslessness, composite (layer,component)
keying, node attribution, feature→feature edges, intra-layer edges, and faithfulness/completeness
all behaved correctly on real GPT-2 + real SAEs, and the analytic edge scores matched an
independent bruteforce oracle (sign-agreement 1.00, Pearson 0.923) on a *real nonlinear* model.
Gemini's deep-correctness pass confirmed the sign conventions, eps-freezing, lossless splice,
edge VJP, node IG, keying, and sever-guard point by point. The v1.7 resolver refactor did **not**
regress the EAP/ATP/ACDC backends (exact top-20 match vs the pre-v1.7 baseline).

The real problems are **older and lower in the stack** — in `core/` primitives and the live
Recorder, the parts we *assumed* were solid because they shipped first:

1. 🔴 **The entire SVD-derived weight-diagnostic family is silently biased and non-deterministic
   on any matrix wider than 512** (i.e. every layer of every real LLM). `singular_values`
   defaults to `max_dim=512` random column subsampling; the recorder never overrides it.
   `effective_rank` 38% low, `heavy_tail_alpha` 33% low, `condition_number` up to 290× wrong,
   all varying run-to-run. **This is the headline finding.**
2. 🔴 **`attention_pattern_entropy` reports the induction-score probe's attention, not the
   training batch's.** Proven by exact numerical match. Also wrong in `scan_run`.
3. 🔴 **Faithfulness/completeness/ACDC are wrong for `layer_norm`-normalized SAEs** (which
   includes the standard OpenAI v5 GPT-2 SAEs): `compute_f_per_site` caches `sae.encode(...)`
   output and decodes it later in a different-input frame; layer_norm encode is stateful (proven:
   9.05 activation-unit drift).
4. 🟠 **`variant='ig'` + `include_error_node=True` silently drops all error→feature edges**
   (10 → 0 on the real model). Triple-confirmed (static review, Gemini, real run).

None of these are in the new v1.7 code; #1–#2 predate the SAE work entirely. The lesson: the
oldest, "obviously-fine" primitives were the least validated against real-scale inputs.

**Round 2** (§8) stress-tested the surfaces round 1 skipped — a **parallel-attention** model
(pythia-70m: the no-grad/sever guard fires correctly on the severed intra-layer pair, the exact
case the v1.7 caveat warns about), the **HF backend** (fails loudly with a clear migration path for
non-Llama arch), **batch>1** (works, all-finite), the **bounded-memory** claim (peak 28.6 MB vs a
2.42 GB dense Jacobian — confirmed), **bf16** (finite + sign-consistent), and **scan/writers/
compare/CLI** (all functional). Round 2 found **no new correctness bugs** — only doc gaps
(F22–F28). The framework and the v1.7 design are solid; the correctness work is concentrated in the
four §0 items.

**Round 3** (§9) widened to real architecture families — Qwen (Llama-family), a real MoE (OLMoE),
a Vision Transformer, a DLRM recsys (Gemma-2 was **blocked on HF gating**). The SAE/patching/
primitive *cores* are architecture-robust (HF SAE backend positively validated on Qwen; OLMoE
resolves + backprops through sparse routing). But round 3 exposed a **systemic recipe-robustness
gap**: recipe module-matching is naming-coupled and silently under-covers everything that isn't
GPT-2/standard-Llama — **0%** capture on torchvision ViT (F30), standard DLRM (F32), and MoE expert
weights (F36). It also found **3 new real bugs** — `grad_norm_per_module` crashes on sparse
embeddings (F31), `embedding_alignment` silently `{}` on `nn.ModuleList` (F33), weight primitives
return semantically-wrong rank on 3D batched expert tensors (F38) — and a **high-impact** one:
`Recorder.attach()` crashes on any SDPA-default HF model (Qwen2/Llama-3.x/Mistral/Gemma) (F29).

---

## 1. Static code review

Two parallel Sonnet reviewers + a Gemini-pro deep-correctness pass over the SAE stack.

### 1.1 SAE-circuit stack (sae_features, sae_edges, sites, graph)
- **Gemini verdict:** core math correct point-by-point — Δf sign (`f_corrupt−f_clean`)
  consistent across attrib/ig, eps frozen at clean at every site, lossless splice, per-survivor
  edge VJP (no dense `d_sae×d_sae` Jacobian), node IG path/midpoint rule, (layer,component)
  keying with no collisions, sever-guard (raise vs warn). One MAJOR + minor issues → register.
- Notable static findings folded into the register: completeness() missing `ablation_eps` under
  `include_error_node` (F11); IG edge loop omits error writer (F4); IG edge recomputes
  `eps_clean` every step (F12, perf); FeatureACDC doesn't expose `variant`/`n_ig_steps` (F13);
  stale module docstrings still claim "only HF, only resid_post" (F14).

### 1.2 Core primitives + recorder + recipes/writers
- Surfaced the SVD subsampling family (F1), attention-entropy contamination (F2), the
  scan-with-no-forward gap (F8), gradient tag-namespace split `grad/` vs `gradient/` (F15), and
  several doc/spec mismatches (F16–F19). All later confirmed on real models in §4–§5.

### 1.3 Docs ↔ code consistency
- `design.md §4.2` `scan_run` example omits the required `model_factory` arg and the
  `writer="jsonl"` needed for `build_report` (F16). `gate_stats` returns `dict[str,float]` not the
  documented `GateStats` dataclass (F17). No import-linter contract enforces the design's
  "patching/ MUST NOT import cli/" rule (F18). `design.md §10` says the TB adapter writes async
  "by default" but `TensorBoardWriter(async_writes=False)` is the default (F19).

---

## 2. Dynamic — SAE circuits on real GPT-2 + real SAEs  ✅ (math) / 🔴 (layer_norm)

Script: `sae_nodes_real.py`, `sae_edges_real.py`. TL backend, CPU, IOI task
(logit-diff Mary−John, baseline +3.17).

### 2.1 Node attribution, 3 real site types
| site | SAE | FVU | splice max\|Δlogit\| | lossless | component |
|---|---|---|---|---|---|
| resid_post@7 | gpt2-small-res-jb | 0.0011 | 1.9e-05 | ✅ PASS | `None` (legacy) |
| mlp_out@8 | mlp-out-v5-32k | 0.286 | 1.3e-05 | ✅ PASS | `'mlp_out'` |
| attn_out@8 | attn-out-v5-32k | 0.323 | 1.5e-05 | ✅ PASS | `'attn_out'` |

- ✅ **Splice losslessness holds on all three real site types** (the v1.7 routed splice).
- ✅ **Composite keying correct on a real model**: `component=None` for resid_post (preserves the
  v1.6 node identity / golden freeze), `'mlp_out'`/`'attn_out'` for the new sites.
- ✅ **Mechanistically sensible**: attn_out@8 carries a strong IOI feature (−1.13) while mlp_out@8
  features barely move the metric (±0.05) — correct, IOI is attention-mediated (name-movers ~L8–9).
  The machinery produces *meaningful* attributions, not noise.
- 🟡 **Real-world note (F20):** the OpenAI v5 mlp/attn SAEs reconstruct poorly here
  (FVU 0.29/0.32) vs res-jb (0.001) — likely a `prepend_bos`/context mismatch (v5 trained with
  `prepend_bos=False`, ctx 64). Attribution at those sites leans heavily on the error term; the
  error node matters more than at resid_post.

### 2.2 Feature→feature edges + bruteforce oracle + FeatureACDC
- ✅ **Edge math vs independent bruteforce oracle** (`resid_post@6→@7`, both res-jb): the analytic
  attrib edges agree with the bruteforce ground truth on a real nonlinear model —
  **sign-agreement 1.00, Pearson 0.923** over the top-6 edges. (e.g. `L6:18518→L7:22430`
  attrib −1.218 vs bruteforce −1.110.)
- ✅ **Intra-layer `attn_out@8 → mlp_out@8` edges** (v1.7 headline): 50 edges, **0 forward-order
  violations** (writer rank < reader rank enforced for same-layer composite components).
- ✅ faithfulness/completeness machinery runs (full-circuit returns the trivial boundary value
  1.0000 — correct; a thresholded sub-circuit would give non-trivial values).
- 🟠 **F5 — IG edges have no independent oracle.** `bruteforce_feature_edge_scores` uses a
  single-point grad@clean, so it grounds the *attrib* estimator (matches it, above) but **not**
  the IG path-integral; IG edge correctness currently rests on internal consistency only.

### 2.3 The two real-model SAE bugs
- 🔴 **F3 — `layer_norm` SAE statefulness breaks faithfulness/ACDC.** Decisive isolation test on
  the v5 mlp_out SAE (`normalize_activations='layer_norm'`):
  `max|decode(f1)_immediate − decode(f1)_after_encode(a2)| = 9.05`. `sae.encode` stores per-call
  normalization state that `decode` consumes; `compute_f_per_site` (`sae_edges.py:117`) calls bare
  `encode` and the cached value is decoded later in the clean-input frame → ablation values land in
  the wrong normalization frame. Affects every `layer_norm`-normalized SAE — including the standard
  OpenAI v5 GPT-2 SAEs. `sae/grad.py:92-108` already documents the "never cache f, always decode in
  the same call" rule; `compute_f_per_site` violates it.
- 🔴 **F4 — `variant='ig'` + `include_error_node=True` silently drops error→feature edges.**
  Real run: attrib produced **10** error→feature edges, ig produced **0**. The IG writer hook
  (`sae_edges.py:1446-1577`) never builds `err_leaf_U`, so `delta_eps_U` stays `None` and the
  error-writer branch is skipped with no error/warning. Confirmed independently by static review and
  Gemini (MAJOR).

---

## 3. Dynamic — patching backends (EAP/ATP/ACDC) post-v1.7 refactor  ✅

Script: `track1_patching_revalidation.py`. **Verdict: no regression.**

| metric | pre-v1.7 baseline | v1.7 | status |
|---|---|---|---|
| EAP/TL top-20 heads | [(9,9),(5,5),(3,0),…] | identical | exact |
| AtP*/TL top-20 heads | [(8,6),(6,9),(8,10),…] | identical | exact |
| HF-Qwen logit-diff clean / corrupt | +5.5728 / +0.0762 | identical | exact |
| HF-Qwen Spearman-distinct / sign-agree | 0.853 / 1.0 | identical | exact |
| golden-freeze (SAE resid_post) | — | rtol=0/atol=0 | ✅ |

- ✅ Structural reason it can't regress: `eap.py`/`atp.py`/`acdc.py` have **zero** `ResolvedSite`
  / `resolver.resolve` calls — they wire hooks directly; the resolver is used only by the SAE
  splice + `intervene.patch_site`. P1 was well-isolated.
- 🟠 **F6 — non-deterministic faithfulness metric under tied scores.** `Spearman-all` differed
  0.790 vs 0.858 across runs because two equal-ATP-score nodes get dict-order-dependent pairing.
  Not a regression; a reproducibility smell. (The `Spearman-distinct` variant is stable.)
- (ACDC live CPU sweep of 32k edges didn't finish in budget; graph structure `n_edges=32491`
  matches baseline and the code path is statically unchanged.)

---

## 4. Dynamic — Recorder training-time metrics  ✅ (end-to-end) / 🔴 (two diagnostics)

Script: `recorder_correctness.py`. 2-layer HF Llama (656K params), 100 steps, CPU, jsonl.

- ✅ End-to-end clean: `attach`→100 steps→`detach`, 2466 rows / 244 tags across
  train/weight/activation/grad/spectral families; 36/36 hookpoints matched; `build_report` and
  `scan_run` both succeed.
- 🔴 **F1 — SVD-derived weight diagnostics biased & non-deterministic on min-dim>512** (see §5;
  surfaced here on real weights, confirmed catastrophic on GPT-2 in §5).
- 🔴 **F2 — `attention_pattern_entropy` is probe-contaminated.** At each emit step,
  `induction_score` runs `model(probe, output_attentions=True)` (`live.py:869`) which re-fires the
  permanent `_main_pass_attn` hook and overwrites the training-forward attention; the
  later-ordered `attention_pattern_entropy` block (`live.py:892-916`) then reads probe attention.
  Proven: emitted entropy = [2.9682, 2.9685, …] matches a fresh **probe** forward to <1e-5, while
  the true training-forward entropy is ~2.55 (off by 0.42). Also affects `scan_run`. Fix: snapshot
  `_main_pass_attn` before the probe, or guard the hook during the probe forward.
- 🟠 **F8 — `scan_run` runs activation diagnostics with no forward pass.** `logit_lens_kl`,
  `dead_fraction`, `kurtosis`, `participation_ratio`, `gate_stats` get empty `ctx.activations` →
  silently emit nothing; `logit_lens_kl` logs a warning every step and **permanently** sets
  `_lens_meta=None`. `scan.py` docstring implies weight-only. Fix: guard `if not ctx.activations:
  continue`, or add a `forward_fn` to enable activation diagnostics on checkpoints.
- 🟠 **F9 — `condition_number` is in `_WEIGHT_DIAGS` + design §4.1 but absent from the `llm`
  recipe** (`recipes/llm.py:36-39`). Default-llm users get no condition_number at all (slightly
  mitigates F1 for *that one* metric — but `effective_rank`/`stable_rank`/`heavy_tail_alpha` *are*
  in the recipe and *are* biased).
- 🟡 **F15 — gradient tag namespace split:** `norms_per_param` (the only grad diag in the llm
  recipe) emits under `grad/…`, while `grad_norm_per_module` emits under `gradient/…`. A user
  grepping `gradient/` finds nothing from the default recipe.
- **Perf budget:** not re-measured this session (the §10 ≤10% target was validated on rtx at
  +5.3% in v1.4.2). ⚠️ **Tension worth flagging:** the SVD subsampling behind F1 is a *perf*
  optimization (bounding SVD cost) that silently corrupts correctness — the F1 fix must trade some
  wide-matrix SVD speed for accuracy (or use a deterministic randomized/Lanczos top-k instead of
  naive column subsampling). See F1 suggested action.

---

## 5. Dynamic — core primitives + scan/report + recipes  ✅ (mostly) / 🔴 (SVD family) / 🟠 (vision)

Scripts: `part1_core_primitives.py`, `part2_vision_recipe.py`, `part3_two_tower_recipe.py`.

### 5.1 🔴 F1 — SVD subsampling, fully quantified on GPT-2 (768×3072 MLP `c_fc`, min-dim 768>512)
| primitive | subsampled (default) | full / true | error |
|---|---|---|---|
| `effective_rank` | 433 | 700 | **38% low** |
| `heavy_tail_alpha` | 1.11 | 1.65 | **33% low** |
| `condition_number` (this matrix) | 38.8–44.6 (varies!) | 44.4 (numpy) | ratio survives by luck |
| `condition_number` (1024² random) | 5.68 | 1649 | **290× low** |
| `singular_values` σ_max | 18.1 | 42.1 | 57% low |

- Root cause: `singular_values(W, …, max_dim=512)` truncates to a random 512-column subsample
  (`weight.py:68`). `condition_number` (`weight.py:157`) doesn't pass `max_dim=None` despite its
  docstring promising "full SVD path"; and it exposes **no `max_dim` kwarg** to override
  (`condition_number(W, max_dim=None)` → `TypeError`). `effective_rank`/`stable_rank`/
  `heavy_tail_alpha` likewise have no passthrough.
- **Non-determinism:** `seed=None` default → values vary run-to-run for any min-dim>512 matrix.
- **Scope:** every weight matrix of every real LLM (d_model ≥ 768 > 512). The live recorder emits
  these biased, non-reproducible numbers for production-scale models. **Headline finding.**
- Suggested action: route `condition_number`/`effective_rank`/`stable_rank`/`heavy_tail_alpha`
  through an accurate default (full SVD, or a deterministic randomized/Lanczos top-k that doesn't
  bias σ_min/the tail); add a `max_dim` passthrough + a stable default `seed`; add a regression
  test `condition_number(randn(1024,1024))` vs `np.linalg.cond`. Update docstrings.

### 5.2 🟠 F7 — vision recipe under-captures standard backbones
- Pattern `(conv\d+|fc\d+|…)` requires a digit suffix → **misses ResNet's `fc` classifier head**
  and all `downsample` convs. Only **20 of 68** ResNet-18 modules captured (29%). Fix: make the
  digit optional (`conv\d*|fc\d*`) or add explicit `fc$`; document supported naming.

### 5.3 🟡 F10 — `participation_ratio` on 4D conv outputs counts elements, not channels
- Reports ~164k for a 4×16×64×64 tensor (≈ B·C·H·W) — scales with batch·resolution, not capacity.
  Misleading for conv health. Fix: document, or add a `channel_first` reduction mode.

### 5.4 ✅ what ran clean on real data
- All of `activation.py` (`norm_stats`, `dead_fraction`, `kurtosis`, `participation_ratio`,
  `gate_stats`, `token_similarity`, `repr_drift` ×3 methods) — 0 NaN/inf on real GPT-2 hidden
  states; self-drift exactly 0.
- `gradient.py` (incl. 4D conv grads), `spectral.py` (`esd`, `rank_trajectory` incl. 4D conv
  weights), `attention.py` (entropy/induction on real patterns), `lens.py` (`logit_lens_kl` exact
  0.0 at the last layer — correct by construction; F21: document it).
- **Vision** + **two_tower** recipes both emit metrics end-to-end; two_tower's custom
  `embedding_alignment` diagnostic fires every step.

---

## 6. Findings register (prioritized)

| # | Sev | Area | Finding | Evidence | Suggested action |
|---|-----|------|---------|----------|------------------|
| **F1** | 🔴 | core/weight | SVD-derived diags (cond/eff_rank/stable_rank/heavy_tail) biased + non-deterministic for min-dim>512 (every real LLM layer) | eff_rank 433 vs 700; cond 5.68 vs 1649 (1024²); seed=None | full/deterministic SVD by default; `max_dim`+`seed` passthrough; regression test |
| **F2** | 🔴 | recorder/live | `attention_pattern_entropy` emits induction-probe attention, not training attention; also in scan | emitted 2.968 ≡ probe vs 2.55 training; `live.py:869,892-916` | snapshot `_main_pass_attn` before probe / guard hook during probe |
| **F3** | 🔴 | patching/sae_edges | faithfulness/completeness/ACDC wrong for `layer_norm` SAEs (incl. OAI v5) — `compute_f_per_site` bare `encode` cached then decoded in wrong frame | 9.05 drift isolation test; `sae_edges.py:117`; rule at `grad.py:92-108` | decode in same call / re-encode at ablation site; or reject layer_norm SAEs loudly |
| **F4** | 🟠 | patching/sae_edges | `variant='ig'`+`include_error_node` silently drops all error→feature edges | attrib 10 → ig 0 (real run); `sae_edges.py:1446-1577` | build interpolated `err_leaf_U` in IG writer hook, or raise NotImplementedError |
| **F5** | 🟠 | patching/sae_edges | IG edges have no independent oracle (bruteforce is single-point-grad, grounds attrib only) | §2.2 | add a path-integral bruteforce for IG edge validation |
| **F6** | 🟠 | patching | faithfulness `Spearman-all` non-deterministic under tied scores (dict order) | 0.790 vs 0.858 same data | dedupe by distinct (atp,true) pairs |
| **F7** | 🟠 | recipes/vision | pattern needs digit suffix → misses ResNet `fc` head + downsample convs (29% capture) | 20/68 ResNet-18 modules | `conv\d*|fc\d*` / explicit `fc$`; document backbones |
| **F8** | 🟠 | recorder/scan | `scan_run` runs activation diags with no forward → silent empty + warnings + disables `_lens_meta` | 0 lens rows; `scan.py:33-74`, `live.py:781-787` | guard empty activations; or add `forward_fn` |
| **F9** | 🟠 | recipes/llm | `condition_number` in `_WEIGHT_DIAGS`+design but absent from llm recipe | 0 cond rows; `llm.py:36-39` | add to recipe (post-F1) or document exclusion |
| **F10** | 🟡 | core/activation | `participation_ratio` counts elements not channels on 4D | 164k for 4×16×64×64 | doc + optional channel mode |
| **F11** | 🟠 | patching/sae_edges | `completeness()` omits `ablation_eps` under `include_error_node` (out-of-circuit error not ablated) | `sae_edges.py:707-731` vs `faithfulness:612-648` | thread `ablation_eps` through completeness |
| **F12** | 🟡 | patching/sae_edges | IG edge hook recomputes `eps_clean` (encode+decode) every one of N steps | `sae_edges.py:1464-1468` | hoist `eps_clean` out of the k-loop |
| **F13** | 🟠 | patching/sae_edges | `FeatureACDCRunner` doesn't expose `variant`/`n_ig_steps`; Stage-1 always attrib | `sae_edges.py:2055-2099` | thread params through run()/sweep() |
| **F14** | 🟡 | patching | module docstrings stale: "only HF / only resid_post" (false since P2/P3) | `sae_features.py:1-9`, `sae_edges.py:1-16` | update docstrings |
| **F15** | 🟡 | recorder/live | grad tag namespace split `grad/` vs `gradient/` | `live.py:1049-1064` | unify prefix |
| **F16** | 🟡 | docs | `design.md §4.2` `scan_run` example omits `model_factory` + `writer="jsonl"` | `scan.py` sig | fix example |
| **F17** | 🟢 | docs | `gate_stats` returns `dict` not documented `GateStats` | `activation.py:348` | update §4.1 |
| **F18** | 🟠 | CI | no import-linter contract for "patching/ must-not import cli/" (design claims enforced) | `.importlinter` | add contract |
| **F19** | 🟢 | docs | §10 says TB writes async "by default"; default is `async_writes=False` | `tensorboard.py:23` | align doc or default |
| **F20** | 🟡 | sae/real-world | OAI v5 mlp/attn SAEs reconstruct poorly on default-processed gpt2 (FVU 0.29/0.32) | §2.1 | document prepend_bos/ctx matching guidance |
| **F21** | 🟢 | core/lens | `logit_lens_kl`=0 at last layer is correct-by-construction but undocumented | §5.4 | docstring note |
| **F22** | 🟠 | recorder/live | `logit_lens_kl` warning text promises an `lm_head` fallback the code doesn't implement (only `get_output_embeddings()` tried) | `live.py:240-258` | add the fallback or fix the warning text |
| **F23** | 🟡 | recorder/compare | `compare_runs` scan-vs-live is silently asymmetric (scan has no activation families) — NaN sentinel, no warning | `compare.py` | warn on family-set mismatch |
| **F24** | 🟡 | cli | `circuitry scan` is a non-functional placeholder (no `--model-factory`); exits rc=2 (helpful msg) but is listed in `--help` | `cli/main.py:41-48` | add `--model-factory` or hide until functional |
| **F25** | 🟡 | patching/docs | IG "N× cost" holds for Stage-2 only; total wall-time ≈ N×(1−f₁), f₁≈0.5 (Stage-1 runs once) | §8.2 timings | clarify docstring |
| **F26** | 🟠 | patching/sae | bf16 SAE splice losslessness degrades to ~1.5e-1 logit err (fp32-trained SAE weights quantized) — finite + sign-consistent, but not lossless | §8.2 | doc: keep SAE weights ≥ model precision (cast acts up before encode) |
| **F27** | 🟠 | patching/docs | SAE runners' HF backend is **Llama-family-only** (`AtPRunner._locate_layers`); `HFSiteResolver` `layer_pattern` overrides are moot for non-Llama HF models — must convert via `to_hooked_transformer`+TL | `_layout.py:29`; §8.3 | doc the limitation; fix the misleading reviewer-style "GPT-2 via HFSiteResolver" snippet |
| **F28** | 🟡 | patching/sae | TL-trained SAEs are unusable on **raw HF** activations (FVU 12613 for res-jb on HF GPT-2) | §8.3 | doc: SAE attribution needs the activation processing the SAE was trained on (use TL backend) |
| **F29** | 🟠 | recorder/live | `Recorder.attach()` **crashes on any SDPA-attention HF model** (Qwen2 / Llama-3.x / Mistral / Gemma — all default `attn_implementation='sdpa'`): `_set_output_attentions_true()` raises `ValueError` | Qwen2.5-0.5B; `live.py:534` | try/except → fall back + skip induction/attn-entropy diags, or doc the `eager` requirement |
| **F30** | 🟠 | recipes/vision | vision recipe matches **0/151** torchvision ViT-B/16 modules → hard RuntimeError on attach; pattern is timm/DeiT (`blocks.N.*`)-locked, doesn't cover torchvision (`encoder.layers.encoder_layer_N.*`) | §9.2 | broaden pattern or add a `vision_vit` recipe; fix docstring claiming ViT support |
| **F31** | 🔴 | core/gradient | `grad_norm_per_module` **crashes** (`NotImplementedError`, SparseCPU) on sparse `nn.Embedding` gradients — standard in recsys | `gradient.py:21` | `g = g.to_dense() if g.is_sparse else g` before `vector_norm` |
| **F32** | 🟠 | recipes/two_tower | matches **0%** of a standard-named DLRM (`embed_tables`/`bottom_mlp`); naming-locked to `query_tower|item_tower`; no `recsys`/`dlrm` recipe exists | §9.2 | add a flexible recsys recipe; doc the naming requirement |
| **F33** | 🔴 | recipes/two_tower | `embedding_alignment` silently returns `{}` when item_tower is an `nn.ModuleList` (forward hook never fires on a ModuleList) | §9.2 | register hooks on each child, or doc the nn.Sequential requirement |
| **F34** | 🟡 | recipes/two_tower | activation hooks fire only on top-level towers, not individual embedding tables (`item_tower.{0..3}`) → no per-table dead_fraction | §9.2 | extend output pattern to `(query_tower\|item_tower)(\.\d+)?$` |
| **F35** | 🟡 | recipes/two_tower | weight hook pattern matches container modules (ReLU/ModuleList) → 6 "no resolvable weight" warnings per attach | §9.2 | filter container/no-weight module types |
| **F36** | 🟠 | recipes/llm | MoE expert weights **0%** covered (0/4856 metrics) — OLMoE stores experts as batched 3D tensors (`[64,2048,2048]`), no leaf Linears; recipe mlp patterns match nothing | §9.3 | add MoE-aware hook points on batched expert tensors |
| **F37** | 🟡 | recorder/live | no user-facing summary when all MLP weight hook points match 0 modules (silent under-coverage on MoE) | §9.3 | INFO note when a weight family matches 0 modules |
| **F38** | 🔴 | core/weight | `effective_rank`/`stable_rank` on a 3D batched expert tensor silently flattens the expert axis to rows → **semantically wrong** rank (63.95 ≈ #experts vs true per-expert ~1599) | `weight.py:_as_2d`; §9.3 | raise on ndim>2, or document the ≤2D contract |
| **F39** | 🟡 | recipes/llm | MoE router weights (`mlp.gate`, `[64,2048]`) unmatched → load-balance/imbalance invisible | §9.3 | opt-in router-weight hook point |

---

## 7. Recommended next-release backlog

**Priority 1 — correctness (ship before any new features):**
- **F1** — fix the SVD subsampling family. This silently corrupts weight diagnostics on *every*
  real LLM and is the single most important issue found (confirmed up to 45,800× wrong on a real
  MoE expert slice). Make accurate + deterministic by default; keep a perf escape hatch.
  Regression-test against numpy.
- **F2** — fix `attention_pattern_entropy` probe contamination (snapshot/guard `_main_pass_attn`).
- **F3** — fix `compute_f_per_site` for `layer_norm` SAEs (decode-in-same-call), or refuse them
  loudly. Without this, SAE faithfulness/ACDC on the most common public SAEs is wrong.
- **F4** — make the IG error→feature edge case correct or explicitly unsupported (no silent drop).
- **F29** — `Recorder.attach()` crashes on every SDPA-default HF model (Qwen2/Llama-3.x/Mistral/
  Gemma). High blast radius; fix the `output_attentions` enablement to degrade gracefully.
- **F31 / F33 / F38** — three silent/crashing correctness bugs surfaced by real ViT/recsys/MoE:
  sparse-embedding grad crash, `embedding_alignment` ModuleList silent-`{}`, 3D-expert-weight
  wrong-rank.

**Priority 2 — recipe robustness (cross-cutting theme from round 3):**
- The recipes silently under-cover non-(GPT-2/Llama) architectures: **F30** (torchvision ViT 0%),
  **F32** (DLRM 0%), **F36/F37/F39** (MoE experts + router invisible), **F7** (ResNet fc/downsample),
  **F34/F35** (recsys table/container matching). Decide the contract: either make matching
  arch-aware (introspect module types, not names) or **fail loud with guidance** instead of silent
  under-coverage. Add a per-family "matched N modules" coverage line to the report.
- Other robustness/honesty: F8 (scan-no-forward), F9 (recipe/cond), F11 (completeness ablation_eps),
  F13 (ACDC variant passthrough), F6 (tied-score determinism), F18 (import-linter contract),
  F22 (lens warning), F23 (compare asymmetry), F27 (HF SAE Llama-only doc).

**Priority 3 — ergonomics / docs / polish:**
- F10, F12, F14, F15, F16, F17, F19, F20, F21, F24, F25, F26, F28.

**What's validated and safe to build on:**
- The v1.5–1.7 SAE-circuit *math* (splice, keying, node/edge attribution, intra-layer edges,
  faithfulness machinery) on real models. The EAP/ATP/ACDC backends (no v1.7 regression). The
  activation/gradient/spectral/attention/lens primitives on real data. The vision/two_tower
  recipe plumbing (modulo F7's matching breadth). The end-to-end Recorder lifecycle.

**Process note:** the bugs clustered in the *oldest, first-shipped* code (`core/weight`,
`recorder/live`) — the code we stopped scrutinizing because it "already worked." The newest,
most-reviewed code (v1.7 SAE circuits) was the cleanest. Future validation should periodically
re-exercise the foundation against real-scale inputs, not only the latest layer.

---

## 8. Round 2 — additional coverage (parallel-attn, HF backend, batch>1, scan/writers/CLI, memory/bf16)

A second pass targeting the surfaces round 1 didn't touch. **Net: the architecture and robustness
held up well — round 2 found mostly ✅ confirmations + a handful of doc gaps (F22–F28), no new
correctness bugs.** This reinforces that the v1.7 design is sound; the real issues are the
specific primitives in §0 (#1–#4), not the framework.

### 8.1 Parallel-attention model — the v1.7 caveat, verified  ✅
Script: `sae_parallel_attn.py`. **pythia-70m-deduped** (`parallel_attn_mlp=True`, GPTNeoX) + real
pythia SAEs. The v1.7 docs scope attn_out/mlp_out equivalence to *sequential* blocks and warn
parallel-attn "differs." Verified circuitry behaves **safely**:
- ✅ splice losslessness holds at resid_post/mlp_out/attn_out (max|Δlogit| ~2.5–3e-03); node
  attribution works with correct component fields.
- ✅ connected `resid_post@2→resid_post@3`: 100 edges, **0 warnings**.
- ✅ severed intra-layer `attn_out@3→mlp_out@3` (mlp is parallel to attn, doesn't read it): the
  v1.7 no-grad guard **warns + returns empty edges** ("f_D.grad is None for pair attn_out@3 →
  mlp_out@3") — **no raise, no crash, no silent spurious edges.** Exactly the documented contract.
  The guard correctly distinguishes the parallel-severed pair from the connected resid pair.

### 8.2 SAE machinery claims — memory / IG-cost / bf16
Scripts: `inv1_bounded_memory.py`, `inv2_ig_cost.py`, `inv3_bf16_robustness.py` (d_sae=24576).
- ✅ **Bounded-memory claim CONFIRMED** — peak **28.6 MB** at top_k_survivors=256 vs a **2.42 GB**
  dense `d_sae²` Jacobian (85× below). The per-downstream-survivor VJP never materializes the
  dense Jacobian on a real SAE. Growth is sub-quadratic in top_k.
- ✅ **IG memory is O(1) in N** (peak ≈ attrib: 0.23→0.31 MB for N=8→32). F25: the "N× cost" is
  Stage-2-only; total wall-time ≈ N×(1−f₁) since Stage-1 node scoring (≈48% of attrib) runs once.
- 🟠 **bf16 (F26):** no NaN, scores finite, **100% sign-consistent** with fp32, magnitudes <3%
  off for strong features — but splice losslessness degrades to ~1.5e-1 logit error because the
  fp32-trained SAE weights quantize to bf16 (recon err ≈ bf16 eps). Not a circuitry bug; needs a
  doc note to keep SAE weights ≥ model precision.

### 8.3 HF backend + batch>1
Script: `sae_hf_and_batch.py`.
- ✅ **Clear failure, not a silent footgun:** `HFSiteResolver.from_config(gpt2_config)` builds a
  Llama-pathed resolver, but the SAE runner then **raises a clear, actionable error** naming the
  unsupported arch and the migration path (`to_hooked_transformer` + TL backend, design §4.6).
- 🟠 **F27 (doc gap):** consequently the SAE runners' HF path is **Llama-family-only**
  (`AtPRunner._locate_layers` gates before the resolver is used); `HFSiteResolver` `layer_pattern`
  overrides cannot make GPT-2 work. The static-review API snippet implying "GPT-2 via
  HFSiteResolver" is wrong — all SAE work on GPT-2 must use the TL backend (which is what this
  whole eval correctly did).
- 🟡 **F28:** res-jb on **raw HF** GPT-2 activations gives **FVU 12613** (vs 0.001 on TL-processed)
  — TL-trained SAEs require the activation processing they were trained on.
- ✅ **batch>1 works:** batch=3 `(3,15)` input → node attribution (6 feats) + edges (64), all
  **finite, no crash.** The whole-tensor (position=None) splice handles real batched input.

### 8.4 scan / writers / compare / CLI
Script: `part4_scan_writers_compare_cli.py`.
- ✅ Cross-step weight primitives correct on real checkpoints: `update_delta` decays as Adam
  converges, `direction_cosine` rises toward 1.0, `rank_trajectory` stable; the v1.3 copy=True
  snapshot fix holds (**no aliasing bug**; a zero `q_proj` delta was a *correct* unused-param result).
- ✅ All writers work: Jsonl (line-buffered, histogram `.npy` artifacts), TensorBoard (sync +
  async both produce readable tfevents), Null (no-ops). `compare_runs`/`build_compare_report`
  produce correct `FamilyDelta` output. CLI `--help/--version/list-recipes/report/--compact/compare`
  all work.
- New gaps: F22 (`logit_lens_kl` warning promises a non-existent `lm_head` fallback), F23
  (compare scan-vs-live asymmetry unwarned), F24 (CLI `scan` is a non-functional placeholder),
  reinforced F8 (scan silently skips activation diagnostics) + F19 (TB async-default doc mismatch).

---

## 9. Round 3 — architecture-family breadth (Qwen, Gemma, MoE, ViT, recsys)

Coverage before round 3 was narrow: GPT-2 (thorough), pythia (SAE), Qwen (patching-only), a
*random* 2-layer Llama (Recorder-only). Round 3 widened to real Qwen, a real MoE, a Vision
Transformer, and a DLRM-style recsys. **Headline: the SAE/patching/primitive *cores* are
architecture-robust, but the recipe module-matching is naming-coupled and silently under-covers
every architecture that isn't GPT-2/standard-Llama — plus 3 new real bugs (F31, F33, F38) and a
high-impact Recorder crash (F29).**

### 9.1 Qwen2.5-0.5B (Llama-family) — HF SAE backend POSITIVELY validated  ✅
Script: `qwen_hf_sae_validation.py`. The HF SAE backend was only ever shown *rejecting* GPT-2;
here it is confirmed to **work** on a real Llama-family model:
- ✅ `locate_layers` accepts Qwen (24 layers); `HFSiteResolver.from_config` → n_heads=14,
  d_model=896; resid_post/mlp_out/attn_out resolve to correct Qwen submodules; `SAEFeatureRunner`
  runs; splice lossless at activation level (max|recon−act| = 4.77e-07); `AtPNode.component` correct.
- 🟠 **F29 (high impact):** `Recorder.attach()` crashes on Qwen loaded with the **default `sdpa`**
  attention (`_set_output_attentions_true` → ValueError). Needs `attn_implementation="eager"`.
  This hits *most modern HF LLMs* (Qwen2, Llama-3.x, Mistral, Gemma all default to sdpa).
- 🔴 condition_number (F1) on real Qwen 896×896 weights: **1631–2138× wrong** (q_proj 1528 vs
  2,496,216 exact). The most extreme confirmation yet.

### 9.2 Vision Transformer + DLRM recsys — recipes don't cover them  🟠 + 🔴
Scripts: `eval_vit_recipe.py`, `eval_dlrm_recipe.py`.
- 🟠 **F30:** vision recipe matches **0/151** torchvision ViT-B/16 modules → hard RuntimeError on
  attach. The pattern is timm/DeiT (`blocks.N.*`)-locked; torchvision ViT uses
  `encoder.layers.encoder_layer_N.*`. Docstring claims ViT support.
- 🔴 **F31:** `grad_norm_per_module` crashes (`NotImplementedError`, SparseCPU backend) on sparse
  `nn.Embedding` gradients — the standard recsys embedding setup.
- 🟠 **F32:** two_tower matches **0%** of a standard-named DLRM; no recsys/dlrm recipe exists.
- 🔴 **F33:** even with recipe-matching names, `embedding_alignment` silently returns `{}` when
  item_tower is an `nn.ModuleList` (the forward hook never fires on a ModuleList).
- Plus F34 (per-table activations missed), F35 (container modules matched), F1 again (cond 133×
  wrong on ViT 768×768). ✅ weight/spectral primitives don't *crash* on ViT/embedding shapes;
  embedding (10000×64, min-dim 64) handled correctly.

### 9.3 MoE (OLMoE-1B-7B) — framework survives, but blind to experts  🟠 + 🔴
Script: `moe_olmoe_eval.py`. **Verdict: circuitry runs on MoE without crashing but does not
usefully cover it.**
- ✅ `locate_layers` accepts OLMoE (16 layers); resid_post/mlp_out/attn_out resolve (mlp_out →
  the whole `OlmoeSparseMoeBlock` output — correct block-level signal); Recorder runs end-to-end;
  **backward through top-8 sparse routing works** with no circuitry intervention needed.
- 🟠 **F36:** **0 of 4856** emitted metrics cover expert weights. `OlmoeExperts` stores experts as
  batched 3D tensors (`gate_up_proj [64,2048,2048]`, `down_proj [64,2048,1024]`) — *no leaf Linear
  submodules*, so the recipe's mlp patterns match nothing. Experts MISSED (not exploded).
- 🔴 **F38:** `effective_rank`/`stable_rank` on those 3D expert tensors silently flatten the
  64-expert axis to rows → **semantically wrong** (eff_rank 63.95 ≈ #experts, vs true per-expert
  ~1599). No error raised — the dangerous silent-wrong class.
- condition_number (F1) **45,800× wrong** (7.42 vs 339,852) and effective_rank 3.3× low on a
  2048² expert slice. F37 (no 0-match summary), F39 (router weights invisible).

### 9.4 Gemma-2-2b (post-norm) — BLOCKED on HF gating  ⚠️
`google/gemma-2-2b` is a **gated** HF repo; download fails with `GatedRepoError: 401` (no
HF_TOKEN / license not accepted). The **post-norm caveat remains untested.** To run it: accept the
license at huggingface.co/google/gemma-2-2b and `huggingface-cli login` with a token, then re-run.
(We did not route around the gating with mirrors.) OLMoE uses pre-norm, so it does not substitute
for the post-norm test.

---

## 11. Post-1.8.0 follow-ups (deferred TODOs)

All 20 test-worthy findings above were fixed and shipped in **v1.8.0** (each with a red→green
regression test under `tests/**/test_*_eval_findings.py`). The items below were intentionally
deferred — they are tracked here and at the relevant code sites (`grep -rn "TODO(v1.8 follow-up"`):

1. **Re-validate the §10 ≤10% wall-clock budget on GPU.** The F1 fix makes the live recorder
   compute the *full* SVD on wide weights by default (the ≤10% budget was measured with the old
   `max_dim=512` subsample). Run `scripts/bench_50m.py` on the RTX box with the budget scenario
   (`--every-n-steps 200 --batch-size 16 --seq-len 512`) and record the new overhead. If it
   exceeds budget, document an explicit per-recipe `max_dim` for perf-sensitive live use (the
   accuracy/perf trade-off is the user's to make, with the bias documented in §4.1/§10).
2. **Refine the F37 0-match weight-family warning (noise).** Because the `llm` recipe now carries
   MoE-only patterns (`mlp.gate` / `mlp.experts`), the aggregate "N weight pattern(s) matched 0
   modules" warning fires on *every* non-MoE attach. Demote to INFO when the only empty families are
   the MoE-specific ones, or split MoE patterns into an opt-in recipe variant. (`recorder/live.py`.)
3. **Sync the `pyproject.toml` version.** ✅ **Done (2026-06-01).** `pyproject.toml` had read
   `1.4.2` since the v1.5.0 release while `circuitry.__version__` tracked the real version (and the
   installed metadata had frozen at `1.1.0`) — three different numbers. Resolved by making
   `pyproject` derive the version dynamically (`[project] dynamic = ["version"]` +
   `[tool.setuptools.dynamic] version = {attr = "circuitry.__version__"}`), so `__init__.py` is the
   single source of truth. Guarded by `tests/test_version_consistency.py` (fails if README/CHANGELOG
   drift from `__version__` or if a static `version` line returns to pyproject) and documented as a
   repeatable process in `.claude/skills/release-checklist/SKILL.md`.
4. **F6 — not a library bug.** The tied-score Spearman non-determinism lives only in the eval
   *script* (`scripts/v17_validation/track1_patching_revalidation.py`); the library ranking
   (`atp.verify_top_k`) is deterministic. No library fix; left as-is.
5. **Gemma-2-2b post-norm caveat (F-blocked).** Still untested — gated HF repo. Accept the license +
   `huggingface-cli login`, then exercise the `mlp_out`/`attn_out` SAE-site equivalence on a
   post-norm architecture (the v1.7 caveat scopes equivalence to sequential pre-norm blocks).
