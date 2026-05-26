# 2026-05-25 — v1.0 real-model validation campaign

End-to-end validation of the shipped v1.0 surface against **real models and a
published ground-truth circuit**, acting as an MLE. Prior validation (Gemma,
v0.9.x) was static `lr=0` single-step capture; this campaign adds (1) the
patching pillar vs the published IOI circuit, (2) a real `lr>0` training run with
the Recorder, (3) observation primitives on an eager-attention model so
induction/entropy actually emit, and (4) the CLI report/scan workflow.

**Bottom line:** the patching pillar genuinely works on real models — ACDC pruned
GPT-2 from 32,491 edges to a **10-edge circuit whose 8 heads are 100% in the
published IOI circuit**, and EAP/AtP\* recover it at overlap@10 = 100%. The campaign
*also* surfaced two real issues against the design contract: the HF patching backend
is Llama-family-only (silently broken on Gemma/GPT-2 → R1), and the §10 ≤10%
wall-clock budget is an unenforced, workload-dependent target (→ R2/R3). Every other
shipped feature (Tier-1 primitives, SAE, live Recorder, vision/two_tower recipes,
writers, CLI report/scan) validated cleanly.

## Reproducibility

| | |
|---|---|
| Date | 2026-05-25 |
| Host | single RTX 5080 (16 GB), CUDA 13.0 |
| Python | 3.12.13 |
| torch | 2.12.0+cu130 |
| transformers | 5.9.0 |
| transformer_lens | 3.2.1 |
| sae_lens | 6.44.0 |
| circuitry | 1.0.0 (HEAD on `main`) |
| Seeds | torch/numpy seed 0 unless noted |

Scripts: `scripts/v1_validation/` (per-track, re-runnable). Numeric results saved
as JSON alongside. Backend probes: `scripts/v1_probe_backends.py`,
`scripts/v1_probe_atp_acdc.py`.

---

## Scorecard

| Feature | Vehicle | Verdict |
|---|---|---|
| EAP (TL) | GPT-2 IOI | ✅ recovers published circuit (overlap@10 100%) |
| AtP\* (TL) | GPT-2 IOI | ✅ recovers circuit (overlap@10 100%, @26 77%) |
| ACDC (TL) | GPT-2 IOI | ✅ 32491→10 edges, 8/8 heads in published circuit (64min/τ) |
| patch_site / verify_top_k (HF) | Qwen2.5-0.5B GQA | ✅ faithful (Spearman 0.85, sign 100%) |
| EAP/AtP* HF backend | Qwen2.5-0.5B | ✅ runs on real GQA model |
| EAP/AtP*/ACDC HF backend | Gemma-2/3, GPT-2 | ❌ **broken** (head_dim / Conv1D) → R1 |
| Recorder live (lr>0) | tiny Llama / TinyStories | ✅ dynamic capture, diagnostics move |
| §10 ≤10% budget | 50M-class, GPU | ◑ holds at ≥25ms/step; not on synthetic stub → R2/R3 |
| EAP-IG / AtP* GradDrop / neuron | Qwen2.5-0.5B | ✅ all run (EAP-IG 9/10 vs vanilla; 116k neurons scored) |
| PatchRunner denoise / noise | Qwen2.5-0.5B | ✅ both directions move the metric correctly |
| Tier-1 primitives (all) | GPT-2 eager | ✅ all correct (lens falls, induction heads, sparsity, …) |
| SAE workflow | Gemma-2-2b + Gemma Scope | ✅ hits design point (l0=71 ex-BOS) |
| Recipes: vision / two_tower | TinyCNN / TwoTower | ✅ emit tags + report (lr>0) |
| Writers: jsonl / null / tensorboard | various | ✅ all three work |
| scan_run strict=True | Llama checkpoints | ✅ passes on clean model |
| CLI report / list-recipes | Track-2 run | ✅ work |
| CLI scan / compare | — | ◑ scan = stub (use scan_run); compare unshipped → R4 |

---

## Headline finding (negative result) — the patching HF backend does not run on any real cached model

The v1.0 patching pillar's HF-eager backend has exact unit tests, but **only on
synthetic Llama configs where `head_dim == d_model / n_heads`**. Probing every
real model cached on this host surfaced two independent blockers:

1. **`head_dim ≠ d_model / n_heads`** (Gemma-2-2b, Gemma-3-1b). `HFSiteResolver`
   computes `head_dim = d_model // n_heads`; Gemma-2/3 set an *explicit*
   `config.head_dim = 256` that is independent of `d_model / n_heads`. EAP's
   `_o_proj_pre_hook` then reshapes the `o_proj` input with the wrong head_dim:
   ```
   RuntimeError: shape '[1, 15, 8, 288]' is invalid for input of size 30720
   ```
   (Gemma-2-2b: hidden=2304, n_heads=8 → assumed head_dim 288; true head_dim 256,
   concatenated heads = 8×256 = 2048.) This silently breaks the HF backend on
   every Gemma model the user actually has.

2. **Non-Llama module layout** (GPT-2 / distilgpt2). GPT-2 uses `transformer.h.{L}`
   blocks with a fused-QKV `c_attn` Conv1D and `transformer.wte` embeddings. The
   HF path is hardwired to `model.model.layers` / `model.embed_tokens` /
   `self_attn.{q,k,v,o}_proj` (Llama/Gemma nesting), so it raises
   `AttributeError: 'GPT2LMHeadModel' object has no attribute 'layers'`.

**Impact:** documented as a v1.0.1 patch candidate (accept explicit
`config.head_dim`; the GPT-2 Conv1D layout is a larger, separate effort). Not
fixed in this campaign — validation surfaces findings; fixes go through their own
spec/plan/test cycle. See "Recommendations" at the end.

**Deeper framing (Gemini review):** these are two symptoms of one design fact — the
"HF-eager" patching backend is effectively a **Llama-family backend**. It hard-codes
`model.model.layers` / `self_attn.{q,k,v,o}_proj` / `head_dim = d_model/n_heads`.
It works on Llama/Qwen-shaped models and silently fails elsewhere (Gemma's explicit
head_dim, GPT-2's Conv1D, and likely most non-Llama families). The marketing term
"HuggingFace backend" overstates the actual coverage. R1 (read `config.head_dim`)
is the right first fix but only widens the Llama family; broad HF coverage is a
larger design effort.

**Consequence for Track 1:** the published-circuit recovery runs on the
**TransformerLens backend** (the canonical IOI vehicle, and it works cleanly),
and the **HF-eager backend is cross-validated on Qwen2.5-0.5B** — a real
pretrained model that satisfies `head_dim = d_model/n_heads` and exercises the
GQA path. This gives genuine both-backend coverage, on different tasks.

---

## Track 1 — Patching pillar vs the published IOI circuit

Ground truth: the IOI circuit from Wang et al. 2022 (head classes pinned from a
known reference, not recall — see script). Metric: logit-diff(IO − S).

### 1a. EAP edge attribution (TransformerLens, GPT-2 small) — ✅ recovers the published circuit

Single-pair probe (`scripts/v1_probe_backends.py`), top edges into `logits` and
attention reader slots, ranked by |score|:

| Head | Published role | EAP rank signal |
|---|---|---|
| 9.9 | name mover | strongest edge into logits (−4.17) |
| 10.7 | negative name mover | +2.81 into logits |
| 9.6 | name mover | −1.28 into logits |
| 8.10 | S-inhibition | −0.97 into logits |
| 10.0 | name mover | −0.74 into logits |
| 11.10 | negative name mover | +0.81 into logits |
| 8.6 / 7.3 | S-inhibition | present (q-slot / logits) |
| 5.5 / 6.9 | induction | present (v-slot) |
| 3.0 | duplicate-token | present (v-slot) |

The top-ranked edges are exactly the published name-mover / negative-name-mover /
S-inhibition / induction / duplicate-token heads. (Full multi-prompt IOIDataset
recovery table + overlap@k below.)

### 1a/1b recovery table — 50-prompt IOIDataset, per-head aggregated |score|

`scripts/v1_validation/track1_eap_atp_tl.py` (overlap@k vs the 26-head circuit):

| Method | overlap@10 | @15 | @20 | @26 | core movers 9.9/9.6/10.0 |
|---|---|---|---|---|---|
| EAP    | 10/10 (100%) | 13/15 (87%) | 17/20 (85%) | 17/26 (65%) | all 3 ✓ |
| AtP\*  | 10/10 (100%) | 14/15 (93%) | 17/20 (85%) | 20/26 (77%) | all 3 ✓ |

Per-class recall @26: EAP — neg-mover 2/2, S-inhibition 4/4, induction 4/4,
duplicate 3/3, name-mover 4/11, **previous-token 0/2**. AtP\* — name-mover 9/11,
neg 2/2, S-inhib 4/4, induction 3/4, duplicate 2/3, **previous-token 0/2**.

**Both methods recover the published circuit**: the top-10 heads are 100%
in-circuit for both. The only class neither surfaces is **previous-token (2.2,
4.11)** — expected and faithful: previous-token heads have negligible *direct*
attribution to the IOI logit-diff (they act indirectly, feeding induction heads),
so a gradient-attribution method correctly scores them low. EAP's lower @26 is
because the 26-head dict includes 8 small "backup" name-movers; both methods find
the 3 core movers and all the high-impact classes.

### 1c. ACDC circuit recovery (TL, GPT-2) — ✅ recovers a 100%-in-circuit minimal graph

`track1_acdc_tl.py` (EAP-seeded ordering, last-token KL recovery, τ=0.05, 12 prompts).
ACDC pruned the full graph **32,491 → 10 edges (8 heads)**, final KL=0.715. **All 8
surviving heads are in the published IOI circuit** (100% precision):

| Head | Published role |
|---|---|
| 9.9, 9.6, 10.0, 10.10 | name mover |
| 10.7 | negative name mover |
| 8.6, 8.10 | S-inhibition |
| 5.5 | induction |

The 10-edge circuit reproduces the full model's last-token distribution to within
0.72 nats. This is the strongest single result of the campaign — a forward-only
greedy prune over 32k edges lands on a minimal circuit that is *entirely* known IOI
heads. **Cost: 3,873 s (~64.5 min) for one τ** — confirming ACDC is O(edges)
forwards and impractical for a τ-sweep at GPT-2 scale without the `eap_skip_threshold`
follow-on (TODO). Single τ here; Pareto sweep deferred (see "Still not covered").

### 1d. HF-eager backend on a real GQA model + patch_site faithfulness (Qwen2.5-0.5B)

`scripts/v1_validation/track1_hf_qwen.py`. Qwen2.5-0.5B (24 layers, 14 q-heads,
**2 kv-heads GQA**, head_dim=64=896/14) — the one real model where the HF backend
runs.

- **HF backend runs end-to-end on a real model**: EAP scored 179,749 edges, AtP\*
  scored 1,033 nodes — no crash, GQA path exercised. (This is the capability that
  *fails* on every cached Gemma/GPT-2; see headline finding.)
- **Model does the task**: IOI logit-diff clean **+5.57** vs corrupt **+0.08**.
- **patch_site faithfulness** (`verify_top_k` → real `patch_site` ablation as
  ground truth; HF-path only): over the top-24 AtP\* nodes (12 distinct after
  GQA-dedup), **Spearman(atp, true_patch_effect) = +0.85, sign-agreement = 100%**.
  AtP\* attribution genuinely tracks real ablation effects on a real model.
- **Honest caveat (faithful negative)**: `mlp 0` scored `atp=−0.71` vs
  `true_patch=−5.51` — the known AtP first-order under-scaling of large /
  high-curvature interventions (MLP0 is the extended-embedding layer). Faithful in
  sign and rank, mis-scaled in magnitude — exactly the failure mode `verify_top_k`
  is designed to expose. GQA v-slots within a kv-group correctly share an identical
  effect (e.g. heads 14.0–14.6), confirming the GQA back-map.

---

## Track 2 — Train-time Recorder run (real `lr>0`)

Tiny from-scratch `LlamaConfig` (28.4M params, 4L/4H, eager) trained on TinyStories,
600 steps, batch 16 × seqlen 128, AdamW lr 3e-4. `scripts/v1_validation/track2_train_recorder.py`.

### Works as intended
- **Real training**: loss **10.87 → 3.48** over 600 steps.
- **Dynamic capture confirmed**: report header reads `# circuitry report — dynamic
  (600 steps)` — the first non-static (`lr>0`) run; 352 tags, 2886 rows over training.
- **Eager-only diagnostics emit** (32 tags) — `induction_score` and
  `attention_pattern_entropy`, which were silently zero under SDPA on Gemma (v0.9),
  now emit on the eager model.
- **Diagnostics move meaningfully**: attention-pattern entropy *drops* as heads
  specialize, e.g. `layers.1/head_0` **2.97 → 1.62 nats** (near-uniform → structured).
  A real, interpretable training-dynamics signal.

### §10 wall-clock budget — workload-dependent; met at realistic step cost, not on the cheap synthetic bench

Contract (§10 / README): overhead is a **target** of ≤10% at default `every_n_steps=200`,
full recipe, on a 50M model. Status of enforcement (checked, not assumed):
- **CI does not enforce the numeric budget.** `tests/perf/test_overhead.py` only
  asserts "under 2×" on a tiny model and states the real <10% budget "is in
  `scripts/bench_50m.py`, not enforceable on tiny tests." So §10's "benchmarked in
  CI; regressions block merge" is aspirational, not a numeric gate.
- **README's only published numbers are v0.2.0a0, CPU**, and self-flagged: *"GPU
  re-measurement is on the to-do list."* The numbers below are that missing GPU
  re-measurement.

Measured (RTX 5080, CUDA, full `llm` recipe):

| Workload | step cost | cadence | overhead |
|---|---|---|---|
| `bench_50m` 88M, **script default** | ~13ms | every_n=**25** | **+308%** |
| `bench_50m` 88M, §10 default | ~13ms | every_n=**200** | **+41.7%** |
| 28M Llama, batch 16×128 | ~15ms | every_n=200 | +12.7% |
| 28M Llama, batch 16×128 | ~15ms | every_n=100 | +25% |
| 28M Llama, batch 16×128 | ~15ms | every_n=50 | +49% |
| **28M Llama, batch 16×256** | **~25ms** | **every_n=200** | **+9.6%** ✓ |
| **28M Llama, batch 8×512** | **~26ms** | **every_n=200** | **+9.9%** ✓ |

Findings:
1. Overhead is a **fixed per-emit cost** (~1.07s/emit on the 88M model; ~0.38s on
   28M) scaling linearly with emit frequency. The cost is the **SVD weight
   diagnostics + logit-lens**, **not** the induction probe — ablation showed
   removing `induction_score` changed overhead by <0.1pp (hypothesis falsified).
2. The ≤10% target **is met** once per-step cost reaches ~**25ms** (batch 16×256),
   which any real LLM training exceeds. It is *not* met on the synthetic
   `bench_50m` workload, whose steps (~13ms, batch 4×64, no real attention) are
   ~2× too cheap to amortize the per-emit SVD cost at every_n=200.
3. Two real (non-code) issues, flagged for follow-up (not "violations"):
   (a) `bench_50m`'s **script default `every_n=25` contradicts §10's stated default
   `every_n=200`** — yields a misleading +308% headline; should default to 200.
   (b) The budget is an **unenforced target** — no CI gate on the ≤10% number; a
   GPU benchmark gate would catch regressions §10 claims are blocked today.

## Track 3 — Tier-1 core primitives on a real model (GPT-2 small, eager)

`scripts/v1_validation/track3a_core_primitives.py` — every Tier-1 primitive on
pretrained GPT-2 (124.4M), checking *correct* signals, not just emission.

| Primitive | Result | Correctness check |
|---|---|---|
| `ModelInventory` | 124.4M params, 148 named | resolves params ✓ |
| `effective_rank` / `stable_rank` / `heavy_tail_alpha` | 431.5 / 24.5 / 1.08 on `mlp.c_fc (768,3072)` | 0 < er ≤ min(dims) ✓; α≈1.1 (heavy-tailed, trained) |
| `logit_lens_kl` per layer | 16.11→5.70→…→**0.54** (L11) | **monotone fall toward final** ✓ — residual aligns with prediction |
| `induction_score` | top heads 5.1(.92) 5.5(.91) 6.9(.89) 7.10(.86) 7.2(.78) | all genuine GPT-2 induction heads; 5.5 & 6.9 ∈ IOI subset ✓ |
| `attention_pattern_entropy` | layer-0 heads 0.25–1.85 nats | varied per-head concentration ✓ |
| `token_similarity` | 0.534 (mid-layer) | in [-1,1], positive shared direction ✓ |
| `grad_norm_per_module` / `total_grad_norm` | 148 modules, 41.2 | non-zero after real backward ✓ |
| `update_delta` / `direction_cosine` | ‖Δ‖=0.026, cos=+0.30 (2 SGD steps) | consecutive updates positively correlated ✓ |

Notes:
- **Eager attention** makes `induction_score` + `attention_pattern_entropy`
  available — the diagnostics that were silently zero under SDPA on Gemma (v0.9).
- The IOI-circuit "known induction" set is just {5.5,5.8,5.9,6.9}; the primitive
  also surfaces 5.1/7.2/7.10 (real induction heads outside that subset), so
  top-6-vs-IOI-4 = 2/4 understates it — every top head scores >0.78 prefix-match.
- `logit_lens_kl` on the *very last* `hidden_states` entry reads 2.65 (not ~0)
  because HF GPT-2's final `hidden_states` is already post-`ln_f`; re-applying
  `ln_f` double-normalizes. The lens-convergence signal is the layer-11 minimum
  (0.54). Not a circuitry bug — a known HF hidden-states indexing subtlety.

### 3b. SAE workflow (Gemma-2-2b + Gemma Scope) — short-prompt caveat resolved

`scripts/v1_validation/track3b_sae.py`. `gemma-scope-2b-pt-res` /
`layer_8/width_16k/average_l0_71` on a 112-token paragraph (vs v0.9's 4 tokens).

| metric | v0.9 (4 tok) | realistic, all tokens | realistic, **ex-BOS** | SAE design |
|---|---|---|---|---|
| l0 | 1293 | 125 | **71.3** | ~71 |
| recon_mse | 3130 | 140 | **0.66** | — |
| ce_recovered_proxy | −19.1 | −10.3 | **+0.887** | →1 |
| frac_alive | 0.384 | 0.472 | 0.155 | — |

The v0.9 negative `ce_recovered_proxy` and l0=1293 were an out-of-distribution
short-prompt artifact, compounded by Gemma-2's huge-norm BOS/attention-sink token
(**norm 1312 vs 114** for normal tokens) which dominates MSE and l0. On a realistic
prompt with the BOS sink excluded, the SAE hits its **published design point exactly
(l0 = 71.3)** and recovers **88.7%** of activation variance. SAE workflow validated.

## Track 4 — CLI report + scan

`circuitry report` / `circuitry scan` on the Track 2 run dir. `circuitry compare`
is unshipped (field-feedback #6) — documented as a gap, not tested.

---

## Track 5 — API coverage completion (after Gemini Pro gap review)

A Gemini Pro review of the obs doc + scripts + `design.md` flagged that Tracks 1–4
exercised a narrow path. This track closes the cheap, high-value coverage gaps it
identified. All passed:

**Remaining Tier-1 primitives** (`track5_primitives_full.py`, GPT-2 eager):
`condition_number`=46.7, `singular_values` (512, descending), `attention_head_rank`
(12 heads, rank 58–63.6/64), `dead_fraction`=0.887 (= the known GPT-2 MLP
neuron-sparsity), `norm_stats`, `kurtosis`=212 (heavy-tailed GELU), `participation_ratio`,
`gate_stats` (frac_active 0.999), `signal_propagation_depth`=12/12 (grad reaches all
layers), `esd` (51-bin spectral density), `rank_trajectory` (per-param rank across
snapshots). Every Tier-1 primitive is now exercised on a real model.

**Patching algorithm variants** (`track5_patching_variants.py`, Qwen2.5-0.5B):
- **EAP-IG** (`ig_steps=4`): runs; top-10 heads overlap vanilla 9/10.
- **AtP\* GradDrop**: runs (1033 nodes).
- **AtP\* neuron-level**: 116,736 `mlp_neuron` nodes scored (they rank below heads,
  as expected — individual neurons have small attribution).
- **AtP\* QK-fix** is the real attention-pattern recomputation on the HF backend
  (Track 1d + here); on TL it documents-falls-back to vanilla q/k.
- **PatchRunner denoise & noise**: on head 14.0 (clean logit-diff +5.38, corrupt
  +0.06): denoise (clean→corrupt) lifts to **+1.08**, noise (corrupt→clean) drops to
  **+4.37** — both directions move the metric correctly. Validates `patch_site` +
  both `PatchRunner` modes.

**Recipes + writers + scan strict** (`track5_recipes_writers.py`):
- **`vision` recipe** (TinyCNN, real lr>0): 17 diagnostic tags + report ✓.
- **`two_tower` recipe** (TwoTower): 14 tags + report ✓ (gracefully skips the
  weightless ReLU/Sequential wrappers).
- **`tensorboard` writer**: event file written ✓ (jsonl/null already used).
- **`scan_run(strict=True)`**: passes on clean Llama (all HookPoints resolved) ✓.

### Still not covered (honest scope)
- **ACDC τ sweep / Pareto curve.** A single τ=0.05 gave a strong result (8/8 heads
  in-circuit), but at **64.5 min/τ** a multi-τ Pareto curve (ACDC's
  performance-vs-sparsity trade-off) is impractical here without the
  `eap_skip_threshold` follow-on (TODO). Single τ run; sweep deferred.
- **Faithfulness sample size.** The Qwen `verify_top_k` correlation is over n=12
  distinct high-attribution nodes (top-k by design — the nodes that matter). Sign
  agreement (100%) is the robust statistic; Spearman 0.85 has a wide CI at n=12. A
  full all-node faithfulness sweep is future work (and on this sparse task a random
  node sample would be dominated by near-zero-effect nodes, inflating correlation).

## Recommendations (end-of-campaign decision points)

Ordered by impact. None applied in this campaign — validation surfaces findings;
fixes go through their own spec/plan/test cycle.

**R1 — v1.0.1 code fix: HF backend `head_dim` (HIGH).** `HFSiteResolver.__init__`
(`head_dim = d_model // n_heads`) and `EAPRunner.__init__` (eap.py:106,
`resolver.d_model // resolver.n_heads`) assume `head_dim == d_model/n_heads`. This
is false for Gemma-2/3 (explicit `config.head_dim=256`), so the entire patching
pillar's HF backend silently fails on every Gemma model. Fix: read
`getattr(config, "head_dim", d_model // n_heads)` and thread it through
EAP/AtP*/ACDC per-head reshapes. Toy tests (where the identity holds) stay green;
add a Gemma-2 regression. Without this, the HF backend works only on a subset of
real models (Qwen yes, Gemma no, GPT-2 no).

**R2 — bench/doc fix: §10 budget (MEDIUM).** `scripts/bench_50m.py` defaults to
`every_n_steps=25`, contradicting §10's stated default of 200, producing a
misleading +308% headline. Set the script default to 200 and add `--batch
--seqlen` so the benchmark runs at a realistic step cost. GPU re-measurement (which
the README flags as a TODO) shows the ≤10% target **holds at ≥~25ms/step** but not
on the ~13ms synthetic stub. Update the README's stale v0.2.0a0 CPU numbers with
GPU figures + the step-cost caveat.

**R3 — CI gap (MEDIUM).** The ≤10% budget is an unenforced *target*:
`tests/perf/test_overhead.py` only asserts "<2×" on a tiny model. §10's
"benchmarked in CI; regressions block merge" is not literally true. Consider a
GPU benchmark gate (or downgrade the §10 wording to "target, measured manually").

**R4 — CLI completeness (LOW, already roadmapped).** `circuitry scan` is a stub
(needs the planned `--model-factory dotted.path:fn`); `circuitry compare` is
unshipped (field-feedback #6). Both are known. The programmatic `scan_run()` works.

**R5 — documented method limitations (INFO, not bugs).** (a) AtP* first-order
attribution under-scales large/high-curvature interventions (e.g. MLP-0 as
extended embedding: atp −0.71 vs true −5.51); faithful in sign/rank, and
`verify_top_k` is the intended ground-truth check. (b) On GPT-2, the HF backend is
unsupported entirely (`transformer.h`/Conv1D layout) — TL is the right backend
there. (c) ACDC is O(edges) forwards (~32k on GPT-2 → measured 64.5 min/τ); the
`eap_skip_threshold` follow-on (TODO) would cut this. (d) GQA per-query-head k/v is
a documented follow-on; v-slots correctly share within a kv-group today.
