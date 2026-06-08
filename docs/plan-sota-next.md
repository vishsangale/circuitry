# circuitry — next SOTA milestone plan

**Created:** 2026-06-08  
**Based on:** five-agent parallel SOTA survey covering mechanistic interpretability 2024–2026  
**Current version:** v1.28.0  
**Status:** planning — not yet in flight

---

## Summary of survey findings

Five research agents covered: (1) SAE architectures & evaluation, (2) circuit discovery & causal methods, (3) representation analysis & probing, (4) training dynamics & monitoring, (5) vision/multimodal & recsys. Together they identified ~50 concrete gaps. Below is the distilled plan: four sequential milestones ordered easiest-first to maintain momentum, with a fifth deferred batch for hard items.

---

## Milestone v1.29 — Probing & Representation Geometry

**Theme:** Pure `core/` additions — new named primitives for a cluster of related "what does this layer represent?" workflows that the community uses constantly but that the library lacks entirely. All are pure functions (no hooks, no I/O), ≤100 LOC each.

### Items

#### `core/probe.py` additions

| Primitive | Method | Paper |
|-----------|--------|-------|
| `mdl_probe(acts, labels, *, n_chunks=8) → MDLResult` | Online-coding MDL: sequential cross-entropy over 8 data chunks; returns `(code_length, data_entropy, mdl_ratio)`. MDL ratio < 1 = genuine encoding. | Voita & Titov 2020, arXiv:2003.12298 |
| `mass_mean_probe(acts, labels) → MassMeanProbe` | Concept direction = `μ₁ − μ₀` normalised (identical computation to `leace_erase` binary direction); exposes `predict(acts)`, `accuracy(acts, labels)`, `causal_flip_rate` field. Causally verified truth probe that beats logistic regression under intervention. | Marks & Tegmark, COLM 2024, arXiv:2310.06824 |
| `verify_linear_representation(probe: LinearProbe, steer_vec: Tensor, acts: Tensor) → float` | Returns cosine similarity between `probe.direction()` and `steer_vec`; measures whether the probe and the CAA steering vector agree (validates the linear representation hypothesis for a concept). ~20 lines. | Park et al. arXiv:2311.03658 |

#### `core/steer.py` additions

| Primitive | Method | Paper |
|-----------|--------|-------|
| `repe_direction(stimuli_acts: Tensor) → Tensor` | First PC of the difference-of-activations matrix across a stimuli set (PCA direction, not mean-difference). Top-down concept extraction; orthogonality guarantees the mean-difference lacks. | Zou et al., arXiv:2310.01405 |
| `directional_ablation(acts: Tensor, direction: Tensor) → Tensor` | Orthogonal projection: `acts - (acts @ d_hat) * d_hat`. Removes a direction from activations. Equivalent to `EraseProjection.apply` but exposed as a named function with the refusal/safety framing. | Arditi et al., NeurIPS 2024, arXiv:2406.11717 |

#### `patching/steer.py` addition

| Item | What |
|------|------|
| `apply_ablation(model, site, direction, *, resolver=None)` context manager | Parallel to `apply_steer` — registers a forward hook that removes `direction` component from site output (`x = x - (x @ d) * d`); hook removed in `finally`. | Arditi et al. NeurIPS 2024 |

#### `core/activation.py` additions

| Primitive | Method | Paper |
|-----------|--------|-------|
| `local_intrinsic_dim(acts: Tensor, *, max_samples: int = 2048) → float` | Two-NN estimator: per-point ratio of 2nd-nearest to 1st-nearest neighbor distance; `ID = 1 / mean(log μ)`. Measures manifold dimensionality (nonlinear; complements `effective_rank`). ~20 lines. | Levina & Bickel 2004; applied to LLMs in arXiv:2402.18048, arXiv:2601.22722 |
| `kernel_alignment(acts_a: Tensor, acts_b: Tensor, *, method: str = 'cka', max_samples: int = 256) → float` | Cross-model CKA (reuses existing `_drift_linear_cka` kernel) + MNN overlap. Measures how similar two models' representations are on the same data. Enables Platonic convergence checks and fine-tuning divergence audits. | Huh et al., ICML 2024, arXiv:2405.07987 |
| `embedding_uniformity(E: Tensor, *, n_samples: int = 2048, seed: int = 0) → float` | Mean pairwise cosine similarity over a random sample of embedding rows. Quantifies embedding collapse in recsys (high uniformity = collapsed). | Guo et al., ICML 2024, arXiv:2310.04400 |

#### `sae/metrics.py` addition

| Primitive | Method | Paper |
|-----------|--------|-------|
| `superposition_index(feature_acts: Tensor) → float` | `exp(H)` where `H` = Shannon entropy of activation magnitudes across features. "Effective number of features" — if >> n_neurons, the layer operates under superposition. ~5 lines. | arXiv:2512.13568 |

### Exports
All new primitives exported at top-level `circuitry.*` and via their module. `MDLResult`, `MassMeanProbe` added to `__all__`.

### Tests
~40 new tests across `tests/core/test_probe.py`, `tests/core/test_steer.py`, `tests/core/test_activation.py`, `tests/patching/test_steer.py`, `tests/sae/test_metrics.py`.

---

## Milestone v1.30 — Training Diagnostics

**Theme:** New live-training and post-hoc weight-space diagnostics. All pure `core/` additions except `embedding_uniformity` (already added in v1.29) and `cold_start_score` (wired into `recipes/recsys.py`).

### Items

#### `core/weight.py` additions

| Primitive | Method | Paper |
|-----------|--------|-------|
| `update_weight_ratio(W_prev: Tensor, W_curr: Tensor) → float` | `‖W_curr − W_prev‖_F / ‖W_prev‖_F`. Scale-invariant μP diagnostic: should be constant across widths/layers when LR is tuned correctly. Complements existing `update_delta` (absolute). ~5 lines. | arXiv:2510.19093, arXiv:2601.01306 |
| `finetuning_delta_svd(W_base: Tensor, W_ft: Tensor) → FinetuningDeltaResult` | SVD of `W_ft − W_base`: returns `(sv_scale_factor, left_rotation_similarity, right_rotation_similarity)`. Diagnoses whether fine-tuning changed geometry (rotation) or just scaled it. | arXiv:2509.17866 |

#### `core/spectral.py` addition

| Primitive | Method | Paper |
|-----------|--------|-------|
| `spectral_edge_gap(W_prev: Tensor, W_curr: Tensor, *, k: int = 5) → float` | SVD of the weight update matrix; returns `s[k-1] / s[k]` (gap between top-k and bulk). A growing gap fingerprints circuit formation during grokking. Complements `grokking_step` (which only detects timing). | arXiv:2604.06256 |

#### `core/activation.py` additions

| Primitive | Method | Paper |
|-----------|--------|-------|
| `neural_collapse_score(acts: Tensor, labels: Tensor) → float` | `NC1 = Tr(Σ_W @ pinv(Σ_B)) / n_classes` where `Σ_W` = within-class covariance, `Σ_B` = between-class covariance. → 0 in well-trained networks; rises during continual learning / fine-tuning plasticity loss. ~40 lines. | arXiv:2404.02719, arXiv:2604.00230 |
| `spectral_collapse_rank(acts: Tensor) → float` | `effective_rank` of the activation matrix (the activation-space analogue of the weight-space `effective_rank`). Downward trend = plasticity loss / representation collapse signal. ~5 lines (reuses SVD path). | arXiv:2509.22335 |

#### `core/dynamics.py` addition

| Primitive | Method | Paper |
|-----------|--------|-------|
| `emergence_score(series: list[tuple[int, float]], *, window: int = 5) → float` | Smoothed second log-derivative of a scalar series: `d²M / d(log step)²`. A spike with no corresponding spike in component sub-metrics is the emergence signature. Upgrades `phase_transition_steps` with semantic "emergent vs. smooth" discrimination. | arXiv:2508.04401 |

#### `core/attention.py` addition

| Primitive | Method | Paper |
|-----------|--------|-------|
| `attention_rollout(attn_weights: list[Tensor], *, grads: list[Tensor] \| None = None) → Tensor` | Recursive attention rollout for ViTs. With `grads=None`: uniform rollout (Abnar & Zuidema 2020). With `grads`: gradient-weighted GMAR (per-head importance weighting). Returns patch-level saliency map `(B, T)`. | Abnar & Zuidema ACL 2020; GMAR arXiv:2504.19414 |

#### `recipes/recsys.py` / `recipes/two_tower.py` wiring

| Item | What |
|------|------|
| Wire `embedding_uniformity` into two_tower recipe as custom diagnostic | Tags `embedding/uniformity/<tower>` at each emit step. Flags when `> 0.9` (near-total collapse). |
| `cold_start_score(E: Tensor, interaction_counts: Tensor, *, percentile: int = 10) → float` | Mean nearest-neighbor distance ratio between bottom-decile and top-decile items by interaction count. Flags geometrically isolated cold-start items. ~80 lines. |

#### Recorder wiring

- `update_weight_ratio` added to trajectory diagnostics; emits `weight/update_weight_ratio/<module>` alongside `update_delta`.
- `neural_collapse_score` added as opt-in activation diagnostic (requires label-annotated `probe_batch`).
- `spectral_edge_gap` added to weight diagnostics (requires ≥2 emit steps like `update_delta`).
- `spectral_collapse_rank` added to activation diagnostics (per-module, emits `activation/spectral_collapse_rank/<module>`).

### Tests
~35 new tests.

---

## Milestone v1.31 — SAE Quality & Steering

**Theme:** Closes the most important gaps in the SAE pillar: true faithfulness metric, gradient-weighted attribution, architecture coverage, and SAE-based steering.

### Items

#### `sae/metrics.py` / `sae/grad.py`

| Item | Method | Paper |
|------|--------|-------|
| `sae_downstream_loss(sae, model, tokens: Tensor, *, site: Site, resolver=None) → dict[str, float]` | KL divergence between model output with and without SAE substitution at `site`. The gold-standard SAE faithfulness metric (distinct from `ce_recovered_proxy` which is variance-based). Returns `{'kl_divergence', 'ce_delta', 'l0'}`. | arXiv:2406.04093, SAEBench arXiv:2503.09532 |
| `sae_influence_scores(sae, x: Tensor, loss_fn: Callable) → Tensor` | GradSAE: `∂loss/∂f_i · |f_i|` per feature. Weights activation magnitude by output-side gradient. Filters "active but irrelevant" features (e.g. punctuation). ~30 lines on top of existing `encode_features`. | arXiv:2505.08080 |
| Register `"p_anneal"` and `"hierarchical_topk"` in `SUPPORTED_SAE_ARCHITECTURES` | Both use standard ReLU/TopK forward pass at inference; annealing is training-time only. Add to `assert_supported_sae` allowlist. | SAEBench arXiv:2503.09532; arXiv:2505.24473 |
| Add reliability caveat + `UNRELIABLE_METRICS = {"tpp", "scr"}` guard in `sae/metrics.py` | Emit `UserWarning` when user requests TPP or SCR metrics (found to fail reliability across seeds/architectures in arXiv:2605.18229). | arXiv:2605.18229 |

#### `sae/steer.py` (new file)

| Item | Method | Paper |
|------|--------|-------|
| `fgaa_steering_vector(sae, positive_acts: Tensor, negative_acts: Tensor, *, n_features: int = 10) → Tensor` | Feature-Guided Activation Addition: encode both polarities, select discriminative features by `|f_pos − f_neg|`, return weighted sum of their decoder columns. Outperforms raw CAA and naive decoder steering on AxBench. | arXiv:2501.09929 |

#### `core/erase.py` addition

| Item | Method | Paper |
|------|--------|-------|
| `rlace_erase(acts: Tensor, labels: Tensor, *, rank: int = 1) → EraseProjection` | Rank-k adversarial concept erasure (RLACE). Finds the rank-k orthogonal projection that most aggressively removes concept from a linear classifier. `rank=1` recovers the LEACE direction. Uses `torch.linalg.eigh` on the between-class scatter. | Ravfogel et al., ICML 2022, arXiv:2201.12091 |

### Tests
~30 new tests.

---

## Milestone v1.32 — Attribution Quality

**Theme:** Drop-in improvements to the existing circuit-discovery stack. RelP replaces EAP as the default attribution method; MIB gains three new tasks; a certified-stability wrapper adds robustness guarantees.

### Items

#### `patching/relp.py` (new file) or extend `patching/runner.py`

| Item | Method | Paper |
|------|--------|-------|
| `ReLPRunner(model, resolver)` | Replaces EAP gradient terms with LRP propagation coefficients. Same 2-forward + 1-backward cost as EAP; Pearson correlation to true patching = 0.956 vs. 0.006 for EAP on GPT-2 IOI. API mirrors `EAPRunner`: `.run(clean, corrupted, metric) → EAPResult` (same result type for drop-in compatibility). | arXiv:2508.21258; github.com/FarnoushRJ/RelP |

#### `benchmarks/mib.py` additions

| Item | Method | Paper |
|------|--------|-------|
| `load_ravel(n, *, seed, entity_type, attribute)` | RAVEL entity-attribute disentanglement task: prompts that vary an entity while holding an attribute constant (or vice versa). Returns `MIBTask`. Required for MIB causal-variable-localization track (pairs with `DASRunner`). | arXiv:2402.17700; MIB arXiv:2504.13151 |
| `load_arithmetic(n, *, seed, op)` | Arithmetic circuit task (addition / modular addition). Returns `MIBTask`. Used in MIB circuit-localization track. | MIB arXiv:2504.13151 |
| `load_mcqa(n, *, seed)` | Multiple-choice Q&A task. Returns `MIBTask`. | MIB arXiv:2504.13151 |
| `mib_circuit_f1(circuit_edges: set, ground_truth_edges: set) → float` | Edge-set F1 for MIB circuit localization track. Enables leaderboard-compatible evaluation. | MIB arXiv:2504.13151 |
| `mib_iia_score(das_result: DASResult, task: MIBTask, *, threshold: float = 0.5) → float` | IIA-at-threshold metric for MIB causal variable localization. | MIB arXiv:2504.13151 |

#### `patching/certified.py` (new file)

| Item | Method | Paper |
|------|--------|-------|
| `CertifiedCircuitRunner(base_runner, *, n_subsamples: int = 20, confidence: float = 0.95)` | Wraps any runner (`EAPRunner`, `ACDCRunner`, `EdgePruningRunner`, `ReLPRunner`) with randomized data subsampling. For each edge inclusion decision, votes across `n_subsamples` subsets; abstains on edges below the confidence threshold. Returns `CertifiedCircuitResult` with `.certified_edges` (stable under perturbation) and `.abstained_edges`. | arXiv:2602.22968 |

### Tests
~40 new tests.

---

## Deferred (v1.33+) — hard items

These are genuine SOTA but architecturally heavy. Deferred until a dedicated sprint.

| Item | Why deferred | Paper |
|------|-------------|-------|
| Critical sharpness λ_c | Requires Hessian-vector products + gradient access at hook time; new Recorder infrastructure | arXiv:2601.16979 |
| Gradient subspace saturation | Requires buffering per-layer gradient matrices (not just norms) across steps | arXiv:2508.07370 |
| Inference-Time Intervention (ITI) | Per-head probe training + calibrated steering; ~150 LOC integration of existing primitives | arXiv:2306.03341 |
| HyperDAS | Hypernetwork on top of DAS; requires training loop + saved weights | arXiv:2503.10894 |
| CLT Attribution Graphs | Requires pre-trained cross-layer transcoders or transcoder training | transformer-circuits.pub/2025/attribution-graphs; arXiv:2603.21014 |
| DAAM for diffusion | Requires new DiffusionRecorder mode; iterative denoising loop | arXiv:2210.04885 |
| CD-T Contextual Decomposition | Recursive contribution propagation through transformer modules | arXiv:2407.00886; ICLR 2025 |
| MoE pathway complexity | Requires per-sample expert routing access; needs real MoE model to test | arXiv:2506.21551 |
| SAGE automated feature labeling | LLM API dependency; not a pure `core/` primitive | arXiv:2511.20820 |

---

## Implementation order rationale

v1.29 before v1.30 because pure primitive additions (no Recorder wiring) are the fastest to test and ship. v1.30 before v1.31 because the training diagnostics build on the v1.29 activation primitives (`spectral_collapse_rank` reuses `spectral_edge_gap`'s SVD; `neural_collapse_score` reuses activation captures). v1.31 before v1.32 because the attribution improvements in v1.32 benefit from the better SAE scoring in v1.31.

---

## What each milestone adds to key use cases

| Use case | v1.29 gains | v1.30 gains | v1.31 gains | v1.32 gains |
|----------|-------------|-------------|-------------|-------------|
| Alignment / safety research | `directional_ablation`, `repe_direction`, `mass_mean_probe` | — | `rlace_erase` rank-k, `fgaa_steering_vector` | — |
| SAE interpretability | `superposition_index` | — | `sae_downstream_loss`, `sae_influence_scores`, p_anneal/HierTopK arch | — |
| Circuit discovery | `verify_linear_representation` | — | — | `ReLPRunner` (better EAP), `CertifiedCircuitRunner` |
| Training monitoring | `kernel_alignment`, `local_intrinsic_dim` | `update_weight_ratio`, `neural_collapse_score`, `spectral_edge_gap` | — | — |
| Recsys | `embedding_uniformity` | `cold_start_score` (wired in recipe) | — | — |
| Vision / ViT | — | `attention_rollout` (GMAR) | — | — |
| Grokking analysis | — | `spectral_edge_gap`, `emergence_score` | — | — |
| Fine-tuning auditing | — | `finetuning_delta_svd` | — | — |
| MIB benchmark compat | — | — | — | 3 new tasks + F1/IIA metrics |
