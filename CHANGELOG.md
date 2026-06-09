# Changelog

All notable changes to this project will be documented in this file. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.34.0] — 2026-06-09

**CLT Attribution Graphs.**

### Added
- **`patching/clt.py`** (new module) — `CLTNode`, `CLTEdge`, `CLTGraphResult`,
  `CLTGraphRunner(model, layer_transcoders)`: builds a feature-level attribution
  graph using cross-layer transcoders. For each consecutive layer pair `(l, l+1)`,
  scores every feature-to-feature edge using the EAP approximation
  `score(fi→fj) = Σ delta_fi · grad_fj`. Clean forward pass splices each
  transcoder losslessly (`decode(encode(x)) + sg(output − decode(encode(x)))`) so
  PyTorch autograd gives `f.grad = ∂metric/∂f` after `backward()`.
  `CLTGraphResult` exposes `.top_k()`, `.threshold()`, `.to_markdown()`,
  `.node_scores` (per-feature importance), `.layer_order` (arXiv:2603.21014).
- 12 new tests in `tests/patching/test_clt.py`.

## [1.33.0] — 2026-06-09

**Inference-Time Diagnostics & Deeper Attribution.**

### Added
- **`patching/iti.py`** (new module) — `ITIConfig`, `fit_iti(head_acts, labels, *, coeff=15.0)`,
  `apply_iti(model, config, *, attn_modules, resolver)`: Inference-Time Intervention.
  Trains per-(layer, head) mass-mean probes on labelled activation data; at inference adds
  `coeff × direction` to each head's output slice. Direct integration of existing
  `mass_mean_probe` primitive (arXiv:2306.03341, Li et al.).
- **`patching/cd.py`** (new module) — `CDResult`, `cd_token_contributions(attn_weights, *,
  head_agg, add_residual)`: CD-T contextual decomposition. Propagates per-source-token
  contribution scores through the attention stack via iterative left-multiplication of the
  aggregated attention matrix; optional 50/50 residual blend models skip connections.
  Returns `contributions[q, s]` = fraction of position *q* attributable to source *s*
  (arXiv:2407.00886, Jain et al. ICLR 2025).
- **`core/weight.py`** additions:
  - `critical_sharpness(model, loss_fn, *, n_iters=20, tol=1e-4) → float`: largest Hessian
    eigenvalue λ_max via power iteration using double backpropagation (HVP). High sharpness
    correlates with poor generalisation (arXiv:2601.16979, Damian et al.).
  - `gradient_subspace_saturation(grad_history, *, k=10) → float`: fraction of the current
    gradient lying in the top-*k* principal directions of historical gradients. High
    saturation = gradient has settled into a low-rank subspace (plasticity loss signal)
    (arXiv:2508.07370, Chen et al.).
- 34 new tests across `tests/patching/test_iti.py`, `tests/patching/test_cd.py`,
  `tests/core/test_critical_sharpness.py`.

## [1.32.0] — 2026-06-09

**Attribution Quality.**

### Added
- **`patching/relp.py`** (new module) — `ReLPRunner(model, resolver, *, eps=1e-6)`:
  RelP attribution — replaces EAP's gradient term with an LRP-epsilon residual-stream
  coefficient `lrp_coeff_w = act_clean_w / (|Σ_w act_clean_w| + eps)`. Same 2-forward + 1-backward
  cost as EAP; Pearson correlation to ground-truth patching = 0.956 vs 0.006 for EAP on
  GPT-2 IOI. Returns `EAPResult` for drop-in compatibility (arXiv:2508.21258).
- **`patching/certified.py`** (new module) — `CertifiedCircuitRunner(base_runner, *,
  n_subsamples=20, confidence=0.95, subsample_frac=0.5)`: wraps any attribution runner
  with randomised batch subsampling — an edge is "certified" (stable) if it appears in the
  top-K of ≥ ``confidence × n_subsamples`` subsets; otherwise "abstained".
  Returns `CertifiedCircuitResult` with `.certified_edges`, `.abstained_edges`,
  `.vote_counts`, `.certified_set()`, `.n_certified()`, `.n_abstained()` (arXiv:2602.22968).
- **`benchmarks/mib.py`** — three new synthetic task loaders:
  `load_ravel(n, *, entity_type, attribute)` (RAVEL entity-attribute disentanglement;
  arXiv:2402.17700), `load_arithmetic(n, *, op, modulus)` (addition / modular addition
  circuits), `load_mcqa(n, *, n_choices)` (multiple-choice Q&A). Plus two evaluation
  helpers: `mib_circuit_f1(circuit_edges, gt_edges) → float` (edge-set F1 for the MIB
  localisation leaderboard) and `mib_iia_score(das_result, *, threshold=0.5) → float`
  (IIA-at-threshold for causal variable localisation). All MIB arXiv:2504.13151.
- `ReLPRunner`, `CertifiedCircuitResult`, `CertifiedCircuitRunner` exported at `circuitry.*`;
  version bumped to `1.32.0`.
- Tests: ~40 new tests across test_relp, test_certified, test_mib_v132, test_public_api.

---

## [1.31.0] — 2026-06-09

**SAE Quality & Steering.**

### Added
- **`core/erase.py`** — `rlace_erase(acts, labels, *, rank=1) → EraseProjection`: rank-k
  adversarial concept erasure (Ravfogel et al. ICML 2022, arXiv:2201.12091). Finds `P = I − U Uᵀ`
  where U spans the top-`rank` eigenvectors of the between-class scatter `B = M_c^T M_c`; for
  `rank=1` recovers the LEACE direction.
- **`sae/metrics.py`** — `UNRELIABLE_METRICS: frozenset` containing `{"tpp", "scr"}` — a
  frozen guard for metrics with high-variance estimates on standard SAE benchmarks.
  `warn_if_unreliable(metric_name)`: emits a `UserWarning` when `metric_name ∈ UNRELIABLE_METRICS`.
  `sae_downstream_loss(sae, model, tokens, *, site, resolver=None) → dict`: KL-faithfulness
  metric — runs the model clean then with a SAE hook at `site`, returns
  `{"kl_divergence", "ce_delta", "l0"}`.
- **`sae/grad.py`** — `sae_influence_scores(sae, x, loss_fn) → Tensor`: GradSAE per-feature
  influence `|∂loss/∂f_i| · |f_i|`, mean over batch/positions (arXiv:2505.08080). `p_anneal`
  and `hierarchical_topk` added to `SUPPORTED_SAE_ARCHITECTURES`.
- **`sae/steer.py`** (new module) — `fgaa_steering_vector(sae, positive_acts, negative_acts, *,
  n_features=10) → Tensor`: Feature-Guided Activation Addition — selects top-`n_features`
  discriminative SAE features by `|mean_pos − mean_neg|` and returns a weighted sum of their
  decoder columns as a `(d_model,)` CPU float32 steering vector (arXiv:2501.09929).
- All 4 new top-level names (`rlace_erase`, `sae_downstream_loss`, `sae_influence_scores`,
  `fgaa_steering_vector`) exported at `circuitry.*`; version bumped to `1.31.0`.
- Tests: ~35 new tests across test_erase (rlace_erase), test_metrics (UNRELIABLE_METRICS +
  sae_downstream_loss), test_grad (sae_influence_scores), test_steer (fgaa_steering_vector),
  test_sae_architectures (p_anneal + hierarchical_topk), test_public_api.

---

## [1.30.0] — 2026-06-09

**Training Diagnostics.**

### Added
- **`core/weight.py`** — `update_weight_ratio(W_prev, W_curr) → float`: Frobenius
  relative update `‖ΔW‖_F / ‖W_prev‖_F`; the μP scaling diagnostic. `FinetuningDeltaResult`
  / `finetuning_delta_svd(W_base, W_ft) → FinetuningDeltaResult`: SVD of `W_ft − W_base`
  returning scale factor and left/right rotation similarity; distinguishes geometry-changing
  from geometry-preserving fine-tuning (arXiv:2509.17866).
- **`core/spectral.py`** — `spectral_edge_gap(W_prev, W_curr, *, k=5) → float`: ratio
  `s[k-1] / s[k]` of the weight-update singular values; a growing gap fingerprints
  circuit formation during grokking (arXiv:2604.06256).
- **`core/activation.py`** — `neural_collapse_score(acts, labels) → float`: NC1 metric
  `Tr(Σ_W · Σ_B⁺) / C`; → 0 in terminal training, rising during plasticity loss /
  continual learning (Papyan et al. 2020; arXiv:2404.02719). `spectral_collapse_rank(acts)
  → float`: effective rank of the activation matrix — a downward trend signals
  representation collapse (arXiv:2509.22335).
- **`core/dynamics.py`** — `emergence_score(series, *, window=5) → float`: maximum
  smoothed second log-derivative `d²M / d(log step)²`; a spike discriminates abrupt
  emergent capabilities from smooth learning (arXiv:2508.04401).
- **`core/attention.py`** — `attention_rollout(attn_weights, *, grads=None) → Tensor`:
  recursive attention rollout for ViTs returning `(B, T)` patch saliency; uniform rollout
  (Abnar & Zuidema ACL 2020) with optional gradient-weighted GMAR variant
  (arXiv:2504.19414).
- All 7 new names exported at `circuitry.*`; version bumped to `1.30.0`.
- Tests: ~35 new tests across test_weight, test_spectral, test_activation, test_dynamics,
  test_attention.

---

## [1.29.0] — 2026-06-09

**Probing & Representation Geometry.**

### Added
- **`core/probe.py`** — `MDLResult` / `mdl_probe`: MDL probing (Voita & Titov
  2020, arXiv:2003.12298) — computes online-coding code length across n_chunks
  data splits; `mdl_ratio < 1` indicates genuine encoding. `MassMeanProbe` /
  `mass_mean_probe`: binary mass-mean probe (Marks & Tegmark COLM 2024,
  arXiv:2310.06824) — direction = normalised (μ₁ − μ₀), with threshold at the
  midpoint; `.predict()` and `.accuracy()` methods. `verify_linear_representation`:
  cosine similarity between a probe direction and a steering vector (Park et al.
  arXiv:2311.03658), with graceful dimension-mismatch handling via truncation.
- **`core/steer.py`** — `repe_direction`: first-PC concept direction from
  activation differences (Zou et al. 2023, arXiv:2310.01405). `directional_ablation`:
  orthogonal projection removing a concept direction from activations (Arditi et al.
  NeurIPS 2024, arXiv:2406.11717).
- **`patching/steer.py`** — `apply_ablation`: context manager that registers a
  forward hook removing a concept direction at a site; mirrors `apply_steer`
  structure (eval-mode, try/finally cleanup).
- **`core/activation.py`** — `local_intrinsic_dim`: Two-NN manifold dimensionality
  estimator (Levina & Bickel 2004). `kernel_alignment`: CKA or MNN cross-model
  alignment score (Huh et al. ICML 2024, arXiv:2405.07987). `embedding_uniformity`:
  mean off-diagonal cosine similarity for collapse detection (Guo et al. ICML 2024,
  arXiv:2310.04400).
- **`sae/metrics.py`** — `superposition_index`: effective feature count via SAE
  activation entropy — `exp(H(|feature_acts|))`; `>> n_neurons` signals
  superposition (arXiv:2512.13568).
- All 12 new names exported at `circuitry.*`; version bumped to `1.29.0`.
- Tests: 18 probe + 14 steer + 15 activation geometry + 7 SAE metrics = 54 new tests.

---

## [1.28.0] — 2026-06-08

**Causal alignment — DAS and Causal Scrubbing.**

### Added
- **`patching/das.py`** — `DASRunner` and `DASResult`. Learns an orthogonal
  rotation R such that the first `subspace_dim` columns of R·h align with a
  specified causal variable via interchange-intervention training
  (Geiger et al. NeurIPS 2023, arxiv:2303.02536).
  `DASRunner(model).run(base_inputs, source_inputs, labels, *, module,
  subspace_dim, n_steps, lr, loss_fn)` runs Adam on R with Stiefel-manifold
  SVD retraction after each step. The interchange mixes the R-rotated
  subspaces of base and source activations, injects the result at the target
  module via a forward hook, and minimises CE loss against the target labels.
  `DASResult.rotation` — (d, d) orthogonal matrix; `DASResult.iia_score` —
  interchange-intervention accuracy; `DASResult.subspace_directions()` —
  first `subspace_dim` rows of R (the causal directions).
- **`patching/scrubbing.py`** — `CausalScrubRunner`, `CircuitHypothesis`, and
  `CausalScrubResult` (Conmy et al. / Redwood Research 2022).
  `CausalScrubRunner(model).run(clean_inputs, corrupted_inputs, metric,
  hypothesis, *, compute_per_module)` measures faithfulness of a circuit
  hypothesis: circuit modules keep clean activations; non-circuit modules are
  replaced with pre-captured corrupted activations. Faithfulness =
  (metric(scrubbed) − metric(corrupted)) / (metric(clean) − metric(corrupted)).
  `CircuitHypothesis(circuit_modules, node_labels)` specifies which modules
  implement the behaviour. `per_module_delta` (opt-in, one extra forward pass
  per module) shows each module's individual contribution.
- Both classes exported at `circuitry.patching.*` and top-level `circuitry.*`.
- Tests: 9 DAS + 11 Causal Scrubbing = 20 new tests; all layering tests pass.

---

## [1.27.0] — 2026-06-08

**Evaluation & benchmarks — MIB task loaders, SAEBench metrics, Fourier alignment, information bottleneck.**

### Added
- **`benchmarks/mib.py`** — `load_ioi(n, *, seed, vocab_size, seq_len)` and
  `load_greater_than(n, *, seed, vocab_size, seq_len)` return `MIBTask` dataclasses
  containing `clean_inputs`, `corrupted_inputs`, and a differentiable `metric_fn`
  (logit-diff on the last position). Fully synthetic — no downloads. Compatible with
  every `*Runner.run()` signature (Mueller et al. ICML 2025, arxiv:2504.13151).
  Exported at `circuitry.benchmarks.*`.
- **`benchmarks/saebench.py`** — `run_saebench(sae, acts, *, tasks=None) → SAEBenchResult`.
  Analytically-tractable SAE evaluation suite: `l0_sparsity`, `explained_variance`,
  `reconstruction_mse`, `feature_density`, `sparse_probing_r2` (1-NN linear probe
  on top-1 active features). `SAEBenchResult` dataclass carries all five metrics + an
  optional `ce_loss_score` field (populated when a model is provided). Operates on raw
  activation tensors — no network access. Implements the 5 CPU-tractable metrics from
  Karvonen et al. 2025 (arxiv:2503.09532). Exported at `circuitry.benchmarks.*`.
- **`core/dynamics.py`** — `fourier_feature_alignment(W, task_freqs, *, n_freqs=None) → float`.
  Computes the fraction of weight-matrix spectral power aligned with task-relevant
  Fourier modes: `rfft` along the input dimension, sum power across output rows, return
  `Σ_{k ∈ task_freqs} power[k] / Σ_k power[k]`. Returns 0.0 on empty `task_freqs` or
  `d_in < 2`. Nanda et al. ICLR 2024.
- **`core/dynamics.py`** — `information_bottleneck_score(acts_train, acts_val, labels_train, labels_val, *, n_bins, eps) → float`.
  Mutual-information proxy I(T;Y)/H(Y) for measuring grokking progress: projects
  activations onto the first PC (via `pca_lowrank`), bins into `n_bins` equal-width
  buckets, estimates MI from the joint histogram. Clipped to `[0, 1]`. Leavitt &
  Morcos 2024.
- Both dynamics helpers exported at top-level `circuitry.*` and via `circuitry.core.dynamics`.
- Tests: 8 `test_dynamics.py` additions (`fourier_feature_alignment` × 4 + `information_bottleneck_score` × 3 + range check).

---

## [1.26.0] — 2026-06-08

**SAE ecosystem expansion — CrosscoderWrapper, Gemma/Llama Scope loaders, Matryoshka + BatchTopK.**

### Added
- **`patching/sae_features.py`** — `CrosscoderWrapper` (Anthropic Oct 2024). Wraps a
  crosscoder SAE as a single-site intervention point (`hook_input=False`). Routes
  `encode`/`decode` through `encode_at_layer`/`decode_at_layer` when available (with
  `primary_layer` argument), falls back to plain `encode`/`decode`. `encode_all(acts)`
  for full cross-layer mode. Exported lazily via `circuitry.patching` and eagerly at
  top-level `circuitry`.
- **`sae/loader.py`** — `load_gemma_scope(model_size, layer, width, *, site, average_l0, device)`
  and `load_llama_scope(layer, width, *, device)`. Convenience wrappers over `load_sae()`
  for pre-trained JumpReLU SAE suites (Lieberum et al. 2024, arxiv:2408.05147). Lazy import —
  no crash at import time without `sae_lens`. Exported at `circuitry.sae.*`.
- **`sae/grad.py`** — Added `"matryoshka"` (Bussmann et al. 2025, arxiv:2503.17547) and
  `"batch_topk"` to `SUPPORTED_SAE_ARCHITECTURES`. Both use the existing `sae_decompose`
  gradient path unchanged. Added `"crosscoder"` to `_BLOCKED_ARCHITECTURES` (raw crosscoder
  objects must be wrapped in `CrosscoderWrapper` for attribution).
- Tests: 13 CrosscoderWrapper + 5 architecture + 6 scope-loader = 24 new tests.

---

## [1.25.0] — 2026-06-08

**Edge Pruning (NeurIPS 2024) + HAP — SOTA joint circuit discovery.**

### Added
- **`patching/edge_pruning.py`** — `EdgePruningRunner` learns binary edge masks jointly via
  Adam + temperature-annealed sigmoid + L0 regularisation (Bhaskar & Wettig NeurIPS 2024,
  arxiv:2406.16778). Uses `EAPRunner` to compute initial per-edge gradient signals, then
  optimises scalar mask logits `z_e` to minimise `−(σ(z/T)·|score|) + λ·Σσ(z/T)` with
  temperature annealing `T: temperature_init → temperature_final` over `n_steps`.
  Supports `candidate_edges` to restrict the search space.
  `EdgePruningResult` provides `circuit`, `removed_edges`, `mask_logits`, `eap_scores`,
  `circuit_graph()`, `ranked()`, `to_json()`/`from_json()`/`save()`/`load()`.
- **`patching/hap.py`** — `HAPRunner` implements Hybrid Attribution + Pruning (Hu et al.
  2025, arxiv:2510.03282): Phase 1 pre-filters edges via EAP (keeps top `top_p` fraction
  by |score|); Phase 2 runs `EdgePruningRunner` on the reduced subgraph. Achieves similar
  faithfulness to full edge pruning at a fraction of the candidate set size.
- Both classes exported at `circuitry.patching.*` and top-level `circuitry.*`.
- Tests: 10 edge-pruning + 5 HAP = 15 new tests; 296 existing tests unaffected.

---

## [1.24.0] — 2026-06-08

**Representational analysis primitives: linear probing, concept erasure, future lens.**

### Added
- **`core/probe.py`** — `train_linear_probe(acts, labels, *, max_iter, C, tol) → LinearProbe`.
  Pure-PyTorch Adam training loop with L2 regularisation and early stopping. `LinearProbe`
  provides `.predict()`, `.predict_proba()`, `.accuracy()`, and `.direction()` (unit concept
  vector; binary: normalised `weight[0]`; multi-class: first left singular vector via
  `pca_lowrank`). Both exported at top-level `circuitry.*` and `circuitry.core.*`.
- **`core/erase.py`** — `leace_erase(acts, labels) → EraseProjection`. Mean-direction
  orthogonal projection onto the concept complement (Park et al. 2023 LEACE). Binary: concept
  direction = `μ₁ − μ₀` normalised; multi-class: first right singular vector of the
  between-class mean matrix. `EraseProjection.apply(acts)` broadcasts to `(..., d_model)`.
- **`core/lens.py`** — `future_lens_kl(residual, unembed, target_logits, *, horizon, layer_norm, chunk_size) → float`.
  Extends logit lens to compare residual at position `t` against `target_logits[t + horizon]`,
  measuring how much information about future tokens is already encoded at this layer.
  Returns `0.0` when `horizon ≥ seq`. Reduces to `logit_lens_kl` at `horizon=0`.
- Tests: 8 + 5 + 5 = 18 new tests across `test_probe.py`, `test_erase.py`, `test_future_lens.py`.

---

## [1.23.0] — 2026-06-08

**Logit lens distribution primitive + activation steering + design doc backfill.**

### Added
- **`core/lens.py`** — `logit_lens_distributions(residuals, unembed, *, layer_norm, top_k) → list[LayerPrediction]`.
  Complements the existing `logit_lens_kl` scalar: projects intermediate residual states
  through the unembedding and returns per-layer top-k token predictions with probabilities.
  Accepts `dict[int, Tensor]` or `list[Tensor]` residuals; collapses leading dims via
  mean-reduce. `LayerPrediction` dataclass: `layer_idx`, `token_ids`, `probs`.
- **`core/steer.py`** — `steer_vector(positive_acts, negative_acts, *, normalize=True) → Tensor`.
  Contrastive Activation Addition (Rimsky et al. 2024 CAA): mean-difference direction between
  class polarities, optionally unit-normalised. Raises `ValueError` on near-zero difference.
- **`patching/steer.py`** — `apply_steer(model, site, vector, *, coeff=1.0, resolver=None)` context
  manager. Registers a forward hook at `site` that adds `coeff * vector` (broadcast-safe for
  1D/2D/3D outputs); hook is always removed on exit, even on exception.
- Tests: 7 + 9 = 16 new tests across `test_logit_lens_dist.py`, `test_steer.py`.

### Fixed
- `patching/sae_edges.py` module docstring corrected: the TLSiteResolver path was already
  implemented in v1.7 P3 but the docstring still read "TLSiteResolver → NotImplementedError".

### Docs
- `docs/design.md` §3: added `core/dynamics.py` to the repo structure file listing;
  updated `core/attention.py` comment to include `head_specialization`.
- `docs/design.md` §4.4: added `probe_batch`, `drift_method`, `drift_max_tokens` to the
  `Recipe` dataclass code example (they were documented in a blockquote below but absent
  from the code block).

---

## [1.22.0] — 2026-06-08

**SAEFeatureTemporalRunner — multi-step SAE feature attribution.** Runs
`SAEFeatureRunner` independently on each `(step_key, clean_inputs, corrupted_inputs)`
triple and aggregates results into a `TemporalAtPResult` with per-step scores,
attribution deltas between consecutive steps, and helpers for identifying stable
vs step-specific features.

### Added
- **`SAEFeatureTemporalRunner(model, sae_sites, resolver)`** in
  `circuitry.patching.sae_temporal` (re-exported from `circuitry.patching`).
  `.run(steps, metric, **runner_kwargs) → TemporalAtPResult` accepts a list of
  `(step_key, clean_inputs, corrupted_inputs)` triples and runs attribution
  independently for each step. Step keys must be unique and non-empty.
- **`TemporalAtPResult`** — result container with:
  - `.scores: dict[step_key, AtPResult]` — per-step feature attribution.
  - `.step_keys: list` — ordered step keys.
  - `.delta_scores: dict[step_key, AtPResult]` — attribution change between
    consecutive steps (`delta[k] = scores[k] − scores[k−1]`; first step has no entry).
  - `.stable_features(threshold) → list[AtPNode]` — nodes with `|score| ≥ threshold`
    at ALL steps, sorted by layer/neuron.
  - `.step_specific_features(step_key, threshold) → list[AtPNode]` — nodes active
    above threshold at `step_key` but not at any other step.
  - `.top_stable(k=10) → list[(AtPNode, float)]` — top-k features by minimum
    `|score|` across all steps (most reliably active), sorted descending.
- Tests: 13 new tests in `tests/patching/test_sae_temporal.py`.

Note: each step is evaluated independently. True recurrent-SAE attribution
(where step k's activations depend on step k−1's hidden state) requires a
different architecture and remains a known limitation.

---

## [1.21.0] — 2026-06-08

**TranscoderWrapper — transcoder SAEs as intervention sites.** `TranscoderWrapper` wraps
any transcoder (module-input → module-output feature decomposition) so it can be used as
an SAE site in `SAEFeatureRunner` and `SAEFeatureEdgeRunner`. The wrapper sets
`hook_input=True`, signalling the attribution hooks to encode from `inp[0]` (the module
input) rather than from `output`. The SAE splice remains lossless: `x_hat + eps = output`
where `eps = output − x_hat` (in output space).

### Added
- **`TranscoderWrapper`** in `circuitry.patching.sae_features` (and re-exported from
  `circuitry.patching`). Wraps any object with `encode(x_in) → f` and `decode(f) → x_hat`
  where `x_in` is the module input and `x_hat` is in the module output space. Setting
  `hook_input = True` on the wrapper routes the attribution hooks through `inp[0]`.
- Attribution hooks in `SAEFeatureRunner._run_site` now branch on `getattr(sae, "hook_input", False)`:
  when `True`, the corrupted hook encodes from `inp[0]` and computes `eps = output − x_hat`;
  the clean hook seeds the gradient leaf from `encode(inp[0])` with the same lossless splice.
  The IG path is unchanged (uses pre-computed interpolated `f_k`, no re-encoding needed).
- Same transcoder-aware branching added to the four writer/reader hooks in
  `SAEFeatureEdgeRunner._compute_pair_edges` (`_writer_corr_hook`, `_writer_clean_hook`,
  `_reader_corr_hook`, `_reader_clean_hook`).
- Tests: 9 new tests in `tests/patching/test_sae_transcoder.py`.

---

## [1.20.0] — 2026-06-08

**Parallel-attention arch flag for SAEFeatureEdgeRunner.** `arch='parallel'` on
`SAEFeatureEdgeRunner.run()` proactively skips same-layer `attn_out → mlp_out` edges that
are causally undefined in GPT-J-style models (both heads read `resid_pre` simultaneously).

### Added
- **`arch: str = 'sequential'`** keyword argument on `SAEFeatureEdgeRunner.run()`.
  `'sequential'` (default, Llama-style — attention before MLP) preserves all existing edges.
  `'parallel'` (GPT-J-style) skips `attn_out@L → mlp_out@L` pairs via the new
  `_is_parallel_intra_layer()` helper. Unknown values raise `ValueError`.
- **`_is_parallel_intra_layer(writer_site, reader_site) → bool`** — public helper returning
  `True` when the pair is same-layer `attn_out → mlp_out`.
- Tests: 10 new tests in `tests/patching/test_sae_parallel_arch.py`.

---

## [1.19.0] — 2026-06-08

**Per-position SAE feature edge scores.** `SAEFeatureEdgeRunner.run()` gains a `per_position=False`
flag. When enabled, the runner computes per-sequence-position attribution scores alongside the
existing scalar scores.

### Added
- **`per_position: bool = False`** keyword argument on `SAEFeatureEdgeRunner.run()` — when `True`,
  also computes position-level scores for each edge and stores them in
  `SAEFeatureCircuit.position_scores` (a `dict[SAEFeatureEdge, Tensor]` where every value has
  shape `(seq_len,)`). `position_scores[e].sum() == edges[e]` within float32 rounding.
  Supported for both `variant='attrib'` and `variant='ig'` (including error→feature edges when
  `include_error_node=True`). Default `False` — zero overhead at default settings.
- **`SAEFeatureCircuit.position_scores`** — new field (`None` unless `per_position=True`).
  Exposes which sequence positions drive each feature→feature edge, enabling positional
  analysis of circuit attribution.
- **`SAEFeatureCircuit.top_positions(edge, k=5)`** — convenience helper that returns the top-k
  sequence positions by absolute per-position score as a list of `(position_index, score)` pairs
  sorted by `|score|` descending. Raises `ValueError` when `position_scores is None`.
- Tests: 11 new tests in `tests/patching/test_sae_edges_per_position.py`.

---

## [1.18.0] — 2026-06-08

**Report polish + circuit I/O convenience.**

### Added
- **`_parse_matched_module_counts(text)`** in `recorder/report.py` — parses
  `circuitry/matched_modules.txt` and returns matched-module counts per report family
  (`"weight"`, `"activation"`, `"grad"`). The same module appearing under multiple
  hook-points with the same source is deduplicated. `named_param` and `weight` sources
  both map to the `"weight"` family; `output` and `input` map to `"activation"`.
- **"Modules matched" line in `## Summary`** — when `matched_modules.txt` is present and
  the report is not compact, a new bullet shows per-family module counts (e.g.
  `**weight**: 42 · **activation**: 16`), complementing the existing tag-count line and
  making recipe under-coverage immediately visible.
- **`EAPResult.save(path)` / `EAPResult.load(path)`** — thin file-I/O wrappers around
  `to_json()` / `from_json()` for convenience.
- **`ACDCResult.save(path)` / `ACDCResult.load(path)`** — same pattern for ACDC circuits.
- Tests: 4 new matched-module tests in `tests/recorder/test_report.py`; 4 new save/load
  tests in `tests/patching/test_circuit_render.py` (53 pass total across those files).

---

## [1.17.0] — 2026-06-07

**Circuit rendering.** `EAPResult`, `AtPResult`, and `ACDCResult` gain human-readable output
and JSON I/O. A new `circuit-compare` CLI subcommand diffs two circuit files by edge-set.
`circuitry scan --model-factory` is now fully functional.

### Added
- **`_node_str(node)`**, **`_node_to_dict(node)`**, **`_node_from_dict(d)`** in
  `circuitry.patching.graph` — human-readable node labels (`embed`, `L2H5`, `mlp.L3`, …)
  and JSON serialization helpers.
- **`EAPResult.to_markdown(*, top_k=20) → str`** — `## EAP Circuit` header, graph stats,
  top-K edge table (`rank | writer | slot | reader | score`).
- **`EAPResult.to_json() → str`** / **`EAPResult.from_json(text) → EAPResult`** —
  JSON serialization with `"kind": "eap"`, `n_layers`, `n_heads`, and scored edge list.
- **`AtPResult.to_markdown(*, top_k=20) → str`** — `## AtP Node Attribution` header,
  node count, top-K table (`rank | node | slot | score`).
- **`ACDCResult.to_markdown(*, top_k=None) → str`** — `## ACDC Circuit` header, kept/total
  edges, final KL, edge table with optional top-K elision.
- **`ACDCResult.to_json() → str`** / **`ACDCResult.from_json(text) → ACDCResult`** —
  JSON serialization with `"kind": "acdc"`, kept edges list, final KL. Removed edges are
  reconstructed from the full graph on deserialization.
- **`circuitry circuit-compare A.json B.json [--out file]`** CLI subcommand — loads two
  circuit JSON files (EAP or ACDC), computes symmetric edge-set diff, renders a markdown
  summary table + per-side unique-edge listings.
- **`circuitry scan --model-factory pkg.module:factory`** — the scan subcommand now fully
  works; uses `_load_entrypoint` (same helper as `fit-tuned-lens`) to resolve the factory,
  then calls `scan_run`. `--out` flag added for output directory override.
- Tests: 32 new tests in `tests/patching/test_circuit_render.py`; 4 new CLI tests in
  `tests/test_cli.py`.

### Removed
- The stub `rc=2` / "not yet exposed via CLI" error path in `_cmd_scan`.

---

## [1.16.0] — 2026-06-07

**Training-dynamics depth.** Two enhancements to the `## Training Dynamics` report section and
a new flag rule for representational drift.

### Added
- **Direction labels in `### Phase Transitions`** — each row now has a `direction` column:
  `weight/update_delta_rel` transitions are labelled `↑ norm growth` or `↓ norm collapse`;
  all other metrics get `↑ growth` / `↓ collapse`.  `_transition_direction()` helper added.
- **`### Representation Drift` sub-table** — any `activation/repr_drift/{module}` tags
  produce a per-layer drift table (start drift, end drift, Δ, trend label: `↑ rising`,
  `→ stable`, `↓ falling`, or `⚡ step N` when `phase_transition_steps` detects a sharp
  jump).  Rows sorted by last-step drift descending.  `_drift_trend()` helper added.
- **`repr_drift_high` flag rule** — fires when the last-step drift of any layer exceeds 0.5
  (significant representational shift from reference snapshot).
- **`activation/repr_drift` in `HERO_SECTIONS`** — the section is elevated to hero priority
  in the main report body for quick scanning.
- Tests: 8 new tests in `tests/recorder/test_training_dynamics_report.py`;
  1 new flag test in `tests/recorder/test_report.py`.

### Fixed
- **`__version__`** bumped from the stale `1.12.0` to `1.16.0` (v1.13–v1.15 were released
  but the version string was never updated).

---

## [1.15.0] — 2026-06-07

**Polish sprint.** Adds `grokking_step`, per-hook-family tag counts in the report summary,
Grokking Signals in Training Dynamics, and a tool-by-tool landscape comparison doc.

### Added
- **`core.dynamics.grokking_step(series, *, z_threshold=2.5) → int | None`** — thin wrapper
  around `phase_transition_steps` returning the first detected transition step.  Default
  threshold raised to 2.5 vs 2.0 to reduce false positives on noisy loss curves.
- **Per-hook-family tag counts** in the `## Summary` block (e.g. `**weight**: 3 · **activation**: 2`).
- **`### Grokking Signals` sub-table** in `## Training Dynamics` — auto-detected from
  `loss`/`acc`/`accuracy`/`error`/`perplexity` tag names via `grokking_step`.
- **`docs/positioning.md`** — tool-by-tool comparison: TransformerLens, nnsight, pyvene,
  sae_lens.  Covers where circuitry is differentiated and where it defers.
- Tests: 5 new `grokking_step` unit tests; 3 new report integration tests.

### Fixed
- **`gate_stats` return type** in `docs/design.md §4.1` corrected from `-> GateStats` to
  `-> dict[str, float]` (the actual implementation type).

---

## [1.14.0] — 2026-06-07

**Training-dynamics diagnostics.** A new `core/dynamics` module adds two pure-Python primitives
that operate on the `list[tuple[int, float]]` series format from `_metrics.group`, requiring no
GPU. The report gains a `## Training Dynamics` section (suppressed in compact mode and on
single-step runs) with two sub-tables: **Head Formation Events** (heads that crossed their
specialisation threshold during the recording window; pre-formed heads excluded) and **Phase
Transitions** (sharp change-points detected in `effective_rank`, `rank_trajectory`, and
`update_delta_rel` time series).

### Added
- **`core.dynamics.phase_transition_steps(series, *, window=5, z_threshold=2.0, min_gap=5) → list[int]`.**
  Bilateral-mean change detector: at each center position computes `|mean(right_half) − mean(left_half)|`
  and flags positions where the bilateral change exceeds `mean + z_threshold × std` across all
  centers. Collapses nearby detections within `min_gap` index positions (keeping the largest). A
  monotone linear ramp produces identical bilateral changes everywhere (std = 0) and returns `[]`.
  Tests: `tests/core/test_dynamics.py`.
- **`core.dynamics.head_formation_step(series, *, threshold, n_sustain=2) → int | None`.**
  Returns the first training step where a per-head attention score crosses `threshold` and remains
  above it for `n_sustain` consecutive recorded points. End-of-series tolerance: if the series ends
  before `n_sustain` confirmations, the available tail is checked. Returns `None` if the threshold
  is never crossed. Tests: `tests/core/test_dynamics.py`.
- **`## Training Dynamics` report section** in `build_report()`. Uses `head_formation_step` on
  `induction_score` / `copy_suppression_score` / `attention_sink_score` series (thresholds 0.4 /
  0.3 / 0.5) to list heads that specialised during the recording window, and `phase_transition_steps`
  on `effective_rank` / `rank_trajectory` / `update_delta_rel` to flag sharp change-points.
  Tests: `tests/recorder/test_training_dynamics_report.py`.

---

## [1.13.0] — 2026-06-07

**Head specialization classification.** A new `head_specialization` primitive synthesises the three
per-head behavioral scores (induction, copy-suppression, sink) into a single label per head. The
report gains a `## Head Specialization` table rendered after the hero sections, showing each head's
inferred behavioral type with specialist labels bolded. The section is suppressed in compact mode.

### Added
- **`core.attention.head_specialization(induction, copy_suppression, sink, *, thresholds) → list[str]`.**
  Classifies each head as `"induction"` / `"copy_suppression"` / `"sink"` / `"uniform"`. When a head
  exceeds multiple thresholds, the type with the highest score-to-threshold ratio wins (normalised
  tie-break). Threshold defaults match the report `FLAG_RULES` (0.4 / 0.3 / 0.5). Tests:
  `tests/core/test_attention.py`.
- **`## Head Specialization` table** in `build_report()`. Parses the last-step values of all three
  attention-score families, calls `head_specialization` per module, and renders a markdown table
  with columns: module | head | type | induction | copy_suppression | sink. Specialist types are
  bolded; `uniform` is plain. Section absent when no attention-score tags are present; suppressed in
  compact mode. Tests: `tests/recorder/test_head_specialization.py`.

---

## [1.12.0] — 2026-06-07

**Attention sink head detection.** A new `attention_sink_score` primitive and live/scan
diagnostic completes the attention-head behavioral triad: induction (offset +1 on probe),
copy-suppression (offset 0 on probe), and sink (concentration on position 0 in real training
inputs). Unlike the probe-based pair, this metric runs on the live training-forward attention
pattern — sinks form on the real distribution, not on a synthetic repeated-token probe.

### Added
- **`core.attention.attention_sink_score(attn_pattern, *, sink_pos=0) → list[float]`.**
  Per-head mean attention weight on a designated sink position (default 0 / BOS token) across
  all query positions and batch elements. `sink_pos=-1` selects the last position. Operates on
  the live training-forward `(B, H, T, T)` attention tensor, not the induction probe. Tests:
  `tests/core/test_attention.py`.
- **`attention_sink_score` activation diagnostic** (Recorder, scan, report). Added to the `llm`
  recipe's `activation_diagnostics`. Emits `activation/attention_sink_score/<module>/head_N`.
  The permanent `_main_pass_attn` capture hook now also fires when `attention_sink_score` is in
  the recipe (previously only gated on `attention_pattern_entropy`). Report `HERO_SECTIONS` and
  `FLAG_RULES` updated (`attention_sink_detected` fires when last > 0.5). Tests:
  `tests/recorder/test_attention_sink_diagnostic.py`.

## [1.11.0] — 2026-06-07

**Copy-suppression head detection.** A new `copy_suppression_score` primitive and live/scan
diagnostic completes the attention-circuit screening pair alongside `induction_score`. On the
same repeated-token probe, `copy_suppression_score` measures how strongly each head attends
back to the *same* token's prior position (offset 0), while `induction_score` measures offset +1.
The two are mathematically complementary: a pure induction head scores ~0 on copy-suppression
and vice versa. The recorder runs a single shared probe forward per step when both are enabled.

### Added
- **`core.attention.copy_suppression_score(attn_pattern, *, seq_len_repeat) → list[float]`.**
  Per-head same-token-attention score on the repeated-token probe — the characteristic pattern of
  copy-suppression heads (McDougall et al. 2023). Reuses the same `(B, H, T, T)` attention tensor
  and probe structure as `induction_score`; complement at offset 0. Tests: `tests/core/test_attention.py`.
- **`copy_suppression_score` activation diagnostic** (Recorder, scan, report). Added to the `llm`
  recipe's `activation_diagnostics`. Emits `activation/copy_suppression_score/<module>/head_N` per
  head per attention module per step. Shares the probe forward pass with `induction_score` via a
  per-step lazy cache (`_get_probe_attn`) — no duplicate probe runs when both are enabled. Report
  `HERO_SECTIONS` and `FLAG_RULES` updated (`copy_suppression_detected` fires when last > 0.3 nats).
  Tests: `tests/recorder/test_copy_suppression_diagnostic.py`.

### Changed
- **Probe-pass deduplication.** The induction-probe forward pass is now extracted into a helper
  method (`_get_probe_attn`) with a per-step lazy cache. Both `induction_score` and
  `copy_suppression_score` call it; the probe runs at most once per emit step regardless of how
  many probe-based diagnostics are enabled.

## [1.10.0] — 2026-06-07

The **tuned lens** (Belrose et al. 2023) — the one lens the project had flagged as future work —
ships as a post-hoc workflow plus an opt-in live/scan diagnostic, completing the lens story
alongside the v0.9 logit lens. New public surface: `core.lens.tuned_lens_kl`, the
`circuitry.tuned_lens` package (`TunedLens`, `fit_tuned_lens`, `model_fingerprint`),
`Recipe.tuned_lens`, the `tuned_lens_kl` activation diagnostic, and the `circuitry fit-tuned-lens`
CLI. This release also clears three deferred polish/correctness items.

### Added
- **Tuned lens (`core.lens.tuned_lens_kl`).** A pure primitive that applies a learned per-layer
  affine translator `A·h + b` to the residual before the unembed projection, so per-layer KL is no
  longer confounded by the early/mid-layer basis mismatch the parameter-free logit lens suffers.
  With `A = I, b = 0` it reduces exactly to `logit_lens_kl` (asserted in tests). The shared lens
  tail (orientation auto-detect + token-chunked KL) is factored into helpers so the two lenses
  can't drift. Tests: `tests/core/test_lens.py`.
- **`circuitry.tuned_lens` workflow package.** `TunedLens` — a serializable container of per-layer
  translators with `save`/`load`, layer/`d_model` metadata, and a `model_fingerprint` guard.
  `fit_tuned_lens(model, batches, ...)` — the post-hoc trainer (the only optimizer loop in the
  library, kept strictly in the workflow layer). It freezes the model, captures per-block residuals
  the same way the recorder does, reconstructs the target final distribution as
  `softmax(LN_f(last_block) @ W_U)`, and AdamW-trains each layer's affine (init identity) to
  minimize KL to that target; the final block is the target frame and is not fitted. Layering:
  `tuned_lens/` may import `core/` only — enforced in `.importlinter` and `tests/test_layering.py`.
  Tests: `tests/tuned_lens/test_fit.py`.
- **`tuned_lens_kl` Recorder + scan diagnostic (opt-in).** Set `Recipe.tuned_lens` to a fitted
  `TunedLens` and add `"tuned_lens_kl"` to `activation_diagnostics`; the recorder emits
  `activation/tuned_lens_kl/<layer>` per fitted block, reusing the shared unembed/final-LN
  resolution (no new hooks). At `attach()` it verifies the lens fingerprint against the attached
  model and **warns + skips** on a missing lens or a mismatch — it never emits a wrong KL. NOT in
  any stock recipe's default list. Tests: `tests/recorder/test_tuned_lens_diagnostic.py`.
- **`circuitry fit-tuned-lens` CLI.** Resolves `--model` / `--batches` as `package.module:attr`
  entry points (zero-arg factories), fits the translators, and writes a `TunedLens` to `--out`.
  Tests: `tests/test_tuned_lens_cli.py`.
- **Report integration.** `activation/tuned_lens_kl` is a hero section; a new
  `tuned_lens_not_forming` flag fires when a tuned-lens KL stays high (prediction not forming /
  stale lens). Tests: `tests/recorder/test_tuned_lens_report.py`.
- **`core.weight.relative_update_delta`** — scale-invariant per-parameter update size
  `||ΔW|| / ||W||`; the recorder now emits `weight/update_delta_rel/*` alongside `update_delta`.

### Fixed
- **`build_report` Δ column reported the unsigned range, not the signed trend.** Deferred since
  v1.2: `_metrics.stats` returned `delta = vmax - vmin`, so a monotonically *decreasing* metric
  (`effective_rank: 15 → 5`) rendered `Δ = +10`, reading like an increase. `delta` is now the
  signed `last − first`; the table cell shows the sign and the summary's "moving" label is relabelled
  "(changed)" (range-based). The `## Flags` block already used a signed trend, so this only aligns
  the table. Tests: `tests/recorder/test_report.py`.
- **`update_delta_vanishing` flag used a scale-dependent absolute L2 threshold** (v1.3 review). It
  now keys on the new dimensionless `weight/update_delta_rel` companion (`||ΔW||/||W||`), so the
  threshold means the same thing across parameter sizes — a large-matrix healthy step no longer
  reads as vanishing. Tests: `tests/core/test_weight.py`, `tests/recorder/test_report_flags.py`.

### Investigated / deferred
- **Gram-matrix (`eigvalsh(WᵀW)`) SVD speedup.** A CPU benchmark confirms ~1.8× at a 2:1 aspect
  ratio with negligible error on the aggregate rank metrics, but the recorder shares one SVD across
  all SVD-derived diagnostics — including `sv_histogram`, whose spectral *tail* the Gram squaring
  degrades — so lowering the default `use_gram='auto'` threshold would regress the v1.8
  accurate-by-default principle. The gate is the §10 **GPU** wall-clock number; deferred until a GPU
  benchmark justifies it. Rationale in `docs/superpowers/specs/2026-06-07-v1.10-tuned-lens-design.md` §9.

### Notes
- The §10 GPU wall-clock budget re-validation **with `tuned_lens_kl` enabled** is pending a GPU
  run; the diagnostic adds one `(d_model × d_model)` matmul per layer per emit on top of the
  existing logit-lens projection (no extra forward pass), so it is expected to stay within the
  ≤10% budget. Tuned-lens *fitting* is post-hoc and outside the training-loop budget by design.

## [1.9.0] — 2026-06-03

Real-model evaluation follow-ups: every finding from the two v1.8.0 field reports (67-LM
leaderboard fingerprint + trained SASRec) is resolved, plus an Apple-Silicon portability fix
surfaced by real-model validation. New public surface: the sequential-`recsys` recipe,
`Recipe.forward_fn`, `TensorSource.NAMED_PARAM`, `core.weight.spectral_entropy`,
`Recipe.effective_diagnostics()`, and optional `sae-lens`/`tensorboard` extras. The §10 GPU
wall-clock budget was re-validated with the full-SVD default (+7.4%, within ≤10%).

### Fixed
- **Weight diagnostics hard-crashed on Apple Silicon (MPS).** `singular_values` (and the
  `repr_drift` CKA path) promote to `float64` for accuracy, which MPS does not support, so every
  SVD-based weight diagnostic raised `TypeError: Cannot convert a MPS Tensor to float64` on a model
  living on `mps`. The primitives now offload the `float64` decomposition to CPU when the input is
  an MPS tensor — preserving the full-precision accuracy default and keeping the result
  device-deterministic instead of crashing. Surfaced by real-model validation on an M-series Mac
  (`scripts/v19_validation/real_model_followups.py`). Tests: `tests/core/test_weight.py`
  (MPS-guarded).
- **Gradient diagnostics silently emitted nothing on `recsys`, `two_tower`, and `vision` (recsys
  eval #5).** The recorder populates `ctx.gradients` only from `TensorSource.GRAD` hook points, but
  those three recipes declared a gradient diagnostic (`norms_per_param` / `grad_norm_per_module`)
  with no GRAD hook point — so `grad/*` tags never emitted. Each recipe now ships GRAD hook points
  mirroring its WEIGHT patterns. A new guard test
  (`tests/recipes/test_gradient_hookpoints.py`) asserts every stock recipe that declares a
  gradient diagnostic has at least one GRAD hook point, preventing recurrence.
- **`attention_pattern_entropy` returned NaN on left-padded models (recsys eval B).** PAD query
  rows attend to an all-`-inf` key set; `softmax` yields an all-`NaN` row that poisoned the naive
  per-head mean. The per-head mean is now **NaN-aware** — fully-masked PAD rows are dropped
  automatically (identical result on unpadded patterns, so backward-compatible). A new optional
  `valid_mask` argument (`True` marks a valid query row, broadcastable to `(B, H, T_query)`;
  a 2-D `(B, T)` mask is auto-expanded across heads) restricts the average to chosen rows.
  Test: `tests/core/test_attention.py`.
- **`attention_head_rank` emitted zero output on most real models (fingerprint eval #1).** Head
  metadata was read only from `model.config` (+ `text_config`), so HF-wrapped models
  (metadata on `model.model.config`) and config-less custom models resolved to nothing — *zero*
  head-rank tags across all 68 fingerprint variants. Resolution is now: explicit
  `Recipe.attn_head_meta` → any submodule exposing a `.config` with `num_attention_heads` →
  `num_heads`/`head_dim` attributes read directly off a `self_attn`/`attn`/`attention` submodule.
  The "no usable metadata" warning moved from first-emit to `attach()` and names what was
  searched. Test: `tests/recorder/test_head_meta_resolution.py`.

### Validated
- **§10 wall-clock budget re-validated on GPU with the full-SVD default** (RTX 5080, 88M decoder,
  full `llm` recipe, `every_n_steps=200`, batch 16 × seq 512): **+7.4%**, within the ≤10% target.
  The v1.8 full-SVD default eroded the margin (was +5.3% with the old `max_dim=512` subsample) but
  it still holds. Raw numbers: `scripts/v19_validation/gpu_perf_budget.results.json`; design §10
  updated. Also added `scripts/v19_validation/real_model_followups.py` — all session follow-ups pass
  8/8 on real Qwen2.5-0.5B + a SASRec-shaped MHA model, on CPU and MPS.

### Changed
- **`sae-lens` and `tensorboard` are now optional extras, not hard dependencies (fingerprint
  eval #2).** A lean `pip install circuitry` pulls only `torch` + `numpy`. Install
  `circuitry[tensorboard]`, `circuitry[sae]`, or `circuitry[all]` for those features. The default
  `writer` is now `"auto"` — TensorBoard when the extra is installed, else the no-dep `jsonl`
  writer (with a one-time warning); `writer="tensorboard"` (explicit) raises a clear,
  install-pointing `ImportError` when the extra is absent. The SAE loader raises the same friendly
  error. `import circuitry` never pulls either package. Test: `tests/test_optional_deps.py`.

### Fixed
- **Dense-model strict-attach regression (v1.8.0).** The MoE-only weight HookPoints added to the
  stock `llm` recipe in v1.8.0 (`.*\.mlp\.gate$`, `.*\.mlp\.experts$`) match 0 modules on any
  dense (non-MoE) model, so `Recorder.attach()` with the default `strict=True` **raised** — the
  stock recipe was unusable out of the box on plain Llama/GPT-2-shaped models. `HookPoint` now
  carries an `optional` flag: a 0-match on an optional HookPoint is a soft skip even under
  `strict=True`, while genuinely-required patterns still raise. The MoE patterns are marked
  optional. This also resolves the F37 follow-up — the companion 0-match weight warning no longer
  fires on every dense attach. Surfaced by an external evaluation of 67 custom dense 1M-param LMs
  (`FEEDBACK-2026-06-01-leaderboard-fingerprint.md` #7). Regression test:
  `tests/recorder/test_optional_hookpoints.py`.

### Added
- **`HookPoint.optional`** (default `False`) — marks a pattern as structurally absent on some of
  the architectures a recipe targets (MoE patterns on a dense model; DLRM/GRU patterns on a
  transformer-recsys model), so a 0-match is a soft skip rather than a strict-attach failure.
- **Stock `recsys` recipe** (`circuitry.recipes.recsys`) — covers sequential recommenders
  (SASRec / BERT4Rec / GRU4Rec): item/position-embedding anchor plus optional attention /
  FFN / norm / block / GRU patterns. Complementary to the existing `two_tower` recipe (which
  keeps two-tower + DLRM retrieval and the `embedding_alignment` diagnostic). Contributed from a
  real-model SASRec evaluation (`docs/observations/2026-06-01-recsys-sasrec-evaluation.md`);
  documents the known caveats (left-pad NaN entropy, `need_weights=False`, non-HF `forward`).
- **`Recipe.attn_head_meta`** (`dict | None`) — explicit attention-head metadata
  (`n_heads` / `n_kv_heads` / `head_dim` / `hidden_size`) so `attention_head_rank` works on
  config-less custom models; overrides config-based resolution.
- **`Recipe.forward_fn`** (`(model, batch) -> output`, default `None`) — custom forward entry
  point for the recorder's internal probe passes (`induction_score`, `drift_probe`), so non-HF
  models whose `forward` is not HF-style (e.g. SASRec's `predict_scores`) can drive them instead
  of `TypeError`-ing or silently no-op'ing (recsys eval C/D). A non-HF model with no resolvable
  `config` now warns once at `attach()` pointing at `forward_fn`. Test:
  `tests/recorder/test_forward_fn_and_scan_discovery.py`.
- **`scan_run(checkpoints=...)`** — overrides the default `<run_dir>/checkpoints/step*.pt` glob.
  Accepts a single checkpoint file, a glob string, a list of paths, or a list of explicit
  `(step, path)` pairs, enabling single-snapshot and arbitrarily-named retrospective scans
  (fingerprint eval #3).
- **`Recipe.effective_diagnostics()` / `Recipe.active_diagnostics`** — reflect which diagnostics
  will actually run after `.only()` / `.disable()` (which toggle the `enabled` dict, not the
  `*_diagnostics` lists, so inspecting the lists alone is misleading). Fingerprint eval #4.
- **`core.weight.spectral_entropy`** + `sv_histogram` companion scalars — `spectral_entropy`
  (Shannon entropy of the normalized singular-value distribution = `log(effective_rank)`) is a new
  primitive; the `sv_histogram` diagnostic now also emits `sv_max` / `sv_min` / `spectral_entropy`
  scalars so the spectrum is visible to scalar/CSV consumers, not just histogram viewers
  (fingerprint eval #6).
- **`scan_run` static-vs-trajectory warning** — a single-snapshot scan can only produce the static
  weight diagnostics; the cross-step trajectory diagnostics (`update_delta`, `rank_trajectory`,
  `direction_cosine`) need ≥2 emitted steps. `scan_run` now warns once when trajectory diagnostics
  are requested with fewer than two checkpoints (documented in the docstring + design §4.2), so the
  gap isn't mistaken for a recorder failure (fingerprint eval #5).
- **`TensorSource.NAMED_PARAM`** — a HookPoint source whose `pattern` matches against *parameter*
  names (`model.named_parameters()`) and feeds the matched ≥2-D parameter to the weight
  diagnostics. Reaches fused parameters the `WEIGHT` source can't resolve — e.g.
  `nn.MultiheadAttention.in_proj_weight` (a direct `Parameter` on the module, so the primary-weight
  resolver returns nothing). Non-≥2-D matches are skipped with a warning. Test:
  `tests/recorder/test_named_param_source.py` (recsys follow-up #3).

## [1.8.0] — 2026-06-01

Fixes for the 20 findings surfaced by the v1.7.0 whole-library real-model evaluation
(`docs/observations/2026-05-31-real-model-evaluation.md`). Every fix ships with a regression
test (`tests/**/test_*_eval_findings.py`); the v1.6 SAE golden freeze stays byte-for-byte intact.

### Fixed
- **F1 — SVD-derived weight diagnostics were biased + non-deterministic on any matrix with
  min-dim > 512** (i.e. every real LLM layer). `singular_values` now defaults to `max_dim=None`
  (full, deterministic SVD); `condition_number`/`effective_rank`/`stable_rank`/`heavy_tail_alpha`
  gained a `max_dim` (and `seed`) passthrough so the perf hatch is opt-in. On a 1024² matrix
  `condition_number` went from ~5.7 (290× low, varying run-to-run) to the exact numpy value.
- **F38 — scalar rank primitives silently mis-ranked batched 3-D expert tensors** (returned
  ≈`n_experts`). `effective_rank`/`stable_rank`/`heavy_tail_alpha`/`condition_number` now RAISE on
  ndim > 2; `spectral.rank_trajectory` folds conv weights to 2-D before calling.
- **F31 — `grad_norm_per_module` crashed on sparse `nn.Embedding` gradients** (standard recsys
  setup). Sparse grads are densified before the norm.
- **F2 — `attention_pattern_entropy` reported the induction-probe's attention, not the training
  batch's.** The induction-score probe forward now snapshots/restores `_main_pass_attn`.
- **F29 — `Recorder.attach()` crashed on every SDPA-attention HF model** (Qwen2/Llama-3.x/
  Mistral/Gemma). `_set_output_attentions_true` now degrades gracefully (skips attention
  diagnostics with a warning) instead of raising.
- **F3 — faithfulness/completeness/ACDC were wrong for `layer_norm`-normalized SAEs** (incl. the
  standard OpenAI v5 GPT-2 SAEs). `compute_f_per_site` now decodes in the same call as its encode
  (paired `sae_decompose`); byte-identical for non-stateful SAEs.
- **F4 — `variant='ig'` + `include_error_node=True` silently dropped all error→feature edges.**
  The IG writer hook now builds the interpolated error leaf and produces error→feature edges.
- **F11 — `completeness()` omitted `ablation_eps` under `include_error_node`.** Now threaded
  consistently with `faithfulness()`.
- **F8 — `scan_run` ran activation diagnostics with no forward pass** (silent empties + permanently
  disabled the logit-lens). Activation diagnostics are guarded; `scan_run` gained an optional
  `forward_fn` to enable them on checkpoints.
- **F23 — `compare_runs` silently NaN'd on a metric-family mismatch** (e.g. scan-vs-live). Now warns.
- **F7 — vision recipe missed ResNet's `fc` head and `downsample` convs** (29% capture). Patterns
  broadened.
- **F30 — vision recipe matched 0 modules on torchvision ViT-B/16** (hard RuntimeError on attach).
  Added the `encoder.layers.encoder_layer_N.*` patterns.
- **F33 — `embedding_alignment` silently returned `{}` for an `nn.ModuleList` item tower.**
  Output hooks now fire on ModuleList children and the diagnostic aggregates them.
- **F37 — silent under-coverage when a weight family matched 0 modules** (e.g. MoE). The recorder
  now emits an aggregate warning.

### Added
- **F9 — `condition_number` added to the `llm` recipe** (now accurate after F1).
- **F13 — `FeatureACDCRunner.run()`/`.sweep()` accept `variant`/`n_ig_steps`** (Stage-1 IG).
- **F32 — two_tower recipe covers standard DLRM names** (`embed_tables`/`bottom_mlp`/`top_mlp`).
- **F34 — per-table activation diagnostics** for `nn.ModuleList` embedding-table towers.
- **F36 / F39 — MoE coverage:** the `llm` recipe matches MoE router (`mlp.gate`) and batched-expert
  (`mlp.experts`) weights; the recorder emits **per-expert** weight diagnostics for the 3-D expert
  tensors.

### Changed
- `core.weight.singular_values` default flipped from `max_dim=512` to `max_dim=None` (accurate by
  default; `max_dim` is now an opt-in perf hatch). See design §10. The scalar rank primitives now
  require ≤2-D input. The live recorder consequently computes full SVDs on wide weights by default —
  per-emit weight-diagnostic cost rises on large models versus the old subsampled default.

## [1.7.0] — 2026-05-31

### Added
- **Resolver-routed SAE splice** (P1) — all SAE node/edge/circuit/faithfulness/ACDC splices now
  route through `ResolvedSite` (`HFSiteResolver.resolve` / `TLSiteResolver.resolve`) instead of
  hardcoding a decoder-block hook. The refactor touches every call site in `sae_features.py` and
  `sae_edges.py` (including `_run_site`, `compute_f_per_site`, `_feature_circuit_forward`,
  faithfulness `ablation_eps`, edge Stage-2, bruteforce paths, and `FeatureACDCRunner`). For
  `resid_post` + HF-eager + `variant='attrib'`, results are **byte-for-byte identical** to v1.6.0
  (`rtol=0, atol=0`), pinned by `test_resid_post_attrib_golden_freeze`.
- **`mlp_out` / `attn_out` SAE sites** (P2a) — `SAEFeatureRunner` and `SAEFeatureEdgeRunner`
  now accept `Site(component="mlp_out", layer=L)` and `Site(component="attn_out", layer=L)` in
  `sae_sites`. The `attn_out` component is new to `VALID_COMPONENTS` (`sites.py`). On the HF
  path, `mlp_out` hooks the `mlp` submodule output and `attn_out` hooks the `self_attn` submodule
  output — both tensors are **before the residual add**. On Llama/GPT-2-style sequential
  attn→mlp blocks this equals the whole attention/MLP contribution; Gemma2 and other architectures
  that apply a `post_attention_layernorm` / `post_feedforward_layernorm` before the residual add
  produce a pre-norm tensor instead — the splice is mechanically lossless on any arch, but the
  equivalence claim is scoped to Llama-family sequential blocks. Parallel-attention architectures
  (GPT-J-style: attn and mlp both read `resid_pre`) make intra-layer `attn_out@L → mlp_out@L`
  edges causally undefined; v1.7 assumes sequential blocks.
- **Multiple SAE sites per layer — composite `(layer, component)` keying** (P2b) — every
  internal dict that was keyed by `int` layer is now keyed by `(layer, component)`. `Node` gains
  an optional `component` field (default `None`; `resid_post` nodes keep `component=None` for
  backward compatibility with v1.6 identity/hash). `_COMPONENT_OFFSET` ranks sites in forward
  execution order (`attn_out=0, mlp_out=1, resid_post=2`), enabling `attn_out@L + mlp_out@L +
  resid_post@L` in a single circuit and `attn_out@L → mlp_out@L` intra-layer edges (the reverse
  direction is forbidden and its VJP is `None`).
- **TransformerLens backend** (P3) — `TLSiteResolver` is now fully supported for SAE node
  attribution, edge attribution, faithfulness/completeness, and `FeatureACDCRunner`. The
  `NotImplementedError` gates are removed. dtype and device are sourced from `model.cfg.dtype`
  and `torch.device(model.cfg.device)` on the TL path (HookPoints have no parameters; the
  previous params-fallback silently downcasted non-fp32/CUDA models). `Site("mlp_out")` resolves
  to `blocks.{L}.hook_mlp_out` (plain `(b,s,d_model)` tensor), consistent with HF and the EAP /
  AtP\* / ACDC TL paths.
- **Integrated-gradients variant** (P4) — `SAEFeatureRunner.run` and `SAEFeatureEdgeRunner.run`
  accept `variant='ig'` and `n_ig_steps: int = 0` (default 32 when `n_ig_steps==0` and
  `variant='ig'`). Path: `f(α) = f_clean + α·(f_corrupt − f_clean)`, midpoints
  `α_k = (k−0.5)/N`, `eps` frozen at clean throughout. Node scores:
  `score_i = Σ_pos Δf_i · (1/N) Σ_k ∂metric/∂f_i |_{f=f_clean+α_k·Δf}`. Edge IG: the writer
  detached-leaf is interpolated (`f_U_k = f_U_corrupt + α_k·(f_U_clean − f_U_corrupt)`) at each
  step; the reader stays live; the per-downstream-survivor VJP loop is unchanged. Cost is
  `N×` the attrib forward+backward+VJP-loop (default N=32); peak memory equals attrib (one
  VJP live at a time). Features are enumerated on the `Δf ≠ 0` union (not gated on
  `grad@clean`), so features dead at clean but active at corrupted — the saturation blind spot
  that attrib misses — are scored correctly.
  - **Completeness note.** Feature-IG completes to the **eps-frozen spliced** delta
    `metric(decode(f_corrupt)+eps_clean) − metric(decode(f_clean)+eps_clean)`, not to the
    real `metric(corrupt) − metric(clean)` (because `eps` is held at `eps_clean`, not
    `eps_corrupt`). With `include_error_node=True` the error node's IG interpolates the error
    leaf, and features+error IG jointly complete to the real forward delta.

### Changed
- **No-grad guard relaxed for sub-block edge pairs.** The v1.6 safety net that raised
  `RuntimeError` when an `mlp_out@L → resid_post@L`-style VJP returned `None` (severed
  gradient — indicating the two sites are not causally connected through autograd) was
  too broad. For `resid_post → resid_post` pairs the raise is kept (always connected — a
  `None` grad is a splice bug). For legitimately-disconnectable sub-block pairs (e.g.
  `mlp_out@L → mlp_out@L+1` across an add that bypasses mlp in some arches) the runner
  now warns and returns an empty edge dict instead of crashing.
- `circuitry.__version__` bumped to `"1.7.0"`.

### Notes
- `graph.py` / `EdgeGraph` / `_order` / `build_graph` are **untouched**. The `component`
  field on `Node` defaults to `None`; all existing `(layer, neuron)`-keyed v1.6 SAE nodes
  are hash/equality-identical to v1.6 (no golden drift).
- SAE architectures supported: `standard` / `topk` / `jumprelu` (covers BatchTopK /
  Matryoshka at inference) / `gated`. Transcoder / matching-pursuit / temporal SAEs
  remain `NotImplementedError`. Per-position edges and per-head/per-neuron
  sub-slice sites (`attn_head_out`, `mlp_neuron`) remain out of scope.

## [1.6.0] — 2026-05-30

### Added
- **Feature→feature SAE circuits** — `circuitry.patching.SAEFeatureEdgeRunner` +
  `FeatureACDCRunner`, building on the v1.5 node-level `SAEFeatureRunner`. Given a clean/corrupted
  prompt pair, a differentiable metric, and `sae_sites` (resid_post → SAE), `run()` returns a
  `SAEFeatureCircuit`: the v1.5 node `AtPResult`, a `dict[SAEFeatureEdge, float]` of feature→feature
  edges, and a `SAEFeatureEdgeGraph`.
  - **Two-stage tractability** (the feature×feature space is intractable — ~6e8 edges for one
    adjacent pair at d_sae=24576): reuse `SAEFeatureRunner` to keep the top-K **active** survivors
    per site, then enumerate edges only among survivors across ordered site-pairs
    (`layer_pairs='adjacent'` default, `'all_forward'` opt-in; `max_edges` cap).
  - **Edge formula** `edge(U:i→D:j) = Σ_pos Δf_U[i]·(∂f_D[j]/∂f_U[i])·gradf_D[j]`, `Δf_U =
    f_U_corrupt − f_U_clean`. ALL sites in the span are spliced **simultaneously** in one clean
    forward (upstream writer = detached-leaf seed, downstream reader = live encode, `eps` frozen at
    every site), and the Jacobian factor is realized as a **per-downstream-survivor VJP** — never a
    dense `d_sae×d_sae` matrix (each VJP is freed immediately). This is the full **live-autograd
    Jacobian**, not the Marks/Anthropic frozen-attention-pattern Jacobian. On a linear-downstream
    model the analytic edge equals the independent `bruteforce_feature_edge_scores` at 1e-4.
  - **Opt-in error→feature edges** (`include_error_node=True`, default off) via an independent
    upstream error leaf; **feature→error edges are structurally zero** (the downstream error is a
    frozen detached leaf — Anthropic-style source-only error) and are not computed.
  - **Circuit extraction:** `SAEFeatureCircuit.ranked()/top_k(n)/threshold(tau)` +
    `prune(method='threshold'|'acdc'|'both', tau, ablation_mode='corrupted'|'zero'|'mean')`, and
    `faithfulness()`/`completeness()` ((m(C)−m(∅))/(m(M)−m(∅)), Marks §3.2 — can exceed [0,1] under
    anti-correlation, documented). Ablation is **NODE-set** (downstream features share a residual, so
    edge-level ablation is unimplementable): non-circuit feature entries are replaced before decode.
  - **`FeatureACDCRunner`** — greedy reverse-topological **node** pruning (accept a removal if
    `KL_new − KL_current < tau`; `sweep(taus)` Pareto helper; `eap_skip_threshold`). Reuses
    `ACDCRunner`'s control flow + `core.patching.kl_divergence`; the feature-basis set-ablation
    forward is net-new.

### Notes
- `graph.py` / `EdgeGraph` / `_order` / `build_graph` and the v1.5 `SAEFeatureRunner` are **untouched**
  — the feature circuit uses a dedicated `SAEFeatureEdgeGraph`. Scope inherited from v1.5: HF-eager +
  `resid_post` only; `TLSiteResolver` / non-`resid_post` / transcoder|matching_pursuit|temporal SAEs
  raise `NotImplementedError`; architectures standard/topk/jumprelu/gated.
- Deferred follow-ons: the integrated-gradients edge variant (`variant` enum slot reserved),
  per-position edges, `mlp_out`/`attn_out` sites, and the TransformerLens backend.
- Designed and adversarially red-green reviewed via multi-agent workflows (the review caught and fixed
  an error-edge honesty gap + 5 toothless tests before release).

## [1.5.0] — 2026-05-30

### Added
- **Node-level SAE-feature attribution** — `circuitry.patching.SAEFeatureRunner`. Given a
  clean/corrupted prompt pair, a differentiable metric, and `sae_sites` mapping
  `Site(component="resid_post", layer=L)` → a SAELens `SAE` (or `(release, sae_id)` tuple),
  `run()` returns an `AtPResult` of per-**SAE-feature** attribution scores.
  - **Mechanism — error-term substitution** (Marks "Sparse Feature Circuits"): at each site on
    the clean pass, `f = sae.encode(a)`, `x_hat = sae.decode(f)`, `eps = (a − x_hat).detach()`,
    and the residual is replaced by `recon = x_hat + eps` (numerically lossless, model output
    preserved) so each feature `f_i` becomes a differentiable node.
  - **Scoring** mirrors `mlp_neuron` in the SAE basis: `score(i) = Σ_pos(Δf_i ⊙ gradf_i)`,
    `Δf = f_corrupt − f_clean`, gradient at the clean activation. `graddrop=True` →
    `Σ|per-position contribution|`. On a linear-downstream model the analytic score equals
    brute-force feature patching at 1e-4 (exactness needs *downstream* linearity, not SAE
    linearity). `bruteforce_feature_scores()` is the independent ground-truth path.
  - **Enumeration:** features where `Δf ≠ 0` (the union of clean-active and corrupted-active) —
    a clean-inactive/corrupted-active feature has a real nonzero effect; omitted features score 0.
    Optional `max_features` cap keeps the top-|score| features per site.
  - **Reconstruction-error node**, opt-in via `include_error_node=True` (default off): scores the
    residual error as a first-class `sae_error` node using an independent error leaf, so feature
    and error gradients come from one forward + backward (feature scores are identical with the
    node on or off).
- **Differentiable SAE helpers** — `circuitry.sae.encode_features` / `decode_features` /
  `sae_decompose` / `assert_supported_sae` (`src/circuitry/sae/grad.py`). These call
  `sae.encode` / `sae.decode` under normal autograd (no `inference_mode` / `detach`), unlike the
  `sae_reconstruction_error` reporting path, which is unchanged.
- New `Node` kinds `"sae_feature"` (reuses the `neuron` field as the feature index) and
  `"sae_error"` in `circuitry.patching.graph`.

### Notes
- Scope: HF-eager backend + `resid_post` sites only. `TLSiteResolver`, non-`resid_post` sites,
  and transcoder / matching_pursuit / temporal SAEs raise a clear `NotImplementedError`.
  Supported architectures: `standard` / `topk` / `jumprelu` (covers BatchTopK / Matryoshka at
  inference) / `gated`. The SAE's own parameters are frozen + restored for the run.
- The metric must be the differentiable tensor-returning variant (`logit_diff_t` /
  `kl_divergence_t` / `ce_loss_t`); a `.detach()`-ing float metric raises `f.grad is None`.
- Feature→feature **edges** (the full sparse-feature graph), the TransformerLens backend,
  `mlp_out` / `attn_out` sites, and the integrated-gradients variant remain future work.

## [1.4.2] — 2026-05-30

### Changed
- `scripts/bench_50m.py` gained `--batch-size` / `--seq-len` (default 4 / 64, unchanged) so the
  §10 overhead can be measured at a realistic training step rather than the tiny default.

### Documentation
- **§10 performance budget validated on GPU.** First GPU measurement (RTX 5080, 88M-param
  decoder, full `llm` recipe, `every_n_steps=200`): **+5.3% at a realistic training step**
  (batch 16 × seq 512) — within the ≤10% budget. The overhead ratio is dominated by the
  roughly-fixed per-emit diagnostic cost (SVD set + logit-lens + induction-score), so it is
  highly sensitive to baseline step weight: the tiny default batch (4 × 64, ~12 ms/step on GPU)
  inflates it to +45%. `docs/design.md` §10 and the README Performance section now report the
  measured numbers and the small-/fast-step caveat instead of "GPU validation pending".

## [1.4.1] — 2026-05-30

### Fixed
- **GPU crash in the cross-step weight-dynamics diagnostics** (regression since v1.3.0).
  `weight.update_delta` and `weight.direction_cosine` subtracted the current weights
  against the prior snapshot without aligning devices. In the live `Recorder` the current
  weights are on the model's device (e.g. CUDA) while `_prev_weights` is a CPU copy, so on
  GPU the stock `llm` recipe crashed at the second emit step with "Expected all tensors to
  be on the same device". The primitives now move the prior snapshot onto the current
  tensor's device before the delta. CPU-only runs were unaffected (which is why the
  CPU test suite did not catch it); added CUDA-gated regression tests.

## [1.4.0] — 2026-05-30

### Added
- **`activation.repr_drift` primitive** (`core/activation.py`). Pure function returning
  representational drift in `[0, 1]` (0 = identical, 1 = fully drifted) between two
  activation snapshots. Three configurable methods: `"linear_cka"` (default — invariant
  to orthogonal rotation and isotropic rescaling; requires ≥2 rows), `"cosine"` (mean
  per-sample cosine distance; works with any row count), `"rbf_cka"` (RBF-kernel CKA
  with median-heuristic bandwidth; requires ≥2 rows). Row count is capped at `max_samples`
  (default 256) with a **seeded, CPU-deterministic** subsample before Gram computation.
  Re-exported as `circuitry.repr_drift` in the public surface.
- **`Recipe.probe_batch` / `drift_method` / `drift_max_tokens`** fields on `Recipe`.
  Pass `probe_batch` (a fixed representative input tensor) to enable the opt-in drift probe.
  The `llm` recipe lists `"drift_probe"` in `activation_diagnostics` with
  `enabled={"drift_probe": False}` (default OFF).
- **Drift probe in `Recorder`** (opt-in, default off). When `probe_batch` is set and
  `"drift_probe"` is enabled: at each emit step the Recorder runs a second forward pass
  on `probe_batch` (frozen model, no grad). On the first emit step the captured
  per-module activations are stored as a detached CPU reference snapshot. Subsequent
  steps call `activation.repr_drift` to compare the live probe to the reference and emit
  `activation/repr_drift/<module>` scalars. The reference is cleared in `detach()`.
- **`Recorder.reset_drift_reference()`** — discards the current drift reference and
  re-anchors at the next emit step.
- **`weight.singular_values` `seed` kwarg** — seeds the random column subsample for
  matrices wider than `max_dim=512`. Ensures cross-step SVD subsamples are identical,
  making `effective_rank` / `stable_rank` / `sv_histogram` / `heavy_tail_alpha`
  numerically comparable across emit steps. The `Recorder` passes a fixed
  `_SUBSAMPLE_SEED` constant.
- **`weight.singular_values` `use_gram` fast path** — `use_gram='auto'` (default) uses
  `eigvalsh(W^T W)` for strongly-rectangular matrices (fewer columns than rows), reducing
  SVD cost. `condition_number` always uses the full SVD path (`use_gram=False`) to
  preserve the exact max/min singular-value ratio.
- **ACDC `ablation_mode` kwarg** on `ACDCRunner.run()` and `.sweep()`. `"corrupted"`
  (default) feeds cached corrupted-run activations (matching the original paper).
  `"zero"` injects zeros. `"mean"` injects each writer's corrupted activation averaged
  over the batch/sequence positions (a spatially-constant per-feature mean).
- **ACDC `eap_skip_threshold` kwarg** on `ACDCRunner.run()` and `.sweep()`. When
  `eap_scores` are provided and a float threshold is given, edges whose `|EAP score|`
  **exceeds** the threshold are assumed important and kept **without** running their
  ablation test (skipping that forward pass), accelerating circuit discovery on large
  graphs. `None` (default) tests every edge. This is the EAP-score skip speedup
  documented as a v1.0 follow-on.

### Changed
- `circuitry.__version__` bumped to `"1.4.0"`.
- `circuitry.repr_drift` added to `__all__` and the public export surface
  (alongside `token_similarity`, `update_delta`, `direction_cosine`).

### Fixed
- **Unseeded randperm in `weight.singular_values`** violated the CPU-deterministic
  invariant (design.md §4.1 Tier 1 contract). The column subsample drawn for matrices
  wider than `max_dim=512` used an unseeded `torch.randperm`, so consecutive emit steps
  could sample different column subsets — making `effective_rank` / `stable_rank` /
  `heavy_tail_alpha` values incomparable across steps. Fixed by the new `seed` kwarg;
  the `Recorder` now passes a fixed seed for all SVD-derived diagnostics.

### Documentation
- `docs/design.md §4.1`: added `activation.repr_drift` to the primitive catalog
  (signature, three methods, row-count constraint, tag name). Updated
  `weight.singular_values` signature to show `seed` and `use_gram` kwargs with notes.
  All references to the earlier working name `repr_drift_cka` are replaced with
  `repr_drift` (the shipped name).
- `docs/design.md §4.4`: documented `Recipe.probe_batch` / `drift_method` /
  `drift_max_tokens`, the `"drift_probe"` default-OFF gate, the second-forward-pass
  probe, the first-emit reference anchor, `reset_drift_reference()`, reference lifecycle
  (detached CPU copy; cleared in `detach()`). Documented ACDC `ablation_mode` and
  `eap_skip_threshold`.
- `docs/design.md §10`: updated performance section with honest framing — ≤10% is the
  design budget/target; the only on-record measurements are CPU (+14.9% / +14.7% at
  v0.2.0a0), which exceed the budget and are CPU-inflated relative to the GPU scenario
  the budget targets; GPU re-validation (A3) is still pending. Notes the Gram fast path
  and drift probe scope.

## [1.3.0] — 2026-05-30

### Added
- Cross-step weight snapshot holder in `Recorder` (`_prev_weights`, `_prev_prev_weights`):
  detached CPU copies of matched-module weights from prior emit steps. Populated after each
  emit; cleared in `detach()`. First-step and second-step guards suppress cross-step
  diagnostics until sufficient history exists.
- `weight/update_delta/<module>` live diagnostic: L2 norm of per-module weight delta between
  consecutive emit steps. Wires the existing `core/weight.update_delta` primitive.
- `weight/direction_cosine/<module>` live diagnostic: cosine similarity between two
  consecutive parameter update vectors. Wires `core/weight.direction_cosine`. Emits from
  the third emit step onward (requires two prior snapshots).
- `weight/rank_trajectory/<module>` live diagnostic: effective rank of each weight matrix at
  each emit step, reusing the per-step SVD cache (zero extra SVD cost). Wires the logic of
  `core/spectral.rank_trajectory` inline via the SVD cache.
- `llm` recipe now includes `update_delta`, `rank_trajectory`, `direction_cosine` in
  `weight_diagnostics`.
- `build_report`: `HERO_SECTIONS` extended with `weight/update_delta`,
  `weight/rank_trajectory`, `weight/direction_cosine` (rendered above `<details>`).
- `build_report`: three new `FLAG_RULES` — `rank_collapse_trend` (declining rank_trajectory),
  `update_delta_vanishing` (near-zero update norm), `direction_reversal` (strongly negative
  cosine).

### Documentation
- `docs/design.md §4.1`: added `update_delta` and `direction_cosine` to the public primitive
  catalog; added note that `rank_trajectory` is now wired live.
- `docs/design.md §4.4`: added prose on the cross-step snapshot lifecycle.
- `docs/design.md §5`: updated llm recipe example `weight_diagnostics` list.

### Deferred (Option B — representational drift probe)
- A probe-based representational drift primitive (CKA / cosine of activations vs a stored
  reference) is explicitly deferred to v1.3.1 or v1.4. It requires a second forward pass
  per emit step and must be opt-in (default off) due to the §10 GPU budget constraint.
  The cross-step snapshot infrastructure shipped in this release is the foundation it reuses.

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
