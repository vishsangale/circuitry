# circuitry — design spec

**Last updated:** 2026-06-09
**Status:** as-implemented (living document; tracks shipped releases — see [`CHANGELOG.md`](../CHANGELOG.md))
**Owner:** Vishwanath Sangale

`circuitry` is a standalone Python library providing mechanistic-interpretability diagnostics — weight / activation / gradient / spectral primitives, plus a `Recorder` workflow for live training-time capture and a `scan` workflow for post-hoc analysis on saved checkpoints. Modality-agnostic core with per-modality recipes for LLM, vision, and recsys.

This document is the design contract.

## 1. Motivation

A 2026 survey of the field (TransformerLens, nnsight, captum, pyvene, SAELens, tuned-lens, pyhessian, NetDissect) shows the existing ecosystem covers post-hoc analysis well but does **not** unify (a) live training-time hooks, (b) spectral / rank / weight diagnostics, and (c) recsys + vision + LLMs under one API. That is the niche `circuitry` targets.

The library bundles primitives that get re-implemented project-by-project (effective-rank, stable-rank, heavy-tail alpha, ESD, dead-fraction, kurtosis, participation ratio, gradient norms per param, layer-wise signal propagation) behind a single `Recorder` so that adding diagnostics to a new training run is a three-line change, not a refactor.

### Naming clarity

`circuitry` is statistical diagnostics on weights / activations / gradients, usable live during training or post-hoc on saved checkpoints. The statistical core remains modality-agnostic; an opt-in interventional **activation-patching / attribution** pillar (EAP, AtP\*, ACDC — §4.6), SAE-reconstruction metrics (`circuitry.sae`), node-level **SAE-feature attribution** (`SAEFeatureRunner`, v1.5), **feature→feature SAE circuits** (`SAEFeatureEdgeRunner` + `FeatureACDCRunner`, v1.6), and **generalized SAE circuits** over `mlp_out`/`attn_out` sites, TransformerLens backend, and integrated-gradients variant (`variant='ig'`, v1.7) have since shipped. The **tuned lens** (Belrose et al. 2023) shipped in v1.10 (`circuitry.tuned_lens` — post-hoc `fit_tuned_lens` + the opt-in `tuned_lens_kl` diagnostic; §4.1, §4.4). **v1.11 adds `copy_suppression_score`** — the per-head same-token attention metric (McDougall et al. 2023) that identifies copy-suppression heads on the repeated-token probe (complement of `induction_score`; §4.1). **v1.12 adds `attention_sink_score`** — per-head mean attention weight on the initial token (BOS / position 0), the signature of attention sink heads (Xiao et al. 2023); operates on the live training-forward attention rather than a probe (§4.1). The name is borrowed from electronics, not from interpretability research. The README MUST open with a one-line scope statement so users arriving from mechanistic-interpretability work understand what this is and where it's heading.

### Non-goals

- We do **not** rebuild SAE training (SAELens does this well). SAE *reconstruction* metrics shipped in v0.9 (`circuitry.sae`); training is still out of scope.
- We do **not** attempt to be a complete post-hoc interp framework. We focus on monitoring + diagnostics.
- Causal interventions / activation patching shipped as `circuitry.patching` in v1.0 (§4.6) and are now in scope.

## 2. Decisions locked in brainstorming

| Decision | Choice |
| --- | --- |
| Library name | `circuitry` |
| License | MIT |
| Release target | Public open-source, low-key (clean README, no docs site, not on PyPI for the 0.x line) |
| Repo location | `~/workspace/circuitry/`, public GitHub `vishsangale/circuitry` |
| In-scope | Checkpoint inspector (live + scan + report); spectral / rank / weight / activation / gradient primitives; per-modality recipes (LLM / vision / two-tower / sequential-recsys) |
| Out-of-scope | Architecture-specific diagnostics (those live in consumer codebases via custom `Recipe`s) |
| API shape | Two layers: pure primitives in `core/` + thin opinionated `Recorder` workflow above |
| Modality strategy | Modality-agnostic core + per-modality recipes (`recipes/llm.py`, `recipes/vision.py`, `recipes/two_tower.py`, `recipes/recsys.py`) |
| Framework support | PyTorch only, single-process (rank-0 only in DDP runs; multi-process path in §11) |
| Logging | TensorBoard primary, `MetricWriter` Protocol so jsonl / null (and any user-supplied) adapters are 1-file each |

## 3. Repository structure

```
~/workspace/circuitry/
├── pyproject.toml          # PEP 621
├── README.md
├── LICENSE                 # MIT
├── CHANGELOG.md
├── src/circuitry/
│   ├── __init__.py
│   ├── core/               # pure primitives — no torch.nn assumptions, no I/O
│   │   ├── weight.py
│   │   ├── activation.py
│   │   ├── gradient.py
│   │   ├── spectral.py
│   │   ├── dynamics.py     # phase_transition_steps, head_formation_step, grokking_step (v1.14); fourier_feature_alignment, information_bottleneck_score (v1.27)
│   │   ├── lens.py         # logit_lens_kl, tuned_lens_kl; logit_lens_distributions, future_lens_kl (v1.23–v1.24)
│   │   ├── steer.py        # steer_vector (v1.23); repe_direction, directional_ablation (v1.29)
│   │   ├── probe.py        # train_linear_probe / LinearProbe (v1.24); mdl_probe/MDLResult, mass_mean_probe/MassMeanProbe, verify_linear_representation (v1.29)
│   │   ├── erase.py        # leace_erase / EraseProjection — orthogonal concept erasure (v1.24)
│   │   └── attention.py    # induction_score, copy_suppression_score, attention_sink_score, attention_pattern_entropy, head_specialization
│   ├── sae/                # v0.9: SAELens-backed SAE workflow
│   │   ├── loader.py       # load_sae; load_gemma_scope, load_llama_scope (v1.26)
│   │   └── metrics.py      # sae_reconstruction_error; superposition_index (v1.29)
│   ├── benchmarks/         # v1.27: synthetic MIB tasks + SAEBench metrics
│   │   ├── __init__.py
│   │   ├── mib.py          # MIBTask, load_ioi, load_greater_than (Mueller et al. ICML 2025)
│   │   └── saebench.py     # SAEBenchResult, run_saebench (Karvonen et al. 2025)
│   ├── patching/           # v1.0: activation patching (interventional)
│   │   ├── sites.py        # Site dataclass + HF/TL resolution
│   │   ├── intervene.py    # patch_site() context manager
│   │   ├── runner.py       # PatchRunner prompt-pair runner
│   │   ├── steer.py        # apply_steer (v1.23); apply_ablation (v1.29)
│   │   ├── edge_pruning.py # EdgePruningRunner / EdgePruningResult — mask-logit L0 pruning (v1.25)
│   │   ├── hap.py          # HAPRunner — EAP pre-filter + EdgePruningRunner (v1.25)
│   │   ├── das.py          # DASRunner / DASResult — interchange-intervention rotation learning (v1.28)
│   │   └── scrubbing.py    # CausalScrubRunner / CircuitHypothesis / CausalScrubResult (v1.28)
│   ├── recorder/           # opinionated training-time workflow
│   │   ├── live.py         # LiveRecorder
│   │   ├── scan.py         # scan_run
│   │   ├── report.py       # build_report
│   │   └── hooks.py        # HookPoint, TensorSource, module-type → hook strategy
│   ├── recipes/
│   │   ├── llm.py
│   │   ├── vision.py
│   │   ├── two_tower.py    # two-tower + DLRM retrieval
│   │   └── recsys.py       # sequential recsys (SASRec / BERT4Rec / GRU4Rec)
│   ├── writers/
│   │   ├── base.py         # MetricWriter Protocol
│   │   ├── tensorboard.py
│   │   ├── jsonl.py
│   │   └── null.py
│   └── cli/
│       ├── __init__.py
│       └── main.py         # circuitry scan / report / compare / list-recipes / fit-tuned-lens
├── tests/
│   ├── core/
│   ├── recorder/
│   ├── recipes/
│   └── e2e/
├── examples/
│   ├── tiny_llm.py
│   ├── tiny_vision.py
│   └── tiny_two_tower.py
└── docs/                   # markdown notes only
```

### Layering rules (enforced in CI)

- `core/` MUST NOT import from `recorder/`, `recipes/`, `writers/`, `cli/`, or `patching/`.
- `recipes/` MUST NOT import from `cli/`.
- `patching/` may import from `core/` and `recipes/`; MUST NOT import from `cli/`.
- `benchmarks/` may import from `core/` and `sae/`; MUST NOT import from `patching/`, `recorder/`, `recipes/`, or `cli/`.
- The package MUST NOT import from any downstream user codebase. `circuitry` is the consumed dependency, never the consumer.
- `transformer_lens` and `transformers` are approved optional dependencies (lazy import only; `circuitry` must install and run without them). `transformers` is imported lazily by the AtP\* QK fix (HF Llama RoPE recomputation, eager-only); `transformer_lens` by the optional TL backends, including `patching/tl_bridge.py` (imported inside the function body; the `test_layering` allowlist is unchanged).

A simple `import-linter` config or hand-rolled AST test enforces this.

## 4. Public API

### 4.1 Tier 1 — primitives (`circuitry.core.*`)

Pure functions. Tensors / state-dicts in; floats / arrays / small dataclasses out. No hooks, no logging, no side effects.

```python
from circuitry.core import weight, activation, gradient, spectral

# weight-space
weight.effective_rank(W: Tensor, eps: float = 1e-12, *, max_dim: int | None = None, seed: int | None = None) -> float
weight.stable_rank(W: Tensor, *, max_dim: int | None = None, seed: int | None = None) -> float
weight.condition_number(W: Tensor, eps: float = 1e-12, *, max_dim: int | None = None) -> float
weight.singular_values(W: Tensor, k: int | None = None, max_dim: int | None = None, *, seed: int | None = None, use_gram: bool | str = 'auto') -> Tensor
# ACCURATE BY DEFAULT (v1.8): max_dim=None computes the full, deterministic SVD, so every SVD-derived
#   diagnostic is accurate and reproducible on any matrix (every real LLM layer has min-dim > 512).
# max_dim: opt-in PERFORMANCE hatch — caps SVD cost on wide matrices by truncating to a max_dim-column
#   random subsample, which biases sigma_min / the spectral tail and (unless seed is set) is non-deterministic.
# seed: seeds the subsample draw (only relevant when max_dim triggers subsampling) for CPU-deterministic results.
# use_gram: 'auto' uses eigvalsh(W^T W) fast path for strongly-rectangular matrices; False forces svdvals.
# condition_number always uses the full SVD path (use_gram=False) to preserve the exact max/min ratio.
# ≤2-D CONTRACT (v1.8): effective_rank/stable_rank/heavy_tail_alpha/condition_number RAISE on ndim>2 — a
#   batched tensor (e.g. an MoE expert stack [n_experts, d_in, d_out]) must be folded/iterated per-slice by
#   the caller, never silently flattened into rows (which yields ~n_experts instead of the per-expert rank).
weight.heavy_tail_alpha(W: Tensor, top_frac: float = 0.5, *, max_dim: int | None = None, seed: int | None = None) -> float
weight.spectral_entropy(W: Tensor, eps: float = 1e-12, *, max_dim: int | None = None, seed: int | None = None) -> float  # Shannon entropy (nats) of the normalized singular-value distribution = log(effective_rank); the sv_histogram diagnostic emits it (+ sv_max / sv_min) as companion scalars so the spectrum shows up in scalar/CSV exports
weight.attention_head_rank(W: Tensor, n_heads: int, head_dim: int, axis: int = 0) -> Tensor

# weight-space dynamics (v1.3 — shipped in core/; now wired live)
weight.update_delta(sd_now: Mapping[str, Tensor], sd_prev: Mapping[str, Tensor]) -> dict[str, float]
weight.relative_update_delta(sd_now: Mapping[str, Tensor], sd_prev: Mapping[str, Tensor], eps: float = 1e-12) -> dict[str, float]  # (v1.10) scale-invariant ||ΔW||/||W||; the recorder emits weight/update_delta_rel/* alongside update_delta and the update_delta_vanishing flag keys on it
weight.direction_cosine(sd_now: Mapping[str, Tensor], sd_prev: Mapping[str, Tensor],
                        sd_prev_prev: Mapping[str, Tensor]) -> dict[str, float]

# activation-space
activation.dead_fraction(x: Tensor, threshold: float = 0.0) -> float
activation.kurtosis(x: Tensor, dim: int | tuple = -1) -> Tensor
activation.participation_ratio(x: Tensor) -> float
activation.norm_stats(x: Tensor) -> NormStats   # mean, std, max, frac>k*median
activation.gate_stats(x: Tensor, eps: float = 1e-6) -> dict[str, float]  # frac_active, mean_abs, std (returns plain dict; keys: "frac_active", "mean_abs", "std")
activation.repr_drift(ref: Tensor, cur: Tensor, method: str = 'linear_cka', *,
                      max_samples: int = 256, eps: float = 1e-10, seed: int = 0) -> float
# Representational drift between two activation snapshots. Returns a float in [0, 1] where 0 means
# identical representation and larger values indicate more drift.  Three configurable methods:
#   "linear_cka" (default) — invariant to orthogonal rotation and isotropic rescaling; CKA requires >= 2 rows.
#   "cosine"               — mean per-sample cosine distance; O(n d), not scale-invariant.
#   "rbf_cka"              — RBF-kernel CKA with median-heuristic bandwidth; nonlinear; CKA requires >= 2 rows.
# Rows are subsampled (seeded, CPU-deterministic) to max_samples before Gram computation.
# Recorder emits per-layer tags: activation/repr_drift/<module>.

# representation geometry (v1.29)
activation.local_intrinsic_dim(acts: Tensor, *, max_samples: int = 2048, seed: int = 0) -> float  # Two-NN manifold dimensionality estimator (Levina & Bickel 2004); subsample → cdist → topk(2) → 1/mean(log(d2/d1)); requires >= 3 samples
activation.kernel_alignment(acts_a: Tensor, acts_b: Tensor, *, method: str = 'cka', max_samples: int = 256, seed: int = 0) -> float  # cross-model alignment: 'cka' reuses repr_drift CKA (score = 1 - drift); 'mnn' = mutual nearest-neighbour overlap; both ∈ [0, 1] where 1 = identical; Huh et al. ICML 2024
activation.embedding_uniformity(E: Tensor, *, n_samples: int = 2048, seed: int = 0) -> float  # mean off-diagonal cosine similarity after L2 normalisation; ≈ 1 = collapsed, ≈ 0 = spread / uniform; Guo et al. ICML 2024

# gradient-space
gradient.grad_norm_per_module(grads: dict[str, Tensor]) -> dict[str, float]
gradient.total_grad_norm(per_module_norms: dict[str, float]) -> float  # sqrt(sum of squares)
gradient.signal_propagation_depth(grads_by_depth: list[Tensor]) -> int

# spectral
spectral.esd(W: Tensor, bins: int = 100) -> tuple[Tensor, Tensor]
spectral.rank_trajectory(state_dicts: list[dict]) -> dict[str, list[float]]
# Note: rank_trajectory is now wired live in the Recorder (v1.3) via the SVD cache.

# lens (v0.9 logit lens; v1.10 tuned lens; v1.23–v1.24 distributions + future lens)
from circuitry.core import lens
lens.logit_lens_kl(residual: Tensor, unembed: Tensor, final_logits: Tensor, *, layer_norm=None, chunk_size: int = 256) -> float  # chunk_size bounds the (tokens, vocab) transient (v0.9.2)
lens.tuned_lens_kl(residual: Tensor, translator: tuple[Tensor, Tensor], unembed: Tensor, final_logits: Tensor, *, layer_norm=None, chunk_size: int = 256) -> float  # (v1.10) apply learned affine A·h+b before unembed; A=I,b=0 reduces to logit_lens_kl
lens.logit_lens_distributions(residuals: dict[int, Tensor] | list[Tensor], unembed: Tensor, *, layer_norm=None, top_k: int = 5) -> list[LayerPrediction]  # (v1.23) per-layer top-k token predictions; LayerPrediction.layer_idx / .token_ids / .probs
lens.future_lens_kl(residual: Tensor, unembed: Tensor, target_logits: Tensor, *, horizon: int = 1, layer_norm=None, chunk_size: int = 256) -> float  # (v1.24) compares residual@layer to target_logits[t+horizon]; returns 0.0 when horizon ≥ seq; reduces to logit_lens_kl at horizon=0

# attention screening (v0.9 / v1.11)
from circuitry.core import attention
attention.induction_score(attn_pattern: Tensor, *, seq_len_repeat: int) -> list[float]
attention.copy_suppression_score(attn_pattern: Tensor, *, seq_len_repeat: int) -> list[float]  # (v1.11) per-head same-token attention on the repeated-token probe: at position T+i, how much does the head attend back to position i (prior occurrence of the same token)? Complement of induction_score (offset 0 vs +1). High score = copy-suppression head (McDougall et al. 2023). Emitted as activation/copy_suppression_score/<module>/head_N; flagged in the report when last > 0.3.
attention.attention_sink_score(attn_pattern: Tensor, *, sink_pos: int = 0) -> list[float]  # (v1.12) per-head mean attention weight on a designated sink position (default 0 / BOS). Operates on the live training-forward attention (not the probe), so it reflects the real training distribution. High score = head is a likely attention sink (Xiao et al. 2023). Emitted as activation/attention_sink_score/<module>/head_N; flagged in the report when last > 0.5.
attention.attention_pattern_entropy(attn_pattern: Tensor, *, valid_mask=None) -> list[float]  # normalizes each query row before entropy → comparable across attention variants (v0.9.2); per-head mean is NaN-aware (drops fully-(-inf)-masked PAD rows) and valid_mask (True=valid query row, broadcastable to (B,H,T)) restricts the average — left-padded recsys models no longer return NaN
attention.head_specialization(induction: list[float], copy_suppression: list[float], sink: list[float], *, induction_threshold: float = 0.4, copy_suppression_threshold: float = 0.3, sink_threshold: float = 0.5) -> list[str]  # (v1.13) classify each head as "induction" / "copy_suppression" / "sink" / "uniform"; tie-break = highest score/threshold ratio. Synthesised in the report as ## Head Specialization.

# training dynamics (v1.14)
from circuitry.core import dynamics
dynamics.phase_transition_steps(series: list[tuple[int, float]], *, window: int = 5, z_threshold: float = 2.0, min_gap: int = 5) -> list[int]  # bilateral-mean change detector: returns training steps where the time-series has a statistically sharp change (bilateral diff > mean + z_threshold * std). Monotone ramps return []. Nearby detections within min_gap index positions are collapsed. Used in the report ## Training Dynamics / Phase Transitions table.
dynamics.head_formation_step(series: list[tuple[int, float]], *, threshold: float, n_sustain: int = 2) -> int | None  # first training step at which a per-head attention score crosses threshold and sustains for n_sustain consecutive recorded points. Returns None if the threshold is never crossed. Used in the report ## Training Dynamics / Head Formation Events table.
dynamics.grokking_step(series: list[tuple[int, float]], *, z_threshold: float = 2.5) -> int | None  # (v1.15) thin wrapper around phase_transition_steps — returns the FIRST detected sharp-transition step (lowest z_threshold default for loss/accuracy series), or None if none detected. Used in the report ## Training Dynamics / Grokking Signals sub-table.
dynamics.fourier_feature_alignment(W: Tensor, task_freqs: Tensor, *, n_freqs: int | None = None) -> float  # (v1.27) fraction of spectral power at task_freqs via rfft; Nanda et al. ICLR 2024
dynamics.information_bottleneck_score(acts_train: Tensor, acts_val: Tensor, labels_train: Tensor, labels_val: Tensor, *, n_bins: int = 20, eps: float = 1e-10) -> float  # (v1.27) MI proxy I(T;Y)/H(Y) via binned histogram on first PC; Nanda et al. ICLR 2024

# activation steering (v1.23 core direction; v1.23 patching context manager; v1.29 ablation)
from circuitry.core import steer
steer.steer_vector(positive_acts: Tensor, negative_acts: Tensor, *, normalize: bool = True) -> Tensor  # Rimsky et al. 2024 CAA: mean(positive) − mean(negative), optionally unit-normalised; raises ValueError on near-zero norm
steer.repe_direction(diffs: Tensor) -> Tensor  # (v1.29) Zou et al. 2023 RepE: first PC of centred (n_pairs, d_model) difference matrix; unit-normalised; single-sample falls back to uncentered d[0]; zero matrix returns zeros(d_model)
steer.directional_ablation(acts: Tensor, direction: Tensor) -> Tensor  # (v1.29) Arditi et al. NeurIPS 2024: acts − (acts · d̂) d̂; orthogonal-complement projection; no-op for near-zero direction

from circuitry.patching import steer as patching_steer
patching_steer.apply_steer(model, site: Site, vector: Tensor, *, coeff: float = 1.0, resolver=None)  # context manager; adds coeff*vector to site output; hook always removed on exit
patching_steer.apply_ablation(model, site: Site, direction: Tensor, *, resolver=None)  # (v1.29) context manager; removes direction component from site output; hook always removed on exit; Arditi et al. NeurIPS 2024

# linear probing (v1.24)
from circuitry.core import probe
probe.train_linear_probe(acts: Tensor, labels: Tensor, *, max_iter: int = 1000, C: float = 1.0, tol: float = 1e-4, device=None) -> LinearProbe
# LinearProbe: .weight (Tensor), .bias (Tensor), .classes (list), .predict(acts) → Tensor,
#   .predict_proba(acts) → Tensor, .accuracy(acts, labels) → float, .direction() → Tensor
# direction() = unit concept vector: normalised weight[0] for binary, first left singular of
#   between-class weight matrix for multiclass (pca_lowrank)

# MDL probing (v1.29)
probe.mdl_probe(acts: Tensor, labels: Tensor, *, n_chunks: int = 8) -> MDLResult  # Voita & Titov 2020 online-coding MDL; splits into n_chunks; chunk 0 uses log(n_classes) uniform prior; mdl_ratio < 1 = genuine encoding
# MDLResult(frozen): .code_length, .data_entropy, .mdl_ratio (= code_length / (n * data_entropy))

# mass-mean probe (v1.29)
probe.mass_mean_probe(acts: Tensor, labels: Tensor) -> MassMeanProbe  # Marks & Tegmark COLM 2024; binary only; direction = normalised (μ₁ − μ₀) on CPU; threshold = midpoint
# MassMeanProbe: .direction (Tensor, CPU), .threshold (float), .classes (list), .predict(acts), .accuracy(acts, labels)

# linear representation verification (v1.29)
probe.verify_linear_representation(probe, steer_vec: Tensor) -> float  # Park et al. arXiv:2311.03658; cosine(probe.direction(), steer_vec); handles dimension mismatch by truncation

# concept erasure / LEACE (v1.24) / RLACE (v1.31)
from circuitry.core import erase
erase.leace_erase(acts: Tensor, labels: Tensor) -> EraseProjection  # Park et al. 2023 LEACE
# binary: d = μ₁−μ₀ normalised, P = I − d_hat d_hat^T
# multiclass: first right singular vector of between-class mean matrix
# EraseProjection: .P (Tensor), .direction (Tensor), .apply(acts) → Tensor (broadcast to (..., d_model))
erase.rlace_erase(acts: Tensor, labels: Tensor, *, rank: int = 1) -> EraseProjection  # (v1.31) Ravfogel et al. ICML 2022 RLACE
# rank-k adversarial concept erasure: P = I − U Uᵀ where U spans top-rank eigenvectors of
# between-class scatter B = M_c^T M_c; rank=1 recovers the LEACE direction;
# .direction = first erased direction (leading eigenvector)

# SAE workflow (v0.9)
from circuitry import sae
sae.load_sae(release: str, sae_id: str, device: str = "cpu")
sae.sae_reconstruction_error(x: Tensor, sae) -> dict[str, float]
sae.metrics.superposition_index(feature_acts: Tensor) -> float  # (v1.29) exp(H(|feature_acts|.flatten())); >> n_neurons → superposition; 1.0 if all zero or single feature active; arXiv:2512.13568
sae.metrics.UNRELIABLE_METRICS: frozenset  # (v1.31) frozenset{"tpp", "scr"} — metrics whose estimates have high variance on standard SAE benchmarks
sae.metrics.warn_if_unreliable(metric_name: str)  # (v1.31) emits UserWarning if metric_name ∈ UNRELIABLE_METRICS
sae.metrics.sae_downstream_loss(sae, model, tokens, *, site: Site, resolver=None) -> dict[str, float]  # (v1.31) KL-faithfulness: run model clean + with SAE hook; returns {"kl_divergence", "ce_delta", "l0"}
sae.grad.sae_influence_scores(sae, x: Tensor, loss_fn: Callable[[Tensor], Tensor]) -> Tensor  # (v1.31) GradSAE |∂loss/∂f_i|·|f_i|; mean over batch/positions; arXiv:2505.08080
sae.steer.fgaa_steering_vector(sae, positive_acts: Tensor, negative_acts: Tensor, *, n_features: int = 10) -> Tensor  # (v1.31) FGAA steering: top-n discriminative decoder columns weighted by signed mean diff; (d_model,) CPU float32; arXiv:2501.09929

# SAE feature attribution — differentiable encode/decode helpers (v1.5)
# These wrap sae.encode / sae.decode under NORMAL autograd (no inference_mode/detach),
# unlike sae_reconstruction_error (the reporting path). sae/metrics.py is unchanged.
sae.encode_features(sae, x: Tensor) -> Tensor
sae.decode_features(sae, f: Tensor) -> Tensor
sae.sae_decompose(sae, x: Tensor) -> tuple[Tensor, Tensor, Tensor]   # (f, x_hat, eps=(x-x_hat).detach())
sae.assert_supported_sae(sae) -> None                                # gate: standard/topk/jumprelu/gated/matryoshka/batch_topk/p_anneal/hierarchical_topk (v1.31); raw crosscoder blocked — use CrosscoderWrapper
sae.load_gemma_scope(model_size: str, layer: int, width: int, *, site: str = "res", average_l0=None, device: str = "cpu")  # (v1.26) convenience wrapper for Gemma Scope JumpReLU SAEs (Lieberum et al. arxiv:2408.05147)
sae.load_llama_scope(layer: int, width: int, *, device: str = "cpu")  # (v1.26) convenience wrapper for Llama Scope JumpReLU SAEs

# patching metrics (v1.0)
from circuitry.core import patching
patching.logit_diff(logits: Tensor, correct: int, incorrect: int) -> float
patching.kl_divergence(p_logits: Tensor, q_logits: Tensor, *, chunk_size: int = 256) -> float
patching.ce_loss(logits: Tensor, targets: Tensor) -> float

# Benchmarks (v1.27)
from circuitry import benchmarks
benchmarks.load_ioi(n: int = 50, *, seed: int = 42, vocab_size: int = 1000, seq_len: int = 12) -> MIBTask  # Mueller et al. ICML 2025; synthetic IOI task
benchmarks.load_greater_than(n: int = 50, *, seed: int = 42, vocab_size: int = 1000, seq_len: int = 8) -> MIBTask  # synthetic Greater-Than task
# MIBTask: .clean_inputs, .corrupted_inputs, .metric_fn (differentiable logit-diff on last position)

benchmarks.run_saebench(sae, acts: Tensor, *, tasks=None) -> SAEBenchResult  # Karvonen et al. 2025
# SAEBenchResult: .l0, .explained_variance, .mse, .feature_density, .sparse_probing_r2, .ce_loss_score (None if no model)
# Analytically-tractable subset: l0_sparsity, explained_variance, reconstruction_mse,
#   feature_density, sparse_probing_r2 (all computed from activation tensors; no model/network needed)

# Causal alignment (v1.28)
from circuitry.patching import das, scrubbing
das.DASRunner(model).run(base_inputs: _Inputs, source_inputs: _Inputs, labels: Tensor, *,
    module: nn.Module, subspace_dim: int = 1, n_steps: int = 500, lr: float = 0.01,
    loss_fn=None) -> DASResult  # Geiger et al. NeurIPS 2023, arxiv:2303.02536
# DASResult: .rotation (d×d orthogonal), .subspace_dim, .iia_score, .losses list,
#   .subspace_directions() → (subspace_dim, d_model) — the causal direction rows

scrubbing.CausalScrubRunner(model).run(clean_inputs, corrupted_inputs, metric,
    hypothesis: CircuitHypothesis, *, compute_per_module: bool = True) -> CausalScrubResult
# CircuitHypothesis: .circuit_modules (list[nn.Module]), .node_labels (dict[nn.Module, str])
# CausalScrubResult: .faithfulness, .scrubbed_metric, .clean_metric, .corrupted_metric,
#   .per_module_delta (dict[str, float])

# Attribution quality (v1.32)
from circuitry.patching import relp, certified
relp.ReLPRunner(model, resolver=None, *, eps=1e-6)  # drop-in for EAPRunner; returns EAPResult
# Replaces EAP gradient term with LRP-epsilon residual-stream coefficient
# lrp_coeff_w = act_clean_w / (|Σ_w act_clean_w| + eps); same cost as EAP
# Pearson corr to ground-truth patching = 0.956 vs 0.006 for EAP (arXiv:2508.21258)

certified.CertifiedCircuitRunner(base_runner, *, n_subsamples=20, confidence=0.95, subsample_frac=0.5, seed=0)
# .run(clean, corrupted, metric, *, top_k=10) -> CertifiedCircuitResult
# Wraps any runner with randomised batch subsampling; certifies edges stable in ≥ confidence fraction
# CertifiedCircuitResult: .certified_edges, .abstained_edges, .vote_counts,
#   .certified_set(), .n_certified(), .n_abstained() (arXiv:2602.22968)

# MIB benchmarks (v1.32 additions)
from circuitry import benchmarks
benchmarks.load_ravel(n, *, entity_type, attribute, seed, vocab_size, seq_len) -> MIBTask  # RAVEL entity-attribute task (arXiv:2402.17700)
benchmarks.load_arithmetic(n, *, op, modulus, seed, vocab_size, seq_len) -> MIBTask  # op ∈ {"add","mod_add"}; modular arithmetic circuits
benchmarks.load_mcqa(n, *, n_choices, seed, vocab_size, seq_len) -> MIBTask  # multiple-choice Q&A
benchmarks.mib_circuit_f1(circuit_edges, ground_truth_edges) -> float  # edge-set F1 for MIB localisation leaderboard
benchmarks.mib_iia_score(das_result, *, threshold=0.5) -> float  # IIA-at-threshold for causal variable localisation
```

Invariants for everything in `core/`:

- Deterministic on CPU; no implicit `.cuda()`.
- Accept `torch.Tensor` or `numpy.ndarray` where it makes sense.
- All scalar numeric returns are plain Python `float`, not 0-dim tensors.

### 4.2 Tier 2 — recorder

```python
from circuitry import Recorder, scan_run, build_report
from circuitry.recorder.compare import compare_runs, build_compare_report

recorder = Recorder(
    model,
    run_dir="runs/my_run",
    recipe="llm",                  # str name or Recipe instance
    writer="auto",                 # default: tensorboard if installed, else jsonl; or "tensorboard"/"jsonl"/"null"
    every_n_steps=200,
)
recorder.attach()
for step, batch in enumerate(loader):
    loss = train_step(model, batch)
    recorder.step(step, loss=loss)
recorder.detach()

scan_run(
    run_dir="runs/my_run",
    recipe="llm",
    out_dir="runs/my_run/tb_retro",
    # checkpoints=...  # optional: override the default <run_dir>/checkpoints/step*.pt
    #                  # glob with a file, a glob string, a list of paths, or a list
    #                  # of explicit (step, path) pairs (single-snapshot / custom names)
)

build_report(
    run_dir="runs/my_run",
    out_path="runs/my_run/inspect/report.md",
    compact=False,   # True → Summary + Flags only; suppresses per-tag tables (v1.2)
)

# Compare two runs at family/diagnostic granularity (v1.2)
deltas = compare_runs("runs/run_a", "runs/run_b")   # list[FamilyDelta]
build_compare_report("runs/run_a", "runs/run_b", out_path="runs/compare.md")
```

### 4.3 CLI

```bash
circuitry scan    --run runs/my_run --recipe llm --model-factory pkg.mod:make_model [--out dir]
circuitry report  --run runs/my_run [--compact]
circuitry compare run_a run_b [--out path] [--compact]
circuitry circuit-compare A.json B.json [--out path]
circuitry list-recipes
circuitry fit-tuned-lens --model pkg.mod:make_model --batches pkg.mod:make_batches --out lens.pt [--layers 0 1 2] [--steps N] [--lr LR] [--weight-decay WD] [--device DEV]
```

`report` accepts either a live `metrics.jsonl` (written by the Recorder, no `scan` step) or a retrospective `metrics.jsonl` produced by `scan` with `writer="jsonl"`. `--compact` renders only the `## Summary` and `## Flags` blocks, suppressing per-tag tables (v1.2). `compare` loads `metrics.jsonl` from each run directory and writes a family/diagnostic-granular delta table (v1.2). `scan --model-factory` (v1.17) accepts a `package.module:attr` entry point (zero-arg factory returning the model), enabling retrospective analysis on saved checkpoints without writing a custom script; `--out` overrides the default `<run>/scan_report` output directory. `circuit-compare A.json B.json` (v1.17) diffs two circuit JSON files (EAP or ACDC format) by edge-set, printing added / removed / shared edge counts; `--out path` writes the diff table to a file. `fit-tuned-lens` (v1.10) resolves `--model` / `--batches` as `package.module:attr` entry points (zero-arg factories returning the model and an iterable of inputs), fits per-layer tuned-lens translators post-hoc, and writes a `TunedLens` to `--out`; load it into `recipe.tuned_lens` and add `"tuned_lens_kl"` to `activation_diagnostics` to emit the diagnostic.

**Static vs trajectory diagnostics on a scan.** *Static* weight diagnostics (`effective_rank`, `stable_rank`, `condition_number`, `heavy_tail_alpha`, `sv_histogram`) work on a single checkpoint. *Trajectory* diagnostics (`update_delta`, `rank_trajectory`, `direction_cosine`) compare consecutive emitted snapshots and produce nothing until ≥2 (≥3 for `direction_cosine`) checkpoints are scanned. A single-snapshot `scan_run` emits only the static families and warns once when trajectory diagnostics are requested with fewer than two checkpoints.

### 4.4 `Recipe` and hook escape hatches

```python
@dataclass
class Recipe:
    name: str
    hook_points: list[HookPoint]
    weight_diagnostics: list[str]
    activation_diagnostics: list[str]
    gradient_diagnostics: list[str]
    custom: list[DiagnosticFn] = field(default_factory=list)
    expected_min_matches: dict[str, int] = field(default_factory=dict)  # pattern → min modules
    enabled: dict[str, bool] = field(default_factory=dict)  # name → False to suppress; absent = True
    module_prefix: str | None = None  # if set, only modules under this dotted prefix are matched
    attn_head_meta: dict[str, int] | None = None  # explicit n_heads/n_kv_heads/head_dim for attention_head_rank on config-less models
    forward_fn: Callable[..., object] | None = None  # custom (model, batch) -> output for the recorder's probe passes (non-HF models)
    tuned_lens: TunedLens | None = None  # (v1.10) fitted tuned lens for the opt-in tuned_lens_kl diagnostic
    # drift probe fields (v1.4)
    probe_batch: torch.Tensor | None = None      # if set, enables the drift probe (second forward pass per emit step)
    drift_method: str = "linear_cka"             # repr_drift method: "linear_cka" | "cosine" | "rbf_cka"
    drift_max_tokens: int | None = None          # row cap for the drift probe; None = all tokens
```

Use `Recipe.with_prefix(prefix)` to scope a recipe to a sub-tree of the model (e.g. `get_recipe("llm").with_prefix("model.language_model")` for multimodal HF models). Returns a new `Recipe` via `dataclasses.replace`; the original is not mutated. Latest-wins: calling `.with_prefix("a").with_prefix("b")` yields `module_prefix="b"`. If `expected_min_matches` is set, lower the thresholds after scoping — whole-model counts don't hold after a prefix filter.

Use `Recipe.with_sae(mapping)` (v0.9) to attach SAE checkpoints: `mapping` is a `dict[str, tuple[str, str]]` of `module_name → (release, sae_id)`. Returns a new `Recipe` with `sae_checkpoints` populated. SAE checkpoints are loaded lazily at `attach()` time; the user must also add `"sae_reconstruction"` to `activation_diagnostics` to incur per-step encode+decode cost.

Use `Recipe.disable(names)` (v1.2) to drop specific diagnostics by name; returns a new `Recipe` with those names set to `False` in `enabled`. Use `Recipe.only(names)` to keep only the listed diagnostics and disable the rest. Both raise `ValueError` on any name not in `weight_diagnostics + activation_diagnostics + gradient_diagnostics`; custom `DiagnosticFn` callables are not name-addressable and are unaffected.

`attn_head_meta` and `forward_fn` support non-HF / config-less models. `attention_head_rank` resolves head metadata in order: `attn_head_meta` (explicit `n_heads`/`n_kv_heads`/`head_dim`/`hidden_size`) → any submodule's `.config` (covers `model.config`, `text_config`, and HF-wrapped `model.model.config`) → `num_heads`/`head_dim` attributes on a `self_attn`/`attn`/`attention` submodule. The recorder's internal probe passes (`induction_score`, `copy_suppression_score`, `drift_probe`) call `forward_fn(model, batch)` when set — the entry point for models whose `forward` isn't HF-style (e.g. SASRec's `predict_scores`) — otherwise `model(probe, output_attentions=True)` with a `TypeError` fallback. A non-HF model with no resolvable `config` warns once at `attach()` pointing at `forward_fn`.

`tuned_lens` (v1.10) carries a fitted `TunedLens` (from `circuitry.tuned_lens.fit_tuned_lens` or the `circuitry fit-tuned-lens` CLI) for the **opt-in** `tuned_lens_kl` activation diagnostic — the tuned lens (Belrose et al. 2023) applies a learned per-layer affine `A·h + b` to each residual before the unembed, so per-layer KL is no longer confounded by the early/mid-layer basis mismatch. Fitting is **post-hoc only** (an optimizer loop, never in the training loop — see §10). The recorder resolves the unembed + final-LN exactly as for the logit lens (shared `_lens_meta`), verifies `TunedLens.model_fingerprint` against the attached model at `attach()`, and **warns + skips** when the lens is missing or fitted on a different model — it never emits a wrong KL. It emits `activation/tuned_lens_kl/<layer>` for each fitted block (the final block is the target frame and is not fitted). `tuned_lens_kl` is NOT in any stock recipe's default list; enable it by setting `recipe.tuned_lens` and adding the name to `activation_diagnostics`.

> The `Recorder` maintains two internal CPU weight snapshots (`_prev_weights`,
> `_prev_prev_weights`) to support cross-step weight-dynamics primitives. Both are
> empty at `attach()`, populated (detached CPU copies of matched-module weight tensors)
> after each emit step, and cleared in `detach()`. The cross-step diagnostics
> `update_delta`, `direction_cosine`, and `rank_trajectory` silently skip emission on
> the first emit step (or first two, for `direction_cosine`) until enough snapshots
> exist. This is the only internal recorder state change in v1.3; `StepContext` shape
> is unchanged.

> **Representational drift probe (v1.4).** `Recipe` gains three new fields for opt-in
> drift monitoring:
>
> ```python
> probe_batch: torch.Tensor | None = None      # if set, enables the drift probe
> drift_method: str = "linear_cka"             # "linear_cka" | "cosine" | "rbf_cka"
> drift_max_tokens: int | None = None          # row cap for the probe; None = all tokens
> ```
>
> The `llm` recipe lists `"drift_probe"` in `activation_diagnostics` with
> `enabled={"drift_probe": False}` (default OFF). To enable, pass a
> `Recipe.probe_batch` tensor — a small representative input batch held constant
> across emit steps. When `probe_batch` is set and `"drift_probe"` is not suppressed
> via `enabled`, the Recorder runs a **second forward pass** on `probe_batch` at each
> emit step (frozen model, no grad, CPU clone). On the **first emit step**, the
> captured per-module activations are stored as the reference snapshot (a detached CPU
> copy); subsequent steps compare live activations to this reference using
> `activation.repr_drift`. The reference snapshot is cleared in `detach()`.
>
> Call `recorder.reset_drift_reference()` at any time to discard the current reference
> snapshot and re-anchor at the next emit step (e.g. after a phase change or a
> checkpoint reload).
>
> Per-layer drift is written as `activation/repr_drift/<module>` scalars via the
> configured `MetricWriter`.
>
> Because the probe requires a second forward pass, it adds overhead proportional to
> the probe batch size. It is off by default so it adds zero overhead at default
> settings; see §10 for measured overhead.

> **ACDC run/sweep kwargs (v1.4).** `ACDCRunner.run()` and `.sweep()` gained two new
> optional kwargs:
>
> - `ablation_mode: str = "corrupted"` — controls what value is injected for ablated
>   edges. `"corrupted"` (default) feeds the cached corrupted-run activation (matching
>   the original ACDC paper). `"zero"` injects a zero tensor. `"mean"` injects each
>   writer's corrupted activation averaged over the batch/sequence positions (a
>   spatially-constant per-feature mean).
> - `eap_skip_threshold: float | None = None` — if provided along with `eap_scores`,
>   edges whose `|EAP score|` **exceeds** this threshold are assumed important and kept
>   **without** running their ablation test (skipping that forward pass), accelerating
>   circuit discovery on large graphs. `None` (default) tests every edge. This is the
>   EAP-score skip speedup documented as a v1.0 follow-on.

`HookPoint` supports three target specifications:

```python
@dataclass
class HookPoint:
    source: TensorSource                          # WEIGHT, INPUT, OUTPUT, GRAD, or NAMED_PARAM
    pattern: str | None = None                    # regex against named_modules() (recipe default)
    modules: list[nn.Module] | None = None        # explicit instances (advanced)
    selector: Callable[[nn.Module], list[str]] | None = None  # programmatic name selector
    optional: bool = False                        # a 0-match is a soft skip even under strict=True
    # exactly one of {pattern, modules, selector} must be set
```

This gives three matching modes:

1. **Pattern (default)** — used by all stock recipes. Regex against `dict(model.named_modules()).keys()`.
2. **Explicit modules** — pass `nn.Module` instances directly. For Mamba / MoE / custom architectures where regex is fragile.
3. **Programmatic selector** — a function that walks `model` and returns module names. Use when the right hook set depends on runtime structure (e.g. only experts that fired this step).

`source=TensorSource.NAMED_PARAM` is a special case: its `pattern` matches against **parameter** names (`dict(model.named_parameters()).keys()`), not module names, and the matched ≥2-D parameter is fed to the weight diagnostics keyed by its parameter name. Use it to reach a fused parameter that the `WEIGHT` source can't resolve — e.g. `nn.MultiheadAttention.in_proj_weight`, a direct `Parameter` on the module rather than a child `Linear`'s `.weight` (the module owns two 2-D params, so the WEIGHT primary-weight resolver returns nothing). Non-≥2-D matches (e.g. a 1-D bias) are skipped with a warning, mirroring the WEIGHT source's 2-D guarantee.

Brittleness mitigation (addressing recipe-matched-wrong-subset failure mode):

- At `attach()` time, the recorder logs the full list of matched module names per `HookPoint` at INFO level, and writes it to `<run_dir>/circuitry/matched_modules.txt` so it is visible in artifacts.
- If a `HookPoint`'s `expected_min_matches[pattern]` is set and the actual match count is below it, `attach()` raises by default. Pass `strict=False` to `Recorder.__init__` to downgrade this to a warning.
- A `HookPoint` that matches **zero** modules always raises, regardless of `strict`. There is no legitimate use of a hook that hooks nothing.

Users can register custom recipes via `circuitry.register_recipe(my_recipe)`.

#### `DiagnosticFn` signature and step context

Custom diagnostics that need both forward activations and backward gradients on the same step (Fisher information, gradient-activation alignment, etc.) receive a `StepContext`:

```python
@dataclass
class StepContext:
    step: int
    model: nn.Module
    activations: dict[str, Tensor]    # hooked module name → forward output (if captured)
    gradients:   dict[str, Tensor]    # hooked module name → .grad (post-backward, pre-step)
    weights:     dict[str, Tensor]    # hooked module name → parameter (current value)
    loss:        float | None
    user:        dict[str, Any]       # opaque pass-through from Recorder.step(**kwargs)

DiagnosticFn = Callable[[StepContext], dict[str, float | Tensor]]
# returned dict: tag (str, no leading "/") → value. Recorder prefixes "custom/" and writes.
```

The recorder builds the `StepContext` once per emit step (every `every_n_steps`), runs all listed `custom` callables in order, and writes their outputs through the configured `MetricWriter`. Built-in diagnostics in `weight_diagnostics` / `activation_diagnostics` / `gradient_diagnostics` are implemented internally against the same `StepContext` shape; `custom` is the public extension point.

### 4.5 `MetricWriter` protocol

```python
class MetricWriter(Protocol):
    def add_scalar(self, tag: str, value: float, step: int) -> None: ...
    def add_histogram(self, tag: str, values: Tensor, step: int) -> None: ...
    def add_image(self, tag: str, image: Tensor, step: int,
                  dataformats: str = "CHW") -> None: ...
    def add_text(self, tag: str, text: str, step: int) -> None: ...
    def flush(self) -> None: ...
    def close(self) -> None: ...
```

`add_image` is essential for vision recipes (activation maps, weight kernels visualized as heatmaps) and for matrix-as-image debug views even in LLM recipes (e.g. plotting `W_O @ W_V` per head). `dataformats` follows TB's convention.

The TensorBoard adapter is a thin wrapper over `torch.utils.tensorboard.SummaryWriter`. The JSONL adapter writes one JSON line per `add_scalar` call and dumps tensors / images to side files under `<run_dir>/circuitry/artifacts/` (no extra deps); the `scan` / `report` workflow reads this format. The null adapter is a no-op for tests. Third-party loggers (wandb, mlflow, etc.) are not shipped in v0.3.0 — implement `MetricWriter` (~50 LOC) and pass the instance to `Recorder(writer=...)`.

**Optional dependencies (extras).** Core install (`pip install circuitry`) pulls only `torch` + `numpy`. `tensorboard` and `sae-lens` are optional extras — install `circuitry[tensorboard]`, `circuitry[sae]`, or `circuitry[all]`. The default `writer="auto"` resolves to the TensorBoard adapter when the extra is installed and otherwise falls back to the no-dep JSONL adapter (one-time warning); `writer="tensorboard"` (explicit) raises an install-pointing `ImportError` when the extra is missing, and the SAE loader does the same. `import circuitry` never imports either package — both are lazy-loaded at the writer/SAE call site.

### 4.6 Intervention mode (v1.0)

The `patching/` subsystem adds an opt-in **intervention mode** for causal analysis (activation patching, and the attribution methods built on it). It is the first capability in `circuitry` that *modifies* activations rather than only observing them. Contrasted with the observation-only `Recorder` and `scan` workflows, intervention mode upholds these invariants:

- **Opt-in.** Interventions require explicit use of the `circuitry.patching` API. `Recorder` and `scan` remain observation-only and are never affected by patching.
- **Isolated.** Every intervention is scoped to a context manager (`patch_site`). The forward hook is removed and model state is restored on exit, including on exception (`try/finally`, mutation-last — hooks installed as the final setup step so a partial setup can't leak).
- **Frozen model.** Parameter `requires_grad` is forced off for the duration and restored on exit; eval mode is set on entry and restored; no optimizer runs and no parameter values are modified.
- **Activation-grad-only.** The only gradient flow permitted is on activation tensors at intervention sites (for attribution methods such as AtP\* and EAP). Parameter gradients are never enabled.

Sites are resolved to concrete model locations by a resolver: `HFSiteResolver` (recipe/config-declared layout — per-head needs eager attention; per-neuron is Llama-family-first) or `TLSiteResolver` (TransformerLens hook names; lazy `transformer_lens` import). The metric helpers live in `core/patching.py` (pure functions); any `Callable[[Tensor], float]` is accepted as a custom metric.

```python
from circuitry.patching import Site, patch_site, PatchRunner
from circuitry.patching.sites import HFSiteResolver
from circuitry.core.patching import logit_diff

resolver = HFSiteResolver.from_config(model.config)
site = Site(component="attn_head_out", layer=5, head=3)

# Low-level: single intervention (restores on exit)
with patch_site(model, site, value=cached_act, resolver=resolver):
    output = model(**inputs)

# High-level: prompt-pair runner
runner = PatchRunner(model, resolver)
result = runner.run_patching(
    clean_inputs=clean_ids,
    corrupted_inputs=corrupted_ids,
    sites=[site],
    metric=lambda logits: logit_diff(logits, correct=tok_a, incorrect=tok_b),
    direction="denoise",
)
print(result.metric_values)  # {site: metric}
```

Attribution methods (EAP, AtP\*, ACDC — v1.0), node-level SAE-feature attribution (`SAEFeatureRunner`, v1.5), feature→feature SAE circuits (`SAEFeatureEdgeRunner` + `FeatureACDCRunner`, v1.6), and generalized SAE circuits with `mlp_out`/`attn_out` sites, TransformerLens backend, and integrated-gradients variant (v1.7) build on this primitive.

**EAP (sub-spec 2, shipped).** `EAPRunner(model, resolver).run(clean_inputs, corrupted_inputs, metric, ig_steps=1)` returns an `EAPResult` of per-edge attribution scores over the residual-stream graph (`circuitry.patching.graph`: `Node`/`Edge`/`build_graph`). Edges are writer→reader with q/k/v-typed attention reads; nodes are attention heads + MLPs + embed + logits. Scoring is the 2-forward + 1-backward linear approximation (`Δact · grad`, summed over `d_model`), with vanilla EAP (`ig_steps=1`) and activation-path EAP-IG (`ig_steps=N`). Backends: TransformerLens (native per-slot hooks) and HF (eager, Llama-family — per-head `z@W_O` writers, q/k/v reader gradients back-mapped to residual space with the RMSNorm scale as a stop-gradient constant, GQA-aware). The EAP metric must be **differentiable** — use the tensor-returning `circuitry.core.patching.logit_diff_t` / `kl_divergence_t` / `ce_loss_t`, not the `.detach()`-ing float versions.

**AtP\* (sub-spec 3, shipped).** `AtPRunner(model, resolver).run(clean_inputs, corrupted_inputs, metric, neurons=False, graddrop=False, qk_fix=True)` → `AtPResult` of per-node attribution scores (`circuitry.patching.atp`: `AtPNode`, `AtPResult`). Nodes are embed + per-head (q/k/v slots) + mlp per layer + optionally mlp_neuron per neuron. Scoring: `score(node) = Σ(Δact_node ⊙ grad_node)` summed over all dims; q/k nodes use the **QK fix** (attention-pattern recomputation in `d_model` space against `grad_attn_out`) when `qk_fix=True`; `graddrop=True` replaces summation with `Σ|per-position contribution|`. On a fully linear model vanilla AtP is exact (equals brute-force `patch_site` at 1e-4). The differentiable metric (`logit_diff_t` / `kl_divergence_t` / `ce_loss_t`) is required. Backends: HF (eager, Llama-family, GQA-aware — full QK fix implemented) and TransformerLens (native TL hooks — vanilla q/k scoring only; full QK fix with softmax-pattern recomputation is HF-path only). `AtPResult.verify_top_k(k, clean_inputs, corrupted_inputs, metric, resolver, runner)` calibrates the top-K nodes against real `patch_site` ground truth, returning `{node: (atp_score, true_patch_effect)}`.

**ACDC (sub-spec 4, shipped).** `ACDCRunner(model, resolver).run(clean_inputs, corrupted_inputs, tau, ordering="topo", position=-1, metric=None)` → `ACDCResult` of the pruned circuit edges (`circuitry.patching.acdc`: `ACDCRunner`, `ACDCResult`). Greedy forward-only reverse-topological edge pruning with corrupted-resample set ablation: ablated edges feed corrupted-run activations (cached once), kept edges propagate live (current-circuit) activations, deltas injected **pre-LayerNorm** per reader/slot (per-head rebuild on HF eager, native on TransformerLens). Recovery metric: last-token KL to the clean distribution (default, configurable `position`; custom metric callable accepted). Single-threshold pruning via `tau` (per-edge tolerance) and a `sweep(taus)` Pareto helper returning `[(τ, n_kept, final_kl), …]`. Traversal orderings: `"topo"` (reverse-topological determinism with tie-break key) or `"eap"` (lowest `|EAP score|` first, consumes `EAPResult.scores`). Backends: HF (eager, Llama-family, GQA k/v at group granularity) and TransformerLens. Empty- and full-ablation anchors are exact under corrupted−live + pre-LN injection; v1.0 ships edge-traversal ordering only (EAP-score skip speedup is a documented follow-on).

**SAE-feature attribution (sub-spec 5, v1.5; generalized in v1.7; v1.21 TranscoderWrapper).** `SAEFeatureRunner(model, sae_sites, resolver).run(clean_inputs, corrupted_inputs, metric, *, graddrop=False, include_error_node=False, max_features=None, variant='attrib', n_ig_steps=0)` → `AtPResult` of per-**SAE-feature** attribution scores (`circuitry.patching.sae_features`). `sae_sites` maps `Site(component, layer=L)` → a SAELens `SAE` (or a `(release, sae_id)` tuple loaded via `sae.load_sae`). **Supported components (v1.7): `resid_post`, `mlp_out`, `attn_out`.** Per-head/per-neuron sub-slices (`attn_head_out`, `mlp_neuron`) and `resid_pre` raise `NotImplementedError`. Multiple SAE sites per layer are fully supported via composite `(layer, component)` keying (§4 composite keying below). Mechanism — **error-term substitution** (Marks "Sparse Feature Circuits"): on the clean forward at each site, `f = sae.encode(a)`, `x_hat = sae.decode(f)`, `eps = (a − x_hat).detach()` (the *frozen clean reconstruction error*), and the activation is replaced by `recon = x_hat + eps` (numerically lossless to ≈1e-7, so the model output is preserved) — which makes each SAE feature `f_i` a differentiable node. Every splice routes through `ResolvedSite` (v1.7 P1 refactor — byte-for-byte identical to v1.6 for `resid_post` + HF-eager). For `mlp_out`, the hook fires on `model.layers[L].mlp` output (the MLP submodule tensor, **before** the residual add); for `attn_out`, on `model.layers[L].self_attn` output (the attention submodule tensor). **Arch caveat (Llama-family equivalence):** on sequential attn→mlp blocks (Llama/GPT-2) the submodule output equals the attention/MLP contribution to the residual stream. Gemma2 applies a `post_{attention,feedforward}_layernorm` before the residual add, so the submodule output is the pre-norm tensor — still losslessly spliced but different from what a Gemma2-trained SAE would expect. The equivalence claim is scoped to Llama-family sequential blocks. **Parallel-attention arches (GPT-J-style):** attn and mlp both read `resid_pre`, so intra-layer `attn_out@L → mlp_out@L` edges are causally undefined; v1.7 assumes sequential blocks. **`TranscoderWrapper` (v1.21):** wraps any transcoder (module-input → module-output feature decomposition) as a drop-in SAE site. `TranscoderWrapper(tc)` sets `hook_input=True`; attribution hooks branch on this attribute to encode from `inp[0]` (the module input) rather than `output`, with `eps = output − x_hat` in module output space. The splice is always lossless (`x_hat + eps = output`). Works with both `SAEFeatureRunner` and `SAEFeatureEdgeRunner`; the IG path is unchanged (uses pre-computed interpolated `f_k`, no re-encoding). Scoring (default `variant='attrib'`) mirrors `mlp_neuron` in the SAE basis: `score(feature i) = Σ_pos(Δf_i ⊙ gradf_i)`, `Δf = f_corrupt − f_clean`, gradient taken at the clean activation; `graddrop=True` uses `Σ|per-position contribution|`. **`variant='ig'`** (integrated gradients, v1.7 P4): path `f(α) = f_clean + α·Δf`, midpoints `α_k = (k−0.5)/N` (default `N=32` when `n_ig_steps=0`), `eps` frozen at clean. `score_i = Σ_pos Δf_i · (1/N) Σ_k ∂metric/∂f_i |_{f=f_clean+α_k·Δf}`. Fixes attrib's saturation blind spot: features dead at clean but active at corrupted have a real nonzero IG even though `grad@clean ≈ 0`. **Completeness:** feature-IG completes to the eps-frozen spliced delta `metric(decode(f_corrupt)+eps_clean) − metric(decode(f_clean)+eps_clean)` (not the real `metric(corrupt)−metric(clean)` because `eps` is held at `eps_clean`); with `include_error_node=True` the error IG closes the gap and features+error jointly complete to the real forward delta. Features are enumerated where `Δf ≠ 0` on both paths (not gated on `grad@clean`). `include_error_node=True` (opt-in, **default off**) additionally scores the reconstruction-error term as a first-class `sae_error` node via an **independent error leaf**, so feature and error gradients come from a single forward + backward — feature scores are bit-identical with the node on or off. `bruteforce_feature_scores()` is the independent ground-truth path. **On a model with a linear downstream, the analytic score equals brute-force feature patching at 1e-4.** **Node identity (v1.7):** `Node("sae_feature", layer=L, neuron=i, component=c)` where `c=None` for `resid_post` (preserves v1.6 identity) and `c="mlp_out"` / `c="attn_out"` for new sites — nodes at the same `(layer, neuron)` but different components are distinct. New `Node` kinds: `sae_feature` (reuses the `neuron` field as the feature index) and `sae_error`; the edge-graph machinery (`_order`/`build_graph`) is untouched. Supported SAE architectures: `standard` / `topk` / `jumprelu` (covers BatchTopK / Matryoshka at inference) / `gated`. The metric must be the differentiable tensor-returning variant (`logit_diff_t` / `kl_divergence_t` / `ce_loss_t`). **Backends (v1.7): HF-eager and TransformerLens.** On the TL path dtype/device are sourced from `model.cfg.dtype`/`torch.device(model.cfg.device)` (`HookPoint.parameters()==[]`; the params-fallback would silently downcast non-fp32/CUDA models). Feature→feature **edges** + circuit extraction ship in sub-spec 6 (v1.6); generalized to multi-site and IG in v1.7.

**SAE-feature EDGES + circuit (sub-spec 6, v1.6; updated v1.17–v1.21).** `SAEFeatureEdgeRunner(model, sae_sites, resolver).run(clean, corrupted, metric, *, layer_pairs="adjacent", top_k_survivors=32, max_edges=None, include_error_node=False, variant="attrib", n_ig_steps=0, per_position=False, arch="sequential")` → `SAEFeatureCircuit` (`circuitry.patching.sae_edges`): the v1.5 node `AtPResult`, a `dict[SAEFeatureEdge, float]` of feature→feature edges, a `SAEFeatureEdgeGraph`, and optionally per-position scores (v1.19). **Two-stage** (the feature×feature space is intractable — ~6e8 edges for one adjacent pair at d_sae=24576): stage 1 reuses the shipped `SAEFeatureRunner` to score and keep the top-K **active** survivors per site; stage 2 enumerates edges only among survivors across ordered site-pairs (`adjacent` default, `all_forward` opt-in; `max_edges` cap). **Edge formula** `edge(U:i→D:j) = Σ_pos Δf_U[...,i] · (∂f_D[...,j]/∂f_U[...,i]) · gradf_D[...,j]` (the `mlp_neuron` formula lifted to a feature→feature path), `Δf_U = f_U_corrupt − f_U_clean`. **Mechanism:** ALL sites in the span are spliced **simultaneously** in ONE clean forward (the v1.5→v1.6 pivot — so the upstream feature's decode propagates through the real residual to the downstream encode); the upstream-most (writer) site uses the v1.5 detached-leaf seed, downstream (reader) sites use a LIVE non-detached encode, `eps` is detached/frozen at every site. The Jacobian factor is realized as a **per-downstream-survivor VJP** (`torch.autograd.grad(f_D, f_U_leaf, grad_outputs=…)`) — NEVER a dense `d_sae×d_sae` matrix; each VJP is freed immediately. This is the **full live-autograd Jacobian, NOT the Marks/Anthropic frozen-attention-pattern stop-gradient Jacobian** — on a linear-downstream model the analytic edge equals `bruteforce_feature_edge_scores` (independent forward-patch oracle) at 1e-4. `include_error_node=True` (opt-in, default off) additionally emits **error→feature** edges via an independent upstream `err_leaf`; **feature→error edges are structurally zero** and are not computed. **Per-position scores (v1.19):** `run(…, per_position=True)` additionally stores `SAEFeatureCircuit.position_scores: dict[SAEFeatureEdge, Tensor]` where each tensor has shape `(seq_len,)` and `position_scores[e].sum() == edges[e]` within float32 rounding — exposes which sequence positions drive each edge. Supported for both `variant='attrib'` and `variant='ig'`, including error→feature edges. Default `False` (zero overhead). `SAEFeatureCircuit.top_positions(edge, k=5)` returns top-k `(pos_idx, score)` pairs sorted by `|score|` descending; raises `ValueError` if `position_scores is None`. **Parallel-attention arch (v1.20):** `arch='parallel'` proactively skips same-layer `attn_out@L → mlp_out@L` edges (causally undefined in GPT-J-style models where attn and MLP both read `resid_pre`). `arch='sequential'` (default) preserves all edges. Unknown values raise `ValueError`. Helper `_is_parallel_intra_layer(writer_site, reader_site)` returns `True` for the forbidden pair. **TranscoderWrapper sites (v1.21):** any site's SAE may be wrapped in `TranscoderWrapper` to encode from `inp[0]` (module input) rather than `output`; the four writer/reader hooks in `_compute_pair_edges` branch on `getattr(sae, "hook_input", False)`.

**Circuit extraction / pruning (v1.6; updated in v1.7–v1.21).** `SAEFeatureCircuit.ranked()/top_k(n)/threshold(tau)/top_positions(edge, k=5)` and `prune(method="threshold"|"acdc"|"both", tau, ablation_mode="corrupted"|"zero"|"mean")`. **Ablation is NODE-set, not edge-level** — downstream features at a given `(layer, component)` site share one activation tensor `a`, so an edge cannot be ablated independently; instead non-circuit feature entries are replaced (corrupted/zero/mean) before `decode` at each spliced site, propagated forward. `faithfulness(C) = (m(C) − m(∅)) / (m(M) − m(∅))` and `completeness` via the complement (Marks §3.2). `FeatureACDCRunner(model, sae_sites, resolver).run(…, tau, ablation_mode, eap_skip_threshold)` is **greedy reverse-topological NODE pruning** (`sweep(taus) → [(tau, n_kept, final_kl)]` Pareto). `graph.py` / `EdgeGraph` / `_order` / `build_graph` are **untouched** — the feature graph is a dedicated `SAEFeatureEdgeGraph`. **v1.7 edge IG:** `variant='ig'` wraps the loop over `N` interpolation steps; no dense Jacobian; peak memory = attrib. **Circuit rendering (v1.17):** `EAPResult`, `AtPResult`, and `ACDCResult` gain `.to_markdown(top_k)` (human-readable table), `.to_json()` / `.from_json(text)` (JSON serialization), and `.save(path)` / `.load(path)` file-I/O wrappers (v1.18). Helper functions `_node_str`, `_node_to_dict`, `_node_from_dict` in `patching/graph.py`. **v1.22 temporal runner:** `SAEFeatureTemporalRunner(model, sae_sites, resolver).run(steps, metric)` accepts a list of `(step_key, clean_inputs, corrupted_inputs)` triples and runs `SAEFeatureRunner` independently on each. Returns `TemporalAtPResult` with `.scores: dict[step_key, AtPResult]`, `.delta_scores: dict[step_key, AtPResult]` (attribution change between consecutive steps), `.stable_features(threshold)` (active at all steps), `.step_specific_features(step_key, threshold)` (active at one step only), and `.top_stable(k)` (top-k by minimum `|score|` across all steps). Note: each step is independent — true recurrent-SAE attribution (step k activations depend on step k−1 hidden state) remains a known limitation.

**Activation steering (v1.23).** `steer_vector(positive_acts, negative_acts, *, normalize=True)` in `core/steer.py` computes the CAA direction (Rimsky et al. 2024). `apply_steer(model, site, vector, *, coeff=1.0, resolver=None)` in `patching/steer.py` is a context manager that registers a forward hook at `site` adding `coeff * vector` to the output (broadcast-safe for 1D/2D/3D outputs); the hook is removed in a `finally` block.

**CrosscoderWrapper (v1.26).** `CrosscoderWrapper(crosscoder, *, primary_layer=0)` in `patching/sae_features.py` wraps a crosscoder SAE as a single-site attribution point (`hook_input=False`). Routes `encode`/`decode` through `encode_at_layer(x, primary_layer)` / `decode_at_layer(f, primary_layer)` when available, else plain `encode`/`decode`. `encode_all(acts: list[Tensor])` for full cross-layer mode. Raw crosscoder objects passed directly to `assert_supported_sae` are blocked via `_BLOCKED_ARCHITECTURES`; use `CrosscoderWrapper` to attribute through them.

**Edge Pruning (v1.25).** `EdgePruningRunner(model, resolver).run(clean_inputs, corrupted_inputs, metric, *, lambda_l0, n_steps, lr, temperature_init, temperature_final, ig_steps, candidate_edges) → EdgePruningResult` (Bhaskar & Wettig NeurIPS 2024, arxiv:2406.16778). Phase 1: compute per-edge EAP scores. Phase 2: jointly optimise scalar mask logits z_e via Adam — loss is `−σ(z/T)·|score| + λ·Σσ(z/T)` with temperature T annealed from `temperature_init` to `temperature_final` over `n_steps`. Phase 3: threshold at 0.5 for hard binary circuit. `candidate_edges` restricts the search space (used by HAP). `EdgePruningResult` provides `.circuit`, `.removed_edges`, `.mask_logits`, `.eap_scores`, `.circuit_graph()`, `.ranked()`, `.to_json()`/`.from_json()`/`.save()`/`.load()`.

**HAP (v1.25).** `HAPRunner(model, resolver).run(clean, corrupted, metric, *, top_p=0.5, **pruning_kwargs) → EdgePruningResult` (Hu et al. 2025, arxiv:2510.03282). Phase 1: EAP, keep top `top_p` fraction of edges by |score|. Phase 2: `EdgePruningRunner(candidate_edges=top_edges, **pruning_kwargs)`. Achieves similar faithfulness to full edge pruning at a fraction of the candidate set size.

**DAS — Distributed Alignment Search (v1.28).** `DASRunner(model).run(base_inputs, source_inputs, labels, *, module, subspace_dim, n_steps, lr, loss_fn) → DASResult` (Geiger et al. NeurIPS 2023, arxiv:2303.02536). Learns an orthogonal rotation R such that the first `subspace_dim` columns of R·h align with a specified causal variable. Training: (1) capture detached activations h_base and h_source at `module`; (2) construct h_int via differentiable interchange — rotate both to R-space, replace the first `subspace_dim` coords with those from h_source, rotate back; (3) inject h_int via a forward hook and compute CE loss against `labels`; (4) backprop through R, then project R back to the nearest orthogonal matrix via SVD retraction (U@Vh). `DASResult.rotation` — (d, d) orthogonal R; `DASResult.iia_score` — interchange-intervention accuracy; `DASResult.subspace_directions()` — first `subspace_dim` rows of R (the causal directions). No hidden `.cuda()` calls; works on any device the activations live on.

**Causal Scrubbing (v1.28).** `CausalScrubRunner(model).run(clean_inputs, corrupted_inputs, metric, hypothesis, *, compute_per_module) → CausalScrubResult` (Conmy et al. / Redwood Research 2022). `CircuitHypothesis(circuit_modules, node_labels)` specifies which `nn.Module` instances implement the behaviour of interest. Scrubbing pass: pre-capture all module activations on a corrupted forward; then run model on clean inputs with forward hooks that replace non-circuit modules' outputs with the cached corrupted activations; circuit modules pass through unmodified. Faithfulness = (metric(scrubbed) − metric(corrupted)) / (metric(clean) − metric(corrupted)); defined as 1.0 when clean == corrupted (denominator → 0). `per_module_delta` (opt-in, one extra forward pass per circuit module) measures each module's individual contribution to the scrubbed metric. A faithful hypothesis scores ≈ 1.0; a random or empty hypothesis scores ≈ 0.0.

**Backend scope (v1.1):** the HF-eager patching backend (EAP / AtP* / ACDC)
targets Llama-family layouts (`model.model.layers` + `self_attn.{q,k,v,o}_proj`)
and honors an explicit `config.head_dim` (so Gemma-2/3, where `head_dim !=
hidden_size/num_attention_heads`, work). For GPT-2 and other architectures,
wrap the loaded model with `circuitry.patching.to_hooked_transformer(model,
"<tl-name>")` and use the TransformerLens backend (`TLSiteResolver`); pointing
the HF backend at an unsupported layout raises a `ValueError` directing you
there. TransformerLens folds LayerNorm / centers weights, so patching runs on
the TL-processed (logit-equivalent) model.

## 5. Recipe internals — worked example

`recipes/llm.py`:

```python
from circuitry.recorder.hooks import HookPoint, TensorSource

RECIPE = Recipe(
    name="llm",
    hook_points=[
        HookPoint(pattern=r".*\.attn\.(q|k|v|o)_proj$", source=TensorSource.WEIGHT),
        HookPoint(pattern=r".*\.mlp\.(w1|w2|w3|gate_proj|up_proj|down_proj)$",
                  source=TensorSource.WEIGHT),
        HookPoint(pattern=r".*\.attn$",   source=TensorSource.OUTPUT),
        HookPoint(pattern=r".*\.mlp$",    source=TensorSource.OUTPUT),
        HookPoint(pattern=r".*\.ln_[12]$",source=TensorSource.OUTPUT),
        HookPoint(pattern=r".*\.mlp\.down_proj$", source=TensorSource.INPUT),
        HookPoint(pattern=r"embed.*",     source=TensorSource.WEIGHT),
        HookPoint(pattern=r"lm_head$",    source=TensorSource.WEIGHT),
        HookPoint(pattern=r".*\.attn\.(q|k|v|o)_proj$", source=TensorSource.GRAD),
    ],
    weight_diagnostics=["effective_rank", "attention_head_rank", "stable_rank",
                    "heavy_tail_alpha", "sv_histogram",
                    "update_delta", "rank_trajectory", "direction_cosine"],
    activation_diagnostics=["gate_stats", "dead_fraction", "kurtosis", "participation_ratio"],
    gradient_diagnostics=["grad_norm_per_module"],
)
```

`recipes/vision.py` swaps the regex for `conv\d+|fc\d+|patch_embed|blocks\.\d+\.(attn|mlp)`. `recipes/two_tower.py` knows about `query_tower`, `item_tower`, `interaction` (plus DLRM `embed_tables` / `bottom_mlp` / `top_mlp`), and adds an embedding-alignment diagnostic (cosine of query / item tower output means). `recipes/recsys.py` covers **sequential** recommenders (SASRec / BERT4Rec / GRU4Rec): the item/position embedding is the required anchor and the architecture-variant patterns (attention / FFN / norm / block vs. GRU) are marked `HookPoint(optional=True)` so a model from one sub-family attaches under the default `strict=True` without matching the others — it is complementary to, not a replacement for, `two_tower`.

The recorder walks `model.named_modules()`, matches against each recipe's patterns, registers the appropriate forward / backward hook (or pre-reads `WEIGHT` tensors directly), and at each emit step feeds the captured tensors through the listed primitives and writes the resulting scalars / histograms through the configured `MetricWriter`.

## 6. Testing strategy

Four layers, sized to where bugs actually live:

1. **`tests/core/` — property tests on primitives.** Known answers on synthetic matrices (identity → effective_rank = n; rank-1 outer product → 1; orthogonal → cond = 1). Invariance under orthogonal transforms. ~30-50 tests; <10s on CPU.
2. **`tests/recorder/` — hook & writer smoke tests.** Tiny 2-layer MLP; `RecordingWriter` captures every `add_scalar`; assert tags / steps / no hook leaks after `detach()`. ~10 tests; <5s.
3. **`tests/recipes/` — modality fixtures.** Three minimal fixtures: 1M-param toy transformer, 100k-param ResNet block, 50k-param two-tower model. Attach recorder, 3 steps, assert recipe-specific scalars appear. ~5 tests / recipe.
4. **`tests/e2e/` — full pipeline.** Train tiny model 20 steps with `LiveRecorder` → `scan_run` over its 2 checkpoints → `build_report` → assert markdown contains expected sections. <30s.

CI: GitHub Actions, Python 3.10 / 3.11 / 3.12, PyTorch latest stable. No GPU jobs (everything CPU-sized). Performance benchmark (§10) is a separate CI job using `pytest-benchmark`; regressions >15% over baseline block merge.

## 7. Release history

See [`CHANGELOG.md`](../CHANGELOG.md) for the full version log. Public releases are tagged and announced via [GitHub Releases](https://github.com/vishsangale/circuitry/releases).

## 8. Explicitly out of scope today

- SAE training (interop with SAELens later if demand surfaces).
- JAX / Flax support.
- DDP / FSDP-aware reductions — current releases are single-process; non-zero ranks no-op. See §11 for the additive future-release path.
- ~~Logit lens / tuned lens beyond `core/lens.py`'s `logit_lens_kl`.~~ **Tuned lens shipped in v1.10** (`core.lens.tuned_lens_kl` + `circuitry.tuned_lens.fit_tuned_lens` + the opt-in `tuned_lens_kl` Recorder/scan diagnostic; §4.1, §4.4). **SAE-feature circuits shipped: node-level attribution in v1.5** (`SAEFeatureRunner`); **feature→feature edges + greedy `FeatureACDC` in v1.6** (`SAEFeatureEdgeRunner`, §4.6); **`mlp_out`/`attn_out` SAE sites, multi-site-per-layer composite keying, the TransformerLens backend, and the integrated-gradients variant (`variant='ig'`) all shipped in v1.7.** SAE *reconstruction* metrics shipped in v0.9 (`circuitry.sae`). **~~Per-position feature edges~~** shipped in v1.19 (`per_position=True` flag + `SAEFeatureCircuit.position_scores` + `top_positions()`; §4.6). **~~Parallel-attention intra-layer edge skipping~~** shipped in v1.20 (`arch='parallel'` flag + `_is_parallel_intra_layer()`; §4.6). **~~Transcoder SAEs as intervention sites~~** shipped in v1.21 (`TranscoderWrapper` + `hook_input=True` routing in `SAEFeatureRunner` and `SAEFeatureEdgeRunner`; §4.6). **~~Temporal SAE attribution~~** shipped in v1.22 (`SAEFeatureTemporalRunner` + `TemporalAtPResult`; §4.6). Note: each step is independent; true recurrent-SAE attribution remains a known limitation. **~~Logit lens distributions~~** shipped in v1.23 (`logit_lens_distributions` / `LayerPrediction`; §4.1). **~~Activation steering / CAA~~** shipped in v1.23 (`steer_vector` / `apply_steer`; §4.1, §4.6). **~~Linear probing~~** shipped in v1.24 (`train_linear_probe` / `LinearProbe`; §4.1). **~~Concept erasure (LEACE)~~** shipped in v1.24 (`leace_erase` / `EraseProjection`; §4.1). **~~Future lens~~** shipped in v1.24 (`future_lens_kl`; §4.1). **~~Edge Pruning (NeurIPS 2024)~~** shipped in v1.25 (`EdgePruningRunner` + `EdgePruningResult`; §4.6). **~~HAP~~** shipped in v1.25 (`HAPRunner`; §4.6). **~~CrosscoderWrapper~~** shipped in v1.26 (`CrosscoderWrapper`; §4.6). **~~Gemma Scope / Llama Scope loaders~~** shipped in v1.26 (`load_gemma_scope`/`load_llama_scope`; §4.1). **~~MIB task loaders~~** shipped in v1.27 (`load_ioi`/`load_greater_than`; §4.1). **~~SAEBench metrics~~** shipped in v1.27 (`run_saebench`/`SAEBenchResult`; §4.1). **~~Fourier feature alignment + information bottleneck~~** shipped in v1.27 (`fourier_feature_alignment`/`information_bottleneck_score`; §4.1). **~~DAS (Distributed Alignment Search)~~** shipped in v1.28 (`DASRunner`/`DASResult`; §4.6). **~~Causal Scrubbing~~** shipped in v1.28 (`CausalScrubRunner`/`CircuitHypothesis`/`CausalScrubResult`; §4.6). **~~MDL probing~~** shipped in v1.29 (`mdl_probe`/`MDLResult`; §4.1). **~~Mass-mean probe~~** shipped in v1.29 (`mass_mean_probe`/`MassMeanProbe`; §4.1). **~~verify_linear_representation~~** shipped in v1.29 (§4.1). **~~repe_direction~~** shipped in v1.29 (`core/steer.py`; §4.1). **~~directional_ablation + apply_ablation~~** shipped in v1.29 (§4.1, §4.6). **~~local_intrinsic_dim~~** shipped in v1.29 (Two-NN estimator; §4.1). **~~kernel_alignment~~** shipped in v1.29 (CKA/MNN; §4.1). **~~embedding_uniformity~~** shipped in v1.29 (§4.1). **~~superposition_index~~** shipped in v1.29 (`sae/metrics.py`; §4.1).
- **Note:** causal interventions / activation patching shipped as the `circuitry.patching` subsystem in v1.0 — see §4.6. It is no longer out of scope.
- Web dashboard. TB + markdown report is the UI.
- Differentiability guarantees through diagnostics. Primitives may use non-differentiable ops (`torch.linalg.svd`).

## 9. Risks & mitigations

| Risk | Mitigation |
| --- | --- |
| Recipes accumulate modality-specific cruft and leak back into `core/` | CI import-linter rule: `core/` cannot import `recipes/`, `recorder/`, or `writers/`. Periodic code review of `core/`. |
| Recipe regexes match the **wrong** subset of modules silently (worse than matching nothing) | At `attach()` time the full matched-modules list per `HookPoint` is logged at INFO level and written to `<run_dir>/circuitry/matched_modules.txt`. Recipes can declare `expected_min_matches` per pattern; `strict=True` (default) raises on mismatch. Zero matches always raises. |
| Diagnostic overhead doubles wall-clock training time | §10 sets a hard ≤10% wall-clock budget at default settings; benchmark in CI; per-diagnostic `enabled: bool` so users can drop the expensive ones; `every_n_steps` knob defaults are tuned per recipe (see §10). |
| Public release attracts issues we don't have time for | "Low-key" release; README explicitly says "research code, no support promise." Issues triaged when convenient. |
| TB-primary design alienates wandb / mlflow-first users | `MetricWriter` protocol from day 1 keeps any third-party adapter a ~50-LOC subclass. v0.1.0 shipped a wandb adapter; v0.3.0 removed it as there were no active users — trivially re-addable if demand surfaces. |
| Single-process-only design ages into an architectural dead-end as users hit multi-GPU training | Multi-process design constraints baked into the current protocol (see §11); a future-release FSDP upgrade is additive, not a rewrite. |

## 10. Performance & overhead budget

The most likely 6-month failure mode is "this is cool, but it doubled my training time." The design defends against this with explicit constraints:

- **Wall-clock budget (design target):** at default settings (`every_n_steps=200`, full recipe), `circuitry`'s overhead SHOULD be ≤10% of baseline training step time on a 50M-param transformer. This is the design target and CI regression gate. **Re-validated on GPU with the full-SVD default at a realistic training step: +7.4%** (RTX 5080, 88M decoder, batch 16 × seq 512, 2026-06-03) — see measurements below.
- **Per-diagnostic toggle:** every entry in `weight_diagnostics` / `activation_diagnostics` / `gradient_diagnostics` can be disabled via recipe override. The expensive ones (`heavy_tail_alpha`, `singular_values` on large weights) are documented as such.
- **Subsampling knobs (accurate by default since v1.8):** `singular_values` and the SVD-derived
  weight diagnostics now default to `max_dim=None` — the **full, deterministic SVD** — so the numbers
  emitted live for production-scale models are accurate and reproducible. (Through v1.7 the default was
  `max_dim=512` random-column subsampling, which silently biased *every* diagnostic on any matrix with
  min-dim > 512 — i.e. every real LLM layer — and varied run-to-run; this was the headline finding of the
  v1.7 real-model evaluation, fixed in v1.8.) `max_dim` is retained as an **opt-in performance hatch**:
  passing a `max_dim` (with a fixed `seed` for reproducibility) caps SVD cost on very wide weights at the
  cost of biasing σ_min / the spectral tail. The `use_gram='auto'` fast path (eigvalsh(WᵀW) for
  strongly-rectangular matrices) is unchanged; `condition_number` always uses the full `svdvals` path to
  preserve the exact max/min ratio. **Perf note:** because the accurate default does the full SVD on wide
  weights, per-emit weight-diagnostic cost on large models rises versus the old subsampled default.
  **Re-validated with the full-SVD default (2026-06-03, RTX 5080, 88M decoder, full `llm` recipe,
  `every_n_steps=200`, batch 16 × seq 512): +7.4%** — the full SVD eroded the margin (was +5.3% with
  the old `max_dim=512` subsampling) but the ≤10% budget still holds at default settings. If
  wide-matrix SVD becomes the live bottleneck, pass an explicit `max_dim` per recipe (accepting the
  bias) or raise `every_n_steps`. **`use_gram='auto'` threshold is optimal (benchmarked 2026-06-08):**
  for square matrices (4k×4k, dominant in LLMs) eigvalsh is ~26% *slower* than svdvals on CPU, so
  extending the Gram path to square matrices would regress performance. For ≥3:1 rectangular matrices
  eigvalsh wins by 1.3–5× (speedup grows with aspect ratio) — the existing threshold already captures
  all wins. Accuracy at float64 promotion: `rel_tail_err ≤ 1.4e-5`, `Δheavy_tail_alpha ≤ 1e-4` across
  all tested shapes — the pre-v1.8 "degrades spectral tail" concern no longer applies. **MoE expert weights** are
  batched 3-D tensors (`[n_experts, d_in, d_out]`); the scalar rank primitives reject >2-D (see §4.1) and
  the recorder iterates the expert axis, emitting **per-expert** diagnostics rather than a single
  semantically-wrong flattened rank.
- **Lazy hooks:** activation hooks only run the forward pass capture on the emit step (every N steps). The hook checks `self._should_capture()` and is a no-op otherwise, avoiding per-step allocation cost.
- **Async writer option:** `MetricWriter` adapters MAY implement non-blocking writes (a background thread draining a queue). The TB adapter does this by default; tests use the synchronous null writer.
- **Drift probe overhead:** the v1.4 `drift_probe` diagnostic requires a second forward pass on `probe_batch` per emit step. It is **off by default** (`enabled={"drift_probe": False}`) and adds zero overhead at default settings. Its per-emit cost is proportional to probe batch size and is not separately characterised here — benchmark it if you enable it in production.
- **SAE feature attribution / circuits are post-hoc, not live diagnostics.** `SAEFeatureRunner` (§4.6, v1.5) and `SAEFeatureEdgeRunner` / `FeatureACDCRunner` (v1.6) run on a clean/corrupted prompt pair *outside* the training loop, so they do not count against the ≤10% training-loop budget. The node splice adds ~2 matmuls (encode + decode) per forward and a single backward; edge attribution adds one VJP backward per kept downstream survivor per site-pair (NEVER a dense `d_sae×d_sae` Jacobian — each VJP is freed immediately); `FeatureACDC` adds forward-only set-ablation passes during greedy pruning. **`variant='ig'` cost (v1.7):** the integrated-gradients variant costs `N×` the attrib forward+backward+VJP-loop (default `N=32`); peak memory equals attrib (one VJP alive at a time). The ≤10% wall-clock budget is scoped to the non-IG default path (`variant='attrib'`); `variant='ig'` is a higher-fidelity optional mode whose cost scales explicitly with `n_ig_steps`.

**Measured overhead** (88M-param decoder, full `llm` recipe, `every_n_steps=200`):

| device | training step | baseline | instrumented | overhead |
| --- | --- | -------: | -----------: | -------: |
| RTX 5080 (full-SVD default, v1.8+) | batch 16 × seq 512 (8192 tok) | 74.7 s | 80.2 s | **+7.4%** |
| RTX 5080 (old max_dim=512 subsample) | batch 16 × seq 512 (8192 tok) | 124.0 s | 130.6 s | +5.3% |
| RTX 5080 | batch 4 × seq 64 (256 tok) | 1.36 s | 8.53 s | +525% |
| CPU 16-core (v0.2.0a0) | batch 4 × seq 64 (256 tok) | 23.9 s | 27.5 s | +14.9% |

**At a realistic training step the budget holds: +7.4% on GPU with the full-SVD default** (RTX 5080, batch 16 × seq 512), within the ≤10% target. The overhead ratio is dominated by the roughly *fixed* per-emit diagnostic cost (the shared SVD set + `logit_lens_kl` + `induction_score`, ≈1.3 s/emit on this model/GPU), so it is highly sensitive to how heavy the baseline step is. At the tiny default batch (256 tokens, ≈12 ms/step on GPU) that same fixed cost balloons the ratio to +45%; CPU's slow-but-cheap step lands at +15%. Production training (large batches, hundreds of ms/step) amortises the fixed cost well — which the realistic GPU measurement confirms. On small/fast steps, raise `every_n_steps` or drop the expensive diagnostics via `Recipe.disable` / `Recipe.only`.

(The bench harness defaults to `every_n_steps=25` — 8× more pessimistic than the budget's 200 — and `--batch-size 4 --seq-len 64`. Pass `--every-n-steps 200 --batch-size 16 --seq-len 512` to reproduce the budget-scenario row above.)

Reference benchmark workload: a 50M-param decoder-only transformer on synthetic data, 100 steps, with and without `circuitry` attached, full LLM recipe, `every_n_steps=200`.

## 11. Multi-process (DDP / FSDP) design notes

Current releases are single-process. This section locks in *what circuitry does today* so a future-release FSDP upgrade is additive, not a rewrite.

### Current contract (single process, rank-0 semantics)

- `Recorder.attach()` checks `torch.distributed.is_initialized()`. If True and `rank != 0`, the recorder becomes a no-op (`attach()` returns immediately, all hooks are skipped). This means existing multi-rank training scripts can import `circuitry` without crashing and without duplicate writes; they just don't get diagnostics until the multi-process path lands.
- Primitives in `core/` assume **full, unsharded** tensors. They do not gather. They will silently return wrong numbers if given an FSDP-sharded parameter. The docstring and a runtime assertion (`shape sanity check against module's intended shape`) flag this.
- Writers write to the rank-0 process's filesystem; no rank coordination.

### Future-release path (additive, no rewrite)

To enable multi-process diagnostics in a future release without changing the current API surface:

- `HookPoint` already takes a `source` enum; the future release adds `TensorSource.WEIGHT_FULL` and `ACTIVATION_FULL` variants that trigger an `all_gather_into_tensor` before passing to the primitive. The pattern / modules / selector escape hatches are unchanged.
- `core/` primitives stay single-tensor in / single-float out. The future release adds a small `core/distributed.py` with helpers (`all_gather_sharded_param(param) -> Tensor`) that the recorder calls before the primitive; primitives themselves never know about ranks.
- `MetricWriter` gains an optional `rank: int` constructor argument; the default tensorboard adapter writes from rank 0 only (current behavior). A new `DDPMetricWriter` aggregates histogram tensors across ranks before writing.
- The `StepContext.gradients` / `activations` / `weights` dicts gain a "gathered" status flag; built-in diagnostics ignore it (they only see post-gather tensors), but custom diagnostics that want raw shards can opt in.

Net result: same recipes, same primitives, same `Recorder` constructor signature. Only the `source` enum gains values, `MetricWriter` gains an optional kwarg, and one new file (`core/distributed.py`) appears. No existing user code breaks.

### README MUST state

> "v0.x supports single-process training only. In a multi-rank DDP/FSDP run, `circuitry` no-ops on non-zero ranks; FSDP-sharded parameters will produce incorrect diagnostics on rank 0. Multi-process support is planned for a future release; see §11 of the design spec for the upgrade path."

## 12. Open questions

None blocking implementation. Resolved during brainstorming + Gemini Pro review: name, license, release target, layering, modality strategy, framework support, logging strategy, hook escape hatches, custom-diagnostic API, multi-process upgrade path.
