# circuitry — SOTA roadmap (v1.23–v1.28)

Prepared **2026-06-08**. Derives from a three-agent survey of the codebase, design
contract, and 2024–2025 mechanistic-interpretability literature. Each release is a
coherent bundle that can be implemented, tested, and shipped independently. Work
items within a release are ordered by dependency: implement them top-to-bottom.

**Already confirmed as done (do not re-implement):**
- `logit_lens_kl` / `tuned_lens_kl` — `core/lens.py` ✓
- EAP-IG — `EAPRunner.run(ig_steps=N)` ✓
- TopK / JumpReLU / Gated SAE attribution — `sae/grad.py` allowlist ✓
- SAE temporal, transcoders, parallel-arch — v1.20–1.22 ✓

---

## v1.23 — Primitives + TL fix
**Theme:** close the most visible gaps; quick wins that touch the public API surface.
**Estimated effort:** ~1 week total.

| # | Item | Kind | File(s) | Effort |
|---|------|------|---------|--------|
| 1 | **Design doc backfill** — add `core/dynamics.py` to §3 file listing; add `head_specialization`, `phase_transition_steps`, `head_formation_step`, `grokking_step` to §4.1 primitive catalog; fix `Recipe` code example in §4.4 (missing `attn_head_meta`, `forward_fn`, `probe_batch`, `drift_method`) | doc | `docs/design.md` | XS |
| 2 | **TLSiteResolver path in `sae_edges.py`** — close the `NotImplementedError` that blocks SAE feature-edge attribution on TransformerLens models; mirrors the v1.7 P3 work already done in `sae_features.py` | fix | `patching/sae_edges.py` | S |
| 3 | **Logit lens distribution primitive** — `logit_lens_distributions(residuals, unembed, *, layer_norm=None, top_k=5)` → `list[LayerPrediction]` where `LayerPrediction` holds per-layer top-k token ids + probabilities; complements the existing scalar `logit_lens_kl`; this is the interactive analysis form (not training-time KL) | feat | `core/lens.py` | S |
| 4 | **Activation steering / CAA** — `steer_vector(clean_acts, steered_acts)` → `Tensor`; `apply_steer(model, site, vector, *, coeff=1.0)` context manager that hooks a site and adds `coeff * vector` to the output; Rimsky et al. 2024 | feat | `core/steer.py` | S |

### Verification checklist
- [ ] `test_tl_sae_edges.py` — synthetic TL model, 2 sites; edge scores are finite and non-trivially different from zero
- [ ] `test_logit_lens_distributions.py` — shape/dtype assertions; `top_k=1` argmax matches `logit_lens_kl` minimiser; handles batch dim
- [ ] `test_steer.py` — `steer_vector` is unit-normalised option; `apply_steer` hook is removed on context exit; adding zero vector is identity
- [ ] Design doc diff reviewed and merged in same commit as code changes

---

## v1.24 — Probing & concept ops
**Theme:** representational analysis — train probes, measure what's encoded, erase concepts.
**Estimated effort:** ~1 week.

| # | Item | Kind | File(s) | Effort |
|---|------|------|---------|--------|
| 1 | **Linear probing API** — `train_linear_probe(acts, labels, *, max_iter=1000, C=1.0)` → `LinearProbe`; `LinearProbe.accuracy(acts, labels)` → `float`; `LinearProbe.weight` → `Tensor` (probe direction); works on any activations captured via `Recorder` or `patch_site` | feat | `core/probe.py` | S |
| 2 | **Concept erasure (LEACE)** — `leace_erase(acts, labels)` → `EraseProjection` (orthogonal projection matrix that removes the concept direction); `EraseProjection.apply(acts)` → erased activations; Zou et al. 2023 / Park et al. 2023 | feat | `core/erase.py` | S |
| 3 | **Future lens** — `future_lens_kl(residual, unembed, target_logits, *, horizon=1, layer_norm=None)` → `float`; projects layer `L` residual toward token `t+horizon`; reduces to `logit_lens_kl` at `horizon=0` | feat | `core/lens.py` | S |

### Verification checklist
- [ ] `test_probe.py` — linearly separable synthetic data → accuracy ~1.0; random labels → accuracy ~chance; probe weight has norm 1
- [ ] `test_erase.py` — after erasure, probe accuracy drops to chance; non-target directions unaffected (cosine ~1.0 with baseline)
- [ ] `test_future_lens.py` — `horizon=0` KL equals `logit_lens_kl`; `horizon=1` KL is finite on known sequence

---

## v1.25 — SOTA circuit discovery
**Theme:** replace ACDC as the default circuit-discovery engine with NeurIPS 2024 SOTA.
**Estimated effort:** ~1.5 weeks.

| # | Item | Kind | File(s) | Effort |
|---|------|------|---------|--------|
| 1 | **Edge Pruning** — `EdgePruningRunner`; learns a real-valued mask `m_e ∈ [0,1]` on each edge via a straight-through estimator and L0-regularised loss; forward pass uses `m_e · activation` injection; converges to near-binary masks; Bhaskar & Wettig NeurIPS 2024 (arxiv:2406.16778) | feat | `patching/edge_pruning.py` | M |
| 2 | **`EdgePruningResult`** — stores final binary mask, per-edge importance scores, faithfulness/completeness at convergence; serialises to JSON (same interface as `ACDCResult` / `SAEFeatureCircuit`) | feat | `patching/edge_pruning.py` | XS |
| 3 | **HAP (Hybrid Attribution + Pruning)** — `HAPRunner` that (a) runs `EAPRunner` to pre-score all edges, (b) keeps only top-p% by EAP score, (c) runs `EdgePruningRunner` on that subgraph; Hu et al. 2025 (arxiv:2510.03282) | feat | `patching/hap.py` | S |

### Verification checklist
- [ ] `test_edge_pruning.py` — 3-layer toy transformer with known IOI-style circuit; after convergence the masked-in edges match the planted circuit (>90% F1); faithfulness > 0.9
- [ ] `test_hap.py` — HAP recovers same circuit as EdgePruning alone in ≤60% of the EdgePruning iterations; result objects are equal up to tolerance
- [ ] `EdgePruningResult.to_json` / `from_json` round-trip
- [ ] Layering test still passes (no new forbidden imports)

---

## v1.26 — SAE ecosystem expansion
**Theme:** new SAE architectures + first-class pre-trained weight loaders.
**Estimated effort:** ~1 week.

| # | Item | Kind | File(s) | Effort |
|---|------|------|---------|--------|
| 1 | **Crosscoder SAE support** — `CrosscoderWrapper` analogous to `TranscoderWrapper`; multi-layer encoder (`encode(acts_per_layer: list[Tensor])` → shared feature vector) + per-layer decoder; enables cross-layer and cross-model feature diffing; Anthropic Oct 2024 | feat | `patching/sae_features.py`, `sae/grad.py` | M |
| 2 | **Gemma Scope / Llama Scope loader** — `load_gemma_scope(model_id, layer, width, *, cache_dir=None)` and `load_llama_scope(...)` in `sae/loader.py`; maps HuggingFace weight keys to circuitry SAE interface; Lieberum et al. 2024 (arxiv:2408.05147) | feat | `sae/loader.py` | S |
| 3 | **Matryoshka SAE architecture** — add `"matryoshka"` to `assert_supported_sae` allowlist; `sae_decompose` handles nested-width decode (pass `width=` kwarg to `.decode()`); Bussmann et al. 2025 (arxiv:2503.17547) | feat | `sae/grad.py` | S |
| 4 | **BatchTopK SAE variant** — add `"batch_topk"` to allowlist; identical gradient path to `"topk"` but with a note that the per-sample sparsity may vary | feat | `sae/grad.py` | XS |

### Verification checklist
- [ ] `test_crosscoder.py` — 2-layer synthetic crosscoder; `CrosscoderWrapper` attribution scores are finite; `hook_input=True` path from `TranscoderWrapper` extended correctly
- [ ] `test_loader_scope.py` — mock the HuggingFace download; assert correct weight shapes and dtype after mapping
- [ ] `test_matryoshka.py` — `sae_decompose` at each valid width sums to original activation within tolerance
- [ ] `test_batchtopk.py` — smoke test: assert no exception from `assert_supported_sae` and gradient flows

---

## v1.27 — Evaluation & benchmarks
**Theme:** make circuitry results comparable against public leaderboards.
**Estimated effort:** ~1 week.

| # | Item | Kind | File(s) | Effort |
|---|------|------|---------|--------|
| 1 | **MIB task loaders** — `benchmarks/mib.py`; `load_ioi()`, `load_greater_than()`, `load_mcqa()`, `load_arc()` return `(clean_inputs, corrupted_inputs, metric_fn)` triples compatible with all `*Runner.run()` signatures; Mueller et al. ICML 2025 (arxiv:2504.13151) | feat | `benchmarks/mib.py` | S |
| 2 | **SAEBench metric runner** — `benchmarks/saebench.py`; `run_saebench(sae, model, *, tasks=None)` → `SAEBenchResult`; implements all 8 core metrics: CE loss score, L0, sparse probing accuracy, spurious correlation removal, explained variance, absorption score, RAVEL, KL at zero ablation; Karvonen et al. 2025 (arxiv:2503.09532) | feat | `benchmarks/saebench.py` | M |
| 3 | **Grokking monitor improvements** — `core/dynamics.py`; add `fourier_feature_alignment(W, task_freqs)` → `float` (cosine of weight spectrum vs. task-relevant Fourier modes); add `information_bottleneck_score(acts_train, acts_val, labels)` → `float` (mutual-information proxy via binned histogram); Nanda et al. ICLR 2024, Leavitt & Morcos 2024 | feat | `core/dynamics.py` | S |

### Verification checklist
- [ ] `test_mib_loaders.py` — all four loaders return correct token tensor shapes and `metric_fn(clean_logits)` is differentiable
- [ ] `test_saebench.py` — each metric returns a finite float on a 2-layer synthetic model + tiny SAE; CE loss score = 0.0 for a perfect SAE
- [ ] `test_dynamics_grokking.py` — `fourier_feature_alignment` = 1.0 when W is exactly a Fourier mode; `information_bottleneck_score` monotonically increases with label correlation in synthetic data

---

## v1.28 — Causal alignment  *(research-stage, lower confidence)*
**Theme:** causal variable localization — the hardest and most research-stage items.
**Estimated effort:** ~2 weeks.

| # | Item | Kind | File(s) | Effort |
|---|------|------|---------|--------|
| 1 | **DAS (Distributed Alignment Search)** — `DASRunner`; learns an orthogonal rotation `R` such that `R · acts` aligns with specified causal variables; gradient descent with causal interchange-intervention loss; Geiger et al. NeurIPS 2023 (arxiv:2303.02536) | feat | `patching/das.py` | L |
| 2 | **Causal Scrubbing** — `CausalScrubRunner`; given a `CircuitHypothesis` (a subgraph + node→variable assignments), computes a faithfulness score via behavior-preserving resampling ablations; Conmy et al. / Redwood Research 2022 | feat | `patching/scrubbing.py` | L |

### Verification checklist
- [ ] `test_das.py` — 2D synthetic representation with known causal variable; DAS recovers rotation with cosine > 0.99 to ground truth
- [ ] `test_scrubbing.py` — correct circuit hypothesis scores ~1.0; random hypothesis scores near 0 on toy model
- [ ] Both runners pass layering test; no hidden `.cuda()` calls

---

## Cross-cutting notes

**Layering rules (CI-enforced) apply to every release:**
- `benchmarks/` may import `core/` and `patching/` but not vice versa
- `core/steer.py`, `core/probe.py`, `core/erase.py` are pure functions — no I/O, no `.cuda()` calls
- Any new public export must be added to the appropriate `__init__.py` `__all__` and to §4 of `docs/design.md` in the same commit

**Version increments:** each release bumps the minor version (`1.23`, `1.24`, …). Patch releases (`.1`, `.2`) are for bug fixes only and don't introduce new public API.

**Open items NOT on this roadmap (still blocked):**
- Drift probe GPU benchmark — needs GPU; deferred indefinitely
- DDP/FSDP-aware reductions (§11) — multi-GPU env required; design in §11 is a sketch, not a locked contract
