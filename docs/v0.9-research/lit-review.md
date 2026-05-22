# Mech-interp tooling & diagnostics for transformer LMs — 2024-2026 frontier

*Literature review scoped to inform `circuitry` v0.9.0 planning.*
*Compiled 2026-05-22. ~1500 words.*

---

## 1. Sparse autoencoders (SAEs) for feature decomposition

The field moved from L1-ReLU baselines to **architectures that decouple "which features fire" from "how strongly they fire."** Three papers anchor the 2024-2025 consensus, and one release sets the de-facto checkpoint distribution standard.

- **Bricken et al. 2023 — "Towards Monosemanticity"** (Anthropic, transformer-circuits.pub). Prior work the 2024 architectures are improving on.
- **Rajamanoharan et al. 2024 — Gated SAEs** ([arXiv:2404.16014](https://arxiv.org/abs/2404.16014), NeurIPS 2024). Splits encoder into a gating path and a magnitude path, eliminating L1 shrinkage; ~50% fewer firing features at matched reconstruction fidelity.
- **Gao et al. 2024 — Scaling and Evaluating SAEs** ([arXiv:2406.04093](https://arxiv.org/abs/2406.04093)). OpenAI's TopK SAEs (enforce L0 by keeping top-k pre-activations); first clean SAE scaling-law paper (up to 16M latents on GPT-4 residuals).
- **Rajamanoharan et al. 2024 — JumpReLU SAEs** ([arXiv:2407.14435](https://arxiv.org/abs/2407.14435)). Per-feature learned threshold; the architecture shipped in Gemma Scope.
- **Lieberum et al. 2024 — Gemma Scope** ([arXiv:2408.05147](https://arxiv.org/abs/2408.05147), [huggingface.co/google/gemma-scope](https://huggingface.co/google/gemma-scope)). 400+ JumpReLU SAEs across every layer/sub-layer of Gemma 2 2B/9B and selected layers of 27B — the first large public SAE bank.

**Reference implementation:** `SAELens` ([github.com/jbloomAus/SAELens](https://github.com/jbloomAus/SAELens)) is the de-facto hub. `SAE.from_pretrained(release, sae_id, device)` loads any SAELens-format SAE (including Gemma Scope) from HF Hub; inference works against plain HuggingFace models, not just TransformerLens.

**What a diagnostics library should ship:** a **load-and-apply** primitive — given a pretrained SAE checkpoint (SAELens format) and a hooked activation tensor, return reconstruction, sparse code, and reconstruction-error-on-CE-loss as a recorder metric. Do not train SAEs in-process; defer to SAELens.

---

## 2. Activation / attribution / edge-attribution patching

This area is now a **gradient-of-causal-attribution stack** with a clear cost/accuracy frontier.

- **Wang et al. 2022 — IOI Circuit in GPT-2** ([arXiv:2211.00593](https://arxiv.org/abs/2211.00593)). Manual activation patching benchmark; canonical worked example.
- **Conmy et al. 2023 — ACDC** ([arXiv:2304.14997](https://arxiv.org/abs/2304.14997), NeurIPS 2023 Spotlight). Automated edge-importance via patching; selected 68/32k GPT-2-Small edges.
- **Kramár et al. 2024 — AtP*** ([arXiv:2403.00745](https://arxiv.org/abs/2403.00745)). 2 forward + 1 backward approximation of full patching; the "Q/K fix" (linearize attention probs) and "GradDrop" (zero grads through residual skips) remove the two false-negative modes of naive attribution patching.
- **Syed et al. 2023/2024 — EAP** ([github.com/Aaquib111/edge-attribution-patching](https://github.com/Aaquib111/edge-attribution-patching), BlackBoxNLP 2024). First-order linearization at the edge level; outperforms ACDC at constant cost.
- **Marks et al. 2024 — Sparse Feature Circuits** ([arXiv:2403.19647](https://arxiv.org/abs/2403.19647), ICLR 2025). Composes attribution patching with SAEs so circuit nodes are interpretable features, not polysemantic heads.

**Reference implementations:** TransformerLens (`HookedTransformer.run_with_cache` + `interventions`), nnsight (`with model.trace(...)`), and the Marks et al. and EAP repos. AtP* has a PyTorch+nnsight port at [github.com/koayon/atp_star](https://github.com/koayon/atp_star).

**What a diagnostics library should ship:** a **clean-vs-corrupt cache-diff primitive** — capture activations on two batches, replace one tensor with the other at a hooked module, return the metric delta. That is the minimum subroutine all four methods above are built on. Don't ship ACDC's outer loop; do ship the inner patching primitive plus a gradient-attribution variant (AtP) gated behind an optional flag, since the gradient version is a single backward pass and gives you the linear approximation everything else extends.

---

## 3. Logit lens / tuned lens / residual-stream analysis

Mature, low-risk, high-utility area.

- **nostalgebraist 2020 — "interpreting GPT: the logit lens"** (LessWrong). Project hidden state through `unembed`; canonical zero-cost residual readout.
- **Belrose et al. 2023 — "Eliciting Latent Predictions from Transformers with the Tuned Lens"** ([arXiv:2303.08112](https://arxiv.org/abs/2303.08112), [github.com/AlignmentResearch/tuned-lens](https://github.com/AlignmentResearch/tuned-lens)). Trains a per-layer affine probe to fix the logit lens's representational drift; ships pretrained probes for major open models up to 20B.
- **Pal et al. 2023 — "Future Lens"** ([arXiv:2311.04897](https://arxiv.org/abs/2311.04897)). Shows residual states encode tokens ≥ t+2 with >48% accuracy at certain layers — extends the lens idea to multi-token lookahead.

**Reference implementation:** `tuned-lens` library (loads pretrained probes from HF Hub). TransformerLens exposes residual-stream caches directly; `transformer_lens.utils.test_prompt` includes a logit-lens helper.

**What a diagnostics library should ship:** **per-layer logit-lens KL-to-final** as a one-line activation metric on the residual stream. Cheap, deterministic, no probe to train. Tuned-lens is more accurate but needs a per-model probe — defer it to a recipe that *loads* a tuned-lens checkpoint rather than trains one.

---

## 4. Induction-head / circuit detection at scale

- **Olsson et al. 2022 — "In-Context Learning and Induction Heads"** ([arXiv:2209.11895](https://arxiv.org/abs/2209.11895), [transformer-circuits.pub](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html)). Defines the prefix-matching + copy score; shows induction heads form in a sharp phase transition coincident with the in-context-learning bump in loss.
- **McDougall et al. 2023 — "Copy Suppression"** ([arXiv:2310.04625](https://arxiv.org/abs/2310.04625)). Reverse-engineers L10H7 in GPT-2 Small (~77% effect explained); demonstrates that "negative heads" are a stable, screenable category.
- **Gould et al. 2023 — "Successor Heads"** and later **attention-pattern-entropy** screening work continue this line; head-pattern entropy is a cheap one-shot diagnostic.

**Reference implementation:** TransformerLens ships `prefix_matching_score` and `copying_score` utilities; the canonical induction-head screen is ~20 lines on top of `run_with_cache`.

**What a diagnostics library should ship:** **induction-score** (prefix-matching probability on a repeated random-token sequence) per attention head, plus **attention-pattern entropy** per head. Both are cheap weight+activation primitives that fit cleanly alongside the existing `attention_head_rank`.

---

## 5. Training-dynamics interpretability

- **Nanda et al. 2023 — "Progress Measures for Grokking"** ([arXiv:2301.05217](https://arxiv.org/abs/2301.05217), ICLR 2023). Reverse-engineered modular-addition transformer; defines Fourier-basis progress measures and the memorization → circuit-formation → cleanup phase decomposition.
- **Tigges et al. 2024 — "LLM Circuit Analyses Are Consistent Across Training and Scale"** (NeurIPS 2024). Circuits identified at one scale/checkpoint persist; supports cross-checkpoint diagnostic reuse.
- **Xu et al. 2024 — "Tracking Feature Dynamics in LLM Training (SAE-Track)"** ([arXiv:2412.17626](https://arxiv.org/abs/2412.17626)). Tracks SAE feature drift / shifting / grouping across checkpoints.
- **Developmental interpretability review 2025** ([arXiv:2508.15841](https://arxiv.org/abs/2508.15841)). Surveys the area; the metrics that have stuck are induction-score curves, effective-dimensionality curves, and circuit-faithfulness across checkpoints.

**Reference implementation:** Nothing has consolidated. Most work re-implements per-paper.

**What a diagnostics library should ship:** because circuitry already has a `Recorder` workflow, the natural fit is **per-step induction-score and per-step residual-stream effective-rank logging.** This is mostly a matter of wiring existing primitives into a training loop — high value, low new-code cost.

---

## 6. Tooling landscape

| Library | Style | Pattern | Notes |
|---|---|---|---|
| **TransformerLens** | Custom re-implementations of major arches | PyTorch forward hooks + named cache | Mature; lags new model families because each needs hand-conversion. |
| **nnsight** ([arXiv:2407.14561](https://arxiv.org/abs/2407.14561), [github.com/ndif-team/nnsight](https://github.com/ndif-team/nnsight)) | Wraps any PyTorch model | Intervention-graph / tracing proxy (`with model.trace(...)`) | Works against raw HF models; supports remote execution via NDIF. |
| **pyvene** ([arXiv:2403.07809](https://arxiv.org/abs/2403.07809), NAACL 2024 Demo) | Declarative intervention configs | Config-driven hook registration | Stanford NLP; strong for causal-abstraction work. |
| **SAELens** | SAE-centric | Hooks + checkpoint registry | Backend-agnostic; integrates with TransformerLens, nnsight, raw HF. |
| **OpenAI Transformer Debugger** ([github.com/openai/transformer-debugger](https://github.com/openai/transformer-debugger)) | Visualization app | React frontend + activation server | GPT-2-small only; effectively unmaintained since 2024. |

**Pattern divide:** TransformerLens and circuitry are **forward-hook** libraries (`register_forward_hook` + metric). nnsight and pyvene are **intervention-proxy** libraries (defer execution, build a graph, run). Forward-hooks win for low-overhead training-time diagnostics; intervention-proxies win for ad-hoc post-hoc patching. The gap: a forward-hook library with SAE-aware metrics and a lens-style residual readout.

---

## Synthesis — what to ship in v0.9.0

I'd prioritize, in order:

1. **Logit-lens KL-to-final as a per-layer activation primitive.** Highest leverage : cost ratio in this list. It's a single matmul against the unembed, no training, no extra checkpoint, slots straight into the existing recorder. It immediately gives users a per-layer "where does this model commit to the answer" curve — visually compelling, complementary to the existing rank/dead-fraction primitives, and the only thing on this list that's strictly cheaper than what circuitry already does. Sets up tuned-lens *loading* (not training) as a v0.10 follow-up.

2. **Induction-score primitive (prefix-matching + copy score).** Pairs naturally with the existing `attention_head_rank`. Computable from a fixed repeated-random-token probe sequence at any recorder tick; clean phase-transition signal during training; well-defined, well-cited, ~50 lines. Pure observation, no intervention — fits the library's stated contract perfectly.

3. **An SAE *load-and-apply* primitive against the SAELens checkpoint format.** This is the bigger lift and the higher-risk pick — it brings in a dependency on SAELens (or its checkpoint schema) and forces you to decide on a `SparseFeatureMetric` shape. But it's the only way circuitry stays relevant to the post-2024 mech-interp mainstream, and Gemma Scope means there are 400+ free public checkpoints to test against. Ship `load_sae(release, sae_id) → SAE` plus an activation-side primitive that returns `(reconstruction_error, l0, l1)` from a hook. Explicitly do **not** train SAEs in v0.9.

**Defer:**
- **Activation/attribution patching.** It requires a clean-vs-corrupt cache abstraction that doesn't exist in circuitry today and pushes the library from "observation" toward "intervention" — that's a design-doc-amendment-sized change, not a v0.9 feature. AtP via gradients is tempting because it's hook-friendly, but the API surface to do it well is large. Punt to v0.10+.
- **Tuned-lens training.** Adds a training loop the library doesn't otherwise have. Load-only is fine if/when a tuned-lens checkpoint becomes useful.
- **Training-dynamics-specific metrics beyond what falls out of #1 and #2.** Once you ship logit-lens-KL and induction-score per recorder tick, you *already have* the canonical training-dynamics curves. No separate machinery needed.

The unifying thread: **all three picks are pure-observation, fit the existing forward-hook contract, do not require an intervention-proxy abstraction, and align circuitry with the diagnostics-vs-intervention gap identified in §6.**
