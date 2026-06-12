# circuitry — Tooling landscape positioning

> Where circuitry fits, where it defers, and how it complements the existing
> mechanistic-interpretability ecosystem.

---

## Summary

circuitry occupies a distinct position: **training-time diagnostics + post-hoc attribution
in a single, modality-agnostic library**. No other tool in the space covers all three
simultaneously — live training, post-hoc weight analysis, and activation patching — without
requiring a specific model family or framework.

---

## Tool-by-tool comparison

### TransformerLens

**What it does well:** Clean, opinionated residual-stream hook API for transformer circuits;
rich activation cache; built-in logit-lens and direct-logit-attribution; the de-facto standard
for HF-to-interpretable model porting.

**Where circuitry defers:** TransformerLens is the better choice for deep, static circuit
analysis on a single checkpoint of a standard HF transformer. Its hook naming convention
(resid\_pre, z, etc.) and built-in activations cache are more ergonomic for that workflow.

**Where circuitry is differentiated:**
- Live training monitoring — TransformerLens has no live recorder.
- Modality-agnostic — circuitry works on ResNets, SASRec, DLRM, custom nn.Module without
  a TL port.
- Weight diagnostics — effective rank, heavy-tail alpha, spectral entropy, rank trajectory
  have no TransformerLens equivalent.
- SAE circuits interop — circuitry can use TransformerLens as a site-resolution backend
  while layering its own `EAPRunner` / `FeatureACDCRunner` on top.

---

### nnsight

**What it does well:** Remote execution model; works on any PyTorch model via tracing;
single API for local + API-hosted models (NDIF); suitable for large models inaccessible
locally.

**Where circuitry defers:** If the model lives on a remote inference server (NDIF), nnsight
is the right interface. circuitry is local-process only.

**Where circuitry is differentiated:**
- Training-time integration — nnsight is inference-focused; circuitry's Recorder is designed
  to run inside a training loop.
- Quantitative diagnostics — nnsight captures activations; circuitry computes and logs
  interpretability-relevant statistics (rank, entropy, induction scores, etc.) automatically.
- Report generation — circuitry generates a human-readable Markdown report; nnsight returns
  raw tensors for the user to analyse.

---

### pyvene

**What it does well:** Interchange intervention framework; first-class support for
distributed alignment search (DAS / boundless DAS); causal abstraction workflows.

**Where circuitry defers:** For elaborate causal abstraction experiments (boundless DAS,
multi-variable alignment maps), pyvene's intervention framework is more complete.
circuitry ships the core methods (`DASRunner` since v1.28, `HyperDASRunner` since v1.35)
but not pyvene's full intervention-configuration DSL.

**Where circuitry is differentiated:**
- Training monitoring vs single-run intervention — pyvene is a single-forward-pass
  intervention library; circuitry spans the full training lifecycle.
- Weight-space analytics — no equivalent in pyvene.
- Circuit attribution (EAP, AtP\*, ACDC) — pyvene's interventions can be used for
  localisation, but it has no circuit-level graph extraction.

---

### sae\_lens

**What it does well:** First-class SAE loading for all major public releases (EleutherAI,
DeepMind Gemma-scope, Anthropic, etc.); sparse feature visualisation; feature-to-logit
dashboards.

**Where circuitry defers:** For loading, exploring, and visualising public SAE releases,
sae\_lens is far more complete. circuitry's SAE loader is a thin wrapper around sae\_lens
itself (an optional dependency).

**Where circuitry is differentiated:**
- SAE-feature circuits — circuitry's `FeatureACDCRunner` runs greedy circuit extraction over
  feature→feature edges, using the SAE as an intervention site. sae\_lens has no circuit
  extraction.
- Training-time SAE monitoring — circuitry can log SAE reconstruction error and L0 sparsity
  live during training; sae\_lens is inference-only.

---

## What circuitry does not do (deliberate scope limits)

| Out of scope | Why | Better tool |
|---|---|---|
| Model serving / remote inference | Not a training-loop concern | nnsight, NDIF |
| Logit-attribution dashboards / feature vis | Needs a frontend, not a library | Neuroscope, sae\_lens |
| Training data provenance / influence functions | Different analysis axis | kronfluence, trak |
| Sparse fine-tuning / LoRA | Not a diagnostic — is a training method | PEFT, LLaMA-Factory |

---

## How to use circuitry alongside the ecosystem

A typical interpretability workflow might combine all of them:

1. **Train** a model with circuitry's `Recorder` active — live rank, entropy, induction
   scores, and grokking signals in the report.
2. **Export** a checkpoint and load it into **TransformerLens** for static circuit analysis
   and direct logit attribution.
3. **Load** a public SAE via **sae\_lens**; pass it to circuitry's `FeatureACDCRunner` for
   feature-circuit extraction.
4. **Probe** causal structure with circuitry's own `DASRunner` / `EAPRunner` / `AtPRunner`
   (or **pyvene** for boundless-DAS-style experiments).
5. **Scale** to a model hosted on NDIF using **nnsight** for large-model inference, then
   analyse the results in circuitry's compare report.
