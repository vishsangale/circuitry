# AtP\* (Attribution Patching with corrections) — Design Spec

> **Sub-spec 3 of the v1.0 patching pillar.** Builds on sub-spec 1 (core
> primitive: `patch_site` / `PatchRunner` / `Site` / resolvers) and reuses
> sub-spec 2's (EAP) backend machinery — writer-activation caching, `_t`
> differentiable metrics, HF/TL backends, GQA + module-location logic.
> Sub-spec 4 (ACDC) may consume node scores alongside EAP's edge scores.

**Goal:** Localize a model's behaviour to individual **nodes** (components) by
approximating, via gradients, the effect of patching each node's activation —
with AtP\*'s two corrections (QK fix, GradDrop) for the failure modes of vanilla
attribution patching, plus a verification helper that calibrates the top-K
against real patching.

**Architecture:** A new `AtPRunner` (its own runner; reuses EAP's
caching/backends) producing per-node attribution scores. Vanilla AtP for
value/MLP/neuron/embed nodes; the QK fix for query/key nodes; optional GradDrop.
An `AtPResult` mirroring `EAPResult` with a `verify_top_k` helper.

**Tech Stack:** Python 3.12, PyTorch, HF transformers (eager — `output_attentions`
for the QK fix; Llama-family), TransformerLens (optional, lazy).

---

## 1. Scope & taxonomy

AtP\* (Kramár, Conmy et al. 2024, "AtP\*: An efficient and scalable method for
localizing LLM behaviour to components") is **node attribution**: for each node,
approximate the effect of patching that node's activation clean→corrupted.
Distinct from EAP (sub-spec 2), which attributes *edges*. Because AtP\* is
**O(nodes)** (not O(edges)), neuron-level nodes are feasible.

- **Vanilla AtP:** `effect(node) ≈ Δact_node · grad_node`, a first-order Taylor
  term. `grad_node = ∂metric/∂(node's activation)` is the **full** downstream
  gradient (the *total* effect of patching the node) — NOT EAP's component-only
  reader-input gradient. Captured via `retain_grad` on the node's activation.
- **QK fix** (query/key nodes): vanilla AtP is poor for q/k activations because
  the attention softmax saturates. Fix (§4): recompute the attention pattern
  with the patched q/k, propagate the pattern change through the **clean V**,
  project to the head's `d_model` residual contribution via `W_O`, dot with the
  head-output gradient.
- **GradDrop** (optional, `graddrop=True`): mitigate per-position
  sign-cancellation by attributing per-position and combining as the
  **sum of absolute per-position scores** (§5). Costs ~`n_positions` backward
  passes — the expensive correction.

**Inherited approximation:** like EAP, AtP\* treats the pre-LN LayerNorm/RMSNorm
scale as a **stop-gradient constant**. This (plus the first-order Taylor) is why
AtP\* scores are *approximate* — the verification helper (§6) calibrates them.

**In scope:** vanilla AtP + QK fix + GradDrop; nodes = attention q/k/v + MLP
output + MLP neuron + embed; `AtPResult` with `verify_top_k`; HF + TL backends.
**Out of scope:** edges (EAP), ACDC search (sub-spec 4), SAE-feature nodes,
attributing the head's *output* node directly (q/k/v are the attributable
attention activations in v1.0).

---

## 2. Node model

Attributable activations (the things patched), reusing `graph.Node` plus a slot
for attention:

| Node | Activation patched | Space | Scoring |
|------|--------------------|-------|---------|
| `attn_head(L,h)` slot `q` | per-head query (`q_proj` output, head h) | d_model contribution (via QK fix) | **QK fix** |
| `attn_head(L,h)` slot `k` | per-head key (`k_proj` output, head h) | d_model contribution (via QK fix) | **QK fix** |
| `attn_head(L,h)` slot `v` | per-head value (`v_proj` output, head h) | d_head | vanilla AtP |
| `mlp(L)` | MLP block output | d_model | vanilla AtP |
| `mlp_neuron(L,n)` | post-activation intermediate, neuron n | scalar/pos | vanilla AtP |
| `embed` | token-embedding output | d_model | vanilla AtP |

```python
def enumerate_nodes(n_layers, n_heads, d_mlp=None) -> list[AtPNode]: ...
# AtPNode = (Node, slot) where slot ∈ {q,k,v} for attn_head, else None.
# Neuron nodes emitted only when d_mlp is given.
```

`AtPResult` mirrors `EAPResult`: `scores: dict[AtPNode, float]`, with `ranked()`,
`top_k(n)`, `threshold(tau)`, plus `verify_top_k(...)` (§6).

---

## 3. Vanilla AtP scoring (v / mlp / neuron / embed)

`score(node) = Σ (Δact_node ⊙ grad_node)`, summed over the activation's own dims
(d_head for `v`; d_model for `mlp`/`embed`; the scalar neuron value per position
for `mlp_neuron`, summed over positions):

- `Δact_node = corrupted_act_node − clean_act_node` (reuses EAP's writer/activation
  caching at the node's read/output point).
- `grad_node = ∂metric/∂act_node`, the **full** gradient, via `retain_grad` on the
  node's activation tensor (NOT a component-only clone — we want the total effect).
- On a **linear** model this first-order term is **exact**: it equals the
  brute-force `patch_site` metric delta for that node. This is the linear gate (§7).

---

## 4. QK fix (query / key nodes)

Vanilla `Δq·grad_q` mis-estimates q/k effects because softmax is the dominant
nonlinearity. The fix computes the head-output change induced by the q/k patch
through a **local softmax recomputation**, then scores in `d_model` space
(consistent with EAP's attention writer activation `z@W_O`):

For head h, query node (key node symmetric):
1. `Δq_h = q_corrupted_h − q_clean_h` (per-head, d_head).
2. Recompute the attention scores/pattern with the patched query: `pattern' =
   softmax((Q_clean + Δq into head h) · K_clean^T / √d_head + mask)`; the change
   is `Δpattern_h = pattern'_h − pattern_clean_h`.
3. Propagate through the **clean V**: `Δz_h = Δpattern_h @ V_clean_h` (d_head).
4. Project to the residual contribution: `Δhead_out_h = Δz_h @ W_O[h]` (d_model).
5. `score = Σ_d_model (Δhead_out_h ⊙ grad_head_out_h)`, where `grad_head_out_h`
   is `∂metric/∂(head h's d_model residual contribution)` — the SAME activation
   EAP defines as the attention writer (`z[h] @ W_O[h]`), so the gradient-capture
   point is shared with EAP's writer-side machinery.

**Operand space invariant:** `Δhead_out_h` and `grad_head_out_h` are both
`d_model`; a test asserts the shapes/semantics match before the dot product
(mirrors EAP's "writer space" test).

**Backends:** the recomputation needs the clean attention pattern and clean V.
- **HF (eager):** obtain per-head attention probabilities via
  `output_attentions=True` (reuse the v0.9.2 wrapper-safe `config.output_attentions`
  infrastructure); V from the `v_proj` output (per head, GQA-mapped).
- **TL:** `blocks.{L}.attn.hook_pattern` (clean pattern) and `hook_v`.

The ground-truth intervention this approximates is `patch_site` on the per-head
`q_proj`/`k_proj` output (new sites `attn_head_q_out` / `attn_head_k_out`,
resolved per head — added to `sites.py` for both backends). `verify_top_k` (§6)
patches exactly these.

---

## 5. GradDrop (optional, `graddrop=True`)

Vanilla AtP undercounts a node whose per-position gradient flips sign, because
the position sum cancels. GradDrop attributes per-position and combines without
cancellation:

```
score_graddrop(node) = Σ_pos | Δact_node[pos] · grad_node[pos] |
```

where `Δact_node[pos] · grad_node[pos]` is the per-position dot over the node's
feature dims. For the **single-position metrics** this library targets
(e.g. `logit_diff` at the final token), a single backward already yields
`∂metric/∂act[pos]` for every input position `pos` (the gradient tensor carries
the position axis), so the per-input-position contributions `c[pos]` — and thus
the absolute-sum — are computed exactly from **one** backward pass. GradDrop is
opt-in (`graddrop=True`); when off (default), scoring uses the plain summed term
`Σ_pos c[pos]` (§3/§4), and `Σ_pos |c[pos]| ≥ |Σ_pos c[pos]|` always holds.

*Limitation:* for a metric that is itself a **sum over multiple output
positions**, fully isolating cancellation across those output-position loss
terms would need a per-output-position backward variant; the single-backward
input-position absolute-sum here addresses input-position cancellation only.
That variant is a follow-on, not in this sub-spec.

---

## 6. Verification helper

```python
AtPResult.verify_top_k(
    k: int, clean_inputs, corrupted_inputs, metric, resolver
) -> dict[AtPNode, tuple[float, float]]
```

For the top-K nodes by `|score|`, run **real** patching with the shipped
`patch_site` (ground truth): patch the node's activation clean→corrupted, measure
`metric(patched) − metric(clean)`. Returns `{node: (atp_score, true_effect)}`.
This turns the cheap ranked list into a calibrated one and is the **primary
correctness gate for the QK fix** (which the linear toy can't exercise).

Node→`patch_site` mapping: `v` → `attn_head_out`-style per-head v site; `q`/`k` →
the new `attn_head_q_out`/`k_out` sites; `mlp` → `mlp_out`; `mlp_neuron` →
`mlp_neuron` site; `embed` → embed special-case (as in EAP).

---

## 7. Module layout

```
patching/
  atp.py       NEW — AtPRunner, AtPResult, enumerate_nodes, vanilla AtP +
                     QK fix + GradDrop scoring, verify_top_k.
  sites.py     EXTEND — attn_head_q_out / attn_head_k_out per-head sites
                     (q_proj / k_proj output) for HF and TL.
  eap.py       REUSE (refactor shared bits cleanly if needed; do NOT regress
                     EAP) — writer-activation caching, backend/module location,
                     GQA, ln-scale capture.
core/patching.py  REUSE — _t differentiable metrics.
```

Layering unchanged: `patching/` imports `core/` + `recipes/`, never `cli/`;
`transformer_lens` lazy.

---

## 8. Testing

| Test | What it verifies |
|------|-----------------|
| **Vanilla AtP exact (linear)** | On a linear toy, `score(node)` for v/mlp/neuron/embed equals brute-force `patch_site` metric delta per node, tight tolerance (1e-4). The linear gate (q/k are dead on the fixed-pattern toy → trivially 0). |
| **Neuron cross-check (linear)** | `patch_site`-per-neuron on the linear toy matches the neuron AtP score (anchors neuron-node correctness). |
| **QK fix vs brute-force (real)** | On a tiny real HF Llama, the QK-fixed q/k scores approximate brute-force `patch_site(q_h / k_h)` better than vanilla `Δq·grad` does (anti-stub: vanilla q/k attribution is far from ground truth; QK-fixed is close — assert QK-fixed correlation > vanilla correlation, and QK-fixed corr above a floor). |
| **Operand space-match** | For every node type, `Δact` and `grad` have identical shape/semantics before the dot product (defends the §4 space invariant). |
| **GradDrop reduces cancellation** | On a constructed sign-flipping case, `graddrop=True` yields a larger (un-cancelled) magnitude than the plain summed score, and is closer to brute-force for that node. |
| **verify_top_k** | Returns `(atp_score, true_effect)` per top-K node; `true_effect` matches a direct `patch_site` of that node. |
| **GQA** | k/v nodes on a GQA Llama map query→kv-head correctly (reuses EAP's `_kv_head_for`). |
| **Layering / lazy import** | `patching/atp.py` imports core/recipes/torch only; `transformer_lens` not imported at package-import time. |

---

## 9. Risks & sequencing

- The **QK fix** is the main correctness risk (softmax recomputation + V
  propagation + W_O projection + GQA). Its gate is the real-model brute-force
  comparison (§8), since the linear toy can't exercise softmax. Land vanilla AtP
  + the linear/neuron exact gates first (provable), then the QK fix against the
  brute-force comparison, then GradDrop, then `verify_top_k`.
- **GradDrop** is cost (per-position backwards), not correctness, risk — keep it
  behind the flag and test the cancellation behaviour on a small constructed case.
- Reuse EAP machinery but **do not regress EAP** — if shared helpers are
  refactored, EAP's exact tests must stay green.

---

## 10. Public API sketch

```python
from circuitry.patching.atp import AtPRunner
from circuitry.patching.sites import HFSiteResolver
from circuitry.core.patching import logit_diff_t

runner = AtPRunner(model, HFSiteResolver.from_config(model.config))
result = runner.run(
    clean_inputs=clean_ids,
    corrupted_inputs=corrupted_ids,
    metric=lambda logits: logit_diff_t(logits, correct=tok_a, incorrect=tok_b),
    graddrop=False,          # True = GradDrop correction (~n_positions backwards)
    neurons=True,            # attribute individual MLP neurons
)
for node, score in result.top_k(20):
    print(node, score)
verified = result.verify_top_k(
    k=20, clean_inputs=clean_ids, corrupted_inputs=corrupted_ids,
    metric=lambda logits: logit_diff(logits, correct=tok_a, incorrect=tok_b),
    resolver=HFSiteResolver.from_config(model.config),
)  # {node: (atp_score, true_patch_effect)}
```

Signatures are a sketch; exact names settled in the implementation plan.
