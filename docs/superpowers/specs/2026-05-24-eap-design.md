# EAP (Edge Attribution Patching) — Design Spec

> **Sub-spec 2 of the v1.0 patching pillar.** Builds on the core intervention
> primitive (sub-spec 1: `Site` / `HFSiteResolver` / `TLSiteResolver` /
> `patch_site` / `PatchRunner` / `core.patching` metrics), already shipped.
> Sub-spec 4 (ACDC) consumes the edge graph + scores produced here.

**Goal:** Compute an approximate causal-attribution score for every *edge* in
a transformer's residual-stream computation graph, using gradients (a few
forward/backward passes) instead of one forward pass per edge — so circuit
discovery scales to whole models.

**Architecture:** A new `EAPRunner` (its own execution path, not `PatchRunner`'s
per-site loop) that caches node activations on clean + corrupted prompts, takes
metric gradients w.r.t. each reader's residual-stream input, and scores edges
analytically. A `Node`/`Edge`/`Slot` graph data model. Dual backend: native
TransformerLens hook points, and a full HF reader-side decomposition.

**Tech Stack:** Python 3.12, PyTorch, HF transformers (eager attention,
Llama-family first), TransformerLens (optional, lazy).

---

## 1. Scope & taxonomy

**EAP** (Syed, Rager, Conmy 2023) assigns each *edge* in the residual-stream
computation graph an attribution score that **approximates** the effect of
patching that edge, computed from gradients rather than per-edge forward
passes. "Approximate" is load-bearing:

- It is a **first-order Taylor expansion** of the true edge-patch effect.
- On pre-LN transformers, the LayerNorm scale (`1/std` of the read residual) is
  treated as a **stop-gradient constant** so the read is locally linear in the
  residual. This is the standard EAP approximation and a primary source of EAP's
  inexactness.

**EAP-IG** (Hanna, Pezzelle, Belinkov 2024, "Have Faith in Faithfulness")
integrates the gradient along interpolated **node activations** between the
corrupted and clean runs, mitigating the zero-gradient/saturation failure of
vanilla EAP. Unified here via an `ig_steps` knob: `ig_steps=1` = vanilla EAP;
`ig_steps=N>1` = EAP-IG over N activation-interpolation points. (Input-path IG
is explicitly out of scope.)

**In scope:** vanilla EAP + activation-path EAP-IG; the heads + MLPs + embed +
logits node graph; q/k/v-typed edges; TL backend (full) + HF backend (full
q/k/v reader-side, eager, Llama-family first); `EAPResult` with ranked
edges / top-k / threshold helpers.

**Out of scope:** neuron-level edges (the graph explodes to billions of edges at
real scale — neuron-level resolution is served by direct `patch_site` patching
and AtP\* node attribution, sub-spec 3); ACDC iterative search (sub-spec 4,
consumes this graph); non-Llama HF architectures (clean error, follow-on);
input-path integrated gradients.

---

## 2. Data model (`patching/graph.py`)

```python
Slot = Literal["q", "k", "v", "mlp_in", "logits_in"]

@dataclass(frozen=True)
class Node:
    kind: str          # "embed" | "attn_head" | "mlp" | "logits"
    layer: int | None  # None for embed / logits
    head: int | None   # set for attn_head

@dataclass(frozen=True)
class Edge:
    writer: Node       # writes to the residual stream
    reader: Node       # reads from the residual stream
    slot: Slot         # which input of the reader this edge feeds
```

- **Writers:** `embed`, every `attn_head(L,H)`, every `mlp(L)`. (`logits` never
  writes.)
- **Readers:** every `attn_head(L,H)` (via slots `q`, `k`, `v`), every `mlp(L)`
  (slot `mlp_in`), and `logits` (slot `logits_in`). (`embed` never reads.)
- **Causal validity:** an edge `(writer, reader, slot)` exists iff `writer` is
  strictly upstream of `reader` in the forward order (embed < layer 0 attn <
  layer 0 mlp < layer 1 attn < … < logits). Same-layer attn→mlp edges exist;
  attn→same-layer-attn does not.

`build_graph(n_layers, n_heads) -> list[Edge]` enumerates the causally-valid
edge set.

---

## 3. Execution model (the EAP trick)

For a (clean, corrupted) prompt pair — which must be **position-aligned** (same
sequence length), as in `PatchRunner`:

1. **Corrupted forward** — cache each node's **residual-stream contribution**
   (the `d_model` vector it *adds* to the residual stream — see §4 "Writer
   activations"; no grad).
2. **Clean forward + backward** — cache each node's residual-stream contribution
   **and** the gradient of the metric w.r.t. each *reader slot's residual-stream
   input* (also `d_model`).
3. **Analytic scoring** — for every edge, a cheap dot product over `d_model`
   (§5). No per-edge forward passes; cost is `O(forwards + backwards) +
   O(edges)` cheap ops.

Both operands of the score live in the **same `d_model` residual space** — this
matching is the central correctness invariant (see §4 and the exact cross-check
in §8).

**EAP-IG (`ig_steps=N>1`):** repeat step 2 at activation-interpolation points
`a_k = corrupted_act + (k/N)·(clean_act − corrupted_act)` for `k = 1..N`,
accumulating `grad[reader, slot]` at each, then average. The `Δact` term
(§5) is computed **once** from the original clean/corrupted runs; only the
gradient is integrated.

`EAPRunner` is a **new runner**, not `PatchRunner`'s per-site loop. It reuses
`Site`/resolver machinery only for *placing* the caching/gradient hooks.

---

## 4. Model support — writer activations & reader-side gradients

### Writer activations (must be `d_model` residual contributions)

A writer's activation is **what it adds to the residual stream**, a `d_model`
vector — *not* its internal pre-projection activation. This matters most for
attention heads:

- **`attn_head(L,H)`** → `z[h] @ W_O[h]`, where `z[h]` is the per-head `o_proj`
  *input* (`d_head`) and `W_O[h]` is the `(d_head, d_model)` output-projection
  slice for head `h`. The product is the head's `d_model` residual contribution.
  **The primitive's `attn_head_out` resolver captures only `z[h]` (`d_head`
  space)** — EAP must additionally apply `W_O[h]`. Capturing `z[h]` alone and
  dotting it against a `d_model` reader gradient is the latent space-mismatch
  bug this section exists to prevent.
- **`mlp(L)`** → the MLP block output (`down_proj` output), already `d_model`
  (reuse `mlp_out`).
- **`embed`** → the token-embedding output, `d_model`. TL: `hook_embed`. HF:
  output of the token-embedding module (`model.embed_tokens` / resolved via the
  recipe), captured as a special-case writer in `EAPRunner` (the primitive's
  `Site` has no `embed` component; EAP handles it directly rather than extending
  `Site`).

### Reader-side gradients

For each reader `(head, slot)` (and `mlp_in`, `logits_in`), the gradient of the
metric w.r.t. **that reader's residual-stream input**, in residual (`d_model`)
space, kept separate per reader-slot so edges are typed.

### TL backend (clean)

TransformerLens exposes per-slot input hook points:

- `blocks.{L}.hook_q_input`, `hook_k_input`, `hook_v_input` — shape
  `(batch, pos, n_heads, d_model)`, the residual as seen by each head's q/k/v.
- `blocks.{L}.hook_mlp_in` — the MLP's residual input.
- the residual into the unembed — `logits_in`.

Because each slot is a distinct graph node, autograd yields per-slot gradients
directly — no manual back-mapping, no explicit LN handling. Writer activations
(all `d_model`, per the definition above): attention heads via
`blocks.{L}.attn.hook_result` (per-head `z @ W_O`, requires
`use_attn_result=True`) — or compute `hook_z @ W_O` if `hook_result` is
disabled; `blocks.{L}.hook_mlp_out`; `hook_embed`.

### HF backend (the hard part)

HF has no per-slot input hooks. Per reader `(head, slot ∈ {q,k,v})`, use a
**two-hook mechanism**:

1. **Forward hook on the pre-attention `LayerNorm`** capturing its *input* (the
   read residual `x_resid`). From `x_resid` compute the LN scale
   (`1/rms(x_resid)` for RMSNorm; `1/std` for LayerNorm), retained as a
   **stop-gradient constant**. (The LN scale depends on the LN *input* and so
   cannot be recovered from the projection output — hence the separate hook.)
2. **Backward hook on `q_proj`/`k_proj`/`v_proj` output**, reshaped per head, to
   capture `dL/dq[h]`, `dL/dk[h]`, `dL/dv[h]`.

Then map the per-head projection-space gradient back to residual space by
multiplying by the per-head projection-weight slice and the stop-grad LN scale:

```
grad_resid[h, q] = ln_scale_stopgrad ⊙ (dL/dq[h] · W_Q[h])
```

where `W_Q[h]` is the `(d_head, d_model)` weight slice for head `h`. **The exact
operand order / transpose is fixed by a shape-and-correctness test, not pinned
in prose** — `dL/dq[h]` is `(batch, pos, d_head)`, the result must be
`(batch, pos, d_model)`, and the orientation is whatever makes the analytic
score match brute-force patching on the linear toy (§8).

- **GQA:** with `n_kv_heads < n_heads`, query head `h` reads kv-head
  `h // (n_heads // n_kv_heads)` for the `k`/`v` slots.
- **Writer side** (residual-space contributions, per the "Writer activations"
  definition above): for attention heads, capture `z[h]` via the primitive's
  `attn_head_out` resolver (`o_proj` input per head, `d_head`) **and then apply
  `W_O[h]`** (the per-head `o_proj` weight slice) to get the `d_model`
  contribution — `attn_head_out` alone is insufficient. `mlp(L)` reuses
  `mlp_out` directly (already `d_model`). `embed` is captured as a special case
  (token-embedding output).
- **Constraints:** eager attention only (per-head decomposition); Llama-family
  first; a clean, descriptive error for unsupported architectures (not a wrong
  number).

---

## 5. Scoring

For an edge `(writer, reader, slot)`:

```
score = Σ_d_model ( corrupted_act[writer] − clean_act[writer] ) ⊙ grad[reader, slot]
```

- Both operands are `d_model`-dimensional residual-space vectors (per position);
  the dot product is over `d_model`.
- **Position handling:** summed over positions by default (one scalar per edge);
  a config flag enables per-position scores.
- **Direction:** pinned to **noising** — `Δact = corrupted − clean` with the
  gradient taken on the clean run, so `score ≈ L(corrupted_at_edge) − L(clean)`,
  the causal effect of corrupting the edge. (Denoising is a sign flip; not a
  separate code path in v1.0.)
- **EAP-IG:** replace `grad[reader, slot]` with the activation-interpolated
  average (§3).

---

## 6. Module layout

```
patching/
  graph.py     NEW — Node, Edge, Slot, build_graph() causal enumeration.
  eap.py       NEW — EAPRunner, EAPResult, vanilla + IG scoring, the
                     clean/corrupted cache + reader-gradient orchestration.
  sites.py     EXTEND — reader-slot resolution (q/k/v/mlp_in/logits_in →
                     module + per-head back-map metadata) for HF and TL,
                     alongside the existing writer-side resolvers.
core/patching.py  REUSE — logit_diff / kl_divergence / ce_loss as metrics.
```

**Layering:** unchanged invariants — `patching/` imports `core/` + `recipes/`,
not `cli/`; `core/` does not import `patching/`; `transformer_lens` stays a lazy
optional import.

---

## 7. Output

```python
@dataclass
class EAPResult:
    scores: dict[Edge, float]
    def top_k(self, n: int) -> list[tuple[Edge, float]]: ...      # by |score|
    def threshold(self, tau: float) -> list[Edge]: ...            # |score| >= tau
    def ranked(self) -> list[tuple[Edge, float]]: ...             # all, sorted
```

The `(graph, scores)` pair is the circuit-discovery output and the exact input
ACDC (sub-spec 4) prunes.

---

## 8. Testing

| Test | What it verifies |
|------|-----------------|
| **Exact cross-check (key)** | On a **linear** toy (all `nn.Linear`, no GELU/softmax/ReLU in any reader→output path, identity LayerNorm, `logit_diff` metric), the analytic EAP score for each edge equals the brute-force per-edge `patch_site` metric delta to tight tolerance (≈1e-5). The first-order approximation is exact for a linear model, so this pins the gradient formulation, the back-map orientation, and the sign convention. |
| **Approx cross-check** | A mildly non-linear toy (one GELU): EAP scores correlate strongly with brute-force patching (loose tolerance / rank correlation), confirming the approximation degrades gracefully. |
| **Graph enumeration** | `build_graph` emits exactly the causally-valid edges; counts match `(heads·layers + layers)`-derived expectations; no upstream-of-self edges. |
| **TL end-to-end** | On a small `HookedTransformer` (`skipif` not installed): EAP runs and returns finite scores for all edges. |
| **HF reader back-map** | On a tiny Llama-like config: per-head `dL/dq[h]` back-mapped to residual space has correct shape and matches a hand-computed reference; GQA mapping selects the right kv-head. |
| **Writer space** | An attention-head writer activation is `d_model` (= `z[h] @ W_O[h]`), not `d_head`; shape matches the reader gradient so the score dot product is well-formed. (Defends the §4 space-match invariant directly, independent of the end-to-end cross-check.) |
| **IG reduces to vanilla** | `ig_steps=1` produces the vanilla-EAP scores exactly. |
| **IG runs** | `ig_steps=N>1` runs and averages N gradient points; Δact computed once. |
| **Position modes** | Sum-over-positions (default) vs per-position both produce correct shapes. |
| **Unsupported HF arch** | A non-Llama MLP/attention layout raises a clean, descriptive error. |
| **Layering** | `patching/eap.py`, `graph.py` import only core/recipes/torch; `transformer_lens` not imported at package-import time. |

---

## 9. Risks & sequencing

- The **HF reader-side decomposition** (two-hook LN scale + per-head back-map +
  GQA) is the bulk of the effort and the main correctness risk. The exact
  cross-check (§8) is the gate: nothing ships until analytic scores match
  brute-force patching on the linear toy.
- The implementation plan should land the **graph + scoring + TL backend +
  exact cross-check first** (provable correctness on a controlled model), then
  the **HF backend** as a distinct task cluster validated against the same
  cross-check.
- EAP-IG sits on top of the same caching/scoring machinery; it is the last
  increment, gated by the `ig_steps=1`-matches-vanilla test.

---

## 10. Public API sketch

```python
from circuitry.patching.eap import EAPRunner
from circuitry.patching.sites import HFSiteResolver
from circuitry.core.patching import logit_diff

runner = EAPRunner(model, HFSiteResolver.from_config(model.config))
result = runner.run(
    clean_inputs=clean_ids,
    corrupted_inputs=corrupted_ids,
    metric=lambda logits: logit_diff(logits, correct=tok_a, incorrect=tok_b),
    ig_steps=1,            # 1 = vanilla EAP; N>1 = EAP-IG
)
circuit = result.threshold(0.01)   # list[Edge]
for edge, score in result.top_k(20):
    print(edge, score)
```

Signatures are a sketch; exact names are settled in the implementation plan.
