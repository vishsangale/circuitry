# ACDC (Automatic Circuit DisCovery) — Design Spec

> **Sub-spec 4 of the v1.0 patching pillar** (the last of the three attribution/
> search methods). Builds on sub-spec 1 (core primitive: `patch_site`), sub-spec
> 2 (EAP: `patching/graph.py` Node/Edge/build_graph + writer-activation caching +
> HF/TL backends), and optionally consumes EAP's edge scores
> (`EAPResult.scores`) for prune ordering.

**Goal:** Discover a minimal circuit — a subset of the residual-stream
computation graph's edges that reproduces the full model's clean behavior within
a tolerance τ — by greedy, iterative edge pruning with corrupted-resample
ablation.

**Architecture:** A new `ACDCRunner` that walks readers in reverse-topological
order, tentatively ablates each incoming edge (corrupted-resample), keeps the
ablation if it costs less than τ in KL-to-clean, and returns the surviving
circuit. Forward-only (no gradients). Reuses EAP's graph, writer-activation
caching, and backends; the **set-ablation forward** is new.

**Tech Stack:** Python 3.12, PyTorch, HF transformers (eager, Llama-family),
TransformerLens (optional, lazy).

---

## 1. Scope & taxonomy

ACDC (Conmy, Mavor-Parker, Lynch, Heimersheim, Garriga-Alonso 2023, "Towards
Automated Circuit Discovery for Mechanistic Interpretability") is a **search
algorithm**, not an attribution method (contrast EAP/AtP\*, which score
edges/nodes). Starting from the *full* graph (all edges present = the full
model), it greedily removes edges that don't matter, leaving a minimal circuit.

This spec implements **faithful canonical ACDC**: each ablated edge feeds the
reader its *corrupted-run* activation, while kept edges propagate the **live**
(current-circuit) activation. The efficient single-forward realization follows
the edge-ablation pattern popularized by auto-circuit (Miller et al. 2024).

- **Forward-only:** ACDC uses real ablation + a recovery metric; **no
  gradients**. (No frozen-model-grad concerns — simpler than EAP/AtP\* on that
  axis. The complexity is the cumulative greedy loop + the set-ablation forward.)
- **Reuses** EAP's `graph.py` (Node/Edge/build_graph — heads + MLPs + embed +
  logits; q/k/v-typed edges into heads, `mlp_in`, `logits_in`), the writer-
  activation caching (the per-writer residual contribution `z@W_O` / MLP-out /
  embed-out), and the HF/TL backend module-location helpers.
- **Does NOT reuse** EAP's per-edge *additive-Δact* patching. EAP's brute force
  adds a fixed `corrupted − clean` delta for **one** edge on an otherwise-clean
  forward, where it is exact. ACDC ablates **many** edges at once and must
  recapture *live* activations each forward (§3) — a different, new mechanism.

**In scope:** corrupted-resample ablation; last-token KL-to-clean recovery
metric (+ custom metric); single τ + a `sweep(taus)` Pareto helper;
`ordering="topo"` and `"eap"` (traversal order); HF + TL backends.

**Out of scope:** mean / zero ablation (follow-on); the EAP-score *skip* speedup
(see §4 — v1.0 ships traversal-ordering only); SAE-feature circuits; report/
compare integration.

---

## 2. The algorithm — reader-keyed greedy pruning

```
build full graph (build_graph); removed: set[Edge] = {}        # full circuit
full_clean_logits = model(clean)                                # KL reference, computed once
cache CORRUPTED writer contributions once: corr_act[u]          # u's residual contribution on corrupted run
current_kl = 0.0                                                # full circuit matches clean exactly

for reader v in REVERSE-topological order (logits → … → layer 0):
    for each incoming edge (u → v) in the chosen order (§4):
        removed.add((u, v))                                     # tentatively ablate
        circuit_logits = forward_with_ablation(clean_inputs, removed, corr_act)   # §3
        new_kl = recovery_metric(circuit_logits, full_clean_logits)               # §5, default last-token KL
        if new_kl - current_kl < tau:                          # per-edge tolerance
            current_kl = new_kl                                # keep removed (prune)
        else:
            removed.discard((u, v))                            # revert — edge stays in circuit

return ACDCResult(kept_edges = all_edges − removed,
                  removed_edges = removed, final_kl = current_kl)
```

- **Reverse-topological is reader-keyed:** outer loop over readers (logits first,
  embed last), inner loop over that reader's *incoming* edges. The cumulative
  ablation state is naturally per-reader (each reader accumulates the set of
  upstream writers ablated into it).
- **τ is a per-edge tolerance**, not a total budget: an edge is pruned iff its
  *individual* removal raises the metric by less than τ relative to the *current*
  (already-partially-pruned) circuit. Cumulative drift across many small-τ prunes
  is allowed and expected. `current_kl` is always the **actual measured** KL of
  the current circuit (updated to `new_kl` on each accepted prune), never an
  accumulated sum of increments — KL is not additive.
- **Greedy + online:** prunes are permanent; later edges are tested against the
  circuit with all earlier prunes already ablated.

---

## 3. Ablation & the circuit forward  *(the core correctness surface)*

Removing edge `(u → v)` means reader `v` reads writer `u`'s **corrupted-run**
contribution instead of `u`'s **live** (current-circuit) contribution. The
residual stream is an additive sum of writer contributions, so reader `v`'s
pre-LN residual input becomes:

```
resid_in_ablated(v, slot) = resid_in_live(v) + Σ_{u : (u→v,slot) ∈ removed} ( corr_act[u] − live_act[u] )
```

**Why `live_act[u]`, not `clean_act[u]` (the crux).** `corr_act[u]` is `u`'s
contribution on the full corrupted run, cached **once**. But `live_act[u]` — `u`'s
contribution *in the current partially-pruned forward* — must be **recaptured
every forward**, because `u` may itself have had incoming edges ablated, so its
output already differs from clean. Using a fixed `corrupted − clean` delta (as
EAP's single-edge brute force does) **double-counts** whenever an ablated reader
is also an upstream writer of another ablated reader, and the full-ablation
anchor then fails on any nonlinear model. `corrupted − live` is faithful
canonical ACDC and makes the anchors **exact** (§8).

**Where to inject — pre-LN, per slot.** The delta is a `d_model` residual-stream
contribution, so it must be added to the residual **before** the reader's
LayerNorm, letting the LN renormalize the swapped contribution (post-LN
injection would skip renormalization and is only first-order — that is exactly
what makes EAP approximate). Concretely, per reader:

- **attn q/k/v (layer L):** capture `resid_pre(L)` at `input_layernorm`'s input.
  In each of `q_proj`/`k_proj`/`v_proj`'s pre-hooks, override the projection
  input with `input_layernorm( resid_pre(L) + Σ_slot Δ )`, where the Δ-sum is
  over that **slot's** removed incoming edges. Re-applying the (stateless) LN
  module per slot keeps q/k/v **independently ablatable and exact through LN**
  on HF, where the three share one `input_layernorm` (TransformerLens gets this
  natively via `use_split_qkv_input`).
- **mlp_in (layer L):** capture `resid_mid(L)` at `post_attention_layernorm`'s
  input; override `up_proj`'s input with `post_attention_layernorm( resid_mid(L)
  + Σ Δ )`.
- **logits_in:** capture `resid_post` at the final norm's input (HF: `model.norm`
  / `model.model.norm`; TL: `hook_resid_post` before `ln_final`); override
  `lm_head`'s input with `final_norm( resid_post + Σ Δ )`.

**Capturing `live_act[u]` in the same forward.** A forward hook on each writer
captures its live contribution as the forward proceeds — `embed` output,
per-head `z@W_O` at `o_proj`'s input (reusing EAP's per-head contribution
helper, GQA-aware), and `mlp` `down_proj` output. Because every writer is
topologically before its readers, each reader's pre-hook finds its upstream
writers' live contributions already captured. The modified component output
propagates downstream naturally through the residual addition (kept paths stay
live); ablated edges swap in `corr_act[u]`.

**`embed` is a writer into every reader.** `build_graph` emits an `embed→reader`
edge for *every* reader (embed has the lowest topo rank). This is what makes the
full-ablation anchor exact: when all edges into `v` are removed, the `embed`
term's `(corr_act[embed] − live_act[embed])` cancels the clean embedding and
swaps in the corrupted one, so `v` sees the fully-corrupted residual at every
depth — not just layer 0.

- `forward_with_ablation(clean_inputs, removed, corr_act)`: one forward on the
  **clean** inputs with (a) writer forward-hooks capturing `live_act`, and (b)
  per-reader pre-hooks injecting `Σ (corr_act[u] − live_act[u])` over that
  reader/slot's removed edges, pre-LN as above. The injected Δ-sums are recomputed
  from the current `removed` set each call.
- **Anchor invariants** (validate the forward independent of the greedy loop):
  `removed = ∅` → circuit logits == clean logits (KL == 0 exactly);
  `removed = all edges` → circuit logits == the corrupted-run logits (every
  reader reads fully-corrupted contributions). Both are **exact** under
  `corrupted − live` + pre-LN injection (§8).

---

## 4. Prune ordering

- **`ordering="topo"`** — reverse-topological readers; incoming edges in a fixed
  deterministic order. Paper-faithful; tests every edge. **Tie-break:** readers
  sharing a topo rank (e.g. all heads in one block) and the incoming edges of a
  reader are ordered by the key `(writer.kind, writer.layer, writer.head,
  writer.neuron, slot)` (None sorts first) so runs are bit-reproducible.
- **`ordering="eap"`** — incoming edges sorted by `|EAP score|` **ascending**
  (prune lowest-attribution first), via an `eap_scores: dict[Edge, float]`
  parameter (pass `EAPResult.scores`). Integrates sub-spec 2's output and tends
  to produce cleaner greedy circuits (unimportant edges pruned first). Ties fall
  back to the `topo` key above.
- **Default:** `eap` if `eap_scores` is provided, else `topo`.

**v1.0 scope is traversal-ordering only — both orderings test every edge.** The
larger ecosystem speedup (5–10×) comes from *skipping* tests for edges whose
`|EAP score|` exceeds a cutoff (assume kept). That changes semantics (untested
edges are assumed kept), so it is an explicit opt-in **follow-on**
(`eap_skip_threshold`), not a silent default. The recovery gate (kept circuit
matches the full model within the accumulated τ) is identical for both orderings.

---

## 5. Recovery metric

- **Default:** `KL(softmax(circuit_logits) ‖ softmax(full_clean_logits))` at the
  **last token position** — KL to the full model's **clean** distribution (the
  original ACDC formulation: "does the circuit still reproduce clean behavior?").
  Reuses `circuitry.core.patching.kl_divergence`; the ACDC layer slices to the
  target position **before** calling it (default `position=-1`, last token; pass
  `position=None` to average over all positions). `core.kl_divergence` is
  unchanged — the position handling lives in the ACDC wrapper. (NOT KL to the
  corrupted distribution.)
- **Custom:** an optional `metric: Callable[[Tensor, Tensor], float]` taking
  `(circuit_logits, full_clean_logits)` and returning a scalar "degradation"; τ
  compares its per-edge increase. Lets users prune by a task metric (e.g. a
  logit-diff recovery) instead of KL. A custom metric is responsible for its own
  position handling.

---

## 6. Output

```python
@dataclass
class ACDCResult:
    kept_edges: list[Edge]
    removed_edges: list[Edge]
    final_kl: float
    graph: EdgeGraph
    def n_kept(self) -> int: ...
    def circuit_graph(self) -> EdgeGraph: ...   # the kept-edge subgraph
```

`ACDCRunner.sweep(clean_inputs, corrupted_inputs, taus=[...], ...) ->
list[tuple[float, int, float]]` returns the Pareto frontier
`[(τ, n_kept_edges, final_kl), …]` (how the ACDC paper presents results).

---

## 7. Module layout

```
patching/
  acdc.py      NEW — ACDCRunner, ACDCResult, the reverse-topo reader-keyed
                     pruning loop, forward_with_ablation (set-ablation forward
                     with live-capture + pre-LN per-slot injection), sweep.
  graph.py     REUSE — Node/Edge/build_graph; add a reverse-topological reader
                     iterator with the deterministic tie-break key (§4).
  eap.py       REUSE — per-head writer-contribution helper (z@W_O, GQA-aware),
                     embed/lm_head/layer locators, corrupted-writer-activation
                     collection. Factor shared bits cleanly; do NOT regress EAP.
core/patching.py  REUSE — kl_divergence (default recovery metric; position
                     slicing done in the ACDC wrapper, core unchanged).
```

Layering unchanged: `patching/` imports `core/` + `recipes/`, never `cli/`;
`transformer_lens` / `transformers` lazy.

---

## 8. Testing

The two anchors are **exact** (KL == 0 / logits all-close to corrupted run) under
the `corrupted − live` + pre-LN mechanism — no fuzzy tolerances. Their ground
truth is independent of the greedy loop: real clean-run logits and real
corrupted-run logits.

| Test | What it verifies |
|------|-----------------|
| **Empty-circuit anchor** | `removed = ∅` → circuit logits **equal** clean logits (KL == 0). Validates the no-op forward + live-capture path adds nothing. |
| **Full-ablation anchor** | `removed = all edges` → circuit logits **equal** the corrupted-run logits (every reader reads corrupted; embed-edge cancellation reaches every depth). Validates the set-ablation forward end-to-end, independent of the greedy loop. Run on a real (small) HF model AND a toy. |
| **Live-vs-clean delta** | A 2-edge case where edge B's reader is downstream of edge A's reader: ablating {A,B} with `corrupted−live` differs from the (wrong) `corrupted−clean` sum, and matches a brute-force reference that re-runs the model with both ablations composed. Directly guards the crux fix. |
| **Dead-edge pruning** | On a constructed toy where a known subset of edges provably does not affect the output, ACDC at small τ prunes exactly those and keeps the live path. |
| **Monotone sweep** | `sweep` over increasing τ yields monotonically smaller circuits and non-decreasing `final_kl` (Pareto frontier). |
| **τ extremes** | τ = 0 (or tiny) prunes only edges with truly zero effect; τ = ∞ prunes everything (empty circuit). |
| **topo vs eap ordering** | Both run end-to-end and return circuits whose KL is within the accumulated bound; `eap` consumes `EAPResult.scores`; both are deterministic across repeats (tie-break key). |
| **Custom metric** | A user `Callable` is accepted and drives pruning. |
| **Backends** | HF eager + TL (`skipif` not installed) run end-to-end and return finite `final_kl`; the full-ablation anchor holds on both. |
| **Layering / lazy import** | `patching/acdc.py` imports core/recipes/torch only; `transformer_lens`/`transformers` not imported at package-import time. |

No gradient/param-grad tests are needed — ACDC is forward-only.

---

## 9. Risks & sequencing

- The **set-ablation forward** is the core correctness surface. The `corrupted −
  live` (not `corrupted − clean`) distinction and **pre-LN** injection are the
  two places a faithful implementation diverges from a naive one. Gate the
  forward with the empty-, full-ablation, and live-vs-clean tests **before** the
  greedy loop is trusted.
- The **greedy loop** is straightforward once the forward is correct; land it
  after the anchors, validated by the dead-edge-pruning test.
- **Cost** is honest: O(edges) ablation forwards per τ, each forward
  recapturing live writer contributions. `sweep` amortizes the **corrupted**
  writer cache and the clean-reference logits across thresholds; per-τ forward
  count is unchanged. EAP-ordering changes traversal, not the count (skipping is
  the follow-on lever).
- Reuse EAP machinery but **do not regress EAP/AtP\***: shared-helper refactors
  (per-head contribution, locators) must keep their exact gates green.

---

## 10. Public API sketch

```python
from circuitry.patching.acdc import ACDCRunner
from circuitry.patching.sites import HFSiteResolver
# optional: from circuitry.patching.eap import EAPRunner  (for eap ordering)

runner = ACDCRunner(model, HFSiteResolver.from_config(model.config))
result = runner.run(
    clean_inputs=clean_ids,
    corrupted_inputs=corrupted_ids,
    tau=0.05,
    ordering="topo",          # or "eap" with eap_scores=eap_result.scores
    # position=-1,            # last-token KL (default); None = mean over positions
    # metric=...,             # optional custom recovery metric (default KL-to-clean)
)
print(result.n_kept(), "edges kept;  final KL =", result.final_kl)

frontier = runner.sweep(clean_inputs=clean_ids, corrupted_inputs=corrupted_ids,
                        taus=[0.01, 0.05, 0.1, 0.5])
# [(0.01, 312, 0.004), (0.05, 180, 0.03), ...]  (τ, n_kept, final_kl)
```

Signatures are a sketch; exact names settled in the implementation plan.
