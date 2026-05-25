# ACDC (Automatic Circuit DisCovery) — Design Spec

> **Sub-spec 4 of the v1.0 patching pillar** (the last of the three attribution/
> search methods). Builds on sub-spec 1 (core primitive: `patch_site`), sub-spec
> 2 (EAP: `patching/graph.py` Node/Edge/build_graph + per-edge additive-Δact
> patching + writer-activation caching + HF/TL backends), and optionally consumes
> EAP's edge scores (`EAPResult.scores`) for prune ordering.

**Goal:** Discover a minimal circuit — a subset of the residual-stream
computation graph's edges that reproduces the full model's clean behavior within
a tolerance τ — by greedy, iterative edge pruning with corrupted-resample
ablation.

**Architecture:** A new `ACDCRunner` that walks readers in reverse-topological
order, tentatively ablates each incoming edge (corrupted-resample), keeps the
ablation if it costs less than τ in KL-to-clean, and returns the surviving
circuit. Forward-only (no gradients). Reuses EAP's graph, per-edge additive
patching, writer caching, and backends.

**Tech Stack:** Python 3.12, PyTorch, HF transformers (eager, Llama-family),
TransformerLens (optional, lazy).

---

## 1. Scope & taxonomy

ACDC (Conmy, Mavor-Parker, Lynch, Heimersheim, Garriga-Alonso 2023, "Towards
Automated Circuit Discovery for Mechanistic Interpretability") is a **search
algorithm**, not an attribution method (contrast EAP/AtP\*, which score
edges/nodes). Starting from the *full* graph (all edges present = the full
model), it greedily removes edges that don't matter, leaving a minimal circuit.

- **Forward-only:** ACDC uses real ablation + a recovery metric; **no
  gradients**. (No frozen-model-grad concerns — simpler than EAP/AtP\* on that
  axis. The complexity is the cumulative greedy loop + the set-ablation forward.)
- **Reuses** EAP's `graph.py` (Node/Edge/build_graph — heads + MLPs + embed +
  logits; q/k/v-typed edges into heads, `mlp_in`, `logits_in`), the per-edge
  additive-Δact patching, the writer-activation caching, and the HF/TL backends.

**In scope:** corrupted-resample ablation; KL-to-clean recovery metric (+ custom
metric); single τ + a `sweep(taus)` Pareto helper; `ordering="topo"` and
`"eap"` (traversal order); HF + TL backends.

**Out of scope:** mean / zero ablation (follow-on); the EAP-score *skip* speedup
(see §4 — v1.0 ships traversal-ordering only); SAE-feature circuits; report/
compare integration.

---

## 2. The algorithm — reader-keyed greedy pruning

```
build full graph (build_graph); removed: set[Edge] = {}        # full circuit
full_clean_logits = model(clean)                                # KL reference, computed once
cache clean + corrupted writer activations once → Δact[u] = corrupted_u − clean_u
current_kl = 0.0                                                # full circuit matches clean exactly

for reader v in REVERSE-topological order (logits → … → layer 0):
    for each incoming edge (u → v) in the chosen order (§4):
        removed.add((u, v))                                     # tentatively ablate
        circuit_logits = forward_with_ablation(clean_inputs, removed)   # §3
        new_kl = recovery_metric(circuit_logits, full_clean_logits)     # §5, default KL
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
  is allowed and expected.
- **Greedy + online:** prunes are permanent; later edges are tested against the
  circuit with all earlier prunes already ablated.

---

## 3. Ablation & the circuit forward

Removing edge `(u → v)` means reader `v` reads writer `u`'s **corrupted**
contribution instead of its clean one. Since the residual stream is an additive
sum of writer contributions, this is:

```
v_input_ablated = v_input_clean + Σ_{u : (u→v) ∈ removed} Δact[u]
                  where  Δact[u] = corrupted_contribution_u − clean_contribution_u
```

- `forward_with_ablation(clean_inputs, removed)`: run the model on the **clean**
  inputs with **one accumulated additive hook per reader** that adds the sum of
  `Δact[u]` over that reader's removed incoming edges to its residual input
  (reusing EAP's per-edge additive-Δact patching, summed per reader). The hook
  set is rebuilt/updated as `removed` changes.
- `Δact[u]` (the writer's `d_model` residual contribution: `z@W_O` for heads,
  MLP output, embed output) is cached **once** from a clean and a corrupted
  forward (EAP's writer-activation machinery), and reused for every edge test and
  every τ in a sweep.
- **Anchor invariants** (validate the set-ablation forward independent of the
  greedy loop): `removed = ∅` → circuit logits == clean logits (KL ≈ 0);
  `removed = all edges` → every reader reads fully-corrupted contributions →
  circuit logits == the corrupted-resampled output.

---

## 4. Prune ordering

- **`ordering="topo"`** — reverse-topological readers; incoming edges in a fixed
  deterministic order. Paper-faithful; tests every edge.
- **`ordering="eap"`** — incoming edges sorted by `|EAP score|` **ascending**
  (prune lowest-attribution first), via an `eap_scores: dict[Edge, float]`
  parameter (pass `EAPResult.scores`). Integrates sub-spec 2's output and tends
  to produce cleaner greedy circuits (unimportant edges pruned first).
- **Default:** `eap` if `eap_scores` is provided, else `topo`.

**v1.0 scope is traversal-ordering only — both orderings test every edge.** The
larger ecosystem speedup (5–10×) comes from *skipping* tests for edges whose
`|EAP score|` exceeds a cutoff (assume kept). That changes semantics (untested
edges are assumed kept), so it is an explicit opt-in **follow-on**
(`eap_skip_threshold`), not a silent default. The recovery gate (kept circuit
matches the full model within the accumulated τ) is identical for both orderings.

---

## 5. Recovery metric

- **Default:** `KL(softmax(circuit_logits) ‖ softmax(full_clean_logits))` — KL to
  the full model's **clean** distribution (the original ACDC formulation: "does
  the circuit still reproduce clean behavior?"). Reuses
  `circuitry.core.patching.kl_divergence`. (NOT KL to the corrupted distribution.)
- **Custom:** an optional `metric: Callable[[Tensor, Tensor], float]` taking
  `(circuit_logits, full_clean_logits)` and returning a scalar "degradation"; τ
  compares its per-edge increase. Lets users prune by a task metric (e.g. a
  logit-diff recovery) instead of KL.

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
`[(τ, n_kept_edges, final_kl), …]` (how the ACDC paper presents results),
reusing the cached `Δact` across thresholds.

---

## 7. Module layout

```
patching/
  acdc.py      NEW — ACDCRunner, ACDCResult, the reverse-topo reader-keyed
                     pruning loop, forward_with_ablation (set-ablation), sweep.
  graph.py     REUSE — Node/Edge/build_graph, reverse-topological reader order
                     (add a small reverse-topo reader iterator if not present).
  eap.py       REUSE — writer-activation caching + per-edge additive-Δact
                     patching + HF/TL backend/module-location helpers (factor
                     shared bits cleanly; do NOT regress EAP).
core/patching.py  REUSE — kl_divergence (default recovery metric).
```

Layering unchanged: `patching/` imports `core/` + `recipes/`, never `cli/`;
`transformer_lens` / `transformers` lazy.

---

## 8. Testing

| Test | What it verifies |
|------|-----------------|
| **Empty-circuit anchor** | `removed = ∅` → circuit logits equal clean logits (KL ≈ 0). Validates the no-op forward. |
| **Full-ablation anchor** | `removed = all edges` → circuit logits equal the corrupted-resampled output (every reader reads corrupted). Validates the set-ablation forward end-to-end, independent of the greedy loop. |
| **Dead-edge pruning** | On a constructed toy where a known subset of edges provably does not affect the output, ACDC at small τ prunes exactly those and keeps the live path. |
| **Monotone sweep** | `sweep` over increasing τ yields monotonically smaller circuits and larger `final_kl` (Pareto frontier). |
| **τ extremes** | τ = 0 (or tiny) prunes only edges with truly zero effect; τ = ∞ prunes everything (empty circuit). |
| **topo vs eap ordering** | Both run end-to-end and return circuits whose KL is within the accumulated bound; `eap` consumes `EAPResult.scores`. |
| **Custom metric** | A user `Callable` is accepted and drives pruning. |
| **Backends** | HF eager + TL (`skipif` not installed) run end-to-end and return finite `final_kl`. |
| **Layering / lazy import** | `patching/acdc.py` imports core/recipes/torch only; `transformer_lens`/`transformers` not imported at package-import time. |

No gradient/param-grad tests are needed — ACDC is forward-only.

---

## 9. Risks & sequencing

- The **set-ablation forward** (one accumulated additive hook per reader, updated
  as `removed` grows) is the core correctness surface. Gate it with the empty-
  and full-ablation anchors before the greedy loop is trusted.
- The **greedy loop** is straightforward once the forward is correct; land it
  after the anchors, validated by the dead-edge-pruning test.
- **Cost** is honest: O(edges) ablation forwards per τ. `sweep` amortizes the
  Δact caching across thresholds; per-τ cost is unchanged. EAP-ordering changes
  traversal, not the count (skipping is the follow-on lever).
- Reuse EAP machinery but **do not regress EAP/AtP\*** — shared-helper refactors
  must keep their exact gates green.

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
    # metric=...,             # optional custom recovery metric (default KL-to-clean)
)
print(result.n_kept(), "edges kept;  final KL =", result.final_kl)

frontier = runner.sweep(clean_inputs=clean_ids, corrupted_inputs=corrupted_ids,
                        taus=[0.01, 0.05, 0.1, 0.5])
# [(0.01, 312, 0.004), (0.05, 180, 0.03), ...]  (τ, n_kept, final_kl)
```

Signatures are a sketch; exact names settled in the implementation plan.
