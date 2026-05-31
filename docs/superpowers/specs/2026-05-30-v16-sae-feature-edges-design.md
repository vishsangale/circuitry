# v1.6.0 — feature→feature SAE edges + sparse-feature circuit (FeatureACDC)

**Status:** approved for implementation (2026-05-30). Design = a 9-agent design workflow
(survey → 3 proposals → synthesis → adversarial critique with float64 probes) + a focused
empirical pass on the FeatureACDC ablation semantics. Every load-bearing claim below was
verified empirically this session, NOT taken from a summary. User scope decision: **full
FeatureACDC** (edge graph + threshold + faithfulness/completeness + node-pruning ACDC).

## 1. Goal & scope

Extend the shipped v1.5 NODE-level SAE-feature attribution to **feature→feature edges** and a
**sparse-feature circuit** with greedy pruning. Builds strictly on the shipped `SAEFeatureRunner`;
the v1.5 node path is frozen (zero regression risk).

**In scope (v1.6.0):**
- Feature→feature **edge attribution** (`SAEFeatureEdgeRunner`) via the multi-site error-term splice.
- A dedicated **`SAEFeatureEdgeGraph`** + **`SAEFeatureCircuit`** result (graph.py / EdgeGraph UNTOUCHED).
- **Threshold pruning** + **faithfulness/completeness** + **node-pruning `FeatureACDCRunner`** (+ `sweep`).
- HF-eager + `resid_post` only (inherited v1.5 gates), `attrib` variant only.

**Deferred (NOT v1.6.0):**
- Integrated-gradients edge variant (`ig`) — keep a `variant` enum slot so it is additive later.
- Per-(writer_pos, reader_pos) position-resolved edges — v1.6 emits position-AGGREGATED scalars.
- `mlp_out`/`attn_out` sites, TL backend (clear `NotImplementedError`, inherited from v1.5).
- TransformerLens backend; the node-level polish items from the v1.5 backlog.

## 2. Edge attribution — multi-site splice + per-downstream VJP

### 2.1 The splice (the v1.5→v1.6 pivot)
v1.5 spliced each site in its OWN forward. v1.6 splices ALL sites in the ordered span in **one
clean forward** so an upstream feature's decode propagates through the real residual to the
downstream encode. Per site, in forward/layer order:
- **WRITER site** (upstream-most per pass): v1.5 detached-leaf seed —
  `f_U = sae.encode(a).detach().requires_grad_(True); f_U.retain_grad()`. Its `decode(f_U)` is
  affine and propagates downstream; `f_U.grad` receives the VJP.
- **READER site** (downstream): LIVE non-detached encode — `f_D = sae.encode(a_live); f_D.retain_grad()`
  (exactly `sae_decompose`). A site that is both reader and writer uses the LIVE construction.
- **`eps` detached/frozen-at-clean at EVERY spliced site** (`recon = x_hat + eps`). A live `eps`
  cancels `x_hat` and zeros grad — the v1.5 §2.2 trap, now at every chain link.
- Each site casts `a → (sae.device, sae.dtype)` and `recon → (model_device, model_dtype)` independently.

**Verified:** 2-site simultaneous splice lossless to 8.9e-16; `f_U.grad` AND `f_D.grad` both nonzero
from a SINGLE backward.

### 2.2 Edge formula (live-autograd Jacobian — NOT the SFC frozen-pattern Jacobian)
```
edge(U:i → D:j) = Σ_pos Δf_U[...,i] · (∂f_D[...,j]/∂f_U[...,i]) · gradf_D[...,j]
  Δf_U   = f_U_corrupt − f_U_clean        (corrupt−clean, matching v1.5 sae_features.py:345)
  gradf_D = ∂metric/∂f_D                   (component-only FOR FREE — eps_D is detached, so the
                                            only differentiable channel from D is decode(f_D))
```
**IMPORTANT (critic finding):** this uses the FULL LIVE autograd Jacobian through the model — it
does NOT freeze the QK attention softmax. Do **not** claim equivalence to the Marks/Anthropic
frozen-pattern stop-gradient Jacobian; with attention between sites they differ by the QK
nonlinearity (~0.19 in a probe). On a linear-downstream model the live Jacobian is exact (the Gate A
oracle); on nonlinear models `attrib` is approximate (that is what Gate B measures). Do NOT reuse
eap.py's clone-trick or its `mlp_in`/`logits_in` LN `ln_scale` (there is no LN between resid_post
and the SAE encode — copying `_collect_reader_grads` wholesale would be a correctness bug).

### 2.3 VJP realization (NEVER materialize a d_sae×d_sae Jacobian)
After the spliced clean forward + `metric.backward(retain_graph=True)` (populates `f_D.grad = gradf_D`),
for each kept DOWNSTREAM survivor `j`:
```
G_j = zeros_like(f_D); G_j[..., j] = gradf_D[..., j]          # seed scaled by metric relevance of j
vjp_j = torch.autograd.grad(f_D, f_U_leaf, grad_outputs=G_j, retain_graph=True)[0]   # shape of f_U
for i in upstream_survivors:                                  # SLICE to survivors immediately
    edge(i→j) = float((Δf_U[..., i] * vjp_j[..., i]).to(torch.float32).sum())
del vjp_j                                                     # FREE per-j (see memory note)
```
- Cost = Σ_pairs K_down VJP backwards (NOT one per edge); each edge is then a free dot.
- **MEMORY (critic):** a full-width `vjp_j` is `(B,S,d_sae_U)` = 0.134 GB at d_sae=65536, B·S=512.
  MUST free each `vjp_j` after use and slice to upstream survivors immediately — NEVER stack
  K_down rows into a `K_down × d_sae` matrix. `retain_graph=True` holds the spliced-span graph
  alive (memory grows with span length × B·S·d_model) — acceptable for a post-hoc tool.
- **fp32 accumulation (v1.4.1/.2 lesson):** the VJP returns on the upstream SAE's dtype; cast the
  `Δf_U·vjp` product to fp32 before the sum at every site; device-align the cross-site tensors.

### 2.4 Conservation = SANITY check only (critic correction)
`Σ_j edge(U:i→D:j)` over ALL downstream `j` equals `f_U.grad`'s D-path component to ~1e-16 ALWAYS
(a trivial same-backward autograd identity — VJP columns sum to the full VJP). It catches only the
detach-sever, NOT edge correctness, and it does NOT equal the GENUINE v1.5 node score on a lossy
(real) SAE (the edge run's detached `eps_D` severs the identity-residual path; error ~0.87 on a
lossy SAE). → ship it as a `test_edge_columns_sum_to_vjp` SANITY test; do NOT advertise it as the
oracle. **The sole exact oracle is analytic-vs-`bruteforce_feature_edge_scores` (§4 Gate A).**

## 3. Two-stage tractability (mandatory)
Full feature×feature for one adjacent pair = ~6e8 edges at d_sae=24576 — fatal. Two-stage:
- **Stage 1:** reuse the SHIPPED `SAEFeatureRunner.run()` per site → `AtPResult`; keep top-K
  **active** survivors/site (reuse the `max_features` cap; active-feature restriction Δf≠0 inherited
  from v1.5). Downstream survivors MUST be active (edges into never-fired TopK/JumpReLU features have
  a structurally-zero VJP — wasted backward otherwise).
- **Stage 2:** enumerate edges only among per-site survivors across ordered site-pairs.
  `layer_pairs='adjacent'` default (n_sites−1 pairs), `'all_forward'` opt-in (n(n−1)/2). `max_edges`
  global cap (top-|score|). Budget (K=32, 12 sites): 384 nodes; adjacent ≈ 11k edges / ~352 VJP
  backwards; all_forward ≈ 68k / ~2112.

## 4. Circuit extraction — NODE-set ablation + node-pruning FeatureACDC

### 4.1 Node-set ablation forward (empirically verified)
Edge-level ablation is unimplementable (downstream features share `a_D`). Faithfulness uses
**node-set** ablation: at each spliced site, replace NON-circuit feature entries before decode:
```
f_ablated = f.clone()
f_ablated[..., i] = ablation_value[site][..., i]   for i NOT in circuit_nodes[site]
recon = sae.decode(f_ablated) + eps                # eps frozen at clean (see §4.4)
```
propagated naturally (hooks in layer order). `ablation_mode`:
- `corrupted` (default): `ablation_value = f_corrupt` (encode of the corrupted residual).
- `zero`: 0. `mean`: per-(batch,seq) mean of `f_corrupt`, broadcast.
Forward-only, no grad; restore hooks in finally (model clean after — verified).
**Verified:** faithfulness(M)=1.0, faithfulness(∅)=0.0, KL(M‖clean)≈0 (lossless), three modes
give distinct sensible m(∅), hooks clean after.

### 4.2 Faithfulness / completeness
```
faithfulness(C)  = (m(C) − m(∅)) / (m(M) − m(∅))      # Marks SFC §3.2
completeness(C)  = (m(M\C) − m(M)) / (m(∅) − m(M))
```
`m(·)` = scalar metric. **Document:** the metric-diff faithfulness can exceed [0,1] when features
anti-correlate with the metric — that is expected, not a bug; the KL recovery is ACDC's accept
criterion. Faithfulness/completeness need the §4.1 ablation forward (the net-new code the scope
decision opted into).

### 4.3 FeatureACDC (node-pruning; reuse ACDC control flow)
Greedy reverse-topological NODE pruning. Maps onto `ACDCRunner`:
- reverse-topo over sites: later layers first; within a site, weakest |AtP node score| first.
- `removed_nodes: dict[site_layer, set[feat_idx]]`; accept removal if `KL_new − KL_current < tau`.
- `sweep(taus) → [(tau, n_kept_nodes, final_kl)]` Pareto (note: greedy, so final circuit KL may
  exceed tau — standard ACDC behavior, document it).
- `ablation_mode` (corrupted/zero/mean), `eap_skip_threshold` (|node score| above → assume kept, skip
  test), `_recovery_kl` via `circuitry.core.patching.kl_divergence` (reused).
- **Net-new:** `feature_circuit_forward` (§4.1), `compute_f_per_site` (feature acts per site).
  **NOT reusable:** `ACDCRunner._run_capturing_live` (d_model per-head deltas + calls `_order`),
  `_cache_corrupted_acts`, `graph._order`/`build_graph`.
**Verified:** greedy converges on the linear toy (faithful≈1 across the tau sweep) and the nonlinear toy.

### 4.4 Error-node handling (follows the v1.5 `include_error_node` flag — DECIDED, no new fork)
- `include_error_node=False` (default): `eps` frozen at clean everywhere; error is not a node;
  faithfulness has a slight upward bias (documented).
- `include_error_node=True`: `sae_error` per site is a first-class node with its own in-circuit flag;
  when an error node is OUTSIDE the circuit, ablate its `eps` to the corrupted/zero/mean value
  (Marks-style). Reuses v1.5's symmetric `err_leaf` for edge endpoints (feature→error / error→feature).

## 5. Module structure & API (graph.py / EdgeGraph / _order LEFT UNTOUCHED)
New `src/circuitry/patching/sae_edges.py`; shared splice/device helpers factored into private
`src/circuitry/patching/_sae_splice.py` (imported by sae_features and sae_edges). Lazy-export from
`patching/__init__.py` via the existing `_LAZY` + `__getattr__` pattern.
```python
@dataclass(frozen=True)
class SAEFeatureEdge:           # NO slot, NO position field in v1.6
    writer: AtPNode            # sae_feature/sae_error endpoint (exact v1.5 Node objects)
    reader: AtPNode

class SAEFeatureEdgeGraph:      # sites, survivors, edges; local sae_edge_sort_key + feature-aware
                                # reverse-topo keyed on (site forward-order, feature idx) — NEVER calls graph._order

class SAEFeatureEdgeRunner:     # composes (NOT subclasses) SAEFeatureRunner for stage 1
    def __init__(self, model, sae_sites: dict[Site, SAE | tuple[str,str]], resolver): ...
    def run(self, clean, corrupted, metric, *, layer_pairs='adjacent', top_k_survivors=32,
            max_edges=None, include_error_node=False, variant='attrib') -> SAEFeatureCircuit: ...
    def bruteforce_feature_edge_scores(self, clean, corrupted, metric, edges) -> dict[SAEFeatureEdge,float]:
        # INDEPENDENT ground truth: patch ONLY upstream feature i clean→corrupted in the spliced
        # forward, measure induced Δf_D[j]·gradf_D[j] (feature-level, NOT metric-level). eps frozen.

class SAEFeatureCircuit:        # the result
    nodes: AtPResult            # v1.5 node scores, unchanged
    edges: dict[SAEFeatureEdge, float]
    graph: SAEFeatureEdgeGraph
    def ranked(self) / top_k(n) / threshold(tau): ...        # copied from EAPResult
    def prune(self, method='threshold'|'acdc'|'both', tau=..., ablation_mode='corrupted', ...): ...
    def faithfulness(self, clean, corrupted, metric, ablation_mode='corrupted') -> float: ...
    def completeness(self, ...) -> float: ...

class FeatureACDCRunner:        # §4.3; .run(...) -> SAEFeatureCircuit ; .sweep(taus) -> [(tau,n_kept,kl)]
```
Bump `__version__='1.6.0'`; update `tests/test_public_api.py`. Layering: no new root import.

## 6. Validation (acceptance gates) — reuse v1.5 fixtures verbatim (2 SAEs at adjacent resid_post)
**GATE A — exact (default CI, abs=1e-4):**
- `test_feature_edge_matches_bruteforce_linear` — analytic edge == `bruteforce_feature_edge_scores`
  (feature-level oracle; verified ~7e-7). The brute force uses the SAME live attention pattern.
- `test_relu_encode_edge_still_exact`; `test_two_site_splice_lossless` (‖recon−a‖∞<1e-5 both sites).
- `test_edge_columns_sum_to_vjp` — the SANITY conservation identity (interp-a), NOT vs v1.5 node score.
- `test_v15_detach_severs_edge` — reader-site detach ⇒ `f_U.grad` None/zero; live ⇒ nonzero (locks mechanism).
**GATE B — correlation (default CI, numpy-only, NonlinearResidToy):** Spearman≥0.7 / Pearson≥0.6 /
sign≥0.9 vs brute-force edge patching.
**ABLATION / ACDC:** `test_faithfulness_full_circuit_is_one` (≈1), `test_faithfulness_empty_is_zero`
(≈0), `test_ablation_monotonicity` (high-attribution node removal drops recovery more than low —
assert at the extremes), `test_ablation_modes` (corrupted/zero/mean all run, distinct m(∅)),
`test_feature_acdc_converges` (greedy recovers a small faithful circuit on the toy), `test_sweep_pareto`,
`test_model_clean_after_ablation`.
**ROBUSTNESS:** determinism (bit-identical), `test_edge_grad_device_align` (fp16 SAE/fp32 model, finite,
<5e-3 vs bf — assert len>0 + finite, NOT just isinstance), `test_no_sae_param_grad_leak` (all spliced
SAEs), `test_two_stage_keeps_top_k_survivors`, `test_max_edges_cap`, `test_adjacent_vs_all_forward_pair_counts`,
`test_error_node_edges_optin`, `test_metric_must_be_differentiable` (clear error), `test_tl_not_implemented`,
`test_non_resid_post_site_error`, `test_vjp_freed_not_stacked` (memory discipline — assert no K_down×d_sae alloc).
**TIER 3 (opt-in @slow + skipif-offline):** one real `load_sae` pair at two adjacent resid_post sites on a
small HF model — finite edges of correct shape, conservation APPROXIMATE only. Gate A must NEVER need a download.

## 7. Docs (same commit): design.md §2 (node-kinds/graph routing note), §4.6 (new SAE-feature-EDGE
sub-spec 6: edge runner/result/FeatureACDC, live-Jacobian-not-frozen note, node-set ablation, conservation-
is-sanity-not-oracle), §8 (remove feature→feature edges from deferred); CHANGELOG [1.6.0]; README.
