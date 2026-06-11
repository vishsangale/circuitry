# circuitry — SOTA gap plan, round 3 (v1.41+)

**Created:** 2026-06-11
**Based on:** gap survey of the mid-2026 mech-interp landscape, conducted after the
v1.29–v1.40 milestones closed out `docs/plan-sota-next.md` (including all of its
"deferred hard items" except gradient subspace saturation and MoE pathway complexity).
**Current version:** v1.40.0
**Status:** in flight — milestones implemented sequentially on this plan.

---

## Survey conclusion

Method coverage is no longer the gap. The patching/attribution stack (EAP, AtP\*, ACDC,
RelP, CLT attribution graphs, DAS/HyperDAS, ITI, causal tracing, mean ablation, head
knockout, certified circuits) is broader than any single library in the ecosystem.
What now separates circuitry from SOTA:

1. **Output/trust layer** — every result terminates at `.to_markdown()`. The ecosystem
   standard is Anthropic's open-source circuit-tracer JSON + the Neuronpedia interactive
   frontend. circuitry computes attribution graphs but cannot exchange them.
2. **Generation-time analysis** — runners patch a single forward pass; 2026 work targets
   multi-token behaviour (CoT faithfulness, refusal dynamics across decoding).
3. **Scale engineering** — single-process only; FSDP-sharded params give wrong rank-0
   diagnostics (design §11, TODO [XL]). Training-time diagnostics is circuitry's moat
   and is exactly the workflow that is worthless single-process at 2026 scale.
4. **2026 method stragglers** — weight-based (input-independent) transcoder
   connectivity maps, cross-method circuit consensus (CIRCUS-style), MoE routing
   diagnostics, pluggable auto-interp labeling (SAGE/ADAG without an API dependency).
5. **Distribution** — no docs site, no notebooks, no published benchmark numbers.
   Out of library-code scope; tracked as Track D below, not a version milestone.

Key references: circuit-tracer (github.com/decoderesearch/circuit-tracer; Anthropic
open-source release), Neuronpedia graph upload, CIRCUS arXiv:2603.00523, ADAG
arXiv:2604.07615, Circuit Insights arXiv:2510.14936, transcoders-beat-SAEs
arXiv:2501.18823, feature flow arXiv:2502.03032, MoE pathway complexity
arXiv:2506.21551.

---

## Milestone v1.41 — Graph export & interchange (output layer) — SHIPPED 2026-06-11

**Theme:** Make every circuitry graph result exchangeable with the existing
visualization ecosystem instead of competing with it. Pure serialization; stdlib-only
(json); no new dependencies; no layering changes (`patching/export.py` may import
sibling `patching/` result types and `core/`).

### Items

| Item | What |
|------|------|
| `patching/export.py::to_neuronpedia_graph(result, *, slug, scan, prompt, prompt_tokens, node_threshold=0.8) → dict` | Serialize a graph result to the circuit-tracer / Neuronpedia JSON graph schema (top-level `metadata` / `qParams` / `nodes` / `links`; node `feature_type` ∈ {embedding, logit, feature, error, token}). Accepts `CLTGraphResult`, `SAEFeatureCircuit`, and `EAPResult`-family results (anything edge-scored over `EdgeGraph` nodes). Exact field names verified against `circuit_tracer/frontend/graph_models.py`. |
| `patching/export.py::save_neuronpedia_graph(result, path, **kwargs)` | Thin wrapper: `json.dump` of the above. Prints/returns the absolute path. |
| `patching/export.py::to_html(result, *, title=None) → str` and `save_html(result, path)` | Self-contained single-file HTML report: layered DAG rendering (nodes grouped by layer, deterministic layout, inline SVG + vanilla JS for hover/edge-weight inspection). Zero runtime deps; no CDN fetches. |
| CLI: `circuitry export-graph <result.json> --format neuronpedia\|html` | Re-serialize a saved graph artifact. CLI may import patching (allowed). |

### Tests
`tests/patching/test_export.py` (~25 tests): schema shape (golden keys), node/link
counts match source result, threshold filtering, round-trip stability, HTML contains
no external URLs, CLI smoke.

---

## Milestone v1.42 — Weight-based transcoder analysis, consensus, labeling hooks — SHIPPED 2026-06-11

**Theme:** The three cheapest method stragglers. All respect invariant #3 (no API
dependencies — labeling is a user-supplied callable).

### Items

| Item | Method | Paper |
|------|--------|-------|
| `core/circuits.py::transcoder_virtual_weights(W_dec_up, W_enc_down) → Tensor` | Input-independent feature→feature connectivity: `W_dec_up @ W_enc_down` (`(f_up, d) × (d, f_down)`). The weight-derived global circuit map; complements per-prompt CLT graphs. | Circuit Insights arXiv:2510.14936 |
| `core/circuits.py::top_virtual_connections(V, *, k=20) → list[(i, j, w)]` | Top-k entries of a virtual-weight matrix by |w|; pure helper for global connectivity maps. | arXiv:2510.14936 |
| `core/circuits.py::feature_token_alignment(W_dec, W_U, *, k=10)` | Per-feature top-k promoted logit tokens via decoder column → unembedding (weight-only "what does this feature do"). Reuses `top_logit_tokens`. | arXiv:2501.18823 |
| `patching/consensus.py::CircuitConsensus(results: list[set[Edge]\|EAPResult...])` | Cross-method stability ensemble: per-edge agreement fraction across N runner results, `.consensus_edges(min_agreement)`, `.pairwise_jaccard()`, `.to_markdown()`. Complements `CertifiedCircuitRunner` (which subsamples data for ONE method). | CIRCUS arXiv:2603.00523 |
| `sae/labeling.py::describe_features(feature_evidence, label_fn) → dict[int, str]` + `FeatureEvidence` dataclass (top activating token strings, top logit tokens, activation stats) | Auto-interp with a pluggable `label_fn: Callable[[str], str]` — the user brings their own LLM call; circuitry builds the evidence prompt and threads labels into export/`to_markdown`. | SAGE arXiv:2511.20820; ADAG arXiv:2604.07615 |
| `patching/export.py`: `labels=` kwarg | Neuronpedia/HTML export accepts `{(layer, feat_idx): str}` labels from `describe_features`. | — |

### Tests
~30 tests across `tests/core/test_circuits.py`, `tests/patching/test_consensus.py`,
`tests/sae/test_labeling.py`.

---

## Milestone v1.43 — Generation-time analysis — SHIPPED 2026-06-11

**Theme:** Patch, steer, and trace across multi-token decoding, not just one forward
pass. Hook-based and model-agnostic (works with HF `generate` or a hand-rolled decode
loop); KV-cache-aware (hooks stay armed across steps; per-step seq-len of 1 handled).

### Items

| Item | What |
|------|------|
| `patching/generation.py::GenerationTrace` | Per-decode-step capture: chosen token, top-k logits, per-site activation stats (entropy, norm), optional logit-lens KL per step. `.to_markdown()`, export hook. |
| `patching/generation.py::trace_generation(model, step_fn \| generate_fn, *, sites, n_steps)` | Drives or observes a decode loop; returns `GenerationTrace`. |
| `patching/generation.py::apply_steer_steps(model, site, vector, *, steps)` / `patch_site_steps(...)` | Step-indexed interventions: active only on decode steps in `steps` (e.g. steer only after step 10). Builds on existing `apply_steer` / `patch_site` context managers. |
| `patching/generation.py::prepare_generation_attribution(...)` + `generation_attribution(model, clean, corrupted, *, target_step, runner='causal_trace'\|'patch_grid')` | Attribute a *generated* token at step t back through the prompt + earlier generated tokens (re-runs teacher-forced forward over the realized sequence, then delegates to existing runners). |

### Tests
~25 tests with a tiny decode loop over the existing test transformer.

---

## Milestone v1.44 — MoE routing diagnostics — SHIPPED 2026-06-11

**Theme:** Close the last deferred item from plan-sota-next; MoE is the dominant
2026 frontier architecture and the Recorder has no routing story. (Gradient
subspace saturation, originally listed here, turned out to have already shipped
as `core/weight.py::gradient_subspace_saturation`.)

### Items

| Item | Method | Paper |
|------|--------|-------|
| `core/moe.py::routing_entropy(gate_probs) → float` | Mean per-token entropy of router distribution. | arXiv:2506.21551 |
| `core/moe.py::expert_load_balance(expert_ids, n_experts) → float` | Normalized load-balance score (1 = uniform). | Shazeer 2017; arXiv:2506.21551 |
| `core/moe.py::pathway_complexity(expert_ids_per_layer) → float` | Effective number of distinct expert paths (exp of path-distribution entropy) per sample batch. | arXiv:2506.21551 |
| Recorder wiring | Opt-in `"moe_routing"` diagnostic (captures gate outputs via existing hook strategies; emits `moe/routing_entropy/<module>` etc.). | — |

### Tests
~25 tests; OLMoE eval script extension under `scripts/`.

---

## Milestone v1.45 — Multi-process (DDP / FSDP) [XL] — IN PROGRESS (item 1 shipped 2026-06-11 as v1.45.0)

**Theme:** The moat. Implements the design §11 additive path so training-time
diagnostics are correct at real scale. Likely split into sub-releases.

### Items (per design §11 contract)

1. ~~`core/distributed.py` — reduce helpers~~ **SHIPPED v1.45.0**: `is_distributed`,
   `world_size`, `is_main_process`, `all_gather_concat` (activation-side),
   DTensor-aware `full_tensor` (weight-side; FSDP1 flat-param gathering is
   recorder-side work in item 4). 2-process gloo tests in
   `tests/core/test_distributed.py`.
2. ~~`TensorSource.WEIGHT_FULL` / `ACTIVATION_FULL`~~ **SHIPPED v1.46.0** — passthrough
   single-process; DTensor `full_tensor` + `all_gather_named` in distributed runs.
3. ~~DDP: rank-0 emission with cross-rank activation gathering~~ **SHIPPED v1.46.0** —
   participant mode on non-zero ranks (join collectives, write nothing); legacy
   recipes keep the v0.x no-op contract. 2-process gloo tests in
   `tests/recorder/test_full_sources.py`.
4. FSDP: full-param gather at emit steps (summon_full_params-style) behind an explicit
   opt-in flag with wall-clock budget enforcement (§10 still applies).
5. Validation scripts under `scripts/` (2-process CPU gloo CI job).

`docs/design.md` §11 to be amended from "future-release path" to the shipped contract
before this milestone merges.

---

## Addendum (2026-06-11): loose-end milestones v1.47–v1.48

The two survey candidates originally deferred to "a plan-sota-4 survey" are
promoted to milestones — both are implementable without hardware (pure
functions + a small CPU-trainable optimization loop, with precedent in
DASRunner / EdgePruningRunner).

## Milestone v1.47 — Cross-layer feature flow — SHIPPED 2026-06-11

**Theme:** Track how an SAE feature persists, transforms, or first appears
across layers via data-free decoder matching (Laptev et al., arXiv:2502.03032:
match feature i at layer A to argmax cosine among decoder rows at layer B).

| Item | What |
|------|------|
| `core/feature_flow.py::match_features(W_dec_a, W_dec_b, *, k=1) → (indices, sims)` | Per-feature top-k cosine matches between two decoder dictionaries `(n_features, d_model)`. |
| `core/feature_flow.py::feature_flow_graph(decoders, *, layer_ids=None, threshold=0.5) → FeatureFlowGraph` | Adjacent-layer match chains; edges kept at `sim ≥ threshold`. `FlowEdge` / `FeatureFlowGraph` with `.ranked()`, `.top_k()`, `.path_from(layer, feature)` (greedy argmax chain), `.born_at(layer)` (features with no upstream match), `.to_markdown()`. |
| `patching/export.py` | `FeatureFlowGraph` accepted by `to_neuronpedia_graph` / `to_html` (patching may import core). |

## Milestone v1.48 — Stochastic Parameter Decomposition (SPD) — SHIPPED 2026-06-11

**Theme:** The parameter-space pillar (Bushnaq et al., arXiv:2506.20790;
Apollo/Goodfire reference implementation github.com/goodfire-ai/spd).
Decomposes one target Linear's weight into C rank-one subcomponents
`V[:, c] ⊗ U[c, :]` trained so that (1) faithfulness: `V @ U ≈ Wᵀ`;
(2) stochastic reconstruction: with masks `m = ci + (1 − ci)·U(0,1)`
(uniform in `[ci, 1]`) on component activations, the model output matches
the unmodified model; (3) minimality: mean causal importance is penalized.
Causal importance `ci(x) ∈ [0,1]^C` is predicted by a small MLP on the
module input (the nano implementation uses a CI-transformer at LM scale;
the MLP is the original APD/SPD formulation and is documented as such).

| Item | What |
|------|------|
| `patching/spd.py::SPDRunner(model, module, *, n_components, importance_hidden=64)` | `.run(batches, *, n_steps, lr, coeff_faith, coeff_stoch, coeff_imp, loss_fn) → SPDResult`. Hook-based: replaces the target Linear's output with the masked component forward during training; no model surgery. |
| `SPDResult` | `.U (C, d_out)`, `.V (d_in, C)`, `.faithfulness_error`, `.importance(x) → (..., C)`, `.active_components(x, threshold)`, `.component_weight(c) → (d_out, d_in)`, `.to_markdown()`. |

---

## Track D (parallel, not a version) — distribution

Not library code; PRs alongside milestones as time allows:

- mkdocs-material docs site built from existing `docs/*.md` + API reference.
- 3–4 tutorial notebooks (IOI circuit end-to-end; live training monitoring; SAE
  feature circuits; graph export → Neuronpedia).
- Published MIB + SAEBench numbers in README (using `benchmarks/`).
- Short library paper (JOSS or arXiv) for citability.

---

## Ordering rationale

v1.41 first: days of work, immediate ecosystem visibility, and v1.42's labels plug
into its exporters. v1.42 before v1.43 because it is pure functions + one
result-aggregator (fast to ship) while v1.43 introduces a new execution mode.
v1.44 closes the remaining survey debt before the [XL] distributed milestone, which
lands last because it amends the design contract and needs dedicated validation
infrastructure.
