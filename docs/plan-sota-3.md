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

## Milestone v1.43 — Generation-time analysis

**Theme:** Patch, steer, and trace across multi-token decoding, not just one forward
pass. Hook-based and model-agnostic (works with HF `generate` or a hand-rolled decode
loop); KV-cache-aware (hooks stay armed across steps; per-step seq-len of 1 handled).

### Items

| Item | What |
|------|------|
| `patching/generation.py::GenerationTrace` | Per-decode-step capture: chosen token, top-k logits, per-site activation stats (entropy, norm), optional logit-lens KL per step. `.to_markdown()`, export hook. |
| `patching/generation.py::trace_generation(model, step_fn \| generate_fn, *, sites, n_steps)` | Drives or observes a decode loop; returns `GenerationTrace`. |
| `patching/generation.py::apply_steer_steps(model, site, vector, *, steps)` / `patch_site_steps(...)` | Step-indexed interventions: active only on decode steps in `steps` (e.g. steer only after step 10). Builds on existing `apply_steer` / `patch_site` context managers. |
| `patching/generation.py::generation_attribution(model, prompt, *, target_step, method='atp')` | Attribute a *generated* token at step t back through the prompt + earlier generated tokens (re-runs teacher-forced forward over the realized sequence, then delegates to existing runners). |

### Tests
~25 tests with a tiny decode loop over the existing test transformer.

---

## Milestone v1.44 — MoE routing diagnostics

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

## Milestone v1.45 — Multi-process (DDP / FSDP) [XL]

**Theme:** The moat. Implements the design §11 additive path so training-time
diagnostics are correct at real scale. Likely split into sub-releases.

### Items (per design §11 contract)

1. `core/distributed.py` — pure reduce helpers (`all_gather_concat`, sharded-SVD
   composition where exact, documented approximations where not).
2. `TensorSource.WEIGHT_FULL` / `ACTIVATION_FULL` — hook-level sources that gather
   before computing, with budget guards.
3. DDP: rank-0 emission with optional cross-rank activation stats reduction.
4. FSDP: full-param gather at emit steps (summon_full_params-style) behind an explicit
   opt-in flag with wall-clock budget enforcement (§10 still applies).
5. Validation scripts under `scripts/` (2-process CPU gloo CI job).

`docs/design.md` §11 to be amended from "future-release path" to the shipped contract
before this milestone merges.

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
