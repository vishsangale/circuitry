# v1.7 — SAE-circuit generalization (mlp_out/attn_out sites · TransformerLens backend · integrated gradients)

Status: **approved for implementation** (2026-05-30). Bundles the three deferred v1.6 follow-ons.
Builds on the v1.5 (`2026-05-30-v15-sae-feature-circuits-design.md`) and v1.6
(`2026-05-30-v16-sae-feature-edges-design.md`) specs, which remain the math contracts for node
attribution and feature→feature edges respectively. This spec was de-risked pre-code by a 10-agent
design workflow (4 empirical surveys + synthesis + 5 adversarial skeptics); §10 records what the
adversarial pass corrected.

## 0. Scope & decisions

Three follow-ons, **both forks resolved to FULL** by the user:

1. **`mlp_out` / `attn_out` SAE sites** — attribute node + edge circuits over MLP-output and
   whole-attention-output SAEs, not just `resid_post`.
2. **TransformerLens backend** — enable `TLSiteResolver` for SAE attribution (currently
   `NotImplementedError`).
3. **Integrated-gradients variant** (`variant='ig'`) — **nodes AND edges** (full EAP-IG).

**Fork 1 = FULL**: multiple SAE sites *per layer* may coexist in one circuit
(`attn_out@L` + `mlp_out@L` + `resid_post@L`), unlocking intra-layer `attn_out@L → mlp_out@L`
feature edges. Requires composite `(layer, component)` keying (§4).

**Fork 2 = FULL EAP-IG**: `variant='ig'` covers both node and edge attribution (§6).

**Fork 3 (TL dtype source)** decided without a user prompt: read `model.cfg.dtype` /
`torch.device(model.cfg.device)` on the TL path (the only correct option; alternatives are
silently wrong) — §5.

### Regression contract (non-negotiable)

Every v1.5/v1.6 result for **`resid_post` + HF-eager + `variant='attrib'`** stays **byte-for-byte
identical**. This is the literal truth, not an approximation: routing through `ResolvedSite` is the
identity object for `resid_post`+`position=None` (`resolved.extract(t) is t`,
`resolved.inject(full,v) is v`; `sites.py:62-73`), verified `0.000e+00` over 4 node cases + 1536
edge values (Task A + skeptic-1). Pinned by `test_resid_post_attrib_golden_freeze` (§7).

## 1. Architectural spine — route every SAE splice through `ResolvedSite`

The v1.5/v1.6 splices bypass the resolver: they hook the **whole decoder block**
(`self._atp._locate_layers(model)[site.layer]`) and use the module-level `_extract_tensor` /
`_inject_tensor` (`sae_features.py:46-63`) to unwrap/rewrap a tuple block output. This is *why* they
are `resid_post`-only and HF-only. The resolver already abstracts every site:
`HFSiteResolver.resolve(model, site)` (`sites.py:180-256`) and `TLSiteResolver.resolve`
(`sites.py:281+`) return `ResolvedSite(module, is_input_hook, extract, inject)`.

**Spine: replace `resolver_layers[site.layer]` with `resolver.resolve(model, site).module`
everywhere, and compose the resolver's extract/inject *inside* the existing tuple unwrap/rewrap.**

Composition order (verified equivalent to current, `0.0` diff):

```
tuple-or-tensor output
  → _extract_tensor(output)            # unwrap tuple → tensor (no-op for plain-tensor submodules / TL HookPoint)
  → resolved.extract(full)             # slice to sub-activation (identity for resid_post, pos=None)
  → SAE work (encode/decode/eps/leaf)  # UNCHANGED from v1.5/v1.6
  → resolved.inject(full, recon)       # write sub-activation back (returns recon for resid_post, pos=None)
  → _inject_tensor(output, new_full)   # rewrap into tuple (no-op for plain-tensor / TL)
  → output
```

Rules that **must not change**:
- Keep `_extract_tensor`/`_inject_tensor` as the **outer** layer — `resolved.extract` indexes a
  tensor (`x[:, pos]`) and crashes on a raw tuple.
- Keep the `recon.to(model_dtype, model_device)` cast-back (`sae_features.py:301`,
  `sae_edges.py:1135`) — `resolved.inject` does not re-cast. (dtype/device source changes for TL — §5.)
- The edge reader hook must **not** detach `a_in` (live encode; `sae_edges.py:1147`), and the writer
  hook keeps its detached-leaf seed + frozen `eps`.
- `is_input_hook` branch: implement it (`register_forward_pre_hook`, splice via
  `(new_full,) + args[1:]`) for completeness, but `resid_post`/`mlp_out`/`attn_out` are all
  `is_input_hook=False`, and only those are valid SAE sites (§3).
- **Assert `site.position is None`** in both runners' `__init__` (or document whole-stream-only): the
  block-hook splice ignored `site.position` and spliced the full tensor; routed `_pos_inject` with
  `pos!=None` would slice, silently changing semantics (skeptic-1 minor).

### 1.1 ALL splice / `_locate_layers` call sites (the blocker — §10 defect 1+2)

P1 must route **every** site below, not just the obvious three. Missing the ablation/circuit paths
makes an `mlp_out` circuit score edges in mlp-out feature space while faithfulness/ACDC ablate the
*residual stream* — same shape `(b,s,d_model)`, no error, wrong tensor (22× magnitude mismatch
measured).

`sae_features.py`: `_run_site` 210-211; node-bruteforce inner 431, 451, 501, 529 (+ the
`site_by_layer` map 443/450 → must become a `(layer,component)` map, see §4).

`sae_edges.py`: `compute_f_per_site` 64; `_feature_circuit_forward` 139-140;
faithfulness `ablation_eps` hook 513-514; edge Stage-2 965, 1043-1044; edge-bruteforce 1336,
1380-1381; `FeatureACDCRunner`/`SAEFeaturePruner.__init__` 1726.

`_locate_layers` definitions/uses to retire in favor of `resolver.resolve`:
`sae_features.py:210,431`; `sae_edges.py:845-846,965,1336,1726` (six sites; skeptic-3 found 1336 &
1726 were missing from the survey's list).

## 2. Phase plan

Each phase is independently testable; each ships behind its own machine-precision oracle. P1 is a
pure refactor (no behavior change) and is the safety foundation for P2–P4.

| Phase | Deliverable | Gate (oracle) |
|---|---|---|
| **P1** | Resolver-routed splice across ALL call sites (§1.1). `resid_post`+HF only, no behavior change. | `test_resid_post_attrib_golden_freeze` `assert_close(rtol=0, atol=0)` + existing GATE A (node 1.66e-07, edge-linear <1e-4, error→feature 5.96e-07). |
| **P2** | `mlp_out`/`attn_out` sites + FULL composite `(layer,component)` keying (§3,§4). | mlp_out node analytic==bruteforce (float64 8.9e-16); `compute_f_per_site` decomposes the TRUE submodule tensor; `faithfulness(M)==1` for an mlp_out reader; no same-layer node collision; `attn_out@L→mlp_out@L` edge present & `mlp_out@L→attn_out@L` absent. |
| **P3** | TransformerLens backend (§5). | Real tiny `HookedTransformer`: autograd vs central FD ≤1e-10; lossless splice ≤1e-15; linear-downstream analytic==bruteforce. HF golden freeze unchanged. |
| **P4** | IG variant for nodes + edges (§6). | Node IG completeness to the **eps-frozen spliced** delta, `err·N²`≈const (midpoint); edge IG vs path-integral trapezoid oracle 1.97e-7; IG-edge uses per-j VJP (no dense Jacobian). |

## 3. New `attn_out` component

`VALID_COMPONENTS` (`sites.py:11-19`) has `mlp_out` but **no** whole-attention `attn_out` (only
per-head `attn_head_out`). Add it.

- `sites.py`: add `"attn_out"` to `VALID_COMPONENTS`.
- `HFSiteResolver.__init__`: new kwarg `attn_module_full: str = "self_attn"` (distinct from the
  existing `attn_module="self_attn.o_proj"` used by `attn_head_out`). `resolve()` gains an
  `attn_out` branch mirroring `mlp_out` (`sites.py:237-244`): `module=_get_submodule(layer_mod,
  attn_module_full)`, `is_input_hook=False`, full-tensor `_pos_extract`/`_pos_inject`. The
  `self_attn` submodule returns a **tuple** `(attn_out, ...)` — the outer `_extract_tensor`/
  `_inject_tensor` unwrap it (verified round-trip preserves trailing `None`).
- `TLSiteResolver._TL_HOOK_MAP` (`sites.py:261-269`): add `"attn_out": "blocks.{L}.hook_attn_out"`
  (bare tensor; generic `resolve()` tail handles it).
- Gate relaxation (both runners): change `if site.component != "resid_post"` → `if site.component
  not in {"resid_post", "mlp_out", "attn_out"}` (`sae_features.py:108-112`,
  `sae_edges.py:817-821`). Keep rejecting `attn_head_out/_q/_k_out`, `mlp_neuron`, `resid_pre`
  (per-head/per-neuron sub-slices, not whole-vector SAE inputs).

**Arch caveat (§10 defect-adjacent, MAJOR).** `attn_module_full="self_attn"` /
`mlp_module="mlp"` capture the submodule output **before the residual add**. For Llama/GPT-2-style
blocks that equals the whole attention/MLP contribution. For **Gemma2** (`post_attention_layernorm`,
`post_feedforward_layernorm` applied before the residual add) this is the **pre-norm** output —
*not* the post-norm contribution an attn-out/mlp-out SAE was trained on. Scope the equivalence claim
to Llama-family in docs; the splice is still mechanically lossless on any arch (it round-trips
whatever the submodule emits). Document the documented fallback (target `self_attn.o_proj` output, a
plain tensor) for arches where that is the trained SAE's input.

**Forward order & parallel-attention caveat.** The intra-layer edge `attn_out@L → mlp_out@L` is
valid only for **sequential** attn→mlp blocks (`resid_mid = resid_pre + attn; mlp(resid_mid)`),
which the probe verified. **Parallel-attention** architectures (GPT-J-style: attn and mlp both read
`resid_pre`) make `attn_out@L` and `mlp_out@L` causally **unordered** — there is no intra-layer
edge. v1.7 assumes sequential blocks; document the limitation.

## 4. FULL: composite `(layer, component)` keying

Every internal structure currently keyed by the **int** `site.layer` must become keyed by
`(site.layer, site.component)`, and node identity must distinguish components, or two same-layer
sites silently merge (no shape error — the single largest hazard, §10).

### 4.1 Node identity

Add one optional field to `graph.Node` (`graph.py:14-22`):

```python
@dataclass(frozen=True)
class Node:
    kind: str
    layer: int | None = None
    head: int | None = None
    neuron: int | None = None
    component: str | None = None   # NEW: distinguishes same-layer SAE sites; None == resid_post / legacy
```

Backward-compatibility: the field defaults to `None`. Every existing node (`embed`, `attn_head`,
`mlp`, `logits`, and v1.6 `resid_post` SAE nodes) is constructed without `component`, so it keeps
`component=None` and its relative equality/hash is unchanged → **v1.6 goldens are preserved
exactly**. `graph._order` still raises on `sae_feature`/`sae_error` (unchanged); the SAE edge graph
keeps its own ordering (§4.3), so `graph.py` gains only the field.

SAE node construction (`sae_features.py:365,385`; `sae_edges.py:1259,1270,1285`): pass
`component=site.component if site.component != "resid_post" else None`. (resid_post → `None` keeps
v1.6 identity; `mlp_out`/`attn_out` → explicit string.)

### 4.2 Dict keys → `(layer, component)`

Replace the int-layer key with a `(layer, component)` tuple (or key directly by the frozen `Site`)
at **all** of:

`sae_edges.py`: `compute_f_per_site` result 65; `_feature_circuit_forward`
`circuit_nodes/ablation_values/error_in_circuit/ablation_eps` lookups 148-155; `_circuit_node_sets`
355,362; `_all_node_sets` 369,374; mean-ablation `result[layer]` 400; `in_circuit` 461,465;
`abl_eps_dict` 540; `complement` 625-626; pruner `kept_nodes` 722,727; Stage-2 `site_survivors`
945-956,975-976; bruteforce pair lookup `w_layer/r_layer` + site-by-layer 1345-1346,1368-1371;
FeatureACDC `node_scores`/`kept_nodes` 1812,1824-1862,1883,1896-1901,1968,1978.

`sae_features.py`: `site_by_layer` map 443,450 → `site_by_key` keyed `(layer, component)`.

When reconstructing a `Site` from a node (e.g. bruteforce 1368-1371, which does `s.layer ==
w_layer`), match on `(s.layer, s.component) == (node.layer, node.component or "resid_post")`.

### 4.3 Forward-position order

Replace the layer-only sorts (`sae_edges.py:136,293,850,1736`) with a forward-position rank:

```python
_COMPONENT_OFFSET = {"attn_out": 0, "mlp_out": 1, "resid_post": 2}
def _site_rank(site) -> int:
    return 3 * site.layer + _COMPONENT_OFFSET[site.component]
```

Pair `(writer, reader)` is a valid forward edge iff `_site_rank(writer) < _site_rank(reader)`.
`layer_pairs='all_forward'` = all such pairs; `'adjacent'` = consecutive sites in rank order.
This yields `attn_out@L → mlp_out@L` (rank 3L < 3L+1) and forbids `mlp_out@L → attn_out@L`
(verified: the reverse VJP is `None`).

### 4.4 Node-set ablation semantics (doc rewording)

The v1.6 justification — "downstream features share one residual `a_D`, so node-level set-ablation is
well-defined but edge-level is unimplementable" — is stated **per layer**. Under FULL, each site has
its own tensor/decomposition, so restate it **per `(layer, component)`**: node-set ablation stays
well-defined *per site*; a layer no longer maps to a single `a_D`. Update the v1.6 spec note and
`design.md` §4.6. `_feature_circuit_forward` must dedupe ablation by `(layer, component)`.

## 5. TransformerLens backend

Given the spine (§1), TL is **almost free** — the single hard HF dependency is
`_locate_layers`/`resolver_layers[site.layer]` (`_layout.py:29` raises on `model.blocks`), removed by
routing through `resolver.resolve(model, site).module` (a `HookPoint`). Verified on a real random-init
`HookedTransformer`: autograd through the `HookPoint` is exact (per-feature grad vs central FD
≤8e-11), `register_forward_hook` return-override is honored (`HookPoint.forward(x)=x`),
`_extract_tensor`/`_inject_tensor` are no-ops on the bare tensor, and `AtPRunner(model,
TLSiteResolver)` is already TL-aware (`atp.py:120-132`).

**Two net-new pieces:**
1. Remove the TL gates (`sae_features.py:96-100`, `sae_edges.py:805-809`).
2. **Backend-aware dtype/device** (the one real TL bug — `HookPoint.parameters()==[]`, so the
   params-fallback silently downcasts a non-fp32/CUDA TL model, 111% score error measured):

   ```python
   if isinstance(self.resolver, TLSiteResolver):
       model_dtype = self.model.cfg.dtype
       model_device = torch.device(self.model.cfg.device)   # cfg.device is a STR — wrap it
   else:
       # existing params[0] path
   ```
   Apply at `sae_features.py:215-221` and `SAEFeatureEdgeRunner._model_dtype_device`
   (`sae_edges.py:852-856`), plus any cast-back inside the edge writer/reader hooks.

Everything else (`_call_model` = `model(tokens)` → logits tensor, `_freeze_eval`/`_restore`,
`sae_decompose`/`encode`/`decode`, AtPRunner composition) is backend-agnostic and unchanged.
`attn_head_*`/`mlp_neuron` TL hooks return 4D/`d_mlp` tensors — out of SAE scope; `resid_post`/
`mlp_out`/`attn_out` are plain `(b,s,d)`.

## 6. Integrated gradients — `variant='ig'` (nodes + edges)

**Corrected from the survey (§10 defect 3).** IG path with `N` midpoints `α_k=(k-0.5)/N`
(O(1/N²) Riemann), `eps` **frozen at clean** at every site.

### 6.1 Sign & completeness (the corrected statement)

**Keep the existing `attrib` sign convention** — both v1.5 nodes (`sae_features.py`) and v1.6 edges
use **`Δf = f_corrupt − f_clean`**. IG must use the SAME `Δf` so that switching `variant='attrib'
→ 'ig'` refines the scores without flipping their sign (sign-consistency across variants on the same
runner matters more than matching external IG pedagogy). Parameterize the path
`f(α) = f_clean + α·Δf = f_clean + α·(f_corrupt − f_clean)`, `α: 0→1` (so `f(0)=f_clean`,
`f(1)=f_corrupt`), midpoints `α_k=(k-0.5)/N`:

```
score_i = Σ_pos Δf_i · (1/N) Σ_k ∂metric/∂f_i |_{f = f_clean + α_k·Δf}
```

**Completeness (the exact oracle).** By the fundamental theorem of calculus along that path,
```
Σ_i score_i  →  metric(decode(f_corrupt) + eps_clean) − metric(decode(f_clean) + eps_clean)
```
i.e. the **eps-frozen spliced** corrupt-minus-clean metric delta (same sign as `Δf`). This is **not**
the real `metric(corrupt) − metric(clean)` forward difference: because `eps` is held at `eps_clean`,
the corrupt endpoint uses `eps_clean`, whereas a real corrupted forward uses `eps_corrupt` (the two
can differ by the entire reconstruction-error term — measured `|eps_clean − eps_corrupt|` up to 24).
The completeness target is the **eps-frozen spliced** delta, by construction. Derive the test target
from the SAME `Δf`/path the code uses, so the telescoping sign is self-consistent.

**Tie-in to `include_error_node`.** The `sae_error` node gets its own IG by interpolating the
error leaf `eps_corrupt → eps_clean` while features stay at clean. Then:
```
Σ_i feature_IG_i  +  error_IG  →  metric(real corrupt) − metric(real clean)
```
So **features-only IG completes to the eps-frozen spliced delta; features + error IG completes to the
real forward delta.** State both honestly. (Verified: feature-IG + eps-IG matched the full move to
6.4e-7.)

**Do NOT** use "compare to `bruteforce_feature_scores` SUM" as the IG oracle — the v1.5 single-feature
patch sum is not a completeness identity on a nonlinear model (off by 1.498 in the probe). The oracle
is the eps-frozen joint-swap forward difference above.

### 6.2 Enumeration

Score the `Δf ≠ 0` union (clean-active ∪ corrupted-active), **not** gated on `grad@clean` — a feature
dead at clean but live at corrupt has a real nonzero IG even though its single-point attrib grad is
~0. This is exactly the saturation/zero-gradient blind spot IG fixes (attrib was off by 1.69 and
sign-flipped on the probe). Verify `top_k_survivors`/`max_features` caps do not silently re-introduce
a `grad@clean`-based top-K on the IG path (open-concern 3).

### 6.3 Edge IG (full EAP-IG)

Wrap edge Steps B–D in an `N`-loop. At step `k`, set the **writer** detached-leaf to
`f_U_k = f_U_corrupt + α_k·(f_U_clean − f_U_corrupt)` (interpolate **only** the upstream writer);
reader stays **live**; `eps` frozen at both sites; one backward; the per-downstream-survivor VJP loop
(`sae_edges.py:1236-1290`) is **UNCHANGED** — one `autograd.grad(f_D, f_U_leaf, grad_outputs=G_j)`
per surviving `j`, `del vjp_j` each iteration. Accumulate `edge(i→j) += Σ Δf_U[i]·vjp_j[i]`, divide
by `N`.

- **Cost** = `N ×` attrib forward+backward+VJP-loop. **Peak memory == attrib** (one `vjp_j` alive at a
  time; the per-step forward graph is built and freed each iteration — confirm no lingering refs).
- **No dense `d_sae × d_sae` Jacobian** — verified (matched the independent finite-difference
  path-integral oracle to 2.24e-11, float64).
- **Oracle**: IG-edge vs an independent path-integral trapezoid bruteforce (extend
  `bruteforce_feature_edge_scores` with an `n_ig_steps` path variant); N=512 → 1.97e-7.
  **Do NOT** assert `IG-edge == attrib-edge` even on `LinearResidToy`: the edge integrand
  `(∂f_D/∂f_U)·gradf_D` varies along the path whenever the metric is nonlinear in `f_D`, so they
  agree only when **both** the path **and** the metric are linear (machine-zero confirmed there).

### 6.4 API

- `SAEFeatureRunner.run(..., variant: str = "attrib", n_ig_steps: int = 0)` — keyword-only; thread to
  `_run_site`. `variant='exact'`/other → `NotImplementedError`. Default IG steps when `variant='ig'`
  and `n_ig_steps==0`: documented constant (use **32**; probe N=32 → 1.9e-5).
- `SAEFeatureEdgeRunner.run` already has `variant='attrib'` (`sae_edges.py:872`); add `n_ig_steps`,
  relax the edge gate (`sae_edges.py:890-894`) to accept `{'attrib','ig'}`, wire §6.3 into
  `_compute_pair_edges`. `FeatureACDCRunner` and faithfulness inherit `variant` via the edge runner.

## 7. Regression-freeze test (`test_resid_post_attrib_golden_freeze`)

Add in **P1, before** the refactor lands. Build the `LinearResidToy` + affine-SAE fixtures
(`test_sae_features.py:18-201`, `test_sae_edges.py:32-52`) with a **fixed seed**; from the current
(block-hook) v1.6.0 code capture, as checked-in golden constants:
- the full node-score dict (`SAEFeatureRunner.run`), and
- the full edge-value dict (`SAEFeatureEdgeRunner.run`),
- **for `include_error_node ∈ {False, True}` and `graddrop ∈ {False, True}`** (so the freeze is not
  weaker than the existing gates — skeptic-5 minor).

Key goldens by **`(layer, neuron)`** tuples (resid_post fixture has one component/layer, so this is
unambiguous and survives the §4.1 `component` field addition). Assert the routed path reproduces them
with `torch.testing.assert_close(rtol=0, atol=0)` (exact equality — Task A measured `0.0`). Capture
goldens once on HEAD; the test then gates P1–P4. New surfaces (TL, mlp_out, attn_out, IG) have no
pre-existing golden and cannot regress it.

## 8. Docs to amend (same commits)

- `docs/design.md` §4.6 sub-spec 7 (this work): resolver-routed splice, `attn_out` component,
  multi-site keying, TL support, `variant='ig'`/`n_ig_steps`; §3 if `attn_out` touches the public
  component list; §4 (new `attn_out` is public surface); §8 move TL/IG out of future work; §10 add an
  **IG `N×` cost note** so the ≤10% wall-clock budget is read as scoped to non-IG defaults
  (open-concern 7).
- `docs/superpowers/specs/2026-05-30-v16-sae-feature-edges-design.md`: per-`(layer,component)`
  node-set-ablation reword note (§4.4).
- `CHANGELOG.md`: `[1.7.0]` block.
- `src/circuitry/__init__.py` `__version__ = "1.7.0"`; `tests/test_public_api.py` version asserts.

## 9. Layering / CI

No new top-level imports (sae_lens/transformer_lens already on the `tests/test_layering.py`
allowlist). `patching/` may import `sae/` (allowed). `core/` stays pure. `graph.py` change is one
optional field — no import change. Run `lint-imports` + `tests/test_layering.py` each phase.

## 10. What the adversarial pass corrected (pre-code)

1. **🔴 BLOCKER — routing scope under-enumerated.** The synthesis routed only `_run_site`, node
   bruteforce, and `_compute_pair_edges`. `compute_f_per_site`, `_feature_circuit_forward`, and the
   faithfulness `ablation_eps` hook also hook the whole block → for `mlp_out`/`attn_out`,
   faithfulness/completeness/ACDC silently decompose the **residual stream** (wrong tensor, 22×
   magnitude, no error). Fix folded into §1.1 (route **all** sites). + arch-dependent submodule-output
   caveat (Gemma2 pre/post-norm) → §3.
2. **🟠 MAJOR — TL needs routing + dtype source + all 6 `_locate_layers` sites.** `HookPoint.
   parameters()==[]` → silent downcast; `cfg.device` is a `str` (wrap `torch.device`); survey missed
   `sae_edges.py:1336,1726`. Fixed in §5 + §1.1.
3. **🟠 MAJOR — IG completeness oracle was wrong.** Stated target `metric(clean)−metric(corrupt)`
   was both **sign-flipped** and **eps-conflated** (real corrupt uses `eps_corrupt`, IG endpoint uses
   `eps_clean`). Corrected to `Δf=f_clean−f_corrupt` + eps-frozen spliced target, with the
   features+error→real-delta tie-in. The `bruteforce-sum` oracle was dropped. IG-**edge** mechanism
   (per-j VJP, no dense Jacobian) was independently **confirmed sound** (2.24e-11). Fixed in §6.

Not refuted: routing equivalence (`0.0`), regression-freeze (holds; strengthened with
error/graddrop goldens).
