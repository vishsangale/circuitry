# v1.5.0 — SAE-feature circuits (node-level attribution)

**Status:** approved for implementation (2026-05-30). Design produced by a 9-agent design
workflow (survey → 3 proposals → synthesis → adversarial critique). The critique ran live
probes against `sae_lens` 6.44.2 and corrected the synthesis on two points (enumeration set,
error-leaf construction) — those corrections are baked into this spec.

## 1. Goal & scope

Extend AtP\*-style node attribution from architectural nodes (embed / mlp / attn_head /
mlp_neuron) to **individual SAE features**. v1.5.0 ships **node-level** feature attribution:
score each SAE feature's causal effect on a differentiable metric, given a clean/corrupted
prompt pair. This is post-hoc analysis (needs a prompt pair), not a live recorder diagnostic.

**In scope (v1.5.0):**
- A new differentiable SAE helper module `src/circuitry/sae/grad.py`.
- New `Node` kinds `"sae_feature"` and `"sae_error"` in `graph.py`.
- A new `SAEFeatureRunner` in `src/circuitry/patching/sae_features.py` (HF-eager path only).
- The error-term reconstruction node, **opt-in** (`include_error_node=False` by default).
- Site config via `dict[Site, SAE | (release, sae_id)]`, scoped to `resid_post`.
- Two-tier validation gate (exact on linear, correlation on nonlinear) + opt-in real-checkpoint smoke.

**Deferred (NOT v1.5.0):**
- Feature→feature **edges** / full sparse-feature graph (the headline next release).
- TransformerLens backend (clear `NotImplementedError`).
- Transcoders / skip / matching_pursuit / temporal architectures (clear `NotImplementedError`).
- `mlp_out` / `attn_out` SAE attach points (trivial future extension of the same splice).
- `Recipe.with_sae` regex site-config convenience form; live recorder-time feature attribution.
- Integrated-gradients variant for early-layer underestimation (document the limitation, ship single-pass).
- Recorder/report rendering of feature-circuit results.

## 2. Mechanism — error-term substitution (Marks "Sparse Feature Circuits")

At each configured residual site, on the **clean** forward pass, splice the SAE in losslessly:

```
a            = <residual stream at the site>           # layer output (b, s, d_model)
f            = sae.encode(a_in)                          # (b, s, d_sae) — SAE feature acts
x_hat        = sae.decode(f)                             # (b, s, d_model) — reconstruction
eps          = (a_in - x_hat).detach()                   # frozen clean reconstruction error
recon        = x_hat + eps                               # == a_in numerically (lossless ~1e-7)
<splice recon back as the layer output; downstream sees recon, not a>
```

Because `recon == a` numerically, the model output is preserved (faithful to the SAE's
reconstruction quality). But `f` is now a node in the autograd graph, so the metric is
differentiable w.r.t. each feature.

**Feature score** (the existing `mlp_neuron` formula, atp.py:1293-1300, in the SAE basis):

```
score(feature i) = Σ_pos ( Δf[..., i] · gradf[..., i] )
  Δf    = f_corrupt − f_clean        # corrupted − clean  (same convention as atp.py:1311)
  gradf  = ∂metric/∂f at the CLEAN activation (from backward of the spliced clean pass)
```

`graddrop` variant: `Σ_pos | Δf[...,i] · gradf[...,i] |` (per-feature the contribution is already
a per-position scalar, mirroring atp.py:1296-1298).

**Verified properties** (probed live during design):
- On a **linear downstream model** + affine/ReLU SAE, per-feature AtP == brute-force feature
  patching to ~3e-6 (≪ 1e-4 gate). Exactness needs DOWNSTREAM linearity, NOT SAE linearity,
  because `decode` is affine in `f` and the detached `eps` makes `recon` affine in `f`.
- A nonlinear downstream model correctly breaks the exact gate (→ Gate B correlation test).
- Splice is lossless to ~1e-7.

### 2.1 Gradient seeding — reseed at `f` (do NOT rely on embed seeding)

Inside the clean-pass splice hook, seed the autograd graph AT THE FEATURE TENSOR (analogous to
AtPRunner's embed seed, but localized to the site):

```python
a_in   = extract(output).to(sae.device, sae.dtype)     # detached-safe; SAE+model both frozen
f      = sae.encode(a_in).detach().requires_grad_(True) # SEED: leaf with grad
f.retain_grad()
x_hat  = sae.decode(f)
eps    = (a_in - x_hat).detach()
recon  = x_hat + (err_leaf if include_error_node else eps)
out_spliced = inject(output, recon.to(model_dtype/device))
```

`metric(out).backward()` then populates `f.grad`. Model and SAE params are frozen, so no param
grad leaks. Grad flows from the metric through `decode(f)` (and downstream frozen-param ops) to
`f` — confirmed correct.

### 2.2 Error node (opt-in) — independent error leaf, SINGLE pass

**CRITICAL (corrected from synthesis):** you CANNOT get feature grad and error grad from the
same `recon` with a non-detached error: `recon = x_hat + (a − x_hat)` algebraically cancels
`x_hat`, zeroing `f.grad`. Use an **independent leaf** so both gradients come from ONE
forward + ONE backward:

```python
err_leaf = eps.clone().requires_grad_(True)   # eps already detached
err_leaf.retain_grad()
recon    = x_hat + err_leaf
# after backward:
score(sae_error) = Σ_pos ( Δeps · err_leaf.grad ),  Δeps = eps_corrupt − eps_clean
```

When `include_error_node=False` (default), use `recon = x_hat + eps` (no leaf) — identical
feature scores. Lock with a test asserting feature scores are bit-identical with the node on/off.

## 3. New code

### 3.1 `src/circuitry/sae/grad.py` (new; differentiable; `sae/metrics.py` UNTOUCHED)

Thin wrappers that call the SAE's own `encode`/`decode` under **normal autograd** (NO
`inference_mode`, NO `no_grad`, NO `.detach()` on the live path). Device/dtype-aligned (reuse
metrics.py:26-28 pattern). Exported from `sae/__init__.py`.

```python
SUPPORTED_SAE_ARCHITECTURES = {"standard", "topk", "jumprelu", "gated"}

def assert_supported_sae(sae) -> None:
    """Raise NotImplementedError for transcoder/skip_transcoder/jumprelu_transcoder/
    matching_pursuit/temporal and for normalize_activations not in
    {none, layer_norm, constant_norm_rescale}. Mirror the existing non-Llama clear-error."""

def encode_features(sae, x):    # differentiable; returns f (b,s,d_sae) or (n,d_sae)
def decode_features(sae, f):    # differentiable; returns x_hat in x-space

def sae_decompose(sae, x):
    """Single PAIRED encode→decode (required for stateful layer_norm/constant_norm_rescale
    modes — never cache f and decode later). Returns (f, x_hat, eps) with eps=(x-x_hat).detach()
    = frozen clean reconstruction error."""
```

Architecture detection: `sae.cfg.architecture()` (a method in sae_lens 6.x). Note BatchTopK /
Matryoshka load for inference AS jumprelu (architecture string `"jumprelu"`), so they're covered.

### 3.2 `src/circuitry/patching/graph.py` (edit)

Add `"sae_feature"` and `"sae_error"` to the `Node.kind` docstring/allowed set. `sae_feature`
overloads the existing `neuron` field as the feature index (exactly as `mlp_neuron` does).
**Do NOT modify `_order()` or `edge_sort_key()`** — node-level results never route through edge
machinery (`_order` is called only by `build_graph`/`reverse_topo_readers`). Adding to `_order`
would be latent edge coupling. `AtPNode`/`AtPResult.ranked/top_k/threshold` work unchanged
(kind-agnostic, sort by `abs(score)`).

### 3.3 `src/circuitry/patching/sae_features.py` (new)

```python
class SAEFeatureRunner:
    def __init__(self, model, sae_sites: dict[Site, "SAE" | tuple[str, str]], resolver):
        # Resolve (release, sae_id) tuples via circuitry.sae.loader.load_sae (NO recipes import).
        # Resolve each Site via resolver.resolve(model, site) -> ResolvedSite (extract/inject reused).
        # Gate: each site.component must be "resid_post" (else NotImplementedError, v1.5 scope).
        # Gate: assert_supported_sae(sae) per site; TLSiteResolver -> NotImplementedError.
        # Reuse AtPRunner helpers by COMPOSITION: _freeze_eval/_restore/_locate_layers/_embed
        # (import from circuitry.patching.atp; or factor a tiny shared mixin).

    def run(self, clean_inputs, corrupted_inputs, metric, *,
            graddrop=False, include_error_node=False, max_features=None) -> AtPResult:
        # 1. _freeze_eval(model) + freeze SAE params (sae.parameters() if nn.Module) — try/finally restore.
        #    (Real sae_lens params default requires_grad=True and LEAK otherwise.)
        # 2. Corrupted forward (no grad, NO splice): per site capture f_corrupt=encode(a_corr).detach()
        #    and (if include_error_node) eps_corrupt=(a_corr-decode(f_corrupt)).detach().
        # 3. Clean forward + backward (enable_grad), SPLICED per 2.1/2.2: capture f (retain_grad),
        #    f_clean=f.detach(), eps_clean, err_leaf (if enabled). metric(out).backward().
        # 4. Per site, enumeration set = features where Δf != 0 (union of clean-active OR
        #    corrupted-active). Score each; fp32 accumulation; device-aligned. Optional max_features
        #    cap = keep top-|score|. node = AtPNode(Node("sae_feature", layer=site.layer, neuron=i)).
        #    If include_error_node: AtPNode(Node("sae_error", layer=site.layer)) scored from err_leaf.grad.
        # 5. Restore model + SAE; return AtPResult(scores).

    def bruteforce_feature_scores(self, clean, corrupted, metric, nodes) -> dict[AtPNode, float]:
        # INDEPENDENT ground truth (never derived from the analytic score). Spliced clean baseline
        # m0 = metric(spliced clean). For each sae_feature node i: in a spliced clean forward,
        # patch f[...,i]=f_corrupt[...,i] BEFORE decode, re-run, score = metric - m0.
        # For sae_error node: patch eps=eps_corrupt. Mirrors AtPRunner.bruteforce_node_scores.
```

**Enumeration default (corrected from synthesis):** features where **Δf ≠ 0** = union of
clean-active and corrupted-active. The earlier "clean-live-only" idea is WRONG: `∂metric/∂f_i`
flows through `decode(f)=f@W_dec` (affine in EVERY feature), so a clean-INACTIVE /
corrupted-ACTIVE feature has a real nonzero score (probe: AtP=4.95=brute-force). Those are
exactly the features that turn ON to drive the behavior — must not be dropped. Document that an
omitted feature has score 0 by construction (Δf=0 ⇒ score 0).

### 3.4 Public API (`src/circuitry/patching/__init__.py`, `src/circuitry/__init__.py`)

Export `SAEFeatureRunner` (and reuse existing `AtPNode`/`AtPResult`). Export `sae_decompose` /
`encode_features` / `decode_features` from `sae/__init__.py`. Bump `__version__ = "1.5.0"` and
update `tests/test_public_api.py`.

### 3.5 Layering — NO change needed

`sae_lens` is already in `tests/test_layering.py` `ALLOWED_ROOTS`; `.importlinter` only guards
`core/`-purity and `recipes/`→`cli`. `patching → sae → sae_lens` passes as-is. Note: `.importlinter`
does NOT guard `patching` (the pytest `test_patching_does_not_import_cli` does) — don't claim otherwise.

## 4. Validation (acceptance gates)

**Fixtures** (`tests/patching/conftest.py` or local):
- `LinearResidToy` — linear residual stack where each layer is an `nn.Module` that RETURNS the
  residual stream (so a `resid_post` forward hook on `layers[L]` fires). Resolve via
  `HFSiteResolver(layer_pattern="layers.{L}")`.
- `SyntheticSAE` — an `nn.Module` with `W_enc/b_enc/W_dec/b_dec` `nn.Parameter`s
  (`requires_grad=True` so the freeze is actually tested), `.device`/`.dtype`/`.cfg.architecture()`/
  `.cfg.d_sae`/`encode`/`decode`. Affine by default; `relu=True` variant; a `topk`/`jumprelu`-style
  hard-mask variant for the inactive-feature-gradient test.

**GATE A — exact (default CI, `abs=1e-4`):**
- `test_feature_atp_matches_bruteforce_linear` — LinearResidToy + affine SAE: per-feature AtP ==
  `bruteforce_feature_scores` (verified ~3e-6).
- `test_relu_encode_still_exact_on_linear_model` — ReLU-encode SAE, still exact (decode affine).
- `test_error_term_makes_splice_lossless` — `‖recon − a‖∞ < 1e-5`.
- `test_error_node_exact_on_linear` — sae_error AtP == brute-force error patching.
- `test_feature_scores_identical_with_error_node_on_off` — bit-identical feature scores.
- `test_model_clean_after_feature_atp` — model output unchanged after `run()`.
- `test_no_sae_param_grad_leak` + `test_no_model_param_grad_leak` — frozen contract extends to SAE.
- `test_inactive_feature_has_nonzero_score` — clean-inactive/corrupted-active feature scores nonzero
  AND equals brute force (the corrected enumeration — guards the fatal-flaw regression).

**GATE B — correlation (default CI, no network, numpy-only):**
- `test_feature_atp_correlates_on_nonlinear` — nonlinear toy: Spearman ≥ 0.7, Pearson ≥ 0.6,
  sign-agreement ≥ 0.9 over the scored features (+ a seeded random low-rank sample).

**Robustness:**
- `test_unsupported_architecture_error` (transcoder), `test_tl_not_implemented`,
  `test_non_resid_post_site_error`, `test_metric_must_be_differentiable` (clear error if `f.grad`
  is None), `test_feature_grad_device_align` (SAE dtype ≠ model dtype; CUDA cross-device skipif),
  `test_max_features_cap`, `test_graddrop_feature_variant`.

**TIER 3 — opt-in (`@pytest.mark.slow` + skipif-offline):** one real `load_sae` checkpoint; assert
the pipeline runs end-to-end with finite scores of correct shape. NO numeric exactness; Gate A must
NOT depend on a download.

## 5. Risks (carried from design; mitigations in tests)

- Device/dtype misalignment (v1.4.1 class): explicit `.to(device,dtype)`; fp32 score accumulation.
- §10 budget: default single backward; splice adds ~2 matmuls/site/forward (same class as the
  existing `sae_reconstruction` diagnostic). Error node adds no extra pass (dual-leaf). Not a live
  diagnostic, so the ≤10% training-loop budget is not directly implicated, but keep the splice cheap.
- Stateful normalization: single paired encode→decode per hook call; add a `layer_norm` round-trip test.
- Metric must be the tensor-returning variant (`logit_diff_t` etc.); float `.detach()` version zeros grad.
- Baseline semantics: scores are relative to a frozen-clean-error baseline (standard SFC choice) —
  document in design.md §4.

## 6. Docs (same commit as code)

`docs/design.md` §4 (sae_feature/sae_error nodes, `SAEFeatureRunner`, `sae/grad.py` surface,
error-term-substitution contract, active-feature-only-by-Δf semantics, exact-gate-needs-linear-
downstream note, frozen-clean-error baseline), §10 (splice cost), remove "SAE-feature circuit
extraction" from §1/§8 deferred lists (node-level now shipped; edges still future). `CHANGELOG.md`
`[1.5.0]`. README if a feature-attribution example is warranted.
