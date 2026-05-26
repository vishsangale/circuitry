# v1.1 — HF patching-backend generalization (design spec)

**Status:** approved (brainstorming, 2026-05-25). Both parts empirically verified
before writing (see "Evidence").

## 1. Motivation

The v1.0 real-model validation campaign (`docs/observations/2026-05-25-v1-validation.md`)
found that the patching pillar's **HF-eager backend silently fails on every real
non-Llama model**:

- **Gemma-2/3**: `head_dim ≠ d_model / n_heads` (explicit `config.head_dim = 256`)
  → `o_proj` reshape crashes (`RuntimeError: shape '[1,15,8,288]' invalid for size 30720`).
- **GPT-2 / non-Llama**: hard-coded `model.model.layers` / `self_attn.{q,k,v,o}_proj`
  nesting + `nn.Linear` weight orientation → `AttributeError: no attribute 'layers'`;
  GPT-2 uses `transformer.h`, fused-QKV `Conv1D`, LayerNorm, non-gated MLP.

The pillar's exact tests only ever used synthetic Llama configs where
`head_dim == d_model/n_heads`. The "HF backend" is really a Llama-family backend.

## 2. Goals / non-goals

**Goals:**
1. The HF-eager backend (EAP / AtP\* / ACDC) runs correctly on models whose
   `config.head_dim ≠ d_model/n_heads` (Gemma-2/3, and any future model).
2. Non-Llama HF models (GPT-2 and the ~50 families TransformerLens supports) are
   usable for patching via a documented **TL bridge** — without reimplementing
   TL's per-family weight handling inside the HF backend.
3. A user who points the HF backend at an unsupported layout gets a **clear,
   actionable error** (pointer to the bridge), not a cryptic `AttributeError`.

**Non-goals (explicitly out of scope):**
- Reimplementing Conv1D / fused-QKV / LayerNorm / non-gated-MLP handling *inside*
  the HF backend (the TL bridge covers these via TL's converters).
- Truly novel architectures TransformerLens does not know.
- AtP\* QK-fix numerical exactness under Gemma-2 attention logit softcapping /
  sliding-window (documented follow-on).

## 3. Part A — `head_dim` source fix

**Root cause (mapped):** `head_dim` is computed in exactly the resolver/runner
constructors; every per-head reshape downstream *reads* the stored
`self.head_dim` / `resolver.head_dim`. ACDC derives it (`acdc.py:61`,
`self.head_dim = self._eap.head_dim`). So correcting the source corrects all
reshapes — verified (§Evidence).

**Changes:**

1. `patching/sites.py` — `HFSiteResolver.__init__`: add keyword
   `head_dim: int | None = None`; set
   `self.head_dim = head_dim if head_dim is not None else d_model // n_heads`.
2. `patching/sites.py` — `HFSiteResolver.from_config`: read
   `head_dim = getattr(config, "head_dim", None)` and pass it to the constructor.
3. `patching/eap.py` — `EAPRunner.__init__` HF path (eap.py:106, currently
   `self.head_dim = (resolver.d_model // resolver.n_heads) if resolver is not None
   and n_heads > 0 else None`): **preserve the guard**, only swap the computation:
   `self.head_dim = resolver.head_dim if (resolver is not None and n_heads > 0) else None`.
4. `patching/atp.py` — `AtPRunner.__init__` HF path (atp.py:141): same guarded swap
   `self.head_dim = resolver.head_dim if (d_model is not None and n_heads > 0) else None`
   (match the existing guard variables in atp.py).
5. `patching/eap.py` + `patching/atp.py` — TL path (currently `cfg.d_model // n_heads`):
   prefer `getattr(cfg, "d_head", None) or cfg.d_model // cfg.n_heads` for robustness
   on TL models whose `d_head ≠ d_model/n_heads`.
6. `patching/acdc.py` — **no change** (inherits `self._eap.head_dim`).

GQA is unaffected: k/v reshapes use `n_kv_heads * head_dim`; with the correct
`head_dim` the existing slicing (`W_proj[kv_h*head_dim:(kv_h+1)*head_dim]`) is
correct (Qwen GQA already validated in v1.0).

## 4. Part B — TransformerLens bridge for non-Llama models

**New public helper** in `patching/` (exported from `circuitry.patching`):

```python
def to_hooked_transformer(
    hf_model,                      # a loaded HF *ForCausalLM model
    model_name: str,               # TL's name for the architecture, e.g. "gpt2"
    *,
    device: str | None = None,
    dtype=None,
    **tl_kwargs,                   # forwarded to HookedTransformer.from_pretrained
):
    """Wrap a loaded HF model as a TransformerLens HookedTransformer so the
    TL patching backend (TLSiteResolver) can run on it. Lazy-imports
    transformer_lens. Reuses the user's weights via hf_model=, applying TL's
    standard processing (fold_ln / center_writing_weights / center_unembed,
    overridable via tl_kwargs)."""
```

Implementation: lazy `from transformer_lens import HookedTransformer`; return
`HookedTransformer.from_pretrained(model_name, hf_model=hf_model, device=device,
dtype=dtype, **tl_kwargs)`. Usage:

```python
tl = to_hooked_transformer(gpt2_hf, "gpt2", device="cuda")
EAPRunner(tl, TLSiteResolver()).run(clean, corrupt, metric)
```

**Numerical note (documented):** TL folds LayerNorm and centers writing/unembed
weights, so the wrapped model's *activations* differ from the raw HF model's
(logits are equivalent). Patching results are computed on the TL-processed model
— the correct, self-consistent function. Callers needing the exact raw-HF
activations should use a Llama-family model on the HF backend instead.

## 5. Part C — clear error on unsupported HF layout

Today the layer-list location raises `AttributeError: '<Model>' object has no
attribute 'layers'`. Replace the bare attribute access with a check that raises:

```
ValueError: circuitry's HF patching backend supports Llama-family layouts
(model.model.layers + self_attn.{q,k,v,o}_proj). This model (<class>) is not
supported directly. For GPT-2 and other architectures, convert it with
circuitry.patching.to_hooked_transformer(model, "<name>") and use the
TransformerLens backend (TLSiteResolver). See docs/design.md §4.6.
```

**The locator is duplicated — the fix must hit both copies.** Verified:
- `EAPRunner._locate_layers` (eap.py:118) + `EAPRunner._embed` (eap.py:125).
- `AtPRunner._locate_layers` (atp.py:159) + `AtPRunner._embed` (atp.py:166) — an
  independent duplicate; a fix to eap.py alone leaves AtP\* throwing the cryptic
  `AttributeError`.
- `ACDCRunner` composes `EAPRunner` and reads `self._eap._layers_list` /
  `self._eap._embed()` (acdc.py:143-144) — **covered transitively** by the EAP fix.

**Implementation: centralize** to remove the drift risk. Add a small shared helper
`patching/_layout.py::locate_layers(model) -> nn.ModuleList` (and `locate_embed`)
that performs the guarded lookup and raises the `ValueError` above; have both
`EAPRunner` and `AtPRunner` call it. This is a targeted refactor of code we are
already modifying (it also de-duplicates the `_embed`/`_lm_head` helpers that
currently diverge between the two runners).

**Decision (declined scope):** we do *not* extend the locator to try non-Llama
nestings (e.g. `model.transformer.h`). Locating GPT-2's layers without also
handling its Conv1D / fused-QKV / LayerNorm would only push the failure later into
a more cryptic reshape/weight error. The clear error + `to_hooked_transformer`
bridge is the single supported non-Llama path (consistent with the §2 non-goals).

## 6. Testing strategy

1. **head_dim (CI, no network):** build a tiny `LlamaConfig` with an *explicit*
   `head_dim` that differs from `d_model // n_heads` (e.g. `hidden_size=64,
   num_attention_heads=8, head_dim=16` → 8×16=128 ≠ 64). **First assert the config
   actually honors it** (`cfg.head_dim == 16` and the built model's `q_proj.out_features
   == 128`); if a transformers version overrides/ignores `head_dim`, fall back to a
   minimal stub config object exposing `num_attention_heads`/`hidden_size`/`head_dim`
   so the test still isolates the `head_dim != d_model/n_heads` condition. Then
   assert: (a) EAP runs and scores edges; (b) the ACDC empty anchor (removed=∅ →
   KL==0 vs clean) and full anchor (removed=all → logits==corrupted-run) still hold
   exactly; (c) AtP\* `verify_top_k` runs. Regression guard for the Gemma-2 bug.
2. **head_dim (optional, network-gated):** a Gemma-2-2b smoke that EAP runs (skip
   if the model isn't cached / no network).
3. **TL bridge (CI if gpt2 cached, else skip):** load HF gpt2, `to_hooked_transformer`,
   run EAP, assert a known IOI name-mover (e.g. head 9.9) appears in the top edges.
4. **Error path (CI, no network):** a bare non-Llama `nn.Module` (or a stub without
   `.layers`) → **both** `EAPRunner(...).run(...)` *and* `AtPRunner(...).run(...)`
   raise the Part-C `ValueError` mentioning `to_hooked_transformer` (the locator is
   duplicated; test both paths so a one-file fix can't pass).
5. **No regression:** the full existing patching suite (toy Llama exact anchors,
   Qwen GQA, TL gpt2) stays green.

## 7. Layering, invariants, docs

- `transformer_lens` stays a **lazy optional dep** — `to_hooked_transformer`
  imports it inside the function body. `core/` unaffected; `tests/test_layering.py`
  allowlist unchanged.
- No new root dependencies.
- New internal module `patching/_layout.py` (shared `locate_layers`/`locate_embed`)
  imports only `torch`/`torch.nn` — within `patching/`, no layering impact.
- `docs/design.md` §3 (backend support) + §4.6 (intervention mode): document that
  the HF-eager backend targets Llama-family layouts and that
  `to_hooked_transformer` is the supported non-Llama path; note the `head_dim`
  generalization.
- `CHANGELOG.md`: `## [1.1.0]` entry — HF backend honors `config.head_dim`
  (Gemma-2/3 fix); `to_hooked_transformer` bridge for non-Llama models; clearer
  unsupported-layout error. No breaking changes to the v1.0 API.

## 8. Evidence (verified before writing this spec)

- **Part A:** overriding `resolver.head_dim = runner.head_dim = config.head_dim`
  (256) made EAP run end-to-end on Gemma-2-2b — **74,218 edges, no crash**
  (the buggy `d_model//n_heads` gives 288 → reshape crash).
- **Part B:** `HookedTransformer.from_pretrained("gpt2", hf_model=<loaded HF gpt2>)`
  + `EAPRunner(tl, TLSiteResolver())` recovered the IOI circuit — top heads
  **9.9, 10.7, 9.6, 5.5, 8.10** (name-mover / neg-mover / name-mover / induction /
  S-inhibition).
