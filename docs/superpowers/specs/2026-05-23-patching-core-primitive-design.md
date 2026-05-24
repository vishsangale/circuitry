# v1.0 Core Intervention Primitive — Design Spec

> **Sub-spec 1 of the v1.0 patching pillar.** This design covers the core
> intervention primitive and the `docs/design.md` contract amendment.
> Follow-on sub-specs (EAP, AtP\*, ACDC) build on top of this primitive and
> each get their own brainstorm→spec→plan→implement cycle.

**Goal:** Give circuitry the ability to *intervene* on model activations —
not just observe them — so that causal/attribution patching methods can run
on the same Recorder/scan/report surface the library already provides for
observation.

**Architecture:** A new top-level `patching/` orchestration subsystem
(parallel to `recorder/` and `scan/`), with pure metric helpers in
`core/patching.py`, site selectors in `recipes/`, and dual-path site
addressing (HF-recipe fallback + optional TransformerLens interop).

**Tech Stack:** Python 3.12, PyTorch, HF transformers (eager attention
required for per-head sites), TransformerLens (optional, lazy import).

---

## v1.0 roadmap (the release banner)

v1.0 is a release banner spanning several sequential sub-spec cycles:

| Sub-spec | Scope | Depends on |
|----------|-------|------------|
| **1 — core intervention primitive** | hook/replace/restore, prompt-pair runner, metrics, contract amendment | — |
| 2 — EAP | edge attribution patching (gradient-based, single backward) | 1 |
| 3 — AtP\* | attribution patching + corrections (activation gradients) | 1 |
| 4 — ACDC | iterative circuit pruning (real patching per edge) | 1 + edge graph |
| later — SAE-feature circuits | intervention site = SAE feature | 1 + existing SAE primitive |

Each sub-spec ships independently. The core primitive (this spec) must land
first.

---

## 1. Taxonomy

These terms have precise meanings throughout this spec and all follow-on
sub-specs:

- **Activation patching** — the *mechanism*: run a model on a clean prompt
  and a corrupted prompt, cache activations at chosen sites, replace one
  run's activation at a site with the cached value from the other, measure
  the change in a metric. *Denoising* = substitute clean activation into
  the corrupted run (does restoring this site recover performance?).
  *Noising* = substitute corrupted activation into the clean run (does
  corrupting this site break performance?).

- **Attribution methods** (EAP, AtP\*) — approximate the effect of
  patching every site using forward + gradient computations, avoiding
  O(sites) forward passes.

- **ACDC** — a *search algorithm* that uses real patching to iteratively
  prune edges from the computational graph, keeping only edges whose
  removal degrades the metric beyond a threshold.

- **Site** — a named location in the model's computation where an
  activation can be read, cached, or replaced (a node in the component
  graph).

- **Edge** — a connection between two sites in the residual-stream
  computational graph. *Not* a core-primitive concept; derived per-method
  by EAP/ACDC from the residual-stream decomposition.

---

## 2. Module layout

```
src/circuitry/
  core/
    patching.py        PURE helpers (tensor→tensor/number):
                         logit_diff, kl_divergence, ce_loss,
                         residual-stream summand bookkeeping.
                       No model execution, no I/O, no .cuda().

  patching/            NEW top-level orchestration subsystem:
    __init__.py        Public API re-exports.
    sites.py           Site dataclass + resolution logic.
                         HF-recipe layout + TL interop +
                         user-registry escape hatch.
                         Head/neuron slicing with per-architecture
                         docs (Llama-family first).
    intervene.py       Hook install / replace / restore-on-exit
                         context manager. The core intervention
                         primitive.
    runner.py          Clean/corrupted prompt-pair runner.
                         Activation cache. Metric evaluation.

  recipes/
    llm.py             + site selectors: enumerate/resolve sites
                         for Llama-family HF models.
```

### Layering (CI-enforced amendment to design.md §3)

| Module | May import | Must NOT import |
|--------|-----------|-----------------|
| `core/` | stdlib, torch, numpy | recorder/, recipes/, writers/, cli/, **patching/** |
| `patching/` | core/, recipes/, stdlib, torch, numpy, transformer\_lens (lazy) | cli/ |
| `recipes/` | core/ | cli/, patching/ |

`transformer_lens` is added to `tests/test_layering.py`'s import allowlist
as an approved **optional** dependency. The import must be lazy (guarded
behind a try/except or `importlib` check); circuitry must install, import,
and run without it.

---

## 3. Site model

### Site dataclass

```python
@dataclass(frozen=True)
class Site:
    component: str   # resid_pre | resid_post | attn_head_out | mlp_out | mlp_neuron
    layer: int
    head: int | None = None       # required for attn_head_out
    neuron: int | None = None     # required for mlp_neuron
    position: int | slice | None = None  # optional token-position slice
```

The core primitive operates on **nodes** (individual site activations).
Edges (upstream→downstream contributions via the residual stream) are a
derived concept built per-method in the EAP/ACDC sub-specs — not part of
this primitive.

### Site addressing — dual path

**TransformerLens path** (preferred when available):

If `isinstance(model, HookedTransformer)` (lazy import), map `Site` to
native TL hook names:
- `attn_head_out` → `blocks.{L}.attn.hook_z` (already per-head)
- `mlp_out` → `blocks.{L}.mlp.hook_post`
- `mlp_neuron` → index into `blocks.{L}.mlp.hook_post`
- `resid_pre/post` → `blocks.{L}.hook_resid_pre/post`

Clean per-head and per-neuron access; no eager-attention requirement; a
known computational graph.

**HF fallback path:**

Recipe + HF config (`n_layers`, `n_heads`, `d_model`, `d_mlp`):
- `resid_pre/post` → hook on the residual-stream module (e.g.
  `model.layers.{L}` input/output).
- `attn_head_out` → hook on the attention output, reshape
  `(batch, seq, n_heads, d_head)`, slice head. **Requires eager attention**
  (existing v0.9.1 warn applies).
- `mlp_out` → hook on MLP block output.
- `mlp_neuron` → hook on MLP intermediate (e.g. `down_proj` input), index
  neuron. **Architecture-specific**: SwiGLU/GeGLU neuron layout differs
  from standard GELU. Llama-family first; other architectures = follow-on
  extensions with a **clean error** (not a wrong number).
- `position` → slice dim 1 of the hooked activation tensor.

**User-registry escape hatch:**

For architectures neither the recipe nor TL knows:
```python
custom_sites = {
    "attn_head_out": SiteResolver(module="model.layers.{L}.self_attn", slice_fn=my_head_slicer),
    ...
}
runner = PatchRunner(model, site_resolvers=custom_sites)
```

### Value-add for TL interop

circuitry's patching shares the **same** Recorder/scan/report surface, tag
namespace, and per-modality recipe abstraction — one library for
observation + intervention, vs. stitching TransformerLens + auto-circuit +
custom logging. The unified surface is the differentiated value-add over
using TL-native patching directly.

---

## 4. Intervention mechanism

### Core context manager (`intervene.py`)

```python
with patch_site(model, site, value, site_resolver) as handle:
    output = model(**inputs)
    # site activation was replaced with `value` during this forward
# hook removed, activation restored, model state identical to before
```

Implementation:
1. Resolve `site` → `(module, slice_fn)` via the site resolver (TL or HF).
2. Register a forward hook on the target module that overwrites the
   activation (or the relevant slice) with `value`.
3. Run forward inside the context.
4. **Guarantee restore on exit** via `try/finally`: remove the hook, no
   residual state. Follows the v0.9.2 mutation-last discipline — hooks are
   installed as the **last** setup step so partial setup can't leak.

### Gradient discipline

- **Model frozen**: all parameter `requires_grad` attributes stay `False`
  (or are set to `False` and restored). No optimizer, no parameter updates.
  Eval mode is set on entry and restored on exit.
- **Activation gradients**: may be enabled at sites via
  `tensor.requires_grad_(True)` + `retain_grad()` for attribution methods
  (AtP\*, EAP). This is the only gradient flow permitted — it does not
  touch parameters.
- Mirrors the v0.9.2 discipline: mutation-last, restore-on-exit even on
  exception, partial setup can't leak.

### Prompt-pair runner (`runner.py`)

```python
runner = PatchRunner(model, recipe=get_recipe("llm"))

result = runner.run_patching(
    clean_inputs=clean_ids,
    corrupted_inputs=corrupted_ids,
    sites=[site_a, site_b],
    metric=logit_diff,
    direction="denoise",  # or "noise"
)
# result.metric_values: dict[Site, float]
# result.cached_activations: dict[Site, Tensor] (optional)
```

Workflow:
1. Run the corrupted prompt; cache activations at all requested sites.
2. For each site (or batch of sites for efficiency): run the clean prompt
   with the site's activation replaced by the cached corrupted value
   (noising), or vice versa (denoising).
3. Evaluate the metric on the patched output vs. the clean baseline.
4. Return per-site metric deltas.

---

## 5. Metrics (`core/patching.py`)

Pure tensor→number functions, no model execution, no I/O:

| Function | Signature | Notes |
|----------|-----------|-------|
| `logit_diff` | `(logits: Tensor, correct: int, incorrect: int) → float` | Difference in logits at correct vs incorrect token |
| `kl_divergence` | `(p_logits: Tensor, q_logits: Tensor, chunk_size: int = 256) → float` | KL(softmax(p) ‖ softmax(q)), chunked (reuses v0.9.2 lens pattern) |
| `ce_loss` | `(logits: Tensor, targets: Tensor) → float` | Cross-entropy loss |

Users can pass any `Callable[[Tensor], float]` as a custom metric to the
runner.

---

## 6. Contract amendment (`docs/design.md`)

### Current state

design.md defines circuitry as observation-only: primitives are pure
functions; Recorder and scan observe without modifying model state.

### Amendment

Add a new section (or amend §4) introducing a sanctioned **intervention
mode** with these properties:

1. **Opt-in**: interventions require explicit use of the `patching/` API.
   Recorder and scan remain observation-only and are unaffected.
2. **Isolated**: every intervention is scoped to a context manager. Hooks
   are removed and model state is restored on exit, including on exception.
   The mutation-last discipline (hooks installed as the final setup step)
   prevents partial-setup leaks.
3. **Frozen model**: parameter `requires_grad` stays off; eval mode is
   managed and restored; no optimizer or parameter updates occur.
4. **Activation-grad-only**: the only gradient flow permitted is on
   activation tensors at intervention sites, for attribution methods.
   Parameter gradients are never enabled.
5. **No training-state leak**: an intervention run does not modify the
   model's training state (parameters, buffers, optimizer state,
   batch-norm running stats, eval/train mode).

### Layering amendment

- `patching/` added as a new top-level subsystem alongside `recorder/` and
  `scan/`.
- `core/` must NOT import `patching/`.
- `patching/` may import `core/` and `recipes/`; must NOT import `cli/`.
- `transformer_lens` added to `tests/test_layering.py` allowlist as an
  optional dependency; import must be lazy.

---

## 7. Testing strategy

| Test | What it verifies |
|------|-----------------|
| **Deterministic patching** | Toy 2-layer model where a known site causally controls the output. Patching changes the metric to a known value. |
| **Restore-on-exit** | After patching (normal exit), model params + output are bit-identical to before. |
| **Restore-on-exception** | Same guarantee when the forward raises inside the context manager. |
| **HF head slicing** | On a tiny HF config, `attn_head_out` at a specific `(layer, head)` reads/writes the correct slice. |
| **HF neuron slicing** | On a tiny Llama config, `mlp_neuron` at `(layer, neuron)` reads/writes the correct index. |
| **TL path** | `skipif transformer_lens not installed`; Site resolves to correct TL hook name; patching works end-to-end on a `HookedTransformer`. |
| **Lazy TL import** | `import circuitry` does NOT import `transformer_lens`. Layering test. |
| **Gradient discipline** | After a patching run with activation grads enabled: params have `requires_grad=False` and `.grad is None`; activation grads are available at sites. |
| **Frozen-model invariant** | Model parameters, buffers, and batch-norm running stats are identical before and after a patching run. |
| **Prompt-pair runner** | Denoising direction produces expected metric deltas on the toy model. |
| **Custom metric** | Runner accepts a user-supplied `Callable[[Tensor], float]`. |
| **Unsupported architecture** | HF fallback on a non-Llama model for `mlp_neuron` raises a clean, descriptive error. |

---

## 8. Out of scope (this sub-spec)

- EAP, AtP\*, ACDC, SAE-feature circuits (follow-on sub-specs).
- The edge graph / residual-stream decomposition (derived per-method).
- Multi-process / DDP / FSDP (design §11 additive path, unchanged).
- Performance budget: §10's ≤10% training-overhead target does not apply
  to the patching workflow (post-hoc, not live-training). Per-site forward
  cost is the dominant cost of activation patching — attribution methods
  exist precisely to avoid O(sites) forward passes.
- Report/compare integration for patching results (likely part of the
  ergonomics pillar — feedback #6).

---

## 9. Public API sketch

```python
from circuitry.patching import Site, patch_site, PatchRunner
from circuitry.core.patching import logit_diff, kl_divergence, ce_loss

# Define a site
site = Site(component="attn_head_out", layer=5, head=3)

# Low-level: single intervention
with patch_site(model, site, value=cached_activation, resolver=resolver):
    output = model(**inputs)

# High-level: prompt-pair runner
runner = PatchRunner(model, recipe=get_recipe("llm"))
result = runner.run_patching(
    clean_inputs=clean_ids,
    corrupted_inputs=corrupted_ids,
    sites=[site],
    metric=logit_diff,
    direction="denoise",
)
print(result.metric_values)  # {site: delta}
```

This API is a sketch to settle shape and naming. Exact signatures are
refined in the implementation plan.
