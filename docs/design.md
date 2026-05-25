# circuitry — design spec

**Last updated:** 2026-05-21
**Status:** as-implemented (living document; tracks shipped releases — see [`CHANGELOG.md`](../CHANGELOG.md))
**Owner:** Vishwanath Sangale

`circuitry` is a standalone Python library providing mechanistic-interpretability diagnostics — weight / activation / gradient / spectral primitives, plus a `Recorder` workflow for live training-time capture and a `scan` workflow for post-hoc analysis on saved checkpoints. Modality-agnostic core with per-modality recipes for LLM, vision, and recsys.

This document is the design contract.

## 1. Motivation

A 2026 survey of the field (TransformerLens, nnsight, captum, pyvene, SAELens, tuned-lens, pyhessian, NetDissect) shows the existing ecosystem covers post-hoc analysis well but does **not** unify (a) live training-time hooks, (b) spectral / rank / weight diagnostics, and (c) recsys + vision + LLMs under one API. That is the niche `circuitry` targets.

The library bundles primitives that get re-implemented project-by-project (effective-rank, stable-rank, heavy-tail alpha, ESD, dead-fraction, kurtosis, participation ratio, gradient norms per param, layer-wise signal propagation) behind a single `Recorder` so that adding diagnostics to a new training run is a three-line change, not a refactor.

### Naming clarity

`circuitry` is statistical diagnostics on weights / activations / gradients, usable live during training or post-hoc on saved checkpoints. Lens-style and attribution primitives (logit lens, activation patching, SAE probes) are on the future-work list — today's surface is statistical and modality-agnostic. The name is borrowed from electronics, not from interpretability research. The README MUST open with a one-line scope statement so users arriving from mechanistic-interpretability work understand what this is and where it's heading.

### Non-goals

- We do **not** rebuild SAE training (SAELens does this well).
- We do **not** rebuild causal interventions / activation patching (pyvene, nnsight cover it).
- We do **not** attempt to be a complete post-hoc interp framework. We focus on monitoring + diagnostics.

## 2. Decisions locked in brainstorming

| Decision | Choice |
| --- | --- |
| Library name | `circuitry` |
| License | MIT |
| Release target | Public open-source, low-key (clean README, no docs site, not on PyPI for the 0.x line) |
| Repo location | `~/workspace/circuitry/`, public GitHub `vishsangale/circuitry` |
| In-scope | Checkpoint inspector (live + scan + report); spectral / rank / weight / activation / gradient primitives; per-modality recipes (LLM / vision / two-tower) |
| Out-of-scope | Architecture-specific diagnostics (those live in consumer codebases via custom `Recipe`s) |
| API shape | Two layers: pure primitives in `core/` + thin opinionated `Recorder` workflow above |
| Modality strategy | Modality-agnostic core + per-modality recipes (`recipes/llm.py`, `recipes/vision.py`, `recipes/two_tower.py`) |
| Framework support | PyTorch only, single-process (rank-0 only in DDP runs; multi-process path in §11) |
| Logging | TensorBoard primary, `MetricWriter` Protocol so jsonl / null (and any user-supplied) adapters are 1-file each |

## 3. Repository structure

```
~/workspace/circuitry/
├── pyproject.toml          # PEP 621
├── README.md
├── LICENSE                 # MIT
├── CHANGELOG.md
├── src/circuitry/
│   ├── __init__.py
│   ├── core/               # pure primitives — no torch.nn assumptions, no I/O
│   │   ├── weight.py
│   │   ├── activation.py
│   │   ├── gradient.py
│   │   ├── spectral.py
│   │   ├── lens.py         # logit_lens_kl
│   │   └── attention.py    # induction_score, attention_pattern_entropy
│   ├── sae/                # v0.9: SAELens-backed SAE workflow
│   │   ├── loader.py       # load_sae
│   │   └── metrics.py      # sae_reconstruction_error
│   ├── patching/           # v1.0: activation patching (interventional)
│   │   ├── sites.py        # Site dataclass + HF/TL resolution
│   │   ├── intervene.py    # patch_site() context manager
│   │   └── runner.py       # PatchRunner prompt-pair runner
│   ├── recorder/           # opinionated training-time workflow
│   │   ├── live.py         # LiveRecorder
│   │   ├── scan.py         # scan_run
│   │   ├── report.py       # build_report
│   │   └── hooks.py        # HookPoint, TensorSource, module-type → hook strategy
│   ├── recipes/
│   │   ├── llm.py
│   │   ├── vision.py
│   │   └── two_tower.py
│   ├── writers/
│   │   ├── base.py         # MetricWriter Protocol
│   │   ├── tensorboard.py
│   │   ├── jsonl.py
│   │   └── null.py
│   └── cli/
│       ├── __init__.py
│       └── main.py         # circuitry scan / report / list-recipes
├── tests/
│   ├── core/
│   ├── recorder/
│   ├── recipes/
│   └── e2e/
├── examples/
│   ├── tiny_llm.py
│   ├── tiny_vision.py
│   └── tiny_two_tower.py
└── docs/                   # markdown notes only
```

### Layering rules (enforced in CI)

- `core/` MUST NOT import from `recorder/`, `recipes/`, `writers/`, `cli/`, or `patching/`.
- `recipes/` MUST NOT import from `cli/`.
- `patching/` may import from `core/` and `recipes/`; MUST NOT import from `cli/`.
- The package MUST NOT import from any downstream user codebase. `circuitry` is the consumed dependency, never the consumer.
- `transformer_lens` is an approved optional dependency (lazy import only; `circuitry` must install and run without it).

A simple `import-linter` config or hand-rolled AST test enforces this.

## 4. Public API

### 4.1 Tier 1 — primitives (`circuitry.core.*`)

Pure functions. Tensors / state-dicts in; floats / arrays / small dataclasses out. No hooks, no logging, no side effects.

```python
from circuitry.core import weight, activation, gradient, spectral

# weight-space
weight.effective_rank(W: Tensor, eps: float = 1e-12) -> float
weight.stable_rank(W: Tensor) -> float
weight.condition_number(W: Tensor) -> float
weight.singular_values(W: Tensor, k: int | None = None) -> Tensor
weight.heavy_tail_alpha(W: Tensor) -> float
weight.attention_head_rank(W: Tensor, n_heads: int, head_dim: int, axis: int = 0) -> Tensor

# activation-space
activation.dead_fraction(x: Tensor, threshold: float = 0.0) -> float
activation.kurtosis(x: Tensor, dim: int | tuple = -1) -> Tensor
activation.participation_ratio(x: Tensor) -> float
activation.norm_stats(x: Tensor) -> NormStats   # mean, std, max, frac>k*median
activation.gate_stats(x: Tensor, eps: float = 1e-6) -> GateStats  # frac_active, mean_abs, std

# gradient-space
gradient.grad_norm_per_module(grads: dict[str, Tensor]) -> dict[str, float]
gradient.total_grad_norm(per_module_norms: dict[str, float]) -> float  # sqrt(sum of squares)
gradient.signal_propagation_depth(grads_by_depth: list[Tensor]) -> int

# spectral
spectral.esd(W: Tensor, bins: int = 100) -> tuple[Tensor, Tensor]
spectral.rank_trajectory(state_dicts: list[dict]) -> dict[str, list[float]]

# lens (v0.9)
from circuitry.core import lens
lens.logit_lens_kl(residual: Tensor, unembed: Tensor, final_logits: Tensor, *, layer_norm=None, chunk_size: int = 256) -> float  # chunk_size bounds the (tokens, vocab) transient (v0.9.2)

# attention screening (v0.9)
from circuitry.core import attention
attention.induction_score(attn_pattern: Tensor, *, seq_len_repeat: int) -> list[float]
attention.attention_pattern_entropy(attn_pattern: Tensor) -> list[float]  # normalizes each query row before entropy → comparable across attention variants (v0.9.2)

# SAE workflow (v0.9)
from circuitry import sae
sae.load_sae(release: str, sae_id: str, device: str = "cpu")
sae.sae_reconstruction_error(x: Tensor, sae) -> dict[str, float]

# patching metrics (v1.0)
from circuitry.core import patching
patching.logit_diff(logits: Tensor, correct: int, incorrect: int) -> float
patching.kl_divergence(p_logits: Tensor, q_logits: Tensor, *, chunk_size: int = 256) -> float
patching.ce_loss(logits: Tensor, targets: Tensor) -> float
```

Invariants for everything in `core/`:

- Deterministic on CPU; no implicit `.cuda()`.
- Accept `torch.Tensor` or `numpy.ndarray` where it makes sense.
- All scalar numeric returns are plain Python `float`, not 0-dim tensors.

### 4.2 Tier 2 — recorder

```python
from circuitry import Recorder, scan_run, build_report

recorder = Recorder(
    model,
    run_dir="runs/my_run",
    recipe="llm",                  # str name or Recipe instance
    writer="tensorboard",          # or "jsonl", "null"
    every_n_steps=200,
)
recorder.attach()
for step, batch in enumerate(loader):
    loss = train_step(model, batch)
    recorder.step(step, loss=loss)
recorder.detach()

scan_run(
    run_dir="runs/my_run",
    recipe="llm",
    out_dir="runs/my_run/tb_retro",
)

build_report(
    run_dir="runs/my_run",
    out_path="runs/my_run/inspect/report.md",
)
```

### 4.3 CLI

```bash
circuitry scan   --run runs/my_run --recipe llm
circuitry report --run runs/my_run
circuitry list-recipes
```

`report` accepts either a live `metrics.jsonl` (written by the Recorder, no `scan` step) or a retrospective `findings.json` produced by `scan`.

### 4.4 `Recipe` and hook escape hatches

```python
@dataclass
class Recipe:
    name: str
    hook_points: list[HookPoint]
    weight_diagnostics: list[str]
    activation_diagnostics: list[str]
    gradient_diagnostics: list[str]
    custom: list[DiagnosticFn] = field(default_factory=list)
    expected_min_matches: dict[str, int] = field(default_factory=dict)  # pattern → min modules
    module_prefix: str | None = None  # if set, only modules under this dotted prefix are matched
```

Use `Recipe.with_prefix(prefix)` to scope a recipe to a sub-tree of the model (e.g. `get_recipe("llm").with_prefix("model.language_model")` for multimodal HF models). Returns a new `Recipe` via `dataclasses.replace`; the original is not mutated. Latest-wins: calling `.with_prefix("a").with_prefix("b")` yields `module_prefix="b"`. If `expected_min_matches` is set, lower the thresholds after scoping — whole-model counts don't hold after a prefix filter.

Use `Recipe.with_sae(mapping)` (v0.9) to attach SAE checkpoints: `mapping` is a `dict[str, tuple[str, str]]` of `module_name → (release, sae_id)`. Returns a new `Recipe` with `sae_checkpoints` populated. SAE checkpoints are loaded lazily at `attach()` time; the user must also add `"sae_reconstruction"` to `activation_diagnostics` to incur per-step encode+decode cost.

`HookPoint` supports three target specifications:

```python
@dataclass
class HookPoint:
    source: TensorSource                          # WEIGHT, INPUT, OUTPUT, or GRAD
    pattern: str | None = None                    # regex against named_modules() (recipe default)
    modules: list[nn.Module] | None = None        # explicit instances (advanced)
    selector: Callable[[nn.Module], list[str]] | None = None  # programmatic name selector
    # exactly one of {pattern, modules, selector} must be set
```

This gives three matching modes:

1. **Pattern (default)** — used by all stock recipes. Regex against `dict(model.named_modules()).keys()`.
2. **Explicit modules** — pass `nn.Module` instances directly. For Mamba / MoE / custom architectures where regex is fragile.
3. **Programmatic selector** — a function that walks `model` and returns module names. Use when the right hook set depends on runtime structure (e.g. only experts that fired this step).

Brittleness mitigation (addressing recipe-matched-wrong-subset failure mode):

- At `attach()` time, the recorder logs the full list of matched module names per `HookPoint` at INFO level, and writes it to `<run_dir>/circuitry/matched_modules.txt` so it is visible in artifacts.
- If a `HookPoint`'s `expected_min_matches[pattern]` is set and the actual match count is below it, `attach()` raises by default. Pass `strict=False` to `Recorder.__init__` to downgrade this to a warning.
- A `HookPoint` that matches **zero** modules always raises, regardless of `strict`. There is no legitimate use of a hook that hooks nothing.

Users can register custom recipes via `circuitry.register_recipe(my_recipe)`.

#### `DiagnosticFn` signature and step context

Custom diagnostics that need both forward activations and backward gradients on the same step (Fisher information, gradient-activation alignment, etc.) receive a `StepContext`:

```python
@dataclass
class StepContext:
    step: int
    model: nn.Module
    activations: dict[str, Tensor]    # hooked module name → forward output (if captured)
    gradients:   dict[str, Tensor]    # hooked module name → .grad (post-backward, pre-step)
    weights:     dict[str, Tensor]    # hooked module name → parameter (current value)
    loss:        float | None
    user:        dict[str, Any]       # opaque pass-through from Recorder.step(**kwargs)

DiagnosticFn = Callable[[StepContext], dict[str, float | Tensor]]
# returned dict: tag (str, no leading "/") → value. Recorder prefixes "custom/" and writes.
```

The recorder builds the `StepContext` once per emit step (every `every_n_steps`), runs all listed `custom` callables in order, and writes their outputs through the configured `MetricWriter`. Built-in diagnostics in `weight_diagnostics` / `activation_diagnostics` / `gradient_diagnostics` are implemented internally against the same `StepContext` shape; `custom` is the public extension point.

### 4.5 `MetricWriter` protocol

```python
class MetricWriter(Protocol):
    def add_scalar(self, tag: str, value: float, step: int) -> None: ...
    def add_histogram(self, tag: str, values: Tensor, step: int) -> None: ...
    def add_image(self, tag: str, image: Tensor, step: int,
                  dataformats: str = "CHW") -> None: ...
    def add_text(self, tag: str, text: str, step: int) -> None: ...
    def flush(self) -> None: ...
    def close(self) -> None: ...
```

`add_image` is essential for vision recipes (activation maps, weight kernels visualized as heatmaps) and for matrix-as-image debug views even in LLM recipes (e.g. plotting `W_O @ W_V` per head). `dataformats` follows TB's convention.

The TensorBoard adapter (default) is a thin wrapper over `torch.utils.tensorboard.SummaryWriter`. The JSONL adapter writes one JSON line per `add_scalar` call and dumps tensors / images to side files under `<run_dir>/circuitry/artifacts/` (no extra deps); the `scan` / `report` workflow reads this format. The null adapter is a no-op for tests. Third-party loggers (wandb, mlflow, etc.) are not shipped in v0.3.0 — implement `MetricWriter` (~50 LOC) and pass the instance to `Recorder(writer=...)`.

### 4.6 Intervention mode (v1.0)

The `patching/` subsystem adds an opt-in **intervention mode** for causal analysis (activation patching, and the attribution methods built on it). It is the first capability in `circuitry` that *modifies* activations rather than only observing them. Contrasted with the observation-only `Recorder` and `scan` workflows, intervention mode upholds these invariants:

- **Opt-in.** Interventions require explicit use of the `circuitry.patching` API. `Recorder` and `scan` remain observation-only and are never affected by patching.
- **Isolated.** Every intervention is scoped to a context manager (`patch_site`). The forward hook is removed and model state is restored on exit, including on exception (`try/finally`, mutation-last — hooks installed as the final setup step so a partial setup can't leak).
- **Frozen model.** Parameter `requires_grad` is forced off for the duration and restored on exit; eval mode is set on entry and restored; no optimizer runs and no parameter values are modified.
- **Activation-grad-only.** The only gradient flow permitted is on activation tensors at intervention sites (for attribution methods such as AtP\* and EAP). Parameter gradients are never enabled.

Sites are resolved to concrete model locations by a resolver: `HFSiteResolver` (recipe/config-declared layout — per-head needs eager attention; per-neuron is Llama-family-first) or `TLSiteResolver` (TransformerLens hook names; lazy `transformer_lens` import). The metric helpers live in `core/patching.py` (pure functions); any `Callable[[Tensor], float]` is accepted as a custom metric.

```python
from circuitry.patching import Site, patch_site, PatchRunner
from circuitry.patching.sites import HFSiteResolver
from circuitry.core.patching import logit_diff

resolver = HFSiteResolver.from_config(model.config)
site = Site(component="attn_head_out", layer=5, head=3)

# Low-level: single intervention (restores on exit)
with patch_site(model, site, value=cached_act, resolver=resolver):
    output = model(**inputs)

# High-level: prompt-pair runner
runner = PatchRunner(model, resolver)
result = runner.run_patching(
    clean_inputs=clean_ids,
    corrupted_inputs=corrupted_ids,
    sites=[site],
    metric=lambda logits: logit_diff(logits, correct=tok_a, incorrect=tok_b),
    direction="denoise",
)
print(result.metric_values)  # {site: metric}
```

Attribution methods (EAP, AtP\*, ACDC) and SAE-feature circuits build on this primitive and ship in follow-on v1.0 sub-specs.

**EAP (sub-spec 2, shipped).** `EAPRunner(model, resolver).run(clean_inputs, corrupted_inputs, metric, ig_steps=1)` returns an `EAPResult` of per-edge attribution scores over the residual-stream graph (`circuitry.patching.graph`: `Node`/`Edge`/`build_graph`). Edges are writer→reader with q/k/v-typed attention reads; nodes are attention heads + MLPs + embed + logits. Scoring is the 2-forward + 1-backward linear approximation (`Δact · grad`, summed over `d_model`), with vanilla EAP (`ig_steps=1`) and activation-path EAP-IG (`ig_steps=N`). Backends: TransformerLens (native per-slot hooks) and HF (eager, Llama-family — per-head `z@W_O` writers, q/k/v reader gradients back-mapped to residual space with the RMSNorm scale as a stop-gradient constant, GQA-aware). The EAP metric must be **differentiable** — use the tensor-returning `circuitry.core.patching.logit_diff_t` / `kl_divergence_t` / `ce_loss_t`, not the `.detach()`-ing float versions.

**AtP\* (sub-spec 3, shipped).** `AtPRunner(model, resolver).run(clean_inputs, corrupted_inputs, metric, neurons=False, graddrop=False, qk_fix=True)` → `AtPResult` of per-node attribution scores (`circuitry.patching.atp`: `AtPNode`, `AtPResult`). Nodes are embed + per-head (q/k/v slots) + mlp per layer + optionally mlp_neuron per neuron. Scoring: `score(node) = Σ(Δact_node ⊙ grad_node)` summed over all dims; q/k nodes use the **QK fix** (attention-pattern recomputation in `d_model` space against `grad_attn_out`) when `qk_fix=True`; `graddrop=True` replaces summation with `Σ|per-position contribution|`. On a fully linear model vanilla AtP is exact (equals brute-force `patch_site` at 1e-4). The differentiable metric (`logit_diff_t` / `kl_divergence_t` / `ce_loss_t`) is required. Backends: HF (eager, Llama-family, GQA-aware — full QK fix implemented) and TransformerLens (native TL hooks — vanilla q/k scoring only; full QK fix with softmax-pattern recomputation is HF-path only). `AtPResult.verify_top_k(k, clean_inputs, corrupted_inputs, metric, resolver, runner)` calibrates the top-K nodes against real `patch_site` ground truth, returning `{node: (atp_score, true_patch_effect)}`.

## 5. Recipe internals — worked example

`recipes/llm.py`:

```python
from circuitry.recorder.hooks import HookPoint, TensorSource

RECIPE = Recipe(
    name="llm",
    hook_points=[
        HookPoint(pattern=r".*\.attn\.(q|k|v|o)_proj$", source=TensorSource.WEIGHT),
        HookPoint(pattern=r".*\.mlp\.(w1|w2|w3|gate_proj|up_proj|down_proj)$",
                  source=TensorSource.WEIGHT),
        HookPoint(pattern=r".*\.attn$",   source=TensorSource.OUTPUT),
        HookPoint(pattern=r".*\.mlp$",    source=TensorSource.OUTPUT),
        HookPoint(pattern=r".*\.ln_[12]$",source=TensorSource.OUTPUT),
        HookPoint(pattern=r".*\.mlp\.down_proj$", source=TensorSource.INPUT),
        HookPoint(pattern=r"embed.*",     source=TensorSource.WEIGHT),
        HookPoint(pattern=r"lm_head$",    source=TensorSource.WEIGHT),
        HookPoint(pattern=r".*\.attn\.(q|k|v|o)_proj$", source=TensorSource.GRAD),
    ],
    weight_diagnostics=["effective_rank", "attention_head_rank", "stable_rank", "heavy_tail_alpha"],
    activation_diagnostics=["gate_stats", "dead_fraction", "kurtosis", "participation_ratio"],
    gradient_diagnostics=["grad_norm_per_module"],
)
```

`recipes/vision.py` swaps the regex for `conv\d+|fc\d+|patch_embed|blocks\.\d+\.(attn|mlp)`. `recipes/two_tower.py` knows about `query_tower`, `item_tower`, `interaction`, and adds an embedding-alignment diagnostic (cosine of query / item tower output means).

The recorder walks `model.named_modules()`, matches against each recipe's patterns, registers the appropriate forward / backward hook (or pre-reads `WEIGHT` tensors directly), and at each emit step feeds the captured tensors through the listed primitives and writes the resulting scalars / histograms through the configured `MetricWriter`.

## 6. Testing strategy

Four layers, sized to where bugs actually live:

1. **`tests/core/` — property tests on primitives.** Known answers on synthetic matrices (identity → effective_rank = n; rank-1 outer product → 1; orthogonal → cond = 1). Invariance under orthogonal transforms. ~30-50 tests; <10s on CPU.
2. **`tests/recorder/` — hook & writer smoke tests.** Tiny 2-layer MLP; `RecordingWriter` captures every `add_scalar`; assert tags / steps / no hook leaks after `detach()`. ~10 tests; <5s.
3. **`tests/recipes/` — modality fixtures.** Three minimal fixtures: 1M-param toy transformer, 100k-param ResNet block, 50k-param two-tower model. Attach recorder, 3 steps, assert recipe-specific scalars appear. ~5 tests / recipe.
4. **`tests/e2e/` — full pipeline.** Train tiny model 20 steps with `LiveRecorder` → `scan_run` over its 2 checkpoints → `build_report` → assert markdown contains expected sections. <30s.

CI: GitHub Actions, Python 3.10 / 3.11 / 3.12, PyTorch latest stable. No GPU jobs (everything CPU-sized). Performance benchmark (§10) is a separate CI job using `pytest-benchmark`; regressions >15% over baseline block merge.

## 7. Release history

See [`CHANGELOG.md`](../CHANGELOG.md) for the full version log. Public releases are tagged and announced via [GitHub Releases](https://github.com/vishsangale/circuitry/releases).

## 8. Explicitly out of scope today

- SAE training (interop with SAELens later if demand surfaces).
- Causal interventions / activation patching (pyvene, nnsight already cover this).
- JAX / Flax support.
- DDP / FSDP-aware reductions — current releases are single-process; non-zero ranks no-op. See §11 for the additive future-release path.
- Logit lens / tuned lens beyond a stretch in `core/lens.py`.
- Web dashboard. TB + markdown report is the UI.
- Differentiability guarantees through diagnostics. Primitives may use non-differentiable ops (`torch.linalg.svd`).

## 9. Risks & mitigations

| Risk | Mitigation |
| --- | --- |
| Recipes accumulate modality-specific cruft and leak back into `core/` | CI import-linter rule: `core/` cannot import `recipes/`, `recorder/`, or `writers/`. Periodic code review of `core/`. |
| Recipe regexes match the **wrong** subset of modules silently (worse than matching nothing) | At `attach()` time the full matched-modules list per `HookPoint` is logged at INFO level and written to `<run_dir>/circuitry/matched_modules.txt`. Recipes can declare `expected_min_matches` per pattern; `strict=True` (default) raises on mismatch. Zero matches always raises. |
| Diagnostic overhead doubles wall-clock training time | §10 sets a hard ≤10% wall-clock budget at default settings; benchmark in CI; per-diagnostic `enabled: bool` so users can drop the expensive ones; `every_n_steps` knob defaults are tuned per recipe (see §10). |
| Public release attracts issues we don't have time for | "Low-key" release; README explicitly says "research code, no support promise." Issues triaged when convenient. |
| TB-primary design alienates wandb / mlflow-first users | `MetricWriter` protocol from day 1 keeps any third-party adapter a ~50-LOC subclass. v0.1.0 shipped a wandb adapter; v0.3.0 removed it as there were no active users — trivially re-addable if demand surfaces. |
| Single-process-only design ages into an architectural dead-end as users hit multi-GPU training | Multi-process design constraints baked into the current protocol (see §11); a future-release FSDP upgrade is additive, not a rewrite. |

## 10. Performance & overhead budget

The most likely 6-month failure mode is "this is cool, but it doubled my training time." The design defends against this with explicit constraints:

- **Wall-clock budget:** at default settings (`every_n_steps=200`, full recipe), `circuitry`'s overhead MUST be ≤10% of baseline training step time on a 50M-param transformer and ≤5% on a 350M-param transformer. This is benchmarked in CI on a fixed reference workload; regressions block merge.
- **Per-diagnostic toggle:** every entry in `weight_diagnostics` / `activation_diagnostics` / `gradient_diagnostics` can be disabled via recipe override. The expensive ones (`heavy_tail_alpha`, `singular_values` on large weights) are documented as such.
- **Subsampling knobs:** weight-space diagnostics support `max_dim` to truncate SVD to top-k singular values, and `sample_axis` to compute on a random column subset. Default `max_dim=512` keeps SVD cost bounded on wide LLM matrices.
- **Lazy hooks:** activation hooks only run the forward pass capture on the emit step (every N steps). The hook checks `self._should_capture()` and is a no-op otherwise, avoiding per-step allocation cost.
- **Async writer option:** `MetricWriter` adapters MAY implement non-blocking writes (a background thread draining a queue). The TB adapter does this by default; tests use the synchronous null writer.

Reference benchmark workload: a 50M-param decoder-only transformer on synthetic data, 100 steps, with and without `circuitry` attached, full LLM recipe, `every_n_steps=200`. Numbers go in the README.

## 11. Multi-process (DDP / FSDP) design notes

Current releases are single-process. This section locks in *what circuitry does today* so a future-release FSDP upgrade is additive, not a rewrite.

### Current contract (single process, rank-0 semantics)

- `Recorder.attach()` checks `torch.distributed.is_initialized()`. If True and `rank != 0`, the recorder becomes a no-op (`attach()` returns immediately, all hooks are skipped). This means existing multi-rank training scripts can import `circuitry` without crashing and without duplicate writes; they just don't get diagnostics until the multi-process path lands.
- Primitives in `core/` assume **full, unsharded** tensors. They do not gather. They will silently return wrong numbers if given an FSDP-sharded parameter. The docstring and a runtime assertion (`shape sanity check against module's intended shape`) flag this.
- Writers write to the rank-0 process's filesystem; no rank coordination.

### Future-release path (additive, no rewrite)

To enable multi-process diagnostics in a future release without changing the current API surface:

- `HookPoint` already takes a `source` enum; the future release adds `TensorSource.WEIGHT_FULL` and `ACTIVATION_FULL` variants that trigger an `all_gather_into_tensor` before passing to the primitive. The pattern / modules / selector escape hatches are unchanged.
- `core/` primitives stay single-tensor in / single-float out. The future release adds a small `core/distributed.py` with helpers (`all_gather_sharded_param(param) -> Tensor`) that the recorder calls before the primitive; primitives themselves never know about ranks.
- `MetricWriter` gains an optional `rank: int` constructor argument; the default tensorboard adapter writes from rank 0 only (current behavior). A new `DDPMetricWriter` aggregates histogram tensors across ranks before writing.
- The `StepContext.gradients` / `activations` / `weights` dicts gain a "gathered" status flag; built-in diagnostics ignore it (they only see post-gather tensors), but custom diagnostics that want raw shards can opt in.

Net result: same recipes, same primitives, same `Recorder` constructor signature. Only the `source` enum gains values, `MetricWriter` gains an optional kwarg, and one new file (`core/distributed.py`) appears. No existing user code breaks.

### README MUST state

> "v0.x supports single-process training only. In a multi-rank DDP/FSDP run, `circuitry` no-ops on non-zero ranks; FSDP-sharded parameters will produce incorrect diagnostics on rank 0. Multi-process support is planned for a future release; see §11 of the design spec for the upgrade path."

## 12. Open questions

None blocking implementation. Resolved during brainstorming + Gemini Pro review: name, license, release target, layering, modality strategy, framework support, logging strategy, hook escape hatches, custom-diagnostic API, multi-process upgrade path.
