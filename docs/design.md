# circuitry — design spec

**Date:** 2026-05-20
**Status:** draft for implementation
**Owner:** Vishwanath Sangale

`circuitry` is a new standalone Python library that extracts and unifies the mechanistic-interpretability and training-time diagnostic code currently scattered across `mendu` (and not-yet-extracted siblings) into a single reusable package suitable for use across LLM, vision, and recommender projects.

This document is the design contract. A separate implementation plan will follow.

## 1. Motivation

Across mendu we have built repeatedly used mech-interp / diagnostic code:

- A live recorder + scan-checkpoints + markdown-report pipeline (`mendu/tools/inspect_checkpoint/`, also partially extracted to `latent-superpowers-inspect/core/inspect-checkpoint/`).
- Spectral & rank diagnostics that were load-bearing in closing Bet 1 of paper 2 (`mendu/paper2/bet2_daleian/analysis/spectral_diagnostics.py`, `spectral_at_depth.py`).
- Paper-1-era vision diagnostics for CNNs and ViTs (`mendu/scripts/diagnose_{ei_bottlenecks,signal_prop,trained_pc}.py`).

These have been copy-pasted between projects and partially extracted into a shared "skills" repo (`latent-superpowers`) where they sit alongside non-library content. The result is that sibling projects in the workspace — `rl-recsys`, `bumblebee`, `plum`, `bonsai-{llm,vla}`, `gpt-2`, `llm-council` — cannot easily reuse them.

A 2026 survey of the field (TransformerLens, nnsight, captum, pyvene, SAELens, tuned-lens, pyhessian, NetDissect) shows the existing ecosystem covers post-hoc analysis well but does **not** unify (a) live training-time hooks, (b) spectral / rank / weight diagnostics, and (c) recsys + vision + LLMs under one API. That is the niche `circuitry` targets.

### Naming clarity

`circuitry` is statistical diagnostics on weights / activations / gradients during training. It is **not** mechanistic-circuit discovery in the activation-patching / SAE / logit-lens sense. The name is borrowed from electronics, not from interpretability research. The README MUST open with a one-line disambiguation so users arriving from mechanistic-interpretability work are not surprised.

### Non-goals

- We do **not** rebuild SAE training (SAELens does this well).
- We do **not** rebuild causal interventions / activation patching (pyvene, nnsight cover it).
- We do **not** attempt to be a complete post-hoc interp framework. We focus on monitoring + diagnostics.

## 2. Decisions locked in brainstorming

| Decision | Choice |
| --- | --- |
| Library name | `circuitry` |
| License | MIT |
| Release target | Public open-source, low-key (clean README, no docs site, not on PyPI for v1) |
| Repo location | `~/workspace/circuitry/`, public GitHub `vishsangale/circuitry` |
| Relationship to existing extraction | New standalone repo; absorb code from `latent-superpowers-inspect/core/inspect-checkpoint/`; archive that worktree |
| In-scope extractions | Checkpoint inspector (live + scan + report); spectral / rank diagnostics from paper 2; paper-1 vision diagnose scripts |
| Out-of-scope extractions | Daleian E/I-balance diagnostics (stay in `paper2/bet2_daleian`) |
| API shape | Two layers: pure primitives in `core/` + thin opinionated `Recorder` workflow above |
| Modality strategy | Modality-agnostic core + per-modality recipes (`recipes/llm.py`, `recipes/vision.py`, `recipes/two_tower.py`) |
| Framework support v1 | PyTorch only, single-process (rank-0 only in DDP runs; v2 path in §11) |
| Logging | TensorBoard primary, `MetricWriter` Protocol so wandb / jsonl / null adapters are 1-file each |

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
│   │   └── lens.py         # v1 stretch
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
│   │   ├── wandb.py        # optional extra
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

- `core/` MUST NOT import from `recorder/`, `recipes/`, `writers/`, or `cli/`.
- `recipes/` MUST NOT import from `cli/`.
- The package MUST NOT import from `mendu`, `rl-recsys`, or any sibling workspace project.

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

# activation-space
activation.dead_fraction(x: Tensor, threshold: float = 0.0) -> float
activation.kurtosis(x: Tensor, dim: int | tuple = -1) -> Tensor
activation.participation_ratio(x: Tensor) -> float
activation.norm_stats(x: Tensor) -> NormStats   # mean, std, max, frac>k*median

# gradient-space
gradient.layer_norm(grads: dict[str, Tensor]) -> dict[str, float]
gradient.signal_propagation_depth(grads_by_depth: list[Tensor]) -> int

# spectral
spectral.esd(W: Tensor, bins: int = 100) -> tuple[Tensor, Tensor]
spectral.rank_trajectory(state_dicts: list[dict]) -> dict[str, list[float]]
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
    writer="tensorboard",          # or "wandb", "jsonl", "null"
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
```

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

The TensorBoard adapter (default) is a thin wrapper over `torch.utils.tensorboard.SummaryWriter`. The JSONL adapter writes one JSON line per `add_scalar` call and dumps tensors / images to side files under `<run_dir>/circuitry/artifacts/` (no extra deps). The wandb adapter is gated behind `extras_require = {"wandb": ["wandb"]}`. The null adapter is a no-op for tests.

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
        HookPoint(pattern=r"embed.*",     source=TensorSource.WEIGHT),
        HookPoint(pattern=r"lm_head$",    source=TensorSource.WEIGHT),
    ],
    weight_diagnostics=["effective_rank", "stable_rank", "heavy_tail_alpha"],
    activation_diagnostics=["dead_fraction", "kurtosis", "participation_ratio"],
    gradient_diagnostics=["layer_norm"],
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

## 7. Migration plan — bringing mendu over

Three phases; no flag day required.

### Phase M1 — extract & publish (~week 1)

Create `~/workspace/circuitry/`. Seed:

- `mendu/tools/inspect_checkpoint/{live,arch_hooks,__main__}.py` → `src/circuitry/recorder/`.
- `mendu/paper2/bet2_daleian/analysis/spectral_diagnostics.py` + `spectral_at_depth.py` → `src/circuitry/core/{spectral,weight}.py`. Strip paper-2-specific bits.
- `mendu/scripts/diagnose_{ei_bottlenecks,signal_prop,trained_pc}.py` → ported into `recipes/vision.py` + `tests/recipes/test_vision.py`.

Tag `v0.1.0`.

### Phase M2 — mendu cutover (~week 1-2)

- `venv/bin/pip install -e ~/workspace/circuitry`.
- Replace `from tools.inspect_checkpoint...` with `from circuitry...` across mendu.
- Side-by-side sanity check on one canonical run: same TB scalars within `rtol=1e-5, atol=1e-7` for most metrics; `rtol=1e-4` for SVD-derived ones (`effective_rank`, `condition_number`, `heavy_tail_alpha`, `singular_values`) since these are sensitive to LAPACK backend and hardware. Tolerances are parameterized in the parity-check script; ratchet down if a metric is stricter than expected.
- After parity check passes: delete `mendu/tools/inspect_checkpoint/` and the now-redundant copies in `paper2/.../analysis/`.
- Update `mendu/CLAUDE.md`: replace the `latent-superpowers/core/inspect-checkpoint` install line with `pip install -e ~/workspace/circuitry`.
- Archive `~/workspace/latent-superpowers-inspect`: tag `pre-circuitry`, set README to "moved to circuitry."

### Phase M3 — siblings adopt (opportunistic, no timeline)

When `rl-recsys` / `bumblebee` / `plum` / `bonsai-*` / `gpt-2` / `llm-council` next touch their training loops, they pick up `circuitry` and the relevant recipe. `circuitry` itself must never depend on any of these projects — reverse-dependency rule enforced by CI.

## 8. Explicitly NOT in v1

- SAE training (interop with SAELens later if demand surfaces).
- Causal interventions / activation patching (pyvene, nnsight already cover this).
- JAX / Flax support.
- DDP / FSDP-aware reductions — v1 is single-process; non-zero ranks no-op. See §11 for the additive v2 path.
- Logit lens / tuned lens beyond a stretch in `core/lens.py`.
- Web dashboard. TB + markdown report is the UI.
- Differentiability guarantees through diagnostics. Primitives may use non-differentiable ops (`torch.linalg.svd`).

## 9. Risks & mitigations

| Risk | Mitigation |
| --- | --- |
| Extracted primitives change numerics vs in-tree versions, silently breaking paper-2 closeout reproduction | Side-by-side parity check in Phase M2 before any deletion; tolerances in §7 Phase M2; parity script kept in the repo as a regression guard. |
| Recipes accumulate modality-specific cruft and leak back into `core/` | CI import-linter rule: `core/` cannot import `recipes/`, `recorder/`, or `writers/`. Periodic code review of `core/`. |
| Recipe regexes match the **wrong** subset of modules silently (worse than matching nothing) | At `attach()` time the full matched-modules list per `HookPoint` is logged at INFO level and written to `<run_dir>/circuitry/matched_modules.txt`. Recipes can declare `expected_min_matches` per pattern; `strict=True` (default) raises on mismatch. Zero matches always raises. |
| Diagnostic overhead doubles wall-clock training time | §10 sets a hard ≤10% wall-clock budget at default settings; benchmark in CI; per-diagnostic `enabled: bool` so users can drop the expensive ones; `every_n_steps` knob defaults are tuned per recipe (see §10). |
| Public release attracts issues we don't have time for | "Low-key" release; README explicitly says "research code, no support promise." Issues triaged when convenient. |
| TB-primary design alienates wandb-first users | `MetricWriter` protocol from day 1; wandb adapter shipped in v0.1.0 even if jsonl ships first. |
| Single-process-only v1 ages into an architectural dead-end as users hit multi-GPU training | Multi-process design constraints baked into v1 protocol (see §11); v2 FSDP support is additive, not a rewrite. |

## 10. Performance & overhead budget

The most likely 6-month failure mode is "this is cool, but it doubled my training time." The design defends against this with explicit constraints:

- **Wall-clock budget:** at default settings (`every_n_steps=200`, full recipe), `circuitry`'s overhead MUST be ≤10% of baseline training step time on a 50M-param transformer and ≤5% on a 350M-param transformer. This is benchmarked in CI on a fixed reference workload; regressions block merge.
- **Per-diagnostic toggle:** every entry in `weight_diagnostics` / `activation_diagnostics` / `gradient_diagnostics` can be disabled via recipe override. The expensive ones (`heavy_tail_alpha`, `singular_values` on large weights) are documented as such.
- **Subsampling knobs:** weight-space diagnostics support `max_dim` to truncate SVD to top-k singular values, and `sample_axis` to compute on a random column subset. Default `max_dim=512` keeps SVD cost bounded on wide LLM matrices.
- **Lazy hooks:** activation hooks only run the forward pass capture on the emit step (every N steps). The hook checks `self._should_capture()` and is a no-op otherwise, avoiding per-step allocation cost.
- **Async writer option:** `MetricWriter` adapters MAY implement non-blocking writes (a background thread draining a queue). The TB adapter does this by default; tests use the synchronous null writer.

Reference benchmark workload (also v1 deliverable): a 50M-param decoder-only transformer on synthetic data, 100 steps, with and without `circuitry` attached, full LLM recipe, `every_n_steps=200`. Numbers go in the README.

## 11. Multi-process (DDP / FSDP) design notes

v1 is single-process. This section locks in *what v1 does today* so the v2 FSDP upgrade is additive, not a rewrite.

### v1 contract (single process, rank-0 semantics)

- `Recorder.attach()` checks `torch.distributed.is_initialized()`. If True and `rank != 0`, the recorder becomes a no-op (`attach()` returns immediately, all hooks are skipped). This means existing multi-rank training scripts can import `circuitry` without crashing and without duplicate writes; they just don't get diagnostics until v2.
- Primitives in `core/` assume **full, unsharded** tensors. They do not gather. They will silently return wrong numbers if given an FSDP-sharded parameter. The docstring and a runtime assertion (`shape sanity check against module's intended shape`) flag this.
- Writers write to the rank-0 process's filesystem; no rank coordination.

### v2 path (additive, no rewrite)

To enable multi-process diagnostics in v2 without changing the v1 API surface:

- `HookPoint` already takes a `source` enum; v2 adds `TensorSource.WEIGHT_FULL` and `ACTIVATION_FULL` variants that trigger an `all_gather_into_tensor` before passing to the primitive. The pattern / modules / selector escape hatches are unchanged.
- `core/` primitives stay single-tensor in / single-float out. v2 adds a small `core/distributed.py` with helpers (`all_gather_sharded_param(param) -> Tensor`) that the recorder calls before the primitive; primitives themselves never know about ranks.
- `MetricWriter` v2 adds an optional `rank: int` constructor argument; the default tensorboard adapter writes from rank 0 only (current behavior). A new `DDPMetricWriter` aggregates histogram tensors across ranks before writing.
- The `StepContext.gradients` / `activations` / `weights` dicts gain a "gathered" status flag; built-in diagnostics ignore it (they only see post-gather tensors), but custom diagnostics that want raw shards can opt in.

This means in v2: same recipes, same primitives, same `Recorder` constructor signature. Only the `source` enum gains values, `MetricWriter` gains an optional kwarg, and one new file (`core/distributed.py`) appears. No existing user code breaks.

### v1 README MUST state

> "v0.x supports single-process training only. In a multi-rank DDP/FSDP run, `circuitry` no-ops on non-zero ranks; FSDP-sharded parameters will produce incorrect diagnostics on rank 0. Multi-process support lands in v0.next; see §11 of the design spec for the upgrade path."

## 12. Open questions

None blocking implementation. Resolved during brainstorming + Gemini Pro review: name, license, release target, layering, modality strategy, framework support, logging strategy, scope of extractions, hook escape hatches, custom-diagnostic API, multi-process v2 path.
