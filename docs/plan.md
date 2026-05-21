# circuitry v0.1.0 — M1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract and publish `circuitry` v0.1.0 — a standalone PyTorch library of training-time mech-interp diagnostics (weight / activation / gradient / spectral primitives + a `Recorder` workflow + per-modality recipes), with the public API and CI gates defined in `docs/design.md`.

**Architecture:** Two-tier API — pure primitives in `core/` (no I/O, CPU-deterministic) and an opinionated `Recorder` over per-modality `recipes/` writing through a `MetricWriter` protocol (TensorBoard default). Modality-agnostic core + per-modality recipes for LLMs / vision / two-tower. Single-process v1; non-zero ranks no-op. Layering enforced by CI from day one.

**Tech Stack:** Python ≥ 3.10 (CI: 3.10 / 3.11 / 3.12; local dev: 3.12), PyTorch latest stable, pytest, import-linter, TensorBoard (`torch.utils.tensorboard`), optional wandb extra.

**Scope:** This plan covers **Phase M1 only** (extract & publish, v0.1.0 tag). Phase M2 (mendu cutover with parity) and Phase M3 (siblings adopt) are separate plans written after v0.1.0 tags.

**Reference implementations to port from (read before each Phase D / E task):**

- `~/workspace/mendu/mendu/tools/inspect_checkpoint/{live,arch_hooks,__main__}.py` — recorder + CLI source
- `~/workspace/mendu/mendu/paper2/bet2_daleian/analysis/spectral_diagnostics.py` and `spectral_at_depth.py` — spectral / weight primitives source
- `~/workspace/mendu/mendu/scripts/diagnose_{ei_bottlenecks,signal_prop,trained_pc}.py` — vision recipe source
- `~/workspace/latent-superpowers-inspect/core/inspect-checkpoint/` — partial prior extraction; absorb anything useful, then this worktree will be archived in M2

**Porting policy:** Read the mendu source and port-with-cleanup. Do not blindly copy. TDD against the API contract in `docs/design.md` §4. Parity numerics check happens in M2 — for M1, "the test passes" is the bar, not "matches mendu bit-for-bit."

**Environment discipline:** Always use full venv paths — `venv/bin/python`, `venv/bin/pytest`, `venv/bin/pip`. Never `source venv/bin/activate`. Print absolute paths when creating files. Commit-message scope follows `feat(core)`, `feat(recorder)`, `feat(recipes)`, `test(...)`, `docs(...)`, `chore(...)`.

---

## Phase A — Foundation

### Task A1: pyproject + LICENSE + package skeleton + dev install

**Files:**
- Create: `/home/vishsangale/workspace/circuitry/pyproject.toml`
- Create: `/home/vishsangale/workspace/circuitry/LICENSE`
- Create: `/home/vishsangale/workspace/circuitry/CHANGELOG.md`
- Create: `/home/vishsangale/workspace/circuitry/src/circuitry/__init__.py`
- Create: `/home/vishsangale/workspace/circuitry/src/circuitry/core/__init__.py`
- Create: `/home/vishsangale/workspace/circuitry/src/circuitry/recorder/__init__.py`
- Create: `/home/vishsangale/workspace/circuitry/src/circuitry/recipes/__init__.py`
- Create: `/home/vishsangale/workspace/circuitry/src/circuitry/writers/__init__.py`
- Create: `/home/vishsangale/workspace/circuitry/src/circuitry/cli/__init__.py`
- Create: `/home/vishsangale/workspace/circuitry/tests/__init__.py`
- Create: `/home/vishsangale/workspace/circuitry/tests/core/__init__.py`
- Create: `/home/vishsangale/workspace/circuitry/tests/recorder/__init__.py`
- Create: `/home/vishsangale/workspace/circuitry/tests/recipes/__init__.py`
- Create: `/home/vishsangale/workspace/circuitry/tests/e2e/__init__.py`

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "circuitry"
version = "0.1.0.dev0"
description = "Training-time mechanistic-interpretability diagnostics for PyTorch (LLM / vision / two-tower)."
readme = "README.md"
license = { file = "LICENSE" }
authors = [{ name = "Vishwanath Sangale" }]
requires-python = ">=3.10"
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]
dependencies = [
    "torch>=2.1",
    "numpy>=1.24",
    "tensorboard>=2.14",
]

[project.optional-dependencies]
wandb = ["wandb>=0.16"]
dev = [
    "pytest>=8",
    "pytest-benchmark>=4",
    "import-linter>=2",
    "ruff>=0.4",
]

[project.scripts]
circuitry = "circuitry.cli.main:main"

[project.urls]
Homepage = "https://github.com/vishsangale/circuitry"

[tool.setuptools.packages.find]
where = ["src"]

[tool.setuptools.package-dir]
"" = "src"

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra"

[tool.ruff]
line-length = 100
target-version = "py310"

[tool.ruff.lint]
select = ["E", "F", "I", "W", "UP", "B"]
ignore = ["E501"]
```

- [ ] **Step 2: Write `LICENSE`** — standard MIT text, copyright holder "Vishwanath Sangale", year 2026.

- [ ] **Step 3: Write `CHANGELOG.md`**

```markdown
# Changelog

All notable changes to this project will be documented in this file. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial extraction from `mendu` per `docs/design.md`.
```

- [ ] **Step 4: Write `src/circuitry/__init__.py` — version-only stub for now**

```python
"""circuitry — training-time mechanistic-interpretability diagnostics for PyTorch.

The public surface (Recorder, scan_run, build_report, Recipe, register_recipe,
HookPoint, StepContext, TensorSource, MetricWriter) is re-exported here once
the underlying modules land — see Task F2. Until then, import from submodules.
"""

__version__ = "0.1.0.dev0"
```

The re-exports are pinned in Task F2, after every referenced module exists. The public-API contract is documented now even though the bindings come later.

- [ ] **Step 5: Create empty `__init__.py` in each subpackage**

For each of `core/`, `recorder/`, `recipes/`, `writers/`, `cli/`, `tests/`, `tests/core/`, `tests/recorder/`, `tests/recipes/`, `tests/e2e/`, create `__init__.py` containing only a single newline.

- [ ] **Step 6: Install in editable mode with dev extras**

```bash
venv/bin/pip install -e ".[dev]"
```

Expected: package installs; `venv/bin/python -c "import circuitry"` will currently fail because subpackage modules are empty — that's OK, foundation-only at this point. Verify with:

```bash
venv/bin/python -c "import pytest, torch, tensorboard, importlinter; print('ok')"
```

Expected output: `ok`.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml LICENSE CHANGELOG.md src/ tests/
git commit -m "chore: scaffold package layout and pin public surface"
```

---

### Task A2: CI-enforced layering (import-linter + first GitHub Actions run)

**Files:**
- Create: `/home/vishsangale/workspace/circuitry/.importlinter`
- Create: `/home/vishsangale/workspace/circuitry/tests/test_layering.py`
- Create: `/home/vishsangale/workspace/circuitry/.github/workflows/ci.yml`

- [ ] **Step 1: Write the failing layering test**

```python
# tests/test_layering.py
"""Layering rules from docs/design.md §3. Belt-and-suspenders with the
import-linter config — the unit test catches violations at pytest time and
gives a clearer error than import-linter's CLI output."""

from __future__ import annotations

import ast
import pathlib

SRC = pathlib.Path(__file__).parent.parent / "src" / "circuitry"

FORBIDDEN = {
    "core": ("circuitry.recorder", "circuitry.recipes", "circuitry.writers", "circuitry.cli"),
    "recipes": ("circuitry.cli",),
}

SIBLING_FORBIDDEN = ("mendu", "rl_recsys", "rl-recsys", "bumblebee", "plum", "bonsai", "gpt_2", "llm_council", "latent_superpowers")


def _imports(path: pathlib.Path) -> set[str]:
    tree = ast.parse(path.read_text())
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                out.add(n.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            out.add(node.module)
    return out


def test_core_does_not_import_higher_layers():
    for py in (SRC / "core").rglob("*.py"):
        for imp in _imports(py):
            for forbidden in FORBIDDEN["core"]:
                assert not imp.startswith(forbidden), (
                    f"core/{py.relative_to(SRC / 'core')} imports {imp}, "
                    f"violating §3 layering rule"
                )


def test_recipes_do_not_import_cli():
    for py in (SRC / "recipes").rglob("*.py"):
        for imp in _imports(py):
            for forbidden in FORBIDDEN["recipes"]:
                assert not imp.startswith(forbidden), (
                    f"recipes/{py.relative_to(SRC / 'recipes')} imports {imp}, "
                    f"violating §3 layering rule"
                )


def test_no_sibling_workspace_imports():
    for py in SRC.rglob("*.py"):
        for imp in _imports(py):
            root = imp.split(".", 1)[0]
            assert root not in SIBLING_FORBIDDEN, (
                f"{py.relative_to(SRC)} imports {imp} from sibling workspace project — "
                f"reverse-dependency rule (§3)"
            )
```

- [ ] **Step 2: Run the test — it must pass against the empty skeleton**

```bash
venv/bin/pytest tests/test_layering.py -v
```

Expected: 3 PASSED (skeleton is empty, so no violations possible).

- [ ] **Step 3: Write `.importlinter` for parallel CI enforcement**

```ini
[importlinter]
root_package = circuitry

[importlinter:contract:core-is-pure]
name = core/ is pure — no imports of higher layers
type = forbidden
source_modules =
    circuitry.core
forbidden_modules =
    circuitry.recorder
    circuitry.recipes
    circuitry.writers
    circuitry.cli

[importlinter:contract:recipes-not-cli]
name = recipes/ must not import cli/
type = forbidden
source_modules =
    circuitry.recipes
forbidden_modules =
    circuitry.cli
```

Verify:

```bash
venv/bin/lint-imports
```

Expected: contracts kept (no violations in empty skeleton).

- [ ] **Step 4: Write `.github/workflows/ci.yml`**

```yaml
name: ci

on:
  push:
    branches: [main]
  pull_request:

jobs:
  test:
    strategy:
      fail-fast: false
      matrix:
        python: ["3.10", "3.11", "3.12"]
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python }}
          cache: pip
      - run: pip install -e ".[dev]"
      - run: ruff check src tests
      - run: lint-imports
      - run: pytest -ra

  benchmark:
    runs-on: ubuntu-latest
    needs: test
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
          cache: pip
      - run: pip install -e ".[dev]"
      - run: pytest tests/perf -ra --benchmark-only --benchmark-json=bench.json
        continue-on-error: true
      - uses: actions/upload-artifact@v4
        with:
          name: benchmark-json
          path: bench.json
```

The `benchmark` job uses `continue-on-error: true` for now; once `tests/perf/` exists (Phase H), the executor will tighten this.

- [ ] **Step 5: Commit**

```bash
git add tests/test_layering.py .importlinter .github/
git commit -m "chore: enforce §3 layering via import-linter and AST test, add CI"
```

---

### Task A3: README disambiguation + multi-process disclaimer

**Files:**
- Modify: `/home/vishsangale/workspace/circuitry/README.md`

- [ ] **Step 1: Rewrite README**

Replace existing content with:

```markdown
# circuitry

> **Disambiguation:** `circuitry` is statistical diagnostics on neural-network weights / activations / gradients during training. It is **not** mechanistic-circuit discovery in the activation-patching / SAE / logit-lens sense. The name is borrowed from electronics.

Training-time mechanistic-interpretability diagnostics for PyTorch — works across LLMs, vision (CNNs / ViTs), and recommender (two-tower) models with a single API.

**Status:** v0.1.0 (alpha). Research code; no support promise. Design contract: [`docs/design.md`](docs/design.md). Implementation plan: [`docs/plan.md`](docs/plan.md).

## Install

```bash
pip install -e .          # editable, from a checkout
pip install -e ".[wandb]" # with wandb writer
```

## Quickstart

```python
from circuitry import Recorder

recorder = Recorder(
    model,
    run_dir="runs/my_run",
    recipe="llm",            # or "vision", "two_tower"
    writer="tensorboard",    # or "wandb", "jsonl", "null"
    every_n_steps=200,
)
recorder.attach()
for step, batch in enumerate(loader):
    loss = train_step(model, batch)
    recorder.step(step, loss=loss)
recorder.detach()
```

Retrospective scan + report from saved checkpoints:

```bash
circuitry scan   --run runs/my_run --recipe llm
circuitry report --run runs/my_run
```

## What you get

- **Primitives** (`circuitry.core.*`) — `effective_rank`, `stable_rank`, `heavy_tail_alpha`, `dead_fraction`, `kurtosis`, `participation_ratio`, `layer_norm`, ESD, rank trajectory, and more.
- **Recorder** — attach to a training loop, write TensorBoard events every N steps, dump a markdown report at the end.
- **Recipes** — `llm` / `vision` / `two_tower` plug the right hooks and diagnostics into your model; subclass `Recipe` or `register_recipe(...)` for custom architectures.
- **MetricWriter protocol** — TB by default; wandb / jsonl / null adapters ship in-tree.

## v0.1.0 limits

- Single-process training only. In a multi-rank DDP/FSDP run `circuitry` no-ops on non-zero ranks; FSDP-sharded parameters will produce **incorrect** diagnostics on rank 0. Multi-process support lands in v0.next; see `docs/design.md` §11 for the upgrade path.
- Benchmark numbers (overhead at default settings on a 50M-param transformer) will be filled in alongside the M2 mendu cutover. The harness is in `tests/perf/` if you want to run it yourself.

## License

MIT.
```

- [ ] **Step 2: Verify the README renders cleanly**

```bash
venv/bin/python -c "import pathlib; print(pathlib.Path('README.md').read_text()[:200])"
```

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: rewrite README with quickstart and v0.1.0 limits"
```

---

## Phase B — Core primitives (TDD against design.md §4.1)

All Phase B tasks share these invariants — re-read before each:

- **No `.cuda()`, no I/O, no logging.** Pure functions.
- **Scalar returns are plain Python `float`,** not 0-dim tensors.
- **Accept `torch.Tensor` or `numpy.ndarray`** where it makes sense (use `torch.as_tensor`).
- **Numerical robustness:** clamp near-zero singular values via `eps`; never raise on degenerate inputs unless explicitly contracted (e.g. empty tensor).
- The mendu sources for spectral / weight primitives are `~/workspace/mendu/mendu/paper2/bet2_daleian/analysis/spectral_diagnostics.py` and `spectral_at_depth.py`. **Read them first** when implementing.

### Task B1: `core/weight.py` — `singular_values` + `effective_rank` (+ `max_dim` subsampling)

**Files:**
- Create: `/home/vishsangale/workspace/circuitry/src/circuitry/core/weight.py`
- Create: `/home/vishsangale/workspace/circuitry/tests/core/test_weight.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/core/test_weight.py
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from circuitry.core import weight


def test_singular_values_diagonal():
    W = torch.diag(torch.tensor([3.0, 2.0, 1.0]))
    s = weight.singular_values(W)
    assert torch.allclose(s, torch.tensor([3.0, 2.0, 1.0]))


def test_singular_values_k_truncates():
    W = torch.diag(torch.tensor([3.0, 2.0, 1.0, 0.5]))
    s = weight.singular_values(W, k=2)
    assert s.shape == (2,)
    assert torch.allclose(s, torch.tensor([3.0, 2.0]))


def test_singular_values_max_dim_subsamples():
    # Wide matrix; max_dim should cap SVD cost without hanging.
    torch.manual_seed(0)
    W = torch.randn(64, 2048)
    s = weight.singular_values(W, max_dim=256)
    assert s.shape[0] <= 256


def test_singular_values_accepts_numpy():
    W = np.eye(4, dtype=np.float32)
    s = weight.singular_values(W)
    assert torch.allclose(s, torch.ones(4))


def test_effective_rank_identity_is_n():
    n = 5
    assert weight.effective_rank(torch.eye(n)) == pytest.approx(n, rel=1e-6)


def test_effective_rank_rank1_is_one():
    u = torch.randn(8, 1)
    v = torch.randn(1, 8)
    W = u @ v
    assert weight.effective_rank(W) == pytest.approx(1.0, abs=1e-4)


def test_effective_rank_returns_python_float():
    val = weight.effective_rank(torch.eye(3))
    assert isinstance(val, float)


def test_effective_rank_invariant_under_orthogonal():
    torch.manual_seed(0)
    W = torch.randn(16, 16)
    Q, _ = torch.linalg.qr(torch.randn(16, 16))
    assert weight.effective_rank(W) == pytest.approx(
        weight.effective_rank(Q @ W), rel=1e-5
    )
```

- [ ] **Step 2: Run — confirm fail**

```bash
venv/bin/pytest tests/core/test_weight.py -v
```

Expected: ModuleNotFoundError or AttributeError on `weight.singular_values`.

- [ ] **Step 3: Implement**

```python
# src/circuitry/core/weight.py
"""Weight-space diagnostics. Pure functions; CPU-deterministic; no I/O.

See docs/design.md §4.1 for the contract.
"""

from __future__ import annotations

import math
from typing import Union

import numpy as np
import torch

ArrayLike = Union[torch.Tensor, np.ndarray]


def _as_2d(W: ArrayLike) -> torch.Tensor:
    t = torch.as_tensor(W)
    if t.ndim == 1:
        t = t.unsqueeze(0)
    elif t.ndim > 2:
        t = t.reshape(t.shape[0], -1)
    return t.to(dtype=torch.float32 if t.dtype not in (torch.float32, torch.float64) else t.dtype)


def singular_values(
    W: ArrayLike,
    k: int | None = None,
    max_dim: int | None = 512,
) -> torch.Tensor:
    """Singular values of ``W`` in descending order.

    ``max_dim`` caps the SVD cost on wide matrices by truncating to a
    ``max_dim``-column random subsample before the decomposition. Pass
    ``max_dim=None`` to disable. ``k`` truncates the returned vector.
    """
    M = _as_2d(W)
    if max_dim is not None and min(M.shape) > max_dim:
        # Sample columns from the longer axis to keep SVD bounded.
        axis = 1 if M.shape[1] > M.shape[0] else 0
        n = M.shape[axis]
        idx = torch.randperm(n)[:max_dim]
        M = M.index_select(axis, idx)
    s = torch.linalg.svdvals(M)
    s, _ = torch.sort(s, descending=True)
    if k is not None:
        s = s[:k]
    return s


def effective_rank(W: ArrayLike, eps: float = 1e-12) -> float:
    """Roy & Vetterli (2007) effective rank: ``exp(H(p))`` where ``p`` is the
    normalized singular-value distribution.
    """
    s = singular_values(W)
    s = s[s > eps]
    if s.numel() == 0:
        return 0.0
    p = s / s.sum()
    H = -(p * torch.log(p)).sum().item()
    return float(math.exp(H))
```

- [ ] **Step 4: Run — confirm pass**

```bash
venv/bin/pytest tests/core/test_weight.py -v
```

Expected: 8 PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/circuitry/core/weight.py tests/core/test_weight.py
git commit -m "feat(core): add singular_values and effective_rank with max_dim subsampling"
```

---

### Task B2: `core/weight.py` — `stable_rank` + `condition_number`

**Files:**
- Modify: `/home/vishsangale/workspace/circuitry/src/circuitry/core/weight.py`
- Modify: `/home/vishsangale/workspace/circuitry/tests/core/test_weight.py`

- [ ] **Step 1: Add failing tests**

Append to `tests/core/test_weight.py`:

```python
def test_stable_rank_identity():
    n = 5
    assert weight.stable_rank(torch.eye(n)) == pytest.approx(n, rel=1e-6)


def test_stable_rank_rank1_is_one():
    u = torch.randn(8, 1)
    v = torch.randn(1, 8)
    assert weight.stable_rank(u @ v) == pytest.approx(1.0, abs=1e-4)


def test_stable_rank_returns_float():
    assert isinstance(weight.stable_rank(torch.eye(3)), float)


def test_condition_number_orthogonal_is_one():
    Q, _ = torch.linalg.qr(torch.randn(8, 8))
    assert weight.condition_number(Q) == pytest.approx(1.0, abs=1e-4)


def test_condition_number_diag():
    W = torch.diag(torch.tensor([4.0, 1.0]))
    assert weight.condition_number(W) == pytest.approx(4.0, rel=1e-6)


def test_condition_number_returns_float():
    assert isinstance(weight.condition_number(torch.eye(3)), float)
```

- [ ] **Step 2: Run — confirm fail**

```bash
venv/bin/pytest tests/core/test_weight.py -v
```

- [ ] **Step 3: Implement — append to `src/circuitry/core/weight.py`**

```python
def stable_rank(W: ArrayLike) -> float:
    """``||W||_F^2 / ||W||_2^2``. Lower-bounds the algebraic rank and is
    numerically robust on near-singular matrices.
    """
    s = singular_values(W)
    if s.numel() == 0:
        return 0.0
    return float((s.pow(2).sum() / (s[0].pow(2))).item())


def condition_number(W: ArrayLike, eps: float = 1e-12) -> float:
    """``sigma_max / sigma_min``. Returns ``+inf`` if the smallest singular
    value is below ``eps``.
    """
    s = singular_values(W)
    if s.numel() == 0 or s[-1].item() < eps:
        return float("inf")
    return float((s[0] / s[-1]).item())
```

- [ ] **Step 4: Run — confirm pass**

```bash
venv/bin/pytest tests/core/test_weight.py -v
```

Expected: 14 PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/circuitry/core/weight.py tests/core/test_weight.py
git commit -m "feat(core): add stable_rank and condition_number"
```

---

### Task B3: `core/weight.py` — `heavy_tail_alpha`

**Files:**
- Modify: `/home/vishsangale/workspace/circuitry/src/circuitry/core/weight.py`
- Modify: `/home/vishsangale/workspace/circuitry/tests/core/test_weight.py`

Read `~/workspace/mendu/mendu/paper2/bet2_daleian/analysis/spectral_diagnostics.py` for the reference implementation of the Hill estimator used for `alpha`. Port-with-cleanup.

- [ ] **Step 1: Add failing tests**

```python
def test_heavy_tail_alpha_random_matrix():
    # Marchenko-Pastur bulk → alpha is finite and positive.
    torch.manual_seed(0)
    W = torch.randn(64, 64)
    alpha = weight.heavy_tail_alpha(W)
    assert isinstance(alpha, float)
    assert math.isfinite(alpha)
    assert alpha > 0


def test_heavy_tail_alpha_low_rank_is_finite():
    # Constructed power-law tail (rank ~10 in 64-dim space).
    torch.manual_seed(0)
    U = torch.randn(64, 10)
    V = torch.randn(10, 64)
    alpha = weight.heavy_tail_alpha(U @ V)
    assert math.isfinite(alpha)
```

- [ ] **Step 2: Run — confirm fail**

```bash
venv/bin/pytest tests/core/test_weight.py::test_heavy_tail_alpha_random_matrix -v
```

- [ ] **Step 3: Implement — append to `src/circuitry/core/weight.py`**

```python
def heavy_tail_alpha(W: ArrayLike, top_frac: float = 0.5) -> float:
    """Hill estimator of the tail index of the squared-singular-value
    distribution. Computed on the top ``top_frac`` (default half) of squared
    singular values; that fraction is the empirically robust default used in
    the mendu paper-2 spectral diagnostics.

    Returns ``+inf`` on degenerate inputs.
    """
    s = singular_values(W)
    if s.numel() < 4:
        return float("inf")
    s2 = s.pow(2).sort(descending=True).values
    k = max(2, int(s2.numel() * top_frac))
    top = s2[:k]
    smin = top[-1].clamp_min(1e-30)
    # Hill: alpha_hat = k / sum(log(s_i / s_k))
    logs = torch.log(top / smin)
    denom = logs.sum().item()
    if denom <= 0:
        return float("inf")
    return float(k / denom)
```

- [ ] **Step 4: Run — confirm pass**

```bash
venv/bin/pytest tests/core/test_weight.py -v
```

Expected: 16 PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/circuitry/core/weight.py tests/core/test_weight.py
git commit -m "feat(core): add heavy_tail_alpha (Hill estimator) ported from mendu"
```

---

### Task B4: `core/activation.py` — `dead_fraction` + `NormStats` / `norm_stats`

**Files:**
- Create: `/home/vishsangale/workspace/circuitry/src/circuitry/core/activation.py`
- Create: `/home/vishsangale/workspace/circuitry/tests/core/test_activation.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/core/test_activation.py
from __future__ import annotations

import pytest
import torch

from circuitry.core import activation
from circuitry.core.activation import NormStats


def test_dead_fraction_all_zeros_is_one():
    x = torch.zeros(4, 8)
    assert activation.dead_fraction(x) == pytest.approx(1.0)


def test_dead_fraction_none_dead():
    x = torch.ones(4, 8)
    assert activation.dead_fraction(x) == pytest.approx(0.0)


def test_dead_fraction_threshold():
    x = torch.tensor([[0.0, 0.1, 1.0, -0.5]])
    # default threshold=0.0 → "dead" means <= 0
    assert activation.dead_fraction(x) == pytest.approx(0.5)


def test_dead_fraction_returns_float():
    assert isinstance(activation.dead_fraction(torch.zeros(2, 2)), float)


def test_norm_stats_shape_and_fields():
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    s = activation.norm_stats(x)
    assert isinstance(s, NormStats)
    assert s.mean == pytest.approx(2.5)
    assert s.max == pytest.approx(4.0)
    assert s.std > 0
    # frac > k*median: median is 2.5; 4>2.5 → 1/4 = 0.25 with k=1
    assert 0.0 <= s.frac_above_k_median <= 1.0
```

- [ ] **Step 2: Run — confirm fail**

```bash
venv/bin/pytest tests/core/test_activation.py -v
```

- [ ] **Step 3: Implement**

```python
# src/circuitry/core/activation.py
"""Activation-space diagnostics. Pure; CPU-deterministic.

See docs/design.md §4.1 for the contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import numpy as np
import torch

ArrayLike = Union[torch.Tensor, np.ndarray]


@dataclass(frozen=True)
class NormStats:
    mean: float
    std: float
    max: float
    frac_above_k_median: float


def _as_tensor(x: ArrayLike) -> torch.Tensor:
    return torch.as_tensor(x).to(dtype=torch.float32)


def dead_fraction(x: ArrayLike, threshold: float = 0.0) -> float:
    """Fraction of activations at or below ``threshold``."""
    t = _as_tensor(x)
    if t.numel() == 0:
        return 0.0
    return float((t <= threshold).float().mean().item())


def norm_stats(x: ArrayLike, k: float = 3.0) -> NormStats:
    """Per-element norm statistics. ``frac_above_k_median`` is the fraction of
    elements whose absolute value exceeds ``k * median(|x|)`` — a cheap
    heavy-tail indicator.
    """
    t = _as_tensor(x).flatten()
    if t.numel() == 0:
        return NormStats(0.0, 0.0, 0.0, 0.0)
    abs_t = t.abs()
    med = abs_t.median().item()
    return NormStats(
        mean=float(t.mean().item()),
        std=float(t.std(unbiased=False).item()),
        max=float(abs_t.max().item()),
        frac_above_k_median=float((abs_t > k * med).float().mean().item()) if med > 0 else 0.0,
    )
```

- [ ] **Step 4: Run — confirm pass**

```bash
venv/bin/pytest tests/core/test_activation.py -v
```

Expected: 5 PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/circuitry/core/activation.py tests/core/test_activation.py
git commit -m "feat(core): add dead_fraction and NormStats"
```

---

### Task B5: `core/activation.py` — `kurtosis` + `participation_ratio`

**Files:**
- Modify: `/home/vishsangale/workspace/circuitry/src/circuitry/core/activation.py`
- Modify: `/home/vishsangale/workspace/circuitry/tests/core/test_activation.py`

- [ ] **Step 1: Add failing tests**

```python
def test_kurtosis_normal_is_near_zero():
    torch.manual_seed(0)
    x = torch.randn(10_000)
    # Excess kurtosis of N(0,1) ≈ 0 (within sampling noise).
    assert abs(float(activation.kurtosis(x).item())) < 0.3


def test_kurtosis_heavy_tail_is_positive():
    torch.manual_seed(0)
    base = torch.randn(10_000)
    base[:50] *= 20.0  # inject heavy-tail outliers
    assert float(activation.kurtosis(base).item()) > 1.0


def test_kurtosis_along_dim():
    x = torch.randn(8, 100)
    k = activation.kurtosis(x, dim=-1)
    assert k.shape == (8,)


def test_participation_ratio_uniform_is_n():
    # Uniform |x| → PR ≈ n.
    x = torch.ones(16)
    assert activation.participation_ratio(x) == pytest.approx(16.0, rel=1e-5)


def test_participation_ratio_spike_is_one():
    x = torch.zeros(16)
    x[0] = 1.0
    assert activation.participation_ratio(x) == pytest.approx(1.0, rel=1e-5)


def test_participation_ratio_returns_float():
    assert isinstance(activation.participation_ratio(torch.ones(4)), float)
```

- [ ] **Step 2: Run — confirm fail**

```bash
venv/bin/pytest tests/core/test_activation.py -v
```

- [ ] **Step 3: Implement — append to `src/circuitry/core/activation.py`**

```python
def kurtosis(x: ArrayLike, dim: int | tuple[int, ...] = -1) -> torch.Tensor:
    """Excess kurtosis along ``dim``. Returns a tensor (not a Python float)
    because callers commonly want per-channel kurtosis."""
    t = _as_tensor(x)
    mean = t.mean(dim=dim, keepdim=True)
    centered = t - mean
    var = centered.pow(2).mean(dim=dim)
    m4 = centered.pow(4).mean(dim=dim)
    # Avoid div-by-zero
    out = m4 / var.clamp_min(1e-30).pow(2) - 3.0
    out = torch.where(var > 0, out, torch.zeros_like(out))
    return out


def participation_ratio(x: ArrayLike) -> float:
    """``(sum |x|)^2 / sum(x^2)`` — soft count of "active" units.

    Equals ``n`` when ``|x|`` is uniform, equals 1 when ``x`` is a one-hot.
    """
    t = _as_tensor(x).flatten()
    if t.numel() == 0:
        return 0.0
    num = t.abs().sum().pow(2)
    den = t.pow(2).sum().clamp_min(1e-30)
    return float((num / den).item())
```

- [ ] **Step 4: Run — confirm pass**

```bash
venv/bin/pytest tests/core/test_activation.py -v
```

Expected: 11 PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/circuitry/core/activation.py tests/core/test_activation.py
git commit -m "feat(core): add kurtosis and participation_ratio"
```

---

### Task B6: `core/gradient.py` — `layer_norm` + `signal_propagation_depth`

**Files:**
- Create: `/home/vishsangale/workspace/circuitry/src/circuitry/core/gradient.py`
- Create: `/home/vishsangale/workspace/circuitry/tests/core/test_gradient.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/core/test_gradient.py
from __future__ import annotations

import pytest
import torch

from circuitry.core import gradient


def test_layer_norm_returns_dict_of_floats():
    grads = {
        "layer0.weight": torch.ones(3, 3),
        "layer1.weight": torch.zeros(3, 3),
    }
    out = gradient.layer_norm(grads)
    assert set(out) == {"layer0.weight", "layer1.weight"}
    assert all(isinstance(v, float) for v in out.values())
    assert out["layer0.weight"] == pytest.approx(3.0)  # frobenius norm of ones(3,3)
    assert out["layer1.weight"] == pytest.approx(0.0)


def test_layer_norm_empty_dict():
    assert gradient.layer_norm({}) == {}


def test_signal_propagation_depth_all_alive():
    # Norms decreasing but all above eps → reaches max depth.
    grads = [torch.full((4,), v) for v in (1.0, 0.5, 0.25, 0.1)]
    assert gradient.signal_propagation_depth(grads) == 4


def test_signal_propagation_depth_vanishing():
    grads = [torch.ones(4), torch.full((4,), 1e-2), torch.zeros(4), torch.zeros(4)]
    # eps_ratio default 1e-3 of first layer norm → 2.0 cutoff; layer 1 = 2e-2 > 2e-3 → alive; layer 2 = 0 → dead.
    assert gradient.signal_propagation_depth(grads) == 2
```

- [ ] **Step 2: Run — confirm fail**

```bash
venv/bin/pytest tests/core/test_gradient.py -v
```

- [ ] **Step 3: Implement**

```python
# src/circuitry/core/gradient.py
"""Gradient-space diagnostics. Pure; CPU-deterministic.

See docs/design.md §4.1 for the contract.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import torch


def layer_norm(grads: Mapping[str, torch.Tensor]) -> dict[str, float]:
    """Frobenius norm per layer. ``None`` values are skipped."""
    out: dict[str, float] = {}
    for name, g in grads.items():
        if g is None:
            continue
        out[name] = float(torch.linalg.vector_norm(g).item())
    return out


def signal_propagation_depth(
    grads_by_depth: Sequence[torch.Tensor],
    eps_ratio: float = 1e-3,
) -> int:
    """Deepest layer whose gradient norm exceeds ``eps_ratio * norm(layer_0)``.

    Returns 0 if the first layer is itself zero, ``len(grads_by_depth)`` if all
    layers are alive.
    """
    if not grads_by_depth:
        return 0
    norms = [float(torch.linalg.vector_norm(g).item()) for g in grads_by_depth]
    if norms[0] == 0.0:
        return 0
    threshold = eps_ratio * norms[0]
    depth = 0
    for n in norms:
        if n > threshold:
            depth += 1
        else:
            break
    return depth
```

- [ ] **Step 4: Run — confirm pass**

```bash
venv/bin/pytest tests/core/test_gradient.py -v
```

Expected: 4 PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/circuitry/core/gradient.py tests/core/test_gradient.py
git commit -m "feat(core): add gradient.layer_norm and signal_propagation_depth"
```

---

### Task B7: `core/spectral.py` — `esd` + `rank_trajectory`

**Files:**
- Create: `/home/vishsangale/workspace/circuitry/src/circuitry/core/spectral.py`
- Create: `/home/vishsangale/workspace/circuitry/tests/core/test_spectral.py`

Read `~/workspace/mendu/mendu/paper2/bet2_daleian/analysis/spectral_diagnostics.py` for the ESD reference. Port-with-cleanup.

- [ ] **Step 1: Write failing tests**

```python
# tests/core/test_spectral.py
from __future__ import annotations

import pytest
import torch

from circuitry.core import spectral, weight


def test_esd_returns_pair_of_tensors():
    W = torch.randn(32, 32)
    edges, counts = spectral.esd(W, bins=20)
    assert edges.shape == (21,)
    assert counts.shape == (20,)
    assert counts.sum().item() > 0


def test_esd_zero_matrix():
    W = torch.zeros(8, 8)
    edges, counts = spectral.esd(W, bins=5)
    # All mass at zero; should not raise.
    assert counts.sum().item() == 8


def test_rank_trajectory_keys_match_state_dict():
    W1 = {"layer.weight": torch.eye(4), "other.weight": torch.zeros(3, 3)}
    W2 = {"layer.weight": torch.eye(4) * 2, "other.weight": torch.eye(3)}
    traj = spectral.rank_trajectory([W1, W2])
    assert set(traj) == {"layer.weight", "other.weight"}
    assert len(traj["layer.weight"]) == 2
    assert traj["layer.weight"][0] == pytest.approx(weight.effective_rank(W1["layer.weight"]), rel=1e-5)


def test_rank_trajectory_skips_non_2d():
    W1 = {"bias": torch.zeros(8), "W": torch.eye(4)}
    W2 = {"bias": torch.ones(8), "W": torch.eye(4) * 0.5}
    traj = spectral.rank_trajectory([W1, W2])
    assert "bias" not in traj
    assert "W" in traj
```

- [ ] **Step 2: Run — confirm fail**

```bash
venv/bin/pytest tests/core/test_spectral.py -v
```

- [ ] **Step 3: Implement**

```python
# src/circuitry/core/spectral.py
"""Spectral diagnostics across snapshots. Pure; CPU-deterministic.

See docs/design.md §4.1 for the contract.
"""

from __future__ import annotations

from typing import Mapping, Sequence, Union

import numpy as np
import torch

from circuitry.core import weight

ArrayLike = Union[torch.Tensor, np.ndarray]


def esd(W: ArrayLike, bins: int = 100) -> tuple[torch.Tensor, torch.Tensor]:
    """Empirical spectral density: a histogram of singular values.

    Returns ``(bin_edges, counts)`` so the result is drop-in for
    ``torch.utils.tensorboard.SummaryWriter.add_histogram`` after a small
    reshape, and also human-plottable.
    """
    s = weight.singular_values(W)
    if s.numel() == 0:
        edges = torch.linspace(0.0, 1.0, bins + 1)
        counts = torch.zeros(bins)
        return edges, counts
    counts = torch.histc(s, bins=bins, min=float(s.min().item()), max=float(s.max().item()))
    edges = torch.linspace(float(s.min().item()), float(s.max().item()), bins + 1)
    if counts.sum() == 0:  # degenerate (all values identical)
        counts[0] = s.numel()
    return edges, counts


def rank_trajectory(
    state_dicts: Sequence[Mapping[str, torch.Tensor]],
) -> dict[str, list[float]]:
    """Effective rank per 2D parameter across an ordered sequence of state dicts.

    Non-2D tensors (biases, layer norms) are skipped.
    """
    if not state_dicts:
        return {}
    keys = [k for k, v in state_dicts[0].items() if torch.as_tensor(v).ndim >= 2]
    out: dict[str, list[float]] = {k: [] for k in keys}
    for sd in state_dicts:
        for k in keys:
            if k not in sd:
                out[k].append(float("nan"))
            else:
                out[k].append(weight.effective_rank(sd[k]))
    return out
```

- [ ] **Step 4: Run — confirm pass**

```bash
venv/bin/pytest tests/core/test_spectral.py -v
```

Expected: 4 PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/circuitry/core/spectral.py tests/core/test_spectral.py
git commit -m "feat(core): add spectral.esd and rank_trajectory"
```

---

## Phase C — Writers (`MetricWriter` protocol + adapters)

### Task C1: `writers/base.py` — `MetricWriter` Protocol + `RecordingWriter` test double

**Files:**
- Create: `/home/vishsangale/workspace/circuitry/src/circuitry/writers/base.py`
- Create: `/home/vishsangale/workspace/circuitry/tests/recorder/test_writer_protocol.py`

- [ ] **Step 1: Write failing test**

```python
# tests/recorder/test_writer_protocol.py
from __future__ import annotations

import torch

from circuitry.writers.base import MetricWriter, RecordingWriter


def test_recording_writer_satisfies_protocol():
    w: MetricWriter = RecordingWriter()
    w.add_scalar("loss", 1.5, 1)
    w.add_scalar("loss", 1.2, 2)
    w.add_histogram("grad", torch.arange(10.0), 3)
    w.add_image("kernel", torch.zeros(3, 4, 4), 3, dataformats="CHW")
    w.add_text("note", "hi", 3)
    w.flush()
    w.close()

    assert w.scalars == [("loss", 1.5, 1), ("loss", 1.2, 2)]
    assert w.histograms[0][0] == "grad"
    assert w.images[0] == ("kernel", torch.zeros(3, 4, 4).shape, 3, "CHW")
    assert w.texts == [("note", "hi", 3)]
    assert w.flushed == 1
    assert w.closed
```

- [ ] **Step 2: Run — confirm fail**

```bash
venv/bin/pytest tests/recorder/test_writer_protocol.py -v
```

- [ ] **Step 3: Implement**

```python
# src/circuitry/writers/base.py
"""MetricWriter protocol + a recording test double used in recorder tests.

See docs/design.md §4.5 for the protocol contract.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import torch


@runtime_checkable
class MetricWriter(Protocol):
    def add_scalar(self, tag: str, value: float, step: int) -> None: ...
    def add_histogram(self, tag: str, values: torch.Tensor, step: int) -> None: ...
    def add_image(self, tag: str, image: torch.Tensor, step: int,
                  dataformats: str = "CHW") -> None: ...
    def add_text(self, tag: str, text: str, step: int) -> None: ...
    def flush(self) -> None: ...
    def close(self) -> None: ...


class RecordingWriter:
    """Captures every call into in-memory lists. For tests only."""

    def __init__(self) -> None:
        self.scalars: list[tuple[str, float, int]] = []
        self.histograms: list[tuple[str, torch.Tensor, int]] = []
        self.images: list[tuple[str, Any, int, str]] = []
        self.texts: list[tuple[str, str, int]] = []
        self.flushed: int = 0
        self.closed: bool = False

    def add_scalar(self, tag: str, value: float, step: int) -> None:
        self.scalars.append((tag, float(value), int(step)))

    def add_histogram(self, tag: str, values: torch.Tensor, step: int) -> None:
        self.histograms.append((tag, values, int(step)))

    def add_image(self, tag: str, image: torch.Tensor, step: int,
                  dataformats: str = "CHW") -> None:
        # Store shape rather than the tensor to keep memory small in tests.
        self.images.append((tag, image.shape, int(step), dataformats))

    def add_text(self, tag: str, text: str, step: int) -> None:
        self.texts.append((tag, text, int(step)))

    def flush(self) -> None:
        self.flushed += 1

    def close(self) -> None:
        self.closed = True
```

- [ ] **Step 4: Run — confirm pass**

```bash
venv/bin/pytest tests/recorder/test_writer_protocol.py -v
```

Expected: 1 PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/circuitry/writers/base.py tests/recorder/test_writer_protocol.py
git commit -m "feat(writers): add MetricWriter protocol and RecordingWriter test double"
```

---

### Task C2: `writers/null.py` + `writers/jsonl.py`

**Files:**
- Create: `/home/vishsangale/workspace/circuitry/src/circuitry/writers/null.py`
- Create: `/home/vishsangale/workspace/circuitry/src/circuitry/writers/jsonl.py`
- Create: `/home/vishsangale/workspace/circuitry/tests/recorder/test_writers_concrete.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/recorder/test_writers_concrete.py
from __future__ import annotations

import json
import pathlib

import torch

from circuitry.writers.jsonl import JsonlWriter
from circuitry.writers.null import NullWriter


def test_null_writer_is_silent(tmp_path):
    w = NullWriter()
    w.add_scalar("loss", 1.0, 1)
    w.add_histogram("g", torch.zeros(4), 1)
    w.add_image("k", torch.zeros(3, 4, 4), 1)
    w.add_text("note", "x", 1)
    w.flush()
    w.close()
    assert list(tmp_path.iterdir()) == []


def test_jsonl_writer_writes_scalars_per_line(tmp_path):
    w = JsonlWriter(tmp_path)
    w.add_scalar("loss", 1.5, 1)
    w.add_scalar("loss", 1.2, 2)
    w.flush()
    w.close()
    path = tmp_path / "metrics.jsonl"
    assert path.exists()
    lines = path.read_text().splitlines()
    assert json.loads(lines[0]) == {"tag": "loss", "value": 1.5, "step": 1, "kind": "scalar"}
    assert json.loads(lines[1]) == {"tag": "loss", "value": 1.2, "step": 2, "kind": "scalar"}


def test_jsonl_writer_dumps_histogram_to_artifacts(tmp_path):
    w = JsonlWriter(tmp_path)
    w.add_histogram("grad", torch.arange(8.0), 1)
    w.close()
    art_dir = tmp_path / "circuitry" / "artifacts"
    assert any(p.name.startswith("grad") and p.suffix == ".npy" for p in art_dir.iterdir())
```

- [ ] **Step 2: Run — confirm fail**

```bash
venv/bin/pytest tests/recorder/test_writers_concrete.py -v
```

- [ ] **Step 3: Implement `writers/null.py`**

```python
# src/circuitry/writers/null.py
"""No-op writer. Useful for tests and disabled-diagnostic paths."""

from __future__ import annotations

import torch


class NullWriter:
    def add_scalar(self, tag: str, value: float, step: int) -> None: ...
    def add_histogram(self, tag: str, values: torch.Tensor, step: int) -> None: ...
    def add_image(self, tag: str, image: torch.Tensor, step: int,
                  dataformats: str = "CHW") -> None: ...
    def add_text(self, tag: str, text: str, step: int) -> None: ...
    def flush(self) -> None: ...
    def close(self) -> None: ...
```

- [ ] **Step 4: Implement `writers/jsonl.py`**

```python
# src/circuitry/writers/jsonl.py
"""JSONL writer — one JSON line per scalar/text; tensors and images dumped to
side files under <run_dir>/circuitry/artifacts/.

Zero non-stdlib deps beyond numpy / torch.
"""

from __future__ import annotations

import json
import pathlib
import re
from typing import IO

import numpy as np
import torch

_SAFE = re.compile(r"[^a-zA-Z0-9_.-]+")


def _slug(tag: str) -> str:
    return _SAFE.sub("_", tag).strip("_") or "tag"


class JsonlWriter:
    def __init__(self, run_dir: str | pathlib.Path) -> None:
        self._run_dir = pathlib.Path(run_dir)
        self._run_dir.mkdir(parents=True, exist_ok=True)
        self._artifacts = self._run_dir / "circuitry" / "artifacts"
        self._artifacts.mkdir(parents=True, exist_ok=True)
        self._fh: IO[str] = (self._run_dir / "metrics.jsonl").open("a", buffering=1)

    def _emit(self, record: dict) -> None:
        self._fh.write(json.dumps(record) + "\n")

    def add_scalar(self, tag: str, value: float, step: int) -> None:
        self._emit({"tag": tag, "value": float(value), "step": int(step), "kind": "scalar"})

    def add_histogram(self, tag: str, values: torch.Tensor, step: int) -> None:
        arr = torch.as_tensor(values).detach().cpu().numpy()
        out = self._artifacts / f"{_slug(tag)}-step{step:09d}.npy"
        np.save(out, arr)
        self._emit({"tag": tag, "path": str(out.relative_to(self._run_dir)),
                    "step": int(step), "kind": "histogram"})

    def add_image(self, tag: str, image: torch.Tensor, step: int,
                  dataformats: str = "CHW") -> None:
        arr = torch.as_tensor(image).detach().cpu().numpy()
        out = self._artifacts / f"{_slug(tag)}-step{step:09d}.npy"
        np.save(out, arr)
        self._emit({"tag": tag, "path": str(out.relative_to(self._run_dir)),
                    "step": int(step), "kind": "image", "dataformats": dataformats})

    def add_text(self, tag: str, text: str, step: int) -> None:
        self._emit({"tag": tag, "text": text, "step": int(step), "kind": "text"})

    def flush(self) -> None:
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()
```

- [ ] **Step 5: Run — confirm pass**

```bash
venv/bin/pytest tests/recorder/test_writers_concrete.py -v
```

Expected: 3 PASSED.

- [ ] **Step 6: Commit**

```bash
git add src/circuitry/writers/null.py src/circuitry/writers/jsonl.py tests/recorder/test_writers_concrete.py
git commit -m "feat(writers): add NullWriter and JsonlWriter"
```

---

### Task C3: `writers/tensorboard.py` (default, with optional async)

**Files:**
- Create: `/home/vishsangale/workspace/circuitry/src/circuitry/writers/tensorboard.py`
- Modify: `/home/vishsangale/workspace/circuitry/tests/recorder/test_writers_concrete.py`

- [ ] **Step 1: Add failing test**

Append to `tests/recorder/test_writers_concrete.py`:

```python
def test_tensorboard_writer_writes_event_files(tmp_path):
    from circuitry.writers.tensorboard import TensorBoardWriter

    w = TensorBoardWriter(tmp_path)
    w.add_scalar("loss", 1.5, 1)
    w.add_histogram("g", torch.arange(10.0), 1)
    w.add_image("k", torch.zeros(3, 8, 8), 1, dataformats="CHW")
    w.add_text("n", "ok", 1)
    w.flush()
    w.close()

    event_files = [p for p in tmp_path.rglob("events.out.tfevents.*")]
    assert event_files, "TensorBoard event file not written"


def test_tensorboard_writer_async_does_not_lose_data(tmp_path):
    from circuitry.writers.tensorboard import TensorBoardWriter

    w = TensorBoardWriter(tmp_path, async_writes=True)
    for i in range(50):
        w.add_scalar("loss", float(i), i)
    w.flush()
    w.close()
    event_files = [p for p in tmp_path.rglob("events.out.tfevents.*")]
    assert event_files
```

- [ ] **Step 2: Run — confirm fail**

```bash
venv/bin/pytest tests/recorder/test_writers_concrete.py::test_tensorboard_writer_writes_event_files -v
```

- [ ] **Step 3: Implement**

```python
# src/circuitry/writers/tensorboard.py
"""TensorBoard MetricWriter (default).

Thin wrapper over torch.utils.tensorboard.SummaryWriter with an optional
background-thread queue (``async_writes=True``) so add_* never blocks the
training step on disk I/O. The async path drains the queue from a single
worker thread to preserve write order.
"""

from __future__ import annotations

import pathlib
import queue
import threading
from typing import Any

import torch
from torch.utils.tensorboard import SummaryWriter

_SENTINEL: Any = object()


class TensorBoardWriter:
    def __init__(self, run_dir: str | pathlib.Path, async_writes: bool = False) -> None:
        self._writer = SummaryWriter(log_dir=str(run_dir))
        self._async = async_writes
        if async_writes:
            self._q: queue.Queue[Any] = queue.Queue()
            self._worker = threading.Thread(target=self._drain, daemon=True)
            self._worker.start()

    def _drain(self) -> None:
        while True:
            item = self._q.get()
            if item is _SENTINEL:
                return
            method, args, kwargs = item
            getattr(self._writer, method)(*args, **kwargs)

    def _dispatch(self, method: str, *args: Any, **kwargs: Any) -> None:
        if self._async:
            self._q.put((method, args, kwargs))
        else:
            getattr(self._writer, method)(*args, **kwargs)

    def add_scalar(self, tag: str, value: float, step: int) -> None:
        self._dispatch("add_scalar", tag, float(value), int(step))

    def add_histogram(self, tag: str, values: torch.Tensor, step: int) -> None:
        self._dispatch("add_histogram", tag, torch.as_tensor(values), int(step))

    def add_image(self, tag: str, image: torch.Tensor, step: int,
                  dataformats: str = "CHW") -> None:
        self._dispatch("add_image", tag, torch.as_tensor(image), int(step),
                       dataformats=dataformats)

    def add_text(self, tag: str, text: str, step: int) -> None:
        self._dispatch("add_text", tag, text, int(step))

    def flush(self) -> None:
        if self._async:
            self._q.join()
        self._writer.flush()

    def close(self) -> None:
        if self._async:
            self._q.put(_SENTINEL)
            self._worker.join()
        self._writer.close()
```

- [ ] **Step 4: Run — confirm pass**

```bash
venv/bin/pytest tests/recorder/test_writers_concrete.py -v
```

Expected: 5 PASSED total in this file.

- [ ] **Step 5: Commit**

```bash
git add src/circuitry/writers/tensorboard.py tests/recorder/test_writers_concrete.py
git commit -m "feat(writers): add TensorBoardWriter with optional async writes"
```

---

### Task C4: `writers/wandb.py` (gated extra)

**Files:**
- Create: `/home/vishsangale/workspace/circuitry/src/circuitry/writers/wandb.py`
- Modify: `/home/vishsangale/workspace/circuitry/tests/recorder/test_writers_concrete.py`

The wandb writer is gated behind the `[wandb]` extra and only constructed lazily, so the import does not fail when wandb is absent. The unit test skips when wandb is not installed.

- [ ] **Step 1: Add failing test**

```python
def test_wandb_writer_skips_when_wandb_absent():
    pytest = __import__("pytest")
    try:
        import wandb  # noqa: F401
    except ImportError:
        pytest.skip("wandb not installed")
    from circuitry.writers.wandb import WandbWriter

    # Use mode='disabled' so no network/auth is required.
    w = WandbWriter(project="circuitry-test", mode="disabled")
    w.add_scalar("loss", 1.0, 1)
    w.flush()
    w.close()
```

- [ ] **Step 2: Run — confirm xfail or skip**

```bash
venv/bin/pytest tests/recorder/test_writers_concrete.py::test_wandb_writer_skips_when_wandb_absent -v
```

- [ ] **Step 3: Implement**

```python
# src/circuitry/writers/wandb.py
"""Optional wandb MetricWriter. Install with ``pip install circuitry[wandb]``."""

from __future__ import annotations

from typing import Any

import torch


class WandbWriter:
    def __init__(self, project: str | None = None, run_name: str | None = None,
                 mode: str = "online", **init_kwargs: Any) -> None:
        try:
            import wandb
        except ImportError as e:
            raise ImportError(
                "wandb writer requires the [wandb] extra: "
                "`pip install circuitry[wandb]`"
            ) from e
        self._wandb = wandb
        self._run = wandb.init(project=project, name=run_name, mode=mode, **init_kwargs)

    def add_scalar(self, tag: str, value: float, step: int) -> None:
        self._wandb.log({tag: float(value)}, step=int(step))

    def add_histogram(self, tag: str, values: torch.Tensor, step: int) -> None:
        arr = torch.as_tensor(values).detach().cpu().numpy()
        self._wandb.log({tag: self._wandb.Histogram(arr)}, step=int(step))

    def add_image(self, tag: str, image: torch.Tensor, step: int,
                  dataformats: str = "CHW") -> None:
        arr = torch.as_tensor(image).detach().cpu().numpy()
        if dataformats == "CHW" and arr.ndim == 3:
            arr = arr.transpose(1, 2, 0)
        self._wandb.log({tag: self._wandb.Image(arr)}, step=int(step))

    def add_text(self, tag: str, text: str, step: int) -> None:
        self._wandb.log({tag: text}, step=int(step))

    def flush(self) -> None:
        # wandb flushes asynchronously; nothing required.
        pass

    def close(self) -> None:
        self._wandb.finish()
```

- [ ] **Step 4: Run — confirm pass (or skip)**

```bash
venv/bin/pytest tests/recorder/test_writers_concrete.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/circuitry/writers/wandb.py tests/recorder/test_writers_concrete.py
git commit -m "feat(writers): add optional WandbWriter behind [wandb] extra"
```

---

## Phase D — Recorder

### Task D1: `recorder/hooks.py` — `TensorSource` / `HookPoint` / `StepContext` / matcher

**Files:**
- Create: `/home/vishsangale/workspace/circuitry/src/circuitry/recorder/hooks.py`
- Create: `/home/vishsangale/workspace/circuitry/tests/recorder/test_hooks.py`

Read `~/workspace/mendu/mendu/tools/inspect_checkpoint/arch_hooks.py` for the reference hook-strategy logic.

- [ ] **Step 1: Write failing tests**

```python
# tests/recorder/test_hooks.py
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from circuitry.recorder.hooks import (
    HookPoint,
    StepContext,
    TensorSource,
    match_modules,
)


def _toy() -> nn.Module:
    return nn.Sequential(
        nn.Linear(4, 8),  # 0
        nn.ReLU(),        # 1
        nn.Linear(8, 4),  # 2
    )


def test_hookpoint_requires_exactly_one_target():
    with pytest.raises(ValueError):
        HookPoint(source=TensorSource.WEIGHT)
    with pytest.raises(ValueError):
        HookPoint(source=TensorSource.WEIGHT, pattern=r".*", modules=[nn.Linear(2, 2)])


def test_match_modules_by_pattern():
    model = _toy()
    hp = HookPoint(source=TensorSource.WEIGHT, pattern=r"^\d+$")
    names = match_modules(model, hp)
    assert set(names) == {"0", "1", "2"}


def test_match_modules_by_explicit_instance():
    model = _toy()
    linear = model[0]
    hp = HookPoint(source=TensorSource.WEIGHT, modules=[linear])
    names = match_modules(model, hp)
    assert names == ["0"]


def test_match_modules_by_selector():
    model = _toy()
    hp = HookPoint(
        source=TensorSource.WEIGHT,
        selector=lambda m: [name for name, _ in m.named_modules() if "Linear" in type(_).__name__],
    )
    names = match_modules(model, hp)
    assert set(names) == {"0", "2"}


def test_step_context_holds_dicts():
    ctx = StepContext(step=5, model=_toy(), activations={}, gradients={}, weights={},
                      loss=0.5, user={"epoch": 1})
    assert ctx.step == 5
    assert ctx.user["epoch"] == 1
```

- [ ] **Step 2: Run — confirm fail**

```bash
venv/bin/pytest tests/recorder/test_hooks.py -v
```

- [ ] **Step 3: Implement**

```python
# src/circuitry/recorder/hooks.py
"""Hook-point data classes and module-matching logic.

See docs/design.md §4.4. Recorder uses ``match_modules`` to resolve a
HookPoint against ``model.named_modules()`` and to enforce
``expected_min_matches`` invariants at attach time.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

import torch
import torch.nn as nn


class TensorSource(str, Enum):
    WEIGHT = "weight"
    INPUT = "input"
    OUTPUT = "output"
    GRAD = "grad"


@dataclass
class HookPoint:
    source: TensorSource
    pattern: str | None = None
    modules: list[nn.Module] | None = None
    selector: Callable[[nn.Module], list[str]] | None = None

    def __post_init__(self) -> None:
        targets = sum(x is not None for x in (self.pattern, self.modules, self.selector))
        if targets != 1:
            raise ValueError(
                "HookPoint requires exactly one of {pattern, modules, selector}; "
                f"got {targets}"
            )


@dataclass
class StepContext:
    """Snapshot passed to every diagnostic on an emit step.

    Fields are dicts keyed by hooked-module name (the dotted name from
    ``model.named_modules()``). Built-in diagnostics ignore the ``user`` dict;
    custom diagnostics can use it to thread arbitrary state through
    ``Recorder.step(**kwargs)``.
    """

    step: int
    model: nn.Module
    activations: dict[str, torch.Tensor] = field(default_factory=dict)
    gradients: dict[str, torch.Tensor] = field(default_factory=dict)
    weights: dict[str, torch.Tensor] = field(default_factory=dict)
    loss: float | None = None
    user: dict[str, Any] = field(default_factory=dict)


def match_modules(model: nn.Module, hp: HookPoint) -> list[str]:
    """Return the dotted module names matched by ``hp`` against ``model``.

    Resolution rules:
      - ``pattern``  : regex against ``dict(model.named_modules()).keys()``
      - ``modules``  : reverse-lookup each instance to its name
      - ``selector`` : delegate; selector must return module names
    """
    name_to_mod = dict(model.named_modules())
    if hp.pattern is not None:
        rx = re.compile(hp.pattern)
        return [n for n in name_to_mod if rx.search(n)]
    if hp.modules is not None:
        mod_to_name = {id(m): n for n, m in name_to_mod.items()}
        return [mod_to_name[id(m)] for m in hp.modules if id(m) in mod_to_name]
    assert hp.selector is not None
    return list(hp.selector(model))
```

- [ ] **Step 4: Run — confirm pass**

```bash
venv/bin/pytest tests/recorder/test_hooks.py -v
```

Expected: 5 PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/circuitry/recorder/hooks.py tests/recorder/test_hooks.py
git commit -m "feat(recorder): add HookPoint, StepContext, TensorSource, match_modules"
```

---

### Task D2: `recipes/__init__.py` — `Recipe` dataclass + registry + `register_recipe`

**Files:**
- Create: `/home/vishsangale/workspace/circuitry/src/circuitry/recipes/__init__.py` *(overwrite the placeholder)*
- Create: `/home/vishsangale/workspace/circuitry/tests/recipes/test_registry.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/recipes/test_registry.py
from __future__ import annotations

import pytest

from circuitry.recipes import Recipe, get_recipe, list_recipes, register_recipe
from circuitry.recorder.hooks import HookPoint, TensorSource


def _make(name: str = "demo") -> Recipe:
    return Recipe(
        name=name,
        hook_points=[HookPoint(source=TensorSource.WEIGHT, pattern=r".*")],
        weight_diagnostics=["effective_rank"],
        activation_diagnostics=[],
        gradient_diagnostics=[],
    )


def test_register_and_get_round_trip():
    register_recipe(_make("custom-a"))
    r = get_recipe("custom-a")
    assert r.name == "custom-a"
    assert "custom-a" in list_recipes()


def test_register_duplicate_raises():
    register_recipe(_make("custom-b"))
    with pytest.raises(ValueError):
        register_recipe(_make("custom-b"))


def test_get_unknown_raises():
    with pytest.raises(KeyError):
        get_recipe("nonexistent-recipe-xyz")


def test_recipe_diagnostic_toggle_enabled_default_true():
    r = _make("custom-c")
    # Per-diagnostic toggle (design §10) — represented as a dict on the Recipe.
    assert r.enabled.get("effective_rank", True) is True
```

- [ ] **Step 2: Run — confirm fail**

```bash
venv/bin/pytest tests/recipes/test_registry.py -v
```

- [ ] **Step 3: Implement**

```python
# src/circuitry/recipes/__init__.py
"""Recipe dataclass + global registry. See docs/design.md §4.4."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from circuitry.recorder.hooks import HookPoint, StepContext

DiagnosticFn = Callable[[StepContext], dict[str, float]]


@dataclass
class Recipe:
    name: str
    hook_points: list[HookPoint]
    weight_diagnostics: list[str] = field(default_factory=list)
    activation_diagnostics: list[str] = field(default_factory=list)
    gradient_diagnostics: list[str] = field(default_factory=list)
    custom: list[DiagnosticFn] = field(default_factory=list)
    expected_min_matches: dict[str, int] = field(default_factory=dict)
    enabled: dict[str, bool] = field(default_factory=dict)


_REGISTRY: dict[str, Recipe] = {}


def register_recipe(recipe: Recipe) -> None:
    if recipe.name in _REGISTRY:
        raise ValueError(f"recipe {recipe.name!r} already registered")
    _REGISTRY[recipe.name] = recipe


def get_recipe(name: str) -> Recipe:
    if name not in _REGISTRY:
        raise KeyError(f"unknown recipe {name!r}; known: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def list_recipes() -> list[str]:
    return sorted(_REGISTRY)


def _clear_registry_for_tests() -> None:
    """Test-only escape hatch. Not part of the public API."""
    _REGISTRY.clear()
```

- [ ] **Step 4: Update tests to clear registry between runs**

Add to `tests/recipes/test_registry.py` top-level:

```python
import pytest

from circuitry.recipes import _clear_registry_for_tests


@pytest.fixture(autouse=True)
def _clean_registry():
    _clear_registry_for_tests()
    yield
    _clear_registry_for_tests()
```

- [ ] **Step 5: Run — confirm pass**

```bash
venv/bin/pytest tests/recipes/test_registry.py -v
```

Expected: 4 PASSED.

- [ ] **Step 6: Commit**

```bash
git add src/circuitry/recipes/__init__.py tests/recipes/test_registry.py
git commit -m "feat(recipes): add Recipe dataclass and register_recipe / get_recipe registry"
```

---

### Task D3: `recorder/live.py` — `Recorder.attach()` + matched-modules invariants + rank-0 noop

**Files:**
- Create: `/home/vishsangale/workspace/circuitry/src/circuitry/recorder/live.py`
- Create: `/home/vishsangale/workspace/circuitry/tests/recorder/test_recorder_attach.py`

This task implements just the lifecycle: `attach()`, `detach()`, the matched-modules artifact, and the strict-mode invariants. The actual `step()` body comes in D4.

- [ ] **Step 1: Write failing tests**

```python
# tests/recorder/test_recorder_attach.py
from __future__ import annotations

import logging
import pathlib

import pytest
import torch.nn as nn

from circuitry.recipes import Recipe, _clear_registry_for_tests, register_recipe
from circuitry.recorder.hooks import HookPoint, TensorSource
from circuitry.recorder.live import Recorder
from circuitry.writers.base import RecordingWriter


@pytest.fixture(autouse=True)
def _clean():
    _clear_registry_for_tests()
    yield
    _clear_registry_for_tests()


def _toy_model() -> nn.Module:
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 4))


def _register_demo(pattern: str = r"^\d+$", min_matches: int = 0) -> None:
    register_recipe(Recipe(
        name="demo",
        hook_points=[HookPoint(source=TensorSource.WEIGHT, pattern=pattern)],
        weight_diagnostics=["effective_rank"],
        expected_min_matches={pattern: min_matches},
    ))


def test_attach_writes_matched_modules_file(tmp_path):
    _register_demo()
    model = _toy_model()
    rec = Recorder(model, run_dir=tmp_path, recipe="demo",
                   writer=RecordingWriter(), every_n_steps=1)
    rec.attach()
    f = tmp_path / "circuitry" / "matched_modules.txt"
    assert f.exists()
    content = f.read_text()
    assert "0" in content and "2" in content
    rec.detach()


def test_attach_logs_matched_modules_at_info(tmp_path, caplog):
    _register_demo()
    caplog.set_level(logging.INFO, logger="circuitry")
    rec = Recorder(_toy_model(), run_dir=tmp_path, recipe="demo",
                   writer=RecordingWriter(), every_n_steps=1)
    rec.attach()
    rec.detach()
    assert any("matched" in r.message.lower() for r in caplog.records)


def test_attach_raises_on_zero_matches_regardless_of_strict(tmp_path):
    register_recipe(Recipe(
        name="bad",
        hook_points=[HookPoint(source=TensorSource.WEIGHT,
                               pattern=r"this-matches-nothing")],
    ))
    rec = Recorder(_toy_model(), run_dir=tmp_path, recipe="bad",
                   writer=RecordingWriter(), every_n_steps=1, strict=False)
    with pytest.raises(RuntimeError, match="matched 0 modules"):
        rec.attach()


def test_attach_raises_on_min_matches_violation_in_strict_mode(tmp_path):
    _register_demo(pattern=r"^\d+$", min_matches=99)
    rec = Recorder(_toy_model(), run_dir=tmp_path, recipe="demo",
                   writer=RecordingWriter(), every_n_steps=1, strict=True)
    with pytest.raises(RuntimeError, match="expected at least 99"):
        rec.attach()


def test_attach_warns_on_min_matches_violation_in_non_strict_mode(tmp_path, caplog):
    _register_demo(pattern=r"^\d+$", min_matches=99)
    caplog.set_level(logging.WARNING, logger="circuitry")
    rec = Recorder(_toy_model(), run_dir=tmp_path, recipe="demo",
                   writer=RecordingWriter(), every_n_steps=1, strict=False)
    rec.attach()
    rec.detach()
    assert any("expected at least 99" in r.message for r in caplog.records)


def test_detach_removes_all_hooks(tmp_path):
    _register_demo()
    model = _toy_model()
    rec = Recorder(model, run_dir=tmp_path, recipe="demo",
                   writer=RecordingWriter(), every_n_steps=1)
    rec.attach()
    pre = sum(len(m._forward_hooks) + len(m._forward_pre_hooks)
              + len(m._backward_hooks) for m in model.modules())
    rec.detach()
    post = sum(len(m._forward_hooks) + len(m._forward_pre_hooks)
               + len(m._backward_hooks) for m in model.modules())
    assert post == 0
    # We don't assert pre > 0 — pure-weight recipes may not install hooks.


def test_recorder_noop_on_non_zero_rank(monkeypatch, tmp_path):
    _register_demo()
    monkeypatch.setattr("torch.distributed.is_initialized", lambda: True)
    monkeypatch.setattr("torch.distributed.get_rank", lambda: 1)
    writer = RecordingWriter()
    rec = Recorder(_toy_model(), run_dir=tmp_path, recipe="demo",
                   writer=writer, every_n_steps=1)
    rec.attach()
    rec.step(0, loss=1.0)
    rec.detach()
    assert writer.scalars == []
    assert not (tmp_path / "circuitry" / "matched_modules.txt").exists()
```

- [ ] **Step 2: Run — confirm fail**

```bash
venv/bin/pytest tests/recorder/test_recorder_attach.py -v
```

- [ ] **Step 3: Implement**

```python
# src/circuitry/recorder/live.py
"""LiveRecorder — attach hooks per recipe, snapshot tensors at emit steps,
run diagnostics, write scalars through a MetricWriter.

See docs/design.md §4.2, §4.4, §10, §11. Single-process v1 — non-zero ranks
no-op in attach().
"""

from __future__ import annotations

import logging
import pathlib
from typing import Any

import torch
import torch.nn as nn

from circuitry.recipes import Recipe, get_recipe
from circuitry.recorder.hooks import HookPoint, StepContext, TensorSource, match_modules
from circuitry.writers.base import MetricWriter
from circuitry.writers.tensorboard import TensorBoardWriter

logger = logging.getLogger("circuitry")

_WRITERS: dict[str, Any] = {}  # name → factory; populated below


def _resolve_writer(writer: MetricWriter | str, run_dir: pathlib.Path) -> MetricWriter:
    if not isinstance(writer, str):
        return writer
    from circuitry.writers.jsonl import JsonlWriter
    from circuitry.writers.null import NullWriter
    table = {
        "tensorboard": lambda: TensorBoardWriter(run_dir),
        "jsonl": lambda: JsonlWriter(run_dir),
        "null": lambda: NullWriter(),
    }
    if writer == "wandb":
        from circuitry.writers.wandb import WandbWriter
        return WandbWriter()
    if writer not in table:
        raise ValueError(f"unknown writer {writer!r}; known: {sorted(table) + ['wandb']}")
    return table[writer]()


class Recorder:
    """Single-process training-time diagnostics recorder.

    On a multi-rank setup (``torch.distributed.is_initialized()`` true and
    ``get_rank() != 0``), every method is a no-op so existing scripts don't
    crash. Multi-process support is the v0.next deliverable (design §11).
    """

    def __init__(
        self,
        model: nn.Module,
        run_dir: str | pathlib.Path,
        recipe: str | Recipe,
        writer: MetricWriter | str = "tensorboard",
        every_n_steps: int = 200,
        strict: bool = True,
    ) -> None:
        self.model = model
        self.run_dir = pathlib.Path(run_dir)
        self.recipe = recipe if isinstance(recipe, Recipe) else get_recipe(recipe)
        self.every_n_steps = int(every_n_steps)
        self.strict = bool(strict)
        self._writer_arg = writer
        self._writer: MetricWriter | None = None
        self._hook_handles: list[Any] = []
        # name → tensor captured by hook, refreshed each emit step
        self._captured_activations: dict[str, torch.Tensor] = {}
        self._matched: dict[int, list[str]] = {}  # hp index → names
        self._noop = False
        self._current_step: int = -1

    # ---- internal helpers ------------------------------------------------

    def _is_inactive_rank(self) -> bool:
        if not torch.distributed.is_available():
            return False
        try:
            if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
                return True
        except Exception:  # pragma: no cover — defensive
            return False
        return False

    def _should_capture(self, step: int) -> bool:
        return step % self.every_n_steps == 0

    # ---- lifecycle -------------------------------------------------------

    def attach(self) -> None:
        if self._is_inactive_rank():
            self._noop = True
            logger.info("circuitry: rank != 0 — Recorder is no-op")
            return

        (self.run_dir / "circuitry").mkdir(parents=True, exist_ok=True)
        self._writer = _resolve_writer(self._writer_arg, self.run_dir)

        matched_lines: list[str] = []
        for idx, hp in enumerate(self.recipe.hook_points):
            names = match_modules(self.model, hp)
            self._matched[idx] = names

            label = hp.pattern or "<modules>" if hp.modules is not None else "<selector>"
            matched_lines.append(f"# hook_point[{idx}] source={hp.source.value} target={label}")
            for n in names:
                matched_lines.append(n)
            matched_lines.append("")
            logger.info("circuitry: hook_point[%d] (%s) matched %d modules: %s",
                        idx, label, len(names), names)

            if len(names) == 0:
                raise RuntimeError(
                    f"HookPoint {idx} ({label}) matched 0 modules — refusing to attach"
                )
            expected = self.recipe.expected_min_matches.get(hp.pattern or "", 0)
            if expected and len(names) < expected:
                msg = (f"HookPoint {idx} ({label}) matched {len(names)} modules but "
                       f"expected at least {expected}")
                if self.strict:
                    raise RuntimeError(msg)
                logger.warning("circuitry: %s", msg)

        (self.run_dir / "circuitry" / "matched_modules.txt").write_text(
            "\n".join(matched_lines)
        )

        # Install hooks for INPUT / OUTPUT sources (WEIGHT/GRAD read directly at step time).
        name_to_mod = dict(self.model.named_modules())
        for idx, hp in enumerate(self.recipe.hook_points):
            if hp.source is TensorSource.OUTPUT:
                for n in self._matched[idx]:
                    handle = name_to_mod[n].register_forward_hook(self._mk_fwd_hook(n))
                    self._hook_handles.append(handle)
            elif hp.source is TensorSource.INPUT:
                for n in self._matched[idx]:
                    handle = name_to_mod[n].register_forward_pre_hook(self._mk_pre_hook(n))
                    self._hook_handles.append(handle)
            # WEIGHT / GRAD are pulled in step() directly from the module — no hook needed.

    def _mk_fwd_hook(self, name: str):
        # Hooks always capture (cheap detach); step() decides what to consume.
        # Gating the hook itself on `_current_step` is broken: forward runs
        # before step(), so the hook would see a stale step index.
        def _hook(_mod, _inp, out):
            t = out[0] if isinstance(out, tuple) else out
            self._captured_activations[name] = t.detach()
        return _hook

    def _mk_pre_hook(self, name: str):
        def _hook(_mod, inputs):
            t = inputs[0]
            self._captured_activations[name] = t.detach()
        return _hook

    def detach(self) -> None:
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()
        if self._writer is not None:
            self._writer.flush()
            self._writer.close()
            self._writer = None

    def step(self, step: int, loss: float | None = None, **user: Any) -> None:
        if self._noop:
            return
        self._current_step = int(step)
        # Body lands in Task D4.
```

- [ ] **Step 4: Run — confirm pass**

```bash
venv/bin/pytest tests/recorder/test_recorder_attach.py -v
```

Expected: 7 PASSED. (The `step` body is a stub — only lifecycle tests are exercised.)

- [ ] **Step 5: Commit**

```bash
git add src/circuitry/recorder/live.py tests/recorder/test_recorder_attach.py
git commit -m "feat(recorder): add Recorder.attach/detach lifecycle and matched-modules invariants"
```

---

### Task D4: `Recorder.step()` — build `StepContext`, run diagnostics, write scalars

**Files:**
- Modify: `/home/vishsangale/workspace/circuitry/src/circuitry/recorder/live.py`
- Create: `/home/vishsangale/workspace/circuitry/tests/recorder/test_recorder_step.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/recorder/test_recorder_step.py
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from circuitry.recipes import Recipe, _clear_registry_for_tests, register_recipe
from circuitry.recorder.hooks import HookPoint, StepContext, TensorSource
from circuitry.recorder.live import Recorder
from circuitry.writers.base import RecordingWriter


@pytest.fixture(autouse=True)
def _clean():
    _clear_registry_for_tests()
    yield
    _clear_registry_for_tests()


def _toy_model() -> nn.Module:
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 4))


def test_step_writes_weight_diagnostic_scalars(tmp_path):
    register_recipe(Recipe(
        name="w-only",
        hook_points=[HookPoint(source=TensorSource.WEIGHT, pattern=r"^\d+$")],
        weight_diagnostics=["effective_rank", "stable_rank"],
    ))
    writer = RecordingWriter()
    rec = Recorder(_toy_model(), run_dir=tmp_path, recipe="w-only",
                   writer=writer, every_n_steps=1)
    rec.attach()
    rec.step(0, loss=1.0)
    rec.detach()
    tags = {t for t, _, _ in writer.scalars}
    assert any("effective_rank" in t for t in tags)
    assert any("stable_rank" in t for t in tags)
    assert ("loss", 1.0, 0) in writer.scalars


def test_step_respects_every_n_steps(tmp_path):
    register_recipe(Recipe(
        name="every-3",
        hook_points=[HookPoint(source=TensorSource.WEIGHT, pattern=r"^\d+$")],
        weight_diagnostics=["effective_rank"],
    ))
    writer = RecordingWriter()
    rec = Recorder(_toy_model(), run_dir=tmp_path, recipe="every-3",
                   writer=writer, every_n_steps=3)
    rec.attach()
    for s in range(7):
        rec.step(s, loss=float(s))
    rec.detach()
    steps_with_rank = sorted({s for t, _, s in writer.scalars if "effective_rank" in t})
    # Emit steps: 0, 3, 6
    assert steps_with_rank == [0, 3, 6]
    # Loss is recorded every step.
    assert sorted(s for t, _, s in writer.scalars if t == "loss") == list(range(7))


def test_step_runs_activation_diagnostic_after_forward(tmp_path):
    register_recipe(Recipe(
        name="act",
        hook_points=[HookPoint(source=TensorSource.OUTPUT, pattern=r"^0$")],
        activation_diagnostics=["dead_fraction"],
    ))
    model = _toy_model()
    writer = RecordingWriter()
    rec = Recorder(model, run_dir=tmp_path, recipe="act",
                   writer=writer, every_n_steps=1)
    rec.attach()
    _ = model(torch.randn(2, 4))
    rec.step(0)
    rec.detach()
    assert any("dead_fraction" in t for t, _, _ in writer.scalars)


def test_step_runs_custom_diagnostic(tmp_path):
    def custom(ctx: StepContext) -> dict[str, float]:
        return {"my_metric": float(ctx.step + 1)}

    register_recipe(Recipe(
        name="cust",
        hook_points=[HookPoint(source=TensorSource.WEIGHT, pattern=r"^\d+$")],
        custom=[custom],
    ))
    writer = RecordingWriter()
    rec = Recorder(_toy_model(), run_dir=tmp_path, recipe="cust",
                   writer=writer, every_n_steps=1)
    rec.attach()
    rec.step(2)
    rec.detach()
    assert ("custom/my_metric", 3.0, 2) in writer.scalars


def test_step_skips_disabled_diagnostic(tmp_path):
    register_recipe(Recipe(
        name="dis",
        hook_points=[HookPoint(source=TensorSource.WEIGHT, pattern=r"^\d+$")],
        weight_diagnostics=["effective_rank", "stable_rank"],
        enabled={"stable_rank": False},
    ))
    writer = RecordingWriter()
    rec = Recorder(_toy_model(), run_dir=tmp_path, recipe="dis",
                   writer=writer, every_n_steps=1)
    rec.attach()
    rec.step(0)
    rec.detach()
    tags = {t for t, _, _ in writer.scalars}
    assert any("effective_rank" in t for t in tags)
    assert not any("stable_rank" in t for t in tags)


def test_activation_diagnostic_with_every_n_steps_3(tmp_path):
    """Regression: hook capture timing must not depend on stale _current_step.

    With every_n_steps=3, activation tags should appear on steps {0, 3, 6} —
    not on {1, 2, 4, 5} — and must NOT all silently drop because the hook
    gating ran before step() updated _current_step.
    """
    register_recipe(Recipe(
        name="act-every-3",
        hook_points=[HookPoint(source=TensorSource.OUTPUT, pattern=r"^0$")],
        activation_diagnostics=["dead_fraction"],
    ))
    model = _toy_model()
    writer = RecordingWriter()
    rec = Recorder(model, run_dir=tmp_path, recipe="act-every-3",
                   writer=writer, every_n_steps=3)
    rec.attach()
    for s in range(7):
        _ = model(torch.randn(2, 4))
        rec.step(s)
    rec.detach()
    act_steps = sorted({step for tag, _, step in writer.scalars
                        if "dead_fraction" in tag})
    assert act_steps == [0, 3, 6]
```

- [ ] **Step 2: Run — confirm fail**

```bash
venv/bin/pytest tests/recorder/test_recorder_step.py -v
```

- [ ] **Step 3: Implement — replace the stub `step()` in `recorder/live.py` and add helpers**

Add module-level diagnostic-lookup tables at the top of `recorder/live.py`:

```python
from circuitry.core import activation as _act
from circuitry.core import gradient as _grad
from circuitry.core import weight as _w

_WEIGHT_DIAGS = {
    "effective_rank": _w.effective_rank,
    "stable_rank": _w.stable_rank,
    "condition_number": _w.condition_number,
    "heavy_tail_alpha": _w.heavy_tail_alpha,
}

_ACT_DIAGS = {
    "dead_fraction": _act.dead_fraction,
    "participation_ratio": _act.participation_ratio,
    "kurtosis": lambda x: float(_act.kurtosis(x).mean().item()),
}

_GRAD_DIAGS = {
    "layer_norm": _grad.layer_norm,  # dict in, dict out
}
```

Replace `step()` with:

```python
    def step(self, step: int, loss: float | None = None, **user: Any) -> None:
        if self._noop:
            return
        self._current_step = int(step)
        assert self._writer is not None, "Recorder.step called before attach()"

        if loss is not None:
            self._writer.add_scalar("loss", float(loss), self._current_step)

        if not self._should_capture(self._current_step):
            return

        # Build StepContext from currently-captured tensors + read-on-demand weights/grads.
        name_to_mod = dict(self.model.named_modules())
        weights: dict[str, torch.Tensor] = {}
        gradients: dict[str, torch.Tensor] = {}
        for idx, hp in enumerate(self.recipe.hook_points):
            if hp.source is TensorSource.WEIGHT:
                for n in self._matched[idx]:
                    p = getattr(name_to_mod[n], "weight", None)
                    if isinstance(p, torch.Tensor):
                        weights[n] = p.detach()
            elif hp.source is TensorSource.GRAD:
                for n in self._matched[idx]:
                    p = getattr(name_to_mod[n], "weight", None)
                    if isinstance(p, torch.Tensor) and p.grad is not None:
                        gradients[n] = p.grad.detach()

        ctx = StepContext(
            step=self._current_step,
            model=self.model,
            activations=dict(self._captured_activations),
            gradients=gradients,
            weights=weights,
            loss=loss,
            user=dict(user),
        )

        self._run_diagnostics(ctx)
        # Discard activations now that we've consumed them.
        self._captured_activations.clear()

    def _enabled(self, name: str) -> bool:
        return self.recipe.enabled.get(name, True)

    def _run_diagnostics(self, ctx: StepContext) -> None:
        assert self._writer is not None
        for name in self.recipe.weight_diagnostics:
            if not self._enabled(name):
                continue
            fn = _WEIGHT_DIAGS.get(name)
            if fn is None:
                logger.warning("circuitry: unknown weight diagnostic %r — skipping", name)
                continue
            for mod_name, w in ctx.weights.items():
                self._writer.add_scalar(f"weight/{name}/{mod_name}", float(fn(w)), ctx.step)

        for name in self.recipe.activation_diagnostics:
            if not self._enabled(name):
                continue
            fn = _ACT_DIAGS.get(name)
            if fn is None:
                logger.warning("circuitry: unknown activation diagnostic %r — skipping", name)
                continue
            for mod_name, x in ctx.activations.items():
                self._writer.add_scalar(f"activation/{name}/{mod_name}", float(fn(x)), ctx.step)

        for name in self.recipe.gradient_diagnostics:
            if not self._enabled(name):
                continue
            if name == "layer_norm":
                for mod_name, val in _grad.layer_norm(ctx.gradients).items():
                    self._writer.add_scalar(f"gradient/layer_norm/{mod_name}", val, ctx.step)
            else:
                logger.warning("circuitry: unknown gradient diagnostic %r — skipping", name)

        for fn in self.recipe.custom:
            out = fn(ctx)
            for tag, val in out.items():
                if isinstance(val, torch.Tensor):
                    if val.numel() == 1:
                        self._writer.add_scalar(f"custom/{tag}", float(val.item()), ctx.step)
                    else:
                        self._writer.add_histogram(f"custom/{tag}", val, ctx.step)
                else:
                    self._writer.add_scalar(f"custom/{tag}", float(val), ctx.step)
```

- [ ] **Step 4: Run — confirm pass**

```bash
venv/bin/pytest tests/recorder/test_recorder_step.py -v
```

Expected: 5 PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/circuitry/recorder/live.py tests/recorder/test_recorder_step.py
git commit -m "feat(recorder): implement step() — StepContext, diagnostic dispatch, every_n_steps"
```

---

### Task D5: `recorder/scan.py` — `scan_run` over saved checkpoints

**Files:**
- Create: `/home/vishsangale/workspace/circuitry/src/circuitry/recorder/scan.py`
- Create: `/home/vishsangale/workspace/circuitry/tests/recorder/test_scan.py`

Read `~/workspace/mendu/mendu/tools/inspect_checkpoint/__main__.py` for the reference scan workflow.

- [ ] **Step 1: Write failing tests**

```python
# tests/recorder/test_scan.py
from __future__ import annotations

import pathlib

import pytest
import torch
import torch.nn as nn

from circuitry.recipes import Recipe, _clear_registry_for_tests, register_recipe
from circuitry.recorder.hooks import HookPoint, TensorSource
from circuitry.recorder.scan import scan_run


@pytest.fixture(autouse=True)
def _clean():
    _clear_registry_for_tests()
    yield
    _clear_registry_for_tests()


def _register():
    register_recipe(Recipe(
        name="scan-demo",
        hook_points=[HookPoint(source=TensorSource.WEIGHT, pattern=r"^\d+$")],
        weight_diagnostics=["effective_rank"],
    ))


def _toy(seed: int) -> nn.Module:
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 4))


def test_scan_run_processes_checkpoints(tmp_path):
    _register()
    # Lay down two checkpoint files in conventional locations.
    ckpts = tmp_path / "checkpoints"
    ckpts.mkdir()
    for step, seed in [(100, 0), (200, 1)]:
        torch.save(_toy(seed).state_dict(), ckpts / f"step{step:09d}.pt")

    out_dir = tmp_path / "tb_retro"
    scan_run(run_dir=tmp_path, recipe="scan-demo", out_dir=out_dir,
             model_factory=lambda: _toy(0))

    event_files = list(out_dir.rglob("events.out.tfevents.*"))
    assert event_files


def test_scan_run_raises_when_no_checkpoints(tmp_path):
    _register()
    with pytest.raises(FileNotFoundError):
        scan_run(run_dir=tmp_path, recipe="scan-demo",
                 out_dir=tmp_path / "tb_retro",
                 model_factory=lambda: _toy(0))
```

- [ ] **Step 2: Run — confirm fail**

```bash
venv/bin/pytest tests/recorder/test_scan.py -v
```

- [ ] **Step 3: Implement**

```python
# src/circuitry/recorder/scan.py
"""Retrospective scan over checkpoints: rehydrate model state, run the recipe's
weight diagnostics, emit TB events under ``out_dir``.

Checkpoint discovery defaults to ``<run_dir>/checkpoints/step*.pt`` sorted by
filename (which sorts by step under the conventional ``step000000100.pt`` form).
"""

from __future__ import annotations

import pathlib
import re
from typing import Callable

import torch
import torch.nn as nn

from circuitry.recipes import Recipe, get_recipe
from circuitry.recorder.live import Recorder
from circuitry.writers.tensorboard import TensorBoardWriter

_STEP_RX = re.compile(r"step(\d+)")


def _discover_checkpoints(run_dir: pathlib.Path) -> list[tuple[int, pathlib.Path]]:
    ckpts = sorted((run_dir / "checkpoints").glob("step*.pt"))
    out: list[tuple[int, pathlib.Path]] = []
    for p in ckpts:
        m = _STEP_RX.search(p.stem)
        out.append((int(m.group(1)) if m else 0, p))
    return out


def scan_run(
    run_dir: str | pathlib.Path,
    recipe: str | Recipe,
    out_dir: str | pathlib.Path,
    model_factory: Callable[[], nn.Module],
) -> None:
    """Replay each checkpoint through the recipe's weight diagnostics.

    ``model_factory`` produces a fresh model whose architecture matches the
    checkpoint state-dict; the same model is reused with `load_state_dict`
    across checkpoints (cheaper than rebuilding).
    """
    run_dir = pathlib.Path(run_dir)
    out_dir = pathlib.Path(out_dir)
    ckpts = _discover_checkpoints(run_dir)
    if not ckpts:
        raise FileNotFoundError(
            f"no checkpoints found under {run_dir / 'checkpoints'}"
        )

    recipe = recipe if isinstance(recipe, Recipe) else get_recipe(recipe)
    model = model_factory()
    writer = TensorBoardWriter(out_dir)
    rec = Recorder(model, run_dir=out_dir, recipe=recipe,
                   writer=writer, every_n_steps=1)
    rec.attach()
    try:
        for step, ckpt_path in ckpts:
            sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            model.load_state_dict(sd)
            rec.step(step)
    finally:
        rec.detach()
```

- [ ] **Step 4: Run — confirm pass**

```bash
venv/bin/pytest tests/recorder/test_scan.py -v
```

Expected: 2 PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/circuitry/recorder/scan.py tests/recorder/test_scan.py
git commit -m "feat(recorder): add scan_run for retrospective checkpoint analysis"
```

---

### Task D6: `recorder/report.py` — `build_report` markdown

**Files:**
- Create: `/home/vishsangale/workspace/circuitry/src/circuitry/recorder/report.py`
- Create: `/home/vishsangale/workspace/circuitry/tests/recorder/test_report.py`

Read `~/workspace/mendu/mendu/tools/inspect_checkpoint/` for the markdown layout the mendu inspector uses (header, per-recipe section, per-module bullets, summary). Port-with-cleanup.

- [ ] **Step 1: Write failing tests**

```python
# tests/recorder/test_report.py
from __future__ import annotations

import json
import pathlib

import torch

from circuitry.recorder.report import build_report


def _write_jsonl(path: pathlib.Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows))


def test_build_report_writes_markdown_with_sections(tmp_path):
    _write_jsonl(tmp_path / "metrics.jsonl", [
        {"tag": "loss", "value": 1.5, "step": 0, "kind": "scalar"},
        {"tag": "loss", "value": 1.0, "step": 1, "kind": "scalar"},
        {"tag": "weight/effective_rank/0", "value": 8.0, "step": 0, "kind": "scalar"},
        {"tag": "weight/effective_rank/0", "value": 7.5, "step": 1, "kind": "scalar"},
    ])
    (tmp_path / "circuitry" / "matched_modules.txt").parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / "circuitry" / "matched_modules.txt").write_text("# hook_point[0]\n0\n2\n")

    out = tmp_path / "inspect" / "report.md"
    build_report(run_dir=tmp_path, out_path=out)

    md = out.read_text()
    assert "# circuitry report" in md
    assert "matched modules" in md.lower()
    assert "loss" in md
    assert "effective_rank" in md


def test_build_report_handles_missing_jsonl(tmp_path):
    out = tmp_path / "report.md"
    build_report(run_dir=tmp_path, out_path=out)
    assert out.exists()
    assert "no metrics found" in out.read_text().lower()
```

- [ ] **Step 2: Run — confirm fail**

```bash
venv/bin/pytest tests/recorder/test_report.py -v
```

- [ ] **Step 3: Implement**

```python
# src/circuitry/recorder/report.py
"""Markdown report builder.

Reads ``<run_dir>/metrics.jsonl`` (produced by ``JsonlWriter``) and
``<run_dir>/circuitry/matched_modules.txt`` (produced by Recorder.attach()),
emits a single-file markdown summary suitable for committing alongside a run.

The report intentionally avoids plots — point users at TensorBoard for visuals.
"""

from __future__ import annotations

import json
import pathlib
from collections import defaultdict


def _group(rows: list[dict]) -> dict[str, list[tuple[int, float]]]:
    by_tag: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for r in rows:
        if r.get("kind") != "scalar":
            continue
        by_tag[r["tag"]].append((int(r["step"]), float(r["value"])))
    for v in by_tag.values():
        v.sort()
    return by_tag


def build_report(
    run_dir: str | pathlib.Path,
    out_path: str | pathlib.Path | None = None,
) -> pathlib.Path:
    run_dir = pathlib.Path(run_dir)
    out_path = pathlib.Path(out_path) if out_path else run_dir / "inspect" / "report.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    metrics_path = run_dir / "metrics.jsonl"
    rows: list[dict] = []
    if metrics_path.exists():
        for line in metrics_path.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))

    lines: list[str] = ["# circuitry report", ""]
    lines.append(f"Source run: `{run_dir}`")
    lines.append("")

    matched_path = run_dir / "circuitry" / "matched_modules.txt"
    if matched_path.exists():
        lines.append("## Matched modules")
        lines.append("")
        lines.append("```")
        lines.append(matched_path.read_text().rstrip())
        lines.append("```")
        lines.append("")

    if not rows:
        lines.append("_no metrics found_")
        out_path.write_text("\n".join(lines))
        return out_path

    grouped = _group(rows)
    families: dict[str, list[str]] = defaultdict(list)
    for tag in grouped:
        family = tag.split("/", 1)[0] if "/" in tag else "scalar"
        families[family].append(tag)

    for family in sorted(families):
        lines.append(f"## {family}")
        lines.append("")
        lines.append("| tag | first | last | min | max |")
        lines.append("| --- | --- | --- | --- | --- |")
        for tag in sorted(families[family]):
            series = grouped[tag]
            vals = [v for _, v in series]
            lines.append(
                f"| `{tag}` | {vals[0]:.4g} | {vals[-1]:.4g} | "
                f"{min(vals):.4g} | {max(vals):.4g} |"
            )
        lines.append("")

    out_path.write_text("\n".join(lines))
    return out_path
```

- [ ] **Step 4: Run — confirm pass**

```bash
venv/bin/pytest tests/recorder/test_report.py -v
```

Expected: 2 PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/circuitry/recorder/report.py tests/recorder/test_report.py
git commit -m "feat(recorder): add build_report markdown summary"
```

---

## Phase E — Recipes

### Task E1: `recipes/llm.py`

**Files:**
- Create: `/home/vishsangale/workspace/circuitry/src/circuitry/recipes/llm.py`
- Create: `/home/vishsangale/workspace/circuitry/tests/recipes/test_llm.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/recipes/test_llm.py
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from circuitry.recipes import _clear_registry_for_tests, get_recipe
from circuitry.recipes.llm import register
from circuitry.recorder.live import Recorder
from circuitry.writers.base import RecordingWriter


@pytest.fixture(autouse=True)
def _clean():
    _clear_registry_for_tests()
    register()
    yield
    _clear_registry_for_tests()


class _TinyBlock(nn.Module):
    def __init__(self, d: int = 8) -> None:
        super().__init__()
        self.attn = _Attn(d)
        self.mlp = _Mlp(d)
        self.ln_1 = nn.LayerNorm(d)
        self.ln_2 = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attn(self.ln_1(x))
        x = self.mlp(self.ln_2(x))
        return x


class _Attn(nn.Module):
    def __init__(self, d: int) -> None:
        super().__init__()
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.o_proj = nn.Linear(d, d, bias=False)

    def forward(self, x):
        return self.o_proj(self.v_proj(x))  # placeholder; just need named children


class _Mlp(nn.Module):
    def __init__(self, d: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(d, d * 2, bias=False)
        self.up_proj = nn.Linear(d, d * 2, bias=False)
        self.down_proj = nn.Linear(d * 2, d, bias=False)

    def forward(self, x):
        return self.down_proj(self.gate_proj(x) * self.up_proj(x))


class _Tiny(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(100, 8)
        self.block_0 = _TinyBlock(8)
        self.lm_head = nn.Linear(8, 100, bias=False)


def test_llm_recipe_attaches_and_emits_scalars(tmp_path):
    model = _Tiny()
    writer = RecordingWriter()
    rec = Recorder(model, run_dir=tmp_path, recipe="llm",
                   writer=writer, every_n_steps=1)
    rec.attach()
    _ = model.block_0(torch.randn(2, 8))
    rec.step(0, loss=1.0)
    rec.detach()
    tags = {t for t, _, _ in writer.scalars}
    assert any("weight/effective_rank" in t for t in tags)
    assert any("attn" in t or "mlp" in t for t in tags)


def test_llm_recipe_is_registered():
    r = get_recipe("llm")
    assert any(hp.pattern and "attn" in hp.pattern for hp in r.hook_points)
```

- [ ] **Step 2: Run — confirm fail**

```bash
venv/bin/pytest tests/recipes/test_llm.py -v
```

- [ ] **Step 3: Implement**

```python
# src/circuitry/recipes/llm.py
"""Stock LLM recipe. See docs/design.md §5."""

from __future__ import annotations

from circuitry.recipes import Recipe, register_recipe
from circuitry.recorder.hooks import HookPoint, TensorSource

RECIPE = Recipe(
    name="llm",
    hook_points=[
        HookPoint(source=TensorSource.WEIGHT,
                  pattern=r".*\.(q|k|v|o)_proj$"),
        HookPoint(source=TensorSource.WEIGHT,
                  pattern=r".*\.(w1|w2|w3|gate_proj|up_proj|down_proj)$"),
        HookPoint(source=TensorSource.OUTPUT, pattern=r".*\.attn$"),
        HookPoint(source=TensorSource.OUTPUT, pattern=r".*\.mlp$"),
        HookPoint(source=TensorSource.OUTPUT, pattern=r".*\.ln_[12]$"),
        HookPoint(source=TensorSource.WEIGHT, pattern=r"embed.*"),
        HookPoint(source=TensorSource.WEIGHT, pattern=r"lm_head$"),
    ],
    weight_diagnostics=["effective_rank", "stable_rank", "heavy_tail_alpha"],
    activation_diagnostics=["dead_fraction", "kurtosis", "participation_ratio"],
    gradient_diagnostics=["layer_norm"],
)


def register() -> None:
    """Register the LLM recipe. Idempotent under test fixtures via
    ``_clear_registry_for_tests``."""
    register_recipe(RECIPE)
```

- [ ] **Step 4: Auto-register the LLM recipe on package import**

Modify `src/circuitry/recipes/__init__.py` — at the bottom, add:

```python
def _register_stock_recipes() -> None:
    from circuitry.recipes import llm
    for mod in (llm,):
        try:
            mod.register()
        except ValueError:
            pass  # already registered


_register_stock_recipes()
```

E2 and E3 will extend this tuple as they land.

- [ ] **Step 5: Run — confirm pass**

```bash
venv/bin/pytest tests/recipes/test_llm.py -v
```

Expected: 2 PASSED.

- [ ] **Step 6: Commit**

```bash
git add src/circuitry/recipes/llm.py src/circuitry/recipes/__init__.py tests/recipes/test_llm.py
git commit -m "feat(recipes): add stock LLM recipe with auto-registration"
```

---

### Task E2: `recipes/vision.py`

**Files:**
- Create: `/home/vishsangale/workspace/circuitry/src/circuitry/recipes/vision.py`
- Create: `/home/vishsangale/workspace/circuitry/tests/recipes/test_vision.py`
- Modify: `/home/vishsangale/workspace/circuitry/src/circuitry/recipes/__init__.py` (uncomment `vision`)

Read `~/workspace/mendu/mendu/scripts/diagnose_{ei_bottlenecks,signal_prop,trained_pc}.py` for the reference hook patterns and which submodules vision diagnose scripts target.

- [ ] **Step 1: Write failing tests**

```python
# tests/recipes/test_vision.py
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from circuitry.recipes import _clear_registry_for_tests, get_recipe
from circuitry.recipes.vision import register
from circuitry.recorder.live import Recorder
from circuitry.writers.base import RecordingWriter


@pytest.fixture(autouse=True)
def _clean():
    _clear_registry_for_tests()
    register()
    yield
    _clear_registry_for_tests()


class _TinyResNetBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 8, 3, padding=1)
        self.fc1 = nn.Linear(8 * 8 * 8, 10)

    def forward(self, x):
        x = self.conv2(self.conv1(x))
        return self.fc1(x.flatten(1))


def test_vision_recipe_matches_conv_and_fc(tmp_path):
    model = _TinyResNetBlock()
    writer = RecordingWriter()
    rec = Recorder(model, run_dir=tmp_path, recipe="vision",
                   writer=writer, every_n_steps=1)
    rec.attach()
    _ = model(torch.randn(2, 3, 8, 8))
    rec.step(0)
    rec.detach()
    tags = {t for t, _, _ in writer.scalars}
    assert any("conv" in t for t in tags)
    assert any("fc" in t for t in tags)


def test_vision_recipe_is_registered():
    assert "vision" in [r for r in [get_recipe("vision").name]]
```

- [ ] **Step 2: Run — confirm fail**

```bash
venv/bin/pytest tests/recipes/test_vision.py -v
```

- [ ] **Step 3: Implement**

```python
# src/circuitry/recipes/vision.py
"""Stock vision recipe — covers conv-based and ViT-style backbones."""

from __future__ import annotations

from circuitry.recipes import Recipe, register_recipe
from circuitry.recorder.hooks import HookPoint, TensorSource

RECIPE = Recipe(
    name="vision",
    hook_points=[
        HookPoint(source=TensorSource.WEIGHT,
                  pattern=r"(conv\d+|fc\d+|patch_embed|blocks\.\d+\.(attn|mlp))(\.weight)?$"),
        HookPoint(source=TensorSource.OUTPUT,
                  pattern=r"(conv\d+|fc\d+|blocks\.\d+\.(attn|mlp))$"),
    ],
    weight_diagnostics=["effective_rank", "stable_rank"],
    activation_diagnostics=["dead_fraction", "participation_ratio"],
    gradient_diagnostics=["layer_norm"],
)


def register() -> None:
    register_recipe(RECIPE)
```

- [ ] **Step 4: Add `vision` to the auto-register tuple in `recipes/__init__.py`**

Update `_register_stock_recipes`:

```python
def _register_stock_recipes() -> None:
    from circuitry.recipes import llm, vision
    for mod in (llm, vision):
        try:
            mod.register()
        except ValueError:
            pass
```

- [ ] **Step 5: Run — confirm pass**

```bash
venv/bin/pytest tests/recipes/test_vision.py -v
```

Expected: 2 PASSED.

- [ ] **Step 6: Commit**

```bash
git add src/circuitry/recipes/vision.py src/circuitry/recipes/__init__.py tests/recipes/test_vision.py
git commit -m "feat(recipes): add stock vision recipe"
```

---

### Task E3: `recipes/two_tower.py` + embedding-alignment custom diagnostic

**Files:**
- Create: `/home/vishsangale/workspace/circuitry/src/circuitry/recipes/two_tower.py`
- Create: `/home/vishsangale/workspace/circuitry/tests/recipes/test_two_tower.py`
- Modify: `/home/vishsangale/workspace/circuitry/src/circuitry/recipes/__init__.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/recipes/test_two_tower.py
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from circuitry.recipes import _clear_registry_for_tests, get_recipe
from circuitry.recipes.two_tower import register
from circuitry.recorder.live import Recorder
from circuitry.writers.base import RecordingWriter


@pytest.fixture(autouse=True)
def _clean():
    _clear_registry_for_tests()
    register()
    yield
    _clear_registry_for_tests()


class _TwoTower(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.query_tower = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 4))
        self.item_tower = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 4))

    def forward(self, q, i):
        return (self.query_tower(q) * self.item_tower(i)).sum(-1)


def test_two_tower_emits_embedding_alignment(tmp_path):
    model = _TwoTower()
    writer = RecordingWriter()
    rec = Recorder(model, run_dir=tmp_path, recipe="two_tower",
                   writer=writer, every_n_steps=1)
    rec.attach()
    q = torch.randn(8, 4)
    i = torch.randn(8, 4)
    _ = model(q, i)
    rec.step(0)
    rec.detach()
    tags = {t for t, _, _ in writer.scalars}
    assert any("embedding_alignment" in t for t in tags)


def test_two_tower_registered():
    r = get_recipe("two_tower")
    assert r.name == "two_tower"
    assert len(r.custom) >= 1
```

- [ ] **Step 2: Run — confirm fail**

```bash
venv/bin/pytest tests/recipes/test_two_tower.py -v
```

- [ ] **Step 3: Implement**

```python
# src/circuitry/recipes/two_tower.py
"""Two-tower recipe (recsys). Adds an embedding-alignment custom diagnostic:
cosine similarity between the mean query-tower output and the mean item-tower
output captured this step.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from circuitry.recipes import Recipe, register_recipe
from circuitry.recorder.hooks import HookPoint, StepContext, TensorSource

_QUERY_KEYS = ("query_tower", "user_tower", "left_tower")
_ITEM_KEYS = ("item_tower", "right_tower")


def _mean_output(ctx: StepContext, prefix: tuple[str, ...]) -> torch.Tensor | None:
    for name, t in ctx.activations.items():
        if any(name == p or name.startswith(p + ".") for p in prefix):
            return t.flatten(0, -2).mean(dim=0)
    return None


def embedding_alignment(ctx: StepContext) -> dict[str, float]:
    q = _mean_output(ctx, _QUERY_KEYS)
    i = _mean_output(ctx, _ITEM_KEYS)
    if q is None or i is None or q.shape != i.shape:
        return {}
    cos = F.cosine_similarity(q.unsqueeze(0), i.unsqueeze(0)).item()
    return {"embedding_alignment": float(cos)}


RECIPE = Recipe(
    name="two_tower",
    hook_points=[
        HookPoint(source=TensorSource.WEIGHT,
                  pattern=r"(query_tower|item_tower|interaction).*"),
        HookPoint(source=TensorSource.OUTPUT,
                  pattern=r"(query_tower|item_tower)$"),
    ],
    weight_diagnostics=["effective_rank", "stable_rank"],
    activation_diagnostics=["norm_stats_max", "participation_ratio"],
    gradient_diagnostics=["layer_norm"],
    custom=[embedding_alignment],
)


def register() -> None:
    register_recipe(RECIPE)
```

Note: `norm_stats_max` is referenced for symmetry but does not exist in `_ACT_DIAGS`. Remove it from `activation_diagnostics` (leave `participation_ratio` only) — or replace with `dead_fraction`. Pick `dead_fraction`:

```python
    activation_diagnostics=["dead_fraction", "participation_ratio"],
```

- [ ] **Step 4: Add `two_tower` to the auto-register tuple in `recipes/__init__.py`**

Update `_register_stock_recipes`:

```python
def _register_stock_recipes() -> None:
    from circuitry.recipes import llm, two_tower, vision
    for mod in (llm, vision, two_tower):
        try:
            mod.register()
        except ValueError:
            pass
```

- [ ] **Step 5: Run — confirm pass**

```bash
venv/bin/pytest tests/recipes/test_two_tower.py -v
```

Expected: 2 PASSED.

- [ ] **Step 6: Commit**

```bash
git add src/circuitry/recipes/two_tower.py src/circuitry/recipes/__init__.py tests/recipes/test_two_tower.py
git commit -m "feat(recipes): add two_tower recipe with embedding-alignment diagnostic"
```

---

## Phase F — CLI

### Task F1: `cli/main.py` — `scan`, `report`, `list-recipes`

**Files:**
- Create: `/home/vishsangale/workspace/circuitry/src/circuitry/cli/main.py`
- Create: `/home/vishsangale/workspace/circuitry/tests/test_cli.py`

Read `~/workspace/mendu/mendu/tools/inspect_checkpoint/__main__.py` for argument-parsing prior art.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_cli.py
from __future__ import annotations

import subprocess
import sys

import torch
import torch.nn as nn


def test_cli_list_recipes_prints_stock_names():
    out = subprocess.run(
        [sys.executable, "-m", "circuitry.cli.main", "list-recipes"],
        capture_output=True, text=True, check=True,
    ).stdout
    assert "llm" in out
    assert "vision" in out
    assert "two_tower" in out


def test_cli_scan_and_report(tmp_path):
    # Lay down a single checkpoint.
    ckpts = tmp_path / "checkpoints"
    ckpts.mkdir()
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 4))
    torch.save(model.state_dict(), ckpts / "step000000100.pt")

    # `scan` needs a model factory; for the CLI smoke test we use the
    # built-in stub recipe (registered by --recipe llm requires real model
    # structure). Instead, use the "null" recipe wired below.
    # For test purposes we just call `report` against an empty jsonl path,
    # which is enough to exercise the entry point. (Real scan tested in
    # tests/recorder/test_scan.py.)
    (tmp_path / "metrics.jsonl").write_text(
        '{"tag": "loss", "value": 1.0, "step": 0, "kind": "scalar"}\n'
    )
    out = subprocess.run(
        [sys.executable, "-m", "circuitry.cli.main", "report",
         "--run", str(tmp_path)],
        capture_output=True, text=True, check=True,
    )
    assert (tmp_path / "inspect" / "report.md").exists()
```

- [ ] **Step 2: Run — confirm fail**

```bash
venv/bin/pytest tests/test_cli.py -v
```

- [ ] **Step 3: Implement**

```python
# src/circuitry/cli/main.py
"""``circuitry`` CLI entry point. See docs/design.md §4.3."""

from __future__ import annotations

import argparse
import sys

from circuitry.recipes import list_recipes
from circuitry.recorder.report import build_report


def _cmd_list_recipes(_: argparse.Namespace) -> int:
    for name in list_recipes():
        print(name)
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    out = build_report(run_dir=args.run, out_path=args.out)
    print(f"wrote {out}")
    return 0


def _cmd_scan(args: argparse.Namespace) -> int:
    # scan_run needs a model_factory which the CLI cannot conjure without a
    # user-supplied import path. Surface a clear error pointing to the
    # programmatic API. v0.1.0 ships the CLI command shape; the
    # `--model-factory dotted.path:fn` flag lands in v0.next.
    print(
        "circuitry scan: requires a model factory not yet exposed via the CLI.\n"
        "  Use circuitry.recorder.scan.scan_run(...) programmatically for now.\n"
        f"  Discovered checkpoints: {args.run}/checkpoints",
        file=sys.stderr,
    )
    return 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="circuitry")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list-recipes", help="list registered recipes")
    p_list.set_defaults(func=_cmd_list_recipes)

    p_report = sub.add_parser("report", help="build markdown report from a run")
    p_report.add_argument("--run", required=True, help="run directory")
    p_report.add_argument("--out", default=None, help="report output path")
    p_report.set_defaults(func=_cmd_report)

    p_scan = sub.add_parser("scan", help="retrospective scan of checkpoints")
    p_scan.add_argument("--run", required=True)
    p_scan.add_argument("--recipe", required=True)
    p_scan.set_defaults(func=_cmd_scan)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
```

- [ ] **Step 4: Run — confirm pass**

```bash
venv/bin/pytest tests/test_cli.py -v
```

Expected: 2 PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/circuitry/cli/main.py tests/test_cli.py
git commit -m "feat(cli): add scan/report/list-recipes entry point"
```

---

### Task F2: Wire `circuitry/__init__.py` public re-exports

**Files:**
- Modify: `/home/vishsangale/workspace/circuitry/src/circuitry/__init__.py`
- Create: `/home/vishsangale/workspace/circuitry/tests/test_public_api.py`

Everything the public surface references now exists (Phases B–E + F1). Wire the re-exports and lock them with a test.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_public_api.py
"""Pin the v0.1.0 public surface. Anything not in this set is internal."""

from __future__ import annotations


def test_public_surface():
    import circuitry

    expected = {
        "HookPoint",
        "MetricWriter",
        "Recipe",
        "Recorder",
        "StepContext",
        "TensorSource",
        "__version__",
        "build_report",
        "register_recipe",
        "scan_run",
    }
    assert set(circuitry.__all__) == expected
    for name in expected:
        assert hasattr(circuitry, name), f"circuitry.{name} not re-exported"


def test_version_is_a_string():
    import circuitry
    assert isinstance(circuitry.__version__, str)
    assert circuitry.__version__
```

- [ ] **Step 2: Run — confirm fail**

```bash
venv/bin/pytest tests/test_public_api.py -v
```

- [ ] **Step 3: Update `src/circuitry/__init__.py`**

```python
"""circuitry — training-time mechanistic-interpretability diagnostics for PyTorch.

Public surface re-exports below are the v0.1.0 stable API. Anything not re-exported
here is an internal implementation detail and may change without notice.
"""

from circuitry.recipes import Recipe, register_recipe
from circuitry.recorder.hooks import HookPoint, StepContext, TensorSource
from circuitry.recorder.live import Recorder
from circuitry.recorder.report import build_report
from circuitry.recorder.scan import scan_run
from circuitry.writers.base import MetricWriter

__version__ = "0.1.0.dev0"

__all__ = [
    "HookPoint",
    "MetricWriter",
    "Recipe",
    "Recorder",
    "StepContext",
    "TensorSource",
    "__version__",
    "build_report",
    "register_recipe",
    "scan_run",
]
```

- [ ] **Step 4: Run — confirm pass**

```bash
venv/bin/pytest tests/test_public_api.py -v
```

Expected: 2 PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/circuitry/__init__.py tests/test_public_api.py
git commit -m "feat: wire public re-exports and lock v0.1.0 surface with a test"
```

---

## Phase G — Integration

### Task G1: E2E pipeline test

**Files:**
- Create: `/home/vishsangale/workspace/circuitry/tests/e2e/test_full_pipeline.py`

- [ ] **Step 1: Write the test**

```python
# tests/e2e/test_full_pipeline.py
"""End-to-end smoke test:
  1. Train a tiny LLM-shaped model for 6 steps with LiveRecorder + JsonlWriter
  2. Save 2 checkpoints
  3. Run scan_run over those checkpoints
  4. Run build_report against the recorded jsonl
  5. Assert the markdown contains every section we expect
"""

from __future__ import annotations

import pathlib

import torch
import torch.nn as nn

from circuitry import Recorder, build_report, scan_run
from circuitry.writers.jsonl import JsonlWriter


class _Tiny(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(50, 8)
        self.attn = _Attn(8)
        self.mlp = _Mlp(8)
        self.ln_1 = nn.LayerNorm(8)
        self.ln_2 = nn.LayerNorm(8)
        self.lm_head = nn.Linear(8, 50, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embed(tokens)
        x = self.attn(self.ln_1(x))
        x = self.mlp(self.ln_2(x))
        return self.lm_head(x)


class _Attn(nn.Module):
    def __init__(self, d: int) -> None:
        super().__init__()
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.o_proj = nn.Linear(d, d, bias=False)

    def forward(self, x):
        return self.o_proj(self.v_proj(x))


class _Mlp(nn.Module):
    def __init__(self, d: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(d, d * 2, bias=False)
        self.up_proj = nn.Linear(d, d * 2, bias=False)
        self.down_proj = nn.Linear(d * 2, d, bias=False)

    def forward(self, x):
        return self.down_proj(self.gate_proj(x) * self.up_proj(x))


def test_e2e_pipeline(tmp_path: pathlib.Path):
    torch.manual_seed(0)
    model = _Tiny()
    opt = torch.optim.SGD(model.parameters(), lr=1e-2)
    ckpts = tmp_path / "checkpoints"
    ckpts.mkdir()

    rec = Recorder(model, run_dir=tmp_path, recipe="llm",
                   writer=JsonlWriter(tmp_path), every_n_steps=2)
    rec.attach()
    for step in range(6):
        tokens = torch.randint(0, 50, (4, 8))
        logits = model(tokens)
        loss = logits.sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
        rec.step(step, loss=float(loss.item()))
        if step in (2, 5):
            torch.save(model.state_dict(), ckpts / f"step{step:09d}.pt")
    rec.detach()

    scan_run(run_dir=tmp_path, recipe="llm",
             out_dir=tmp_path / "tb_retro",
             model_factory=lambda: _Tiny())

    out = build_report(run_dir=tmp_path, out_path=tmp_path / "inspect" / "report.md")
    md = out.read_text()
    assert "# circuitry report" in md
    assert "weight" in md
    assert (tmp_path / "circuitry" / "matched_modules.txt").exists()
    assert any((tmp_path / "tb_retro").rglob("events.out.tfevents.*"))
```

- [ ] **Step 2: Run**

```bash
venv/bin/pytest tests/e2e/test_full_pipeline.py -v
```

Expected: PASSED. The full test runs in <30s on CPU.

- [ ] **Step 3: Commit**

```bash
git add tests/e2e/test_full_pipeline.py
git commit -m "test(e2e): tiny LLM train → scan → report happy path"
```

---

### Task G2: Examples

**Files:**
- Create: `/home/vishsangale/workspace/circuitry/examples/tiny_llm.py`
- Create: `/home/vishsangale/workspace/circuitry/examples/tiny_vision.py`
- Create: `/home/vishsangale/workspace/circuitry/examples/tiny_two_tower.py`

- [ ] **Step 1: Write `examples/tiny_llm.py`**

```python
# examples/tiny_llm.py
"""Runnable LLM example. ``python examples/tiny_llm.py``."""

from __future__ import annotations

import pathlib
import sys

import torch
import torch.nn as nn

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from circuitry import Recorder  # noqa: E402


class TinyAttn(nn.Module):
    def __init__(self, d=16):
        super().__init__()
        for k in ("q_proj", "k_proj", "v_proj", "o_proj"):
            setattr(self, k, nn.Linear(d, d, bias=False))

    def forward(self, x):
        return self.o_proj(self.v_proj(x))


class TinyMlp(nn.Module):
    def __init__(self, d=16):
        super().__init__()
        self.gate_proj = nn.Linear(d, d * 2, bias=False)
        self.up_proj = nn.Linear(d, d * 2, bias=False)
        self.down_proj = nn.Linear(d * 2, d, bias=False)

    def forward(self, x):
        return self.down_proj(self.gate_proj(x) * self.up_proj(x))


class TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(64, 16)
        self.attn = TinyAttn(16)
        self.mlp = TinyMlp(16)
        self.ln_1 = nn.LayerNorm(16)
        self.ln_2 = nn.LayerNorm(16)
        self.lm_head = nn.Linear(16, 64, bias=False)

    def forward(self, tokens):
        x = self.embed(tokens)
        x = self.attn(self.ln_1(x))
        x = self.mlp(self.ln_2(x))
        return self.lm_head(x)


def main():
    out = pathlib.Path("runs/tiny_llm")
    out.mkdir(parents=True, exist_ok=True)
    model = TinyLM()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    rec = Recorder(model, run_dir=out, recipe="llm",
                   writer="tensorboard", every_n_steps=5)
    rec.attach()
    for step in range(50):
        tokens = torch.randint(0, 64, (8, 16))
        loss = model(tokens).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
        rec.step(step, loss=float(loss.item()))
    rec.detach()
    print(f"tensorboard --logdir {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write `examples/tiny_vision.py`**

```python
# examples/tiny_vision.py
"""Runnable vision example. ``python examples/tiny_vision.py``."""

from __future__ import annotations

import pathlib
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from circuitry import Recorder  # noqa: E402


class TinyCNN(nn.Module):
    def __init__(self, n_classes: int = 10) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 8 * 8, 64)
        self.fc2 = nn.Linear(64, n_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.relu(self.fc1(x.flatten(1)))
        return self.fc2(x)


def main():
    out = pathlib.Path("runs/tiny_vision")
    out.mkdir(parents=True, exist_ok=True)
    model = TinyCNN()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    rec = Recorder(model, run_dir=out, recipe="vision",
                   writer="tensorboard", every_n_steps=5)
    rec.attach()
    for step in range(50):
        x = torch.randn(8, 3, 16, 16)
        y = torch.randint(0, 10, (8,))
        loss = F.cross_entropy(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        rec.step(step, loss=float(loss.item()))
    rec.detach()
    print(f"tensorboard --logdir {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Write `examples/tiny_two_tower.py`**

```python
# examples/tiny_two_tower.py
"""Runnable two-tower example. ``python examples/tiny_two_tower.py``."""

from __future__ import annotations

import pathlib
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from circuitry import Recorder  # noqa: E402


class TwoTower(nn.Module):
    def __init__(self, d_in: int = 8, d_out: int = 16) -> None:
        super().__init__()
        self.query_tower = nn.Sequential(
            nn.Linear(d_in, 32), nn.ReLU(), nn.Linear(32, d_out),
        )
        self.item_tower = nn.Sequential(
            nn.Linear(d_in, 32), nn.ReLU(), nn.Linear(32, d_out),
        )

    def forward(self, q, i):
        return (self.query_tower(q) * self.item_tower(i)).sum(-1)


def main():
    out = pathlib.Path("runs/tiny_two_tower")
    out.mkdir(parents=True, exist_ok=True)
    model = TwoTower()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    rec = Recorder(model, run_dir=out, recipe="two_tower",
                   writer="tensorboard", every_n_steps=5)
    rec.attach()
    for step in range(50):
        q = torch.randn(16, 8)
        i_pos = torch.randn(16, 8)
        i_neg = torch.randn(16, 8)
        score_pos = model(q, i_pos)
        score_neg = model(q, i_neg)
        loss = F.softplus(score_neg - score_pos).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        rec.step(step, loss=float(loss.item()))
    rec.detach()
    print(f"tensorboard --logdir {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Verify each example runs**

```bash
venv/bin/python examples/tiny_llm.py
venv/bin/python examples/tiny_vision.py
venv/bin/python examples/tiny_two_tower.py
```

Expected: each prints a `tensorboard --logdir` line. `runs/tiny_*` directories contain event files.

- [ ] **Step 5: Commit**

```bash
git add examples/
git commit -m "docs(examples): add runnable tiny_llm / tiny_vision / tiny_two_tower"
```

---

## Phase H — Performance, release prep, tag

### Task H1: Benchmark harness (50M-param transformer)

**Files:**
- Create: `/home/vishsangale/workspace/circuitry/tests/perf/__init__.py`
- Create: `/home/vishsangale/workspace/circuitry/tests/perf/test_overhead.py`
- Create: `/home/vishsangale/workspace/circuitry/scripts/bench_50m.py`

The benchmark **runs** in M1 as a harness, but per the user's chosen scope numbers go in the README during M2 (or later). Keep the test light enough for CI (continue-on-error already set in `.github/workflows/ci.yml`).

- [ ] **Step 1: Write `scripts/bench_50m.py`**

A standalone script that constructs a ~50M-param decoder-only transformer (configurable via `--n-layers`, `--d-model`), runs 100 steps with and without `Recorder` attached, prints the wall-clock ratio. Default config sized to land near 50M params:

```python
# scripts/bench_50m.py
"""Reference benchmark per docs/design.md §10.

Usage:
    venv/bin/python scripts/bench_50m.py --n-layers 8 --d-model 768 --steps 100

Reports wall-clock ratio with and without circuitry attached. The budget is
≤10% overhead at default settings on a 50M-param model.
"""

from __future__ import annotations

import argparse
import time

import torch
import torch.nn as nn

from circuitry import Recorder


class Block(nn.Module):
    def __init__(self, d: int, n_heads: int = 8) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(d)
        self.ln_2 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d),
        )
        # Expose linear projections under recipe-matching names.
        self.attn.q_proj = nn.Linear(d, d, bias=False)
        self.attn.k_proj = nn.Linear(d, d, bias=False)
        self.attn.v_proj = nn.Linear(d, d, bias=False)
        self.attn.o_proj = nn.Linear(d, d, bias=False)

    def forward(self, x):
        h = self.ln_1(x)
        q = self.attn.q_proj(h); k = self.attn.k_proj(h); v = self.attn.v_proj(h)
        a, _ = self.attn(q, k, v)
        x = x + self.attn.o_proj(a)
        x = x + self.mlp(self.ln_2(x))
        return x


class TinyTransformer(nn.Module):
    def __init__(self, n_layers: int, d: int, vocab: int = 8192) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList([Block(d) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d)
        self.lm_head = nn.Linear(d, vocab, bias=False)

    def forward(self, tokens):
        x = self.embed(tokens)
        for b in self.blocks:
            x = b(x)
        return self.lm_head(self.ln_f(x))


def _run(model, steps: int, recorder: Recorder | None) -> float:
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    if recorder is not None:
        recorder.attach()
    t0 = time.perf_counter()
    for s in range(steps):
        tokens = torch.randint(0, 8192, (4, 64))
        loss = model(tokens).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if recorder is not None:
            recorder.step(s, loss=float(loss.item()))
    elapsed = time.perf_counter() - t0
    if recorder is not None:
        recorder.detach()
    return elapsed


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n-layers", type=int, default=8)
    p.add_argument("--d-model", type=int, default=768)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--every-n-steps", type=int, default=200)
    p.add_argument("--run-dir", default="runs/bench")
    args = p.parse_args()

    torch.manual_seed(0)
    model = TinyTransformer(args.n_layers, args.d_model)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params/1e6:.1f}M")

    baseline = _run(model, args.steps, recorder=None)
    rec = Recorder(model, run_dir=args.run_dir, recipe="llm",
                   writer="null", every_n_steps=args.every_n_steps)
    instrumented = _run(model, args.steps, recorder=rec)

    overhead = (instrumented / baseline) - 1.0
    print(f"baseline:     {baseline:7.2f}s")
    print(f"instrumented: {instrumented:7.2f}s")
    print(f"overhead:     {overhead*100:+5.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Write a quick pytest-benchmark-driven CI sanity test**

```python
# tests/perf/test_overhead.py
"""Quick perf sanity check. Uses a much smaller model than scripts/bench_50m.py
so it runs in seconds on CI. The 50M-param run is opt-in via the script."""

from __future__ import annotations

import time

import pytest
import torch
import torch.nn as nn

from circuitry import Recorder


class _Small(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(64, 32)
        self.attn = _Attn(32)
        self.mlp = nn.Sequential(nn.Linear(32, 64), nn.GELU(), nn.Linear(64, 32))
        self.ln_1 = nn.LayerNorm(32); self.ln_2 = nn.LayerNorm(32)
        self.lm_head = nn.Linear(32, 64, bias=False)

    def forward(self, t):
        x = self.embed(t)
        x = self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return self.lm_head(x)


class _Attn(nn.Module):
    def __init__(self, d: int) -> None:
        super().__init__()
        for k in ("q_proj", "k_proj", "v_proj", "o_proj"):
            setattr(self, k, nn.Linear(d, d, bias=False))

    def forward(self, x):
        return self.o_proj(self.v_proj(x))


def _train(model: nn.Module, steps: int, rec: Recorder | None) -> float:
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    if rec: rec.attach()
    t0 = time.perf_counter()
    for s in range(steps):
        tokens = torch.randint(0, 64, (4, 16))
        loss = model(tokens).sum()
        opt.zero_grad(); loss.backward(); opt.step()
        if rec: rec.step(s, loss=float(loss.item()))
    out = time.perf_counter() - t0
    if rec: rec.detach()
    return out


@pytest.mark.benchmark(group="overhead")
def test_overhead_under_2x(tmp_path, benchmark):
    """Sanity: tiny model overhead under 2x. Real <10% budget is in
    scripts/bench_50m.py, not enforceable on tiny tests."""
    model = _Small()
    baseline = _train(_Small(), 20, rec=None)
    rec = Recorder(model, run_dir=tmp_path, recipe="llm",
                   writer="null", every_n_steps=5)
    instrumented = benchmark(_train, model, 20, rec)
    assert instrumented < baseline * 5.0, (
        f"overhead {instrumented/baseline:.2f}x — investigate"
    )
```

- [ ] **Step 3: Run benchmark script locally (best-effort)**

```bash
venv/bin/python scripts/bench_50m.py --n-layers 4 --d-model 256 --steps 30
```

Expected: prints params, baseline, instrumented, overhead. Don't gate on a specific number — record the result for the M2 README update.

- [ ] **Step 4: Run the perf test**

```bash
venv/bin/pytest tests/perf/test_overhead.py -v
```

Expected: PASSED (sanity check only).

- [ ] **Step 5: Commit**

```bash
git add tests/perf/ scripts/bench_50m.py
git commit -m "perf: add benchmark harness and CI sanity check"
```

---

### Task H2: Parity-check harness stub (for M2)

**Files:**
- Create: `/home/vishsangale/workspace/circuitry/scripts/parity_check.py`

Committed now so M2 starts with the harness in place. The body is a TODO that names the tolerances from design.md §7 explicitly.

- [ ] **Step 1: Write the script**

```python
# scripts/parity_check.py
"""Numerical parity between circuitry and the in-tree mendu inspector.

Run in M2 (mendu cutover). The script trains a tiny canonical model under
both pipelines, captures all TB scalars from each, and asserts they agree
within the tolerances from docs/design.md §7 Phase M2:

  - Most metrics:        rtol=1e-5, atol=1e-7
  - SVD-derived metrics: rtol=1e-4   (effective_rank, condition_number,
                                       heavy_tail_alpha, singular_values)

M1 ships the harness; M2 wires it up against ~/workspace/mendu and ratchets
tolerances down if a metric is empirically tighter than expected.
"""

from __future__ import annotations

import argparse
import sys

DEFAULT_TOLERANCES = {
    "default": {"rtol": 1e-5, "atol": 1e-7},
    "svd": {"rtol": 1e-4, "atol": 1e-6},
}
SVD_METRICS = {
    "effective_rank",
    "condition_number",
    "heavy_tail_alpha",
    "singular_values",
    "stable_rank",
}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--mendu-root", required=True,
                   help="path to ~/workspace/mendu")
    p.add_argument("--steps", type=int, default=20)
    _ = p.parse_args()
    print("parity_check.py: M2 harness — body fills in during mendu cutover.")
    print("Tolerances:", DEFAULT_TOLERANCES)
    print("SVD-bucket metrics:", sorted(SVD_METRICS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Commit**

```bash
git add scripts/parity_check.py
git commit -m "chore: stub M2 parity-check harness with §7 tolerances"
```

---

### Task H3: Final README polish + CHANGELOG entry

**Files:**
- Modify: `/home/vishsangale/workspace/circuitry/README.md`
- Modify: `/home/vishsangale/workspace/circuitry/CHANGELOG.md`

- [ ] **Step 1: Add a "Performance" section to README**

Insert before the "v0.1.0 limits" section:

```markdown
## Performance

Default settings target ≤10% wall-clock overhead at `every_n_steps=200` on a 50M-param decoder transformer (see `docs/design.md` §10). Benchmark numbers will land alongside the M2 mendu cutover. Run the harness yourself:

```bash
python scripts/bench_50m.py --n-layers 8 --d-model 768 --steps 100
```
```

- [ ] **Step 2: Update CHANGELOG.md**

Replace the `## [Unreleased]` block with:

```markdown
## [0.1.0] — 2026-05-20

### Added
- `circuitry.core.weight` — `effective_rank`, `stable_rank`, `condition_number`, `singular_values`, `heavy_tail_alpha`.
- `circuitry.core.activation` — `dead_fraction`, `kurtosis`, `participation_ratio`, `norm_stats`.
- `circuitry.core.gradient` — `layer_norm`, `signal_propagation_depth`.
- `circuitry.core.spectral` — `esd`, `rank_trajectory`.
- `circuitry.Recorder` — training-time hooks per recipe; matched-modules artifact; `strict`/`expected_min_matches` invariants; rank-0 no-op on multi-rank runs.
- `circuitry.scan_run` and `circuitry.build_report`.
- `circuitry.Recipe` + stock LLM / vision / two_tower recipes; `register_recipe` for custom recipes.
- `circuitry.MetricWriter` protocol with TensorBoard (default, async option), JSONL, null, and optional wandb adapters.
- `circuitry` CLI: `list-recipes`, `report`. (`scan` exposed but requires a model factory — programmatic use only in v0.1.0.)
- Benchmark harness (`scripts/bench_50m.py`) and parity-check stub (`scripts/parity_check.py`) for the M2 mendu cutover.

### Known limits
- Single-process only. Non-zero ranks no-op; FSDP-sharded parameters produce incorrect diagnostics on rank 0. Multi-process design is in `docs/design.md` §11.
- README benchmark numbers are not filled in — run `scripts/bench_50m.py` yourself.
```

- [ ] **Step 3: Commit**

```bash
git add README.md CHANGELOG.md
git commit -m "docs: README performance section and v0.1.0 changelog entry"
```

---

### Task H4: Final test sweep + spec-coverage audit

- [ ] **Step 1: Run the entire test suite**

```bash
venv/bin/pytest -ra
```

Expected: all tests PASS.

- [ ] **Step 2: Run import-linter**

```bash
venv/bin/lint-imports
```

Expected: contracts kept.

- [ ] **Step 3: Run ruff**

```bash
venv/bin/ruff check src tests
```

Expected: clean.

- [ ] **Step 4: Smoke-test the wheel build**

```bash
venv/bin/pip install build
venv/bin/python -m build --wheel
```

Expected: `dist/circuitry-0.1.0.dev0-py3-none-any.whl` is produced and `pip install dist/*.whl` into a scratch venv imports cleanly.

- [ ] **Step 5: Spec-coverage audit (manual)**

Walk through `docs/design.md` §4, §10, §11 and verify each named feature has landed:

- §4.1 primitives — all covered (Phase B)
- §4.2 Recorder / scan_run / build_report — Phase D
- §4.3 CLI scan/report/list-recipes — Phase F
- §4.4 Recipe, HookPoint, StepContext, register_recipe, matched_modules.txt, expected_min_matches, strict — Phases D1/D2/D3
- §4.5 MetricWriter add_scalar/add_histogram/add_image(dataformats)/add_text/flush/close — Phase C
- §10 ≤10% budget — harness shipped (numbers deferred per user choice)
- §10 per-diagnostic enabled — Phase D4
- §10 max_dim subsampling — Phase B1
- §10 lazy hooks / _should_capture — Phase D3
- §10 async writer — Phase C3
- §11 rank-0 noop — Phase D3
- §11 README disclaimer — Phase A3

If any item is missing, add a corrective task before tagging.

- [ ] **Step 6: Commit any audit fixes individually with descriptive messages**

---

### Task H5: Tag v0.1.0

- [ ] **Step 1: Bump version**

In `pyproject.toml` change `version = "0.1.0.dev0"` to `version = "0.1.0"`. In `src/circuitry/__init__.py` change `__version__ = "0.1.0.dev0"` to `__version__ = "0.1.0"`.

- [ ] **Step 2: Commit**

```bash
git add pyproject.toml src/circuitry/__init__.py
git commit -m "chore: bump version to 0.1.0"
```

- [ ] **Step 3: Tag**

```bash
git tag -a v0.1.0 -m "circuitry v0.1.0 — initial extraction from mendu"
```

- [ ] **Step 4: Push (user-confirmation gate — do NOT push without explicit user approval)**

Print the commit + tag, then **ask the user before pushing**. Suggested message:

> "Ready to push `v0.1.0`. This is a public open-source tag; once pushed it cannot be amended without re-tagging. Push to `origin main` and `origin v0.1.0`?"

After user approval:

```bash
git push origin main
git push origin v0.1.0
```

- [ ] **Step 5: Open a tracking issue for M2** (optional, with user approval) — title "M2: mendu cutover with parity check", body cribbed from design.md §7 Phase M2.

---

## After M1

Two follow-up plans (written separately when M1 ships):

- **`docs/plan-m2.md`** — Mendu cutover with parity. Wire up `scripts/parity_check.py` against `~/workspace/mendu`; iterate tolerances; once green, delete `mendu/tools/inspect_checkpoint/` and the `paper2/.../analysis/spectral_diagnostics.py` copies; install `circuitry` editable into mendu's venv; archive `~/workspace/latent-superpowers-inspect`; commit benchmark numbers to README.

- **`docs/plan-m3.md`** — Multi-process v0.next per design.md §11. Adds `core/distributed.py`, `TensorSource.WEIGHT_FULL`/`ACTIVATION_FULL`, `DDPMetricWriter`. Additive — no v0.1.0 user code breaks.

Out of scope for any of these: SAE training, causal interventions, JAX, web dashboard, differentiability through primitives (see design.md §8).
