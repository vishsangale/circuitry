# circuitry M2 — mendu cutover with parity check

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Date:** 2026-05-21
**Status:** draft for implementation
**Owner:** Vishwanath Sangale
**Updates:** `docs/design.md` §7 (rewritten in Phase 5 of this plan to reflect circuitry's actual v0.1.0 API surface).

**Goal:** Cut `~/workspace/mendu` over from its in-tree `tools/inspect_checkpoint/` (which wraps the pip-installed `latent_inspect_checkpoint` package) to `circuitry`. Ship circuitry `v0.2.0`. Gate the cutover on a TB-scalar parity check. Archive the `core/inspect-checkpoint/` subdir of `latent-superpowers-inspect` (not the whole repo — it hosts 13 unrelated subsystems).

**Architecture:** Hybrid approach. Circuitry grows for **universal** features only — `loss_components` dict, gradient-norms-per-param, sv-histograms, weight dynamics (`update_delta`, `direction_cosine`), and a LLaMA-family arch-discovery helper. **Paper2-specific** features (EI balance per layer, MoE route fractions, eval-batch perplexity, Adam moment norms, attention-pattern capture) ship as a **mendu-local custom `Recipe`** registered via `register_recipe`. TB scalar names use circuitry-native conventions — historical paper2 TB runs will not be directly comparable to post-cutover runs (user-accepted name break).

**Tech Stack:** Python 3.12, PyTorch ≥2.1, TensorBoard ≥2.14. Cross-repo: edits land in `~/workspace/circuitry` (Phases 1, 2, 4, 5) and `~/workspace/mendu` (Phase 3). Phase 4 touches `~/workspace/latent-superpowers-inspect`.

**Key user decisions (captured 2026-05-21):**
1. Hybrid strategy (circuitry grows for universal; paper2 stays in mendu Recipe).
2. Accept scalar-name break (no preservation of mendu's historical names).
3. Port the three `diagnose_*.py` kernels into `recipes/vision.py`.
4. Port `token_similarity` into `circuitry.core.activation`.

**Working-directory discipline:** Every task names its repo. Use full venv paths (no `source venv/bin/activate`):
- circuitry: `cd ~/workspace/circuitry && venv/bin/pytest ...`
- mendu: `cd ~/workspace/mendu && venv/bin/pytest ...`

---

## File structure

### New / modified in `~/workspace/circuitry/`

| Phase | Path | Action |
|---|---|---|
| 1 | `src/circuitry/core/activation.py` | extend — add `token_similarity` |
| 1 | `src/circuitry/core/weight.py` | extend — add `update_delta`, `direction_cosine` |
| 1 | `src/circuitry/recipes/_discovery.py` | new — LLaMA-family `arch_discovery` helper |
| 1 | `src/circuitry/recorder/live.py` | extend — `loss_components` kwarg on `step()`; emit per-param grad norms; emit sv histograms |
| 1 | `src/circuitry/recipes/__init__.py` | extend — `Recipe.weight_diagnostics` accepts `"sv_histogram"`; `Recipe.gradient_diagnostics` accepts `"norms_per_param"` |
| 1 | `src/circuitry/recipes/vision.py` | extend — three custom diagnostics: `ei_ratio`, `signal_prop_depth`, `trained_pc_stats` |
| 1 | `src/circuitry/recipes/llm.py` | extend — wire `norms_per_param` + `sv_histogram` into LLM recipe |
| 1 | `tests/core/test_activation.py` | add `token_similarity` tests |
| 1 | `tests/core/test_weight.py` | add dynamics tests |
| 1 | `tests/recipes/test_discovery.py` | new — arch_discovery tests |
| 1 | `tests/recipes/test_vision_custom.py` | new — vision custom-diag tests |
| 1 | `tests/recorder/test_step_extensions.py` | new — `loss_components`, per-param emission |
| 1 | `pyproject.toml` | version bump 0.1.0 → 0.2.0a0 |
| 2 | `scripts/parity_check.py` | extend — body |
| 2 | `tests/parity/test_smoke.py` | new — parity harness smoke (doesn't need mendu) |
| 2 | `parity_results.md` | new — committed numbers from the M2 hardware run |
| 4 | (n/a — Phase 4 touches `latent-superpowers-inspect`) | |
| 5 | `README.md` | extend — Performance section gets numbers from `scripts/bench_50m.py` |
| 5 | `docs/design.md` | §7 rewritten to reflect actual M2 |
| 5 | `CHANGELOG.md` | new `## 0.2.0 — 2026-MM-DD` section |
| 5 | `src/circuitry/__init__.py` | bump `__version__` to `"0.2.0"` |

### Modified / deleted in `~/workspace/mendu/`

| Phase | Path | Action |
|---|---|---|
| 3 | (mendu venv) `latent_inspect_checkpoint` | uninstall (after Q7) |
| 3 | (mendu venv) `circuitry` | `pip install -e ~/workspace/circuitry` |
| 3 | `mendu/paper2/circuitry_recipe.py` | new — paper2 custom Recipe |
| 3 | `mendu/paper2/bet1_surprise/train/train_350m.py` | rewrite recorder call-site (around L18, L817-826) |
| 3 | `mendu/paper2/bet1_surprise/train/train_350m_ste.py` | rewrite recorder call-site (around L18) |
| 3 | `mendu/paper2/bet2_daleian/train/train_350m.py` | rewrite recorder call-site (around L44, L817-826) |
| 3 | `mendu/paper2/tests/inspect_checkpoint/test_live_recorder_smoke.py` | rewrite to use `circuitry.Recorder` |
| 3 | `mendu/paper2/tests/inspect_checkpoint/test_eval_batch_pinning.py` | rewrite — paper2 recipe's `eval_ppl` custom diag |
| 3 | `mendu/paper2/tests/inspect_checkpoint/test_arch_detect.py` | rewrite to use `circuitry.recipes._discovery.discover` |
| 3 | `mendu/paper2/bet2_daleian/tests/test_spectral_diagnostics.py` | rewrite to call `circuitry.core.spectral` |
| 3 | `mendu/paper2/bet2_daleian/tests/test_spectral_at_depth.py` | keep (paper2-specific analysis stays in mendu) — only fix imports |
| 3 | `mendu/tools/inspect_checkpoint/` | delete after parity passes |
| 3 | `mendu/paper2/bet2_daleian/analysis/spectral_diagnostics.py` | delete (functions now in circuitry.core.spectral / .activation) |
| 3 | `mendu/CLAUDE.md` | update install line |

### Touched in `~/workspace/latent-superpowers-inspect/`

| Phase | Path | Action |
|---|---|---|
| 4 | `core/inspect-checkpoint/` | delete |
| 4 | `tests/inspect-checkpoint/` | delete |
| 4 | `adapters/*/inspect-checkpoint/` | delete (3 adapter copies: claude-code, codex, gemini, opencode) |
| 4 | `README.md` | add "inspect-checkpoint extracted to circuitry" note |

---

## Phase 1 — Circuitry v0.2 grow (~10 tasks)

Universal features that mendu needs and that any future user of circuitry plausibly wants. Each task is TDD: failing test first, minimal implementation, verify pass, commit.

### Task N1: `token_similarity` primitive in `core/activation.py`

**Files:**
- Modify: `src/circuitry/core/activation.py`
- Test: `tests/core/test_activation.py`

`token_similarity(h)` computes the mean off-diagonal cosine similarity of token hidden states. Lifted from `mendu/paper2/bet2_daleian/analysis/spectral_diagnostics.py:29`.

- [ ] **Step 1: Write the failing test**

In `tests/core/test_activation.py`, append:

```python
def test_token_similarity_identical_tokens():
    # All tokens identical → cosine similarity = 1.0
    import torch
    from circuitry.core.activation import token_similarity
    h = torch.ones(1, 5, 8)
    sim = token_similarity(h)
    assert torch.allclose(sim, torch.tensor(1.0), atol=1e-6)


def test_token_similarity_orthogonal_tokens():
    # Standard basis tokens → off-diagonal cosine = 0
    import torch
    from circuitry.core.activation import token_similarity
    h = torch.eye(4).unsqueeze(0)  # (1, 4, 4)
    sim = token_similarity(h)
    assert torch.allclose(sim, torch.tensor(0.0), atol=1e-6)


def test_token_similarity_handles_batch():
    import torch
    from circuitry.core.activation import token_similarity
    h = torch.randn(3, 5, 8)
    sim = token_similarity(h)
    assert sim.shape == ()  # scalar, mean across batch
```

- [ ] **Step 2: Run, verify fail**

`cd ~/workspace/circuitry && venv/bin/pytest tests/core/test_activation.py -k token_similarity -v` → FAIL (function not defined).

- [ ] **Step 3: Implement**

In `src/circuitry/core/activation.py`, append:

```python
def token_similarity(h: torch.Tensor) -> torch.Tensor:
    """Mean off-diagonal cosine similarity between token hidden states.

    Args:
        h: ``(batch, seq, dim)`` or ``(seq, dim)`` hidden states.

    Returns:
        Scalar mean off-diagonal cosine similarity (averaged over batch).
    """
    if h.dim() == 2:
        h = h.unsqueeze(0)
    normalized = torch.nn.functional.normalize(h, dim=-1)
    gram = torch.matmul(normalized, normalized.transpose(-2, -1))  # (B, S, S)
    seq = gram.shape[-1]
    if seq < 2:
        return torch.tensor(0.0, dtype=h.dtype, device=h.device)
    off_diag_mask = ~torch.eye(seq, dtype=torch.bool, device=h.device)
    off_diag = gram[..., off_diag_mask].view(gram.shape[0], -1)
    return off_diag.mean()
```

- [ ] **Step 4: Run, verify pass**

`venv/bin/pytest tests/core/test_activation.py -k token_similarity -v` → 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/circuitry/core/activation.py tests/core/test_activation.py
git commit -m "feat(core): add token_similarity primitive (port from mendu)"
```

---

### Task N2: Weight dynamics primitives — `update_delta` and `direction_cosine`

**Files:**
- Modify: `src/circuitry/core/weight.py`
- Test: `tests/core/test_weight.py`

Lifted from `latent_inspect_checkpoint.metrics`. Both operate on state-dict snapshots; both are pure.

- [ ] **Step 1: Write the failing tests**

Append to `tests/core/test_weight.py`:

```python
def test_update_delta_zero_when_unchanged():
    import torch
    from circuitry.core.weight import update_delta
    sd = {"w": torch.ones(3, 3)}
    out = update_delta(sd, sd)
    assert out["w"] == 0.0


def test_update_delta_l2_when_shifted():
    import torch
    from circuitry.core.weight import update_delta
    sd_now = {"w": torch.tensor([[1.0, 0.0], [0.0, 1.0]])}
    sd_prev = {"w": torch.tensor([[0.0, 0.0], [0.0, 0.0]])}
    out = update_delta(sd_now, sd_prev)
    assert abs(out["w"] - (2.0 ** 0.5)) < 1e-6


def test_direction_cosine_collinear_updates():
    import torch
    from circuitry.core.weight import direction_cosine
    # prev_prev -> prev: +I; prev -> now: +2I (same direction)
    sd_pp = {"w": torch.zeros(2, 2)}
    sd_p = {"w": torch.eye(2)}
    sd_n = {"w": torch.eye(2) * 3.0}
    out = direction_cosine(sd_n, sd_p, sd_pp)
    assert abs(out["w"] - 1.0) < 1e-6


def test_direction_cosine_opposite_updates():
    import torch
    from circuitry.core.weight import direction_cosine
    sd_pp = {"w": torch.zeros(2, 2)}
    sd_p = {"w": torch.eye(2)}
    sd_n = {"w": torch.zeros(2, 2)}  # update reverses
    out = direction_cosine(sd_n, sd_p, sd_pp)
    assert abs(out["w"] - (-1.0)) < 1e-6
```

- [ ] **Step 2: Run, verify fail**

`venv/bin/pytest tests/core/test_weight.py -k "update_delta or direction_cosine" -v` → FAIL.

- [ ] **Step 3: Implement**

Append to `src/circuitry/core/weight.py`:

```python
def update_delta(
    sd_now: Mapping[str, torch.Tensor],
    sd_prev: Mapping[str, torch.Tensor],
) -> dict[str, float]:
    """L2 norm of the delta between two state-dict snapshots, per parameter.

    Returns ``{name: ||sd_now[name] - sd_prev[name]||_2}`` for every name
    present in both. Names missing from either side are skipped.
    """
    out: dict[str, float] = {}
    for name in sd_now:
        if name not in sd_prev:
            continue
        diff = (sd_now[name].to(torch.float32) - sd_prev[name].to(torch.float32))
        out[name] = float(diff.norm().item())
    return out


def direction_cosine(
    sd_now: Mapping[str, torch.Tensor],
    sd_prev: Mapping[str, torch.Tensor],
    sd_prev_prev: Mapping[str, torch.Tensor],
) -> dict[str, float]:
    """Cosine similarity between two consecutive parameter updates.

    Update_1 = sd_prev - sd_prev_prev
    Update_2 = sd_now  - sd_prev

    Returns ``{name: cos(Update_1, Update_2)}``. Zero-norm updates return 0.0.
    """
    out: dict[str, float] = {}
    for name in sd_now:
        if name not in sd_prev or name not in sd_prev_prev:
            continue
        u2 = (sd_now[name].to(torch.float32) - sd_prev[name].to(torch.float32)).flatten()
        u1 = (sd_prev[name].to(torch.float32) - sd_prev_prev[name].to(torch.float32)).flatten()
        n2 = float(u2.norm().item())
        n1 = float(u1.norm().item())
        if n1 == 0.0 or n2 == 0.0:
            out[name] = 0.0
        else:
            out[name] = float((u1 @ u2).item()) / (n1 * n2)
    return out
```

Make sure `from typing import Mapping` is present at the top — if not, add `from collections.abc import Mapping`.

- [ ] **Step 4: Run, verify pass**

`venv/bin/pytest tests/core/test_weight.py -v` → all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/circuitry/core/weight.py tests/core/test_weight.py
git commit -m "feat(core): add update_delta + direction_cosine weight dynamics primitives"
```

---

### Task N3: LLaMA-family `arch_discovery` helper

**Files:**
- Create: `src/circuitry/recipes/_discovery.py`
- Test: `tests/recipes/test_discovery.py`

Ports `latent_inspect_checkpoint.arch_discovery.discover` — given a state_dict, classifies each parameter into (role, layer_idx). Lives under `recipes/` because role/layer concepts are recipe-shaped (the LLM recipe uses it to aggregate per-role; vision/two-tower don't).

- [ ] **Step 1: Write the failing tests**

Create `tests/recipes/test_discovery.py`:

```python
import torch
from circuitry.recipes._discovery import discover, ParamInfo


def _make_llama_sd():
    """Synthetic state_dict matching LLaMA-family naming."""
    return {
        "tok_embeddings.weight": torch.empty(100, 64),
        "norm.weight": torch.empty(64),
        "output.weight": torch.empty(100, 64),
        "layers.0.attention.wq.weight": torch.empty(64, 64),
        "layers.0.attention.wk.weight": torch.empty(64, 64),
        "layers.0.attention.wv.weight": torch.empty(64, 64),
        "layers.0.attention.wo.weight": torch.empty(64, 64),
        "layers.0.attention_norm.weight": torch.empty(64),
        "layers.0.feed_forward.w1.weight": torch.empty(128, 64),
        "layers.0.feed_forward.w2.weight": torch.empty(64, 128),
        "layers.0.feed_forward.w3.weight": torch.empty(128, 64),
        "layers.0.ffn_norm.weight": torch.empty(64),
        "layers.1.attention.wq.weight": torch.empty(64, 64),
    }


def test_discover_assigns_layers():
    sd = _make_llama_sd()
    out = discover(sd)
    names = {p.name: p for p in out.params}
    assert names["layers.0.attention.wq.weight"].layer == 0
    assert names["layers.1.attention.wq.weight"].layer == 1
    assert names["tok_embeddings.weight"].layer is None


def test_discover_assigns_roles():
    sd = _make_llama_sd()
    out = discover(sd)
    names = {p.name: p for p in out.params}
    assert names["layers.0.attention.wq.weight"].role == "attn_q"
    assert names["layers.0.attention.wk.weight"].role == "attn_k"
    assert names["layers.0.attention.wv.weight"].role == "attn_v"
    assert names["layers.0.attention.wo.weight"].role == "attn_o"
    assert names["layers.0.feed_forward.w1.weight"].role in ("ffn_in", "ffn_gate")
    assert names["layers.0.feed_forward.w2.weight"].role == "ffn_out"
    assert names["tok_embeddings.weight"].role == "embedding"
    assert names["output.weight"].role == "lm_head"


def test_discover_params_by_role():
    sd = _make_llama_sd()
    out = discover(sd)
    by_role = out.params_by_role()
    assert "attn_q" in by_role
    assert len(by_role["attn_q"]) == 2  # layers 0 and 1
```

- [ ] **Step 2: Run, verify fail**

`venv/bin/pytest tests/recipes/test_discovery.py -v` → FAIL (module missing).

- [ ] **Step 3: Implement**

Create `src/circuitry/recipes/_discovery.py`:

```python
"""LLaMA-family state_dict classifier: param-name → (role, layer_idx).

Lifted from latent_inspect_checkpoint.arch_discovery and trimmed to what
circuitry recipes need. Supports the naming variants seen across mendu's
paper2 bets; if you need a different family, extend ``_ROLE_PATTERNS``.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
import re

import torch

# Block-prefix variants: "layers.N." (LLaMA) and "blocks.N." (paper2 alt).
_LAYER_RE = re.compile(r"^(?:layers|blocks)\.(\d+)\.")

# Role assignment — applied after stripping the layer prefix.
# Order matters: longer / more specific keys first.
_ROLE_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"attention\.wq\.weight$"),   "attn_q"),
    (re.compile(r"attention\.wk\.weight$"),   "attn_k"),
    (re.compile(r"attention\.wv\.weight$"),   "attn_v"),
    (re.compile(r"attention\.wo\.weight$"),   "attn_o"),
    (re.compile(r"attention_norm\.weight$"),  "attn_norm"),
    (re.compile(r"feed_forward\.w1\.weight$"), "ffn_gate"),
    (re.compile(r"feed_forward\.w2\.weight$"), "ffn_out"),
    (re.compile(r"feed_forward\.w3\.weight$"), "ffn_in"),
    (re.compile(r"ffn_norm\.weight$"),         "ffn_norm"),
]

_GLOBAL_ROLES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"^tok_embeddings\.weight$"), "embedding"),
    (re.compile(r"^output\.weight$"),         "lm_head"),
    (re.compile(r"^norm\.weight$"),           "final_norm"),
]


@dataclass(frozen=True)
class ParamInfo:
    name: str
    shape: tuple[int, ...]
    role: str | None
    layer: int | None


@dataclass
class Discovery:
    params: list[ParamInfo] = field(default_factory=list)

    def params_by_role(self) -> dict[str, list[ParamInfo]]:
        out: dict[str, list[ParamInfo]] = defaultdict(list)
        for p in self.params:
            if p.role is not None:
                out[p.role].append(p)
        return dict(out)


def discover(state_dict: Mapping[str, torch.Tensor]) -> Discovery:
    """Classify each param into (role, layer) for per-role / per-layer aggregation."""
    params: list[ParamInfo] = []
    for name, tensor in state_dict.items():
        layer = None
        layer_match = _LAYER_RE.match(name)
        if layer_match:
            layer = int(layer_match.group(1))
            suffix = name[layer_match.end():]
            role = None
            for pat, r in _ROLE_PATTERNS:
                if pat.search(suffix):
                    role = r
                    break
        else:
            role = None
            for pat, r in _GLOBAL_ROLES:
                if pat.search(name):
                    role = r
                    break
        params.append(ParamInfo(
            name=name,
            shape=tuple(tensor.shape),
            role=role,
            layer=layer,
        ))
    return Discovery(params=params)
```

- [ ] **Step 4: Run, verify pass**

`venv/bin/pytest tests/recipes/test_discovery.py -v` → 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/circuitry/recipes/_discovery.py tests/recipes/test_discovery.py
git commit -m "feat(recipes): add LLaMA-family arch_discovery helper"
```

---

### Task N4: `Recorder.step(loss_components=...)`

**Files:**
- Modify: `src/circuitry/recorder/live.py`
- Test: `tests/recorder/test_step_extensions.py`

mendu currently emits `train/lm_loss`, `train/aux_loss`, `train/total_loss`, `train/lr`. Circuitry's `step(loss=...)` only handles a single scalar. Extend it: `step(loss=..., loss_components={"lm_loss": 0.4, "aux_loss": 0.1, "lr": 3e-4})` emits each as `train/<key>`.

- [ ] **Step 1: Write the failing test**

Create `tests/recorder/test_step_extensions.py`:

```python
import torch
import torch.nn as nn

from circuitry import Recorder
from circuitry.recipes import Recipe, _clear_registry_for_tests, register_recipe
from circuitry.recorder.hooks import HookPoint, TensorSource


def _setup_recipe():
    _clear_registry_for_tests()
    register_recipe(Recipe(
        name="probe",
        hook_points=[HookPoint(source=TensorSource.WEIGHT, pattern=r"linear\.weight$")],
        weight_diagnostics=["effective_rank"],
    ))


def test_loss_components_emitted_as_train_scalars(tmp_path):
    _setup_recipe()
    model = nn.Sequential()
    model.add_module("linear", nn.Linear(4, 4))

    rec = Recorder(
        model, run_dir=tmp_path, recipe="probe", writer="jsonl", every_n_steps=1,
    )
    rec.attach()
    rec.step(step=0, loss=0.5, loss_components={"lm_loss": 0.4, "aux_loss": 0.1, "lr": 3e-4})
    rec.detach()

    text = (tmp_path / "circuitry" / "metrics.jsonl").read_text().splitlines()
    keys = set()
    for line in text:
        import json
        rec_obj = json.loads(line)
        keys.add(rec_obj["name"])
    assert "train/lm_loss" in keys
    assert "train/aux_loss" in keys
    assert "train/lr" in keys
    assert "train/loss" in keys  # the existing aggregate-loss tag


def test_loss_components_optional(tmp_path):
    _setup_recipe()
    model = nn.Sequential()
    model.add_module("linear", nn.Linear(4, 4))
    rec = Recorder(model, run_dir=tmp_path, recipe="probe", writer="null", every_n_steps=1)
    rec.attach()
    rec.step(step=0, loss=0.5)  # no loss_components
    rec.detach()
```

- [ ] **Step 2: Run, verify fail**

`venv/bin/pytest tests/recorder/test_step_extensions.py -v` → FAIL.

- [ ] **Step 3: Implement**

Open `src/circuitry/recorder/live.py`. Find the `step()` method. After the existing `train/loss` emission, add per-component emission. Pseudo-diff:

```python
def step(
    self,
    step: int,
    *,
    loss: float | None = None,
    loss_components: dict[str, float] | None = None,
    enabled: bool | None = None,
    **kwargs,
) -> None:
    ...
    if loss is not None and self._writer is not None:
        self._writer.add_scalar("train/loss", float(loss), step)
    if loss_components is not None and self._writer is not None:
        for name, value in loss_components.items():
            self._writer.add_scalar(f"train/{name}", float(value), step)
    ...
```

Make sure the new kwarg lives **before** `**kwargs` and is keyword-only. The dispatch to custom diagnostics already threads `**kwargs` into `ctx.user`; `loss_components` is now an explicit kwarg, so don't double-thread.

- [ ] **Step 4: Run, verify pass**

`venv/bin/pytest tests/recorder/test_step_extensions.py -v` → 2 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/circuitry/recorder/live.py tests/recorder/test_step_extensions.py
git commit -m "feat(recorder): step(loss_components=) emits per-component train/ scalars"
```

---

### Task N5: Per-param gradient norms as a built-in diagnostic

**Files:**
- Modify: `src/circuitry/recipes/__init__.py` (validate `"norms_per_param"` is a recognized gradient diagnostic)
- Modify: `src/circuitry/recorder/live.py` (emit `grad/per_param/<name>/norm` + `grad/global/total_norm` when `"norms_per_param"` in `gradient_diagnostics`)
- Test: `tests/recorder/test_step_extensions.py`

- [ ] **Step 1: Add the failing test**

Append to `tests/recorder/test_step_extensions.py`:

```python
def test_gradient_norms_per_param(tmp_path):
    _clear_registry_for_tests()
    register_recipe(Recipe(
        name="probe_grad",
        hook_points=[
            HookPoint(source=TensorSource.WEIGHT, pattern=r"linear\.weight$"),
            HookPoint(source=TensorSource.GRAD,   pattern=r"linear\.weight$"),
        ],
        gradient_diagnostics=["norms_per_param"],
    ))
    model = nn.Sequential()
    model.add_module("linear", nn.Linear(4, 4))
    rec = Recorder(model, run_dir=tmp_path, recipe="probe_grad", writer="jsonl", every_n_steps=1)
    rec.attach()
    # Run a backward pass to populate .grad
    x = torch.randn(2, 4)
    y = model(x).sum()
    y.backward()
    rec.step(step=0, loss=float(y))
    rec.detach()

    import json
    keys = set()
    for line in (tmp_path / "circuitry" / "metrics.jsonl").read_text().splitlines():
        keys.add(json.loads(line)["name"])
    assert any(k.startswith("grad/per_param/") and k.endswith("/norm") for k in keys)
    assert "grad/global/total_norm" in keys
```

- [ ] **Step 2: Run, verify fail**

→ FAIL.

- [ ] **Step 3: Implement**

In `src/circuitry/recorder/live.py`, find the gradient-diagnostic dispatch. Add handling for `"norms_per_param"`:

```python
if "norms_per_param" in recipe.gradient_diagnostics and gradients:
    global_sq = 0.0
    for name, g in gradients.items():
        n = float(g.detach().to(torch.float32).norm().item())
        self._writer.add_scalar(f"grad/per_param/{name}/norm", n, step)
        global_sq += n * n
    self._writer.add_scalar("grad/global/total_norm", global_sq ** 0.5, step)
```

The `gradients` dict comes from `StepContext`. In `recipes/__init__.py`, the `gradient_diagnostics` list is currently an untyped `list[str]` — no validation change needed. Add a docstring note that `"norms_per_param"` is supported.

- [ ] **Step 4: Run, verify pass**

→ PASS.

- [ ] **Step 5: Commit**

```bash
git add src/circuitry/recorder/live.py src/circuitry/recipes/__init__.py tests/recorder/test_step_extensions.py
git commit -m "feat(recorder): built-in 'norms_per_param' gradient diagnostic"
```

---

### Task N6: Singular-values histogram emission

**Files:**
- Modify: `src/circuitry/recorder/live.py` (when `"sv_histogram"` in `weight_diagnostics`, call `writer.add_histogram`)
- Test: `tests/recorder/test_step_extensions.py`

- [ ] **Step 1: Add the failing test**

Append:

```python
def test_sv_histogram_emitted(tmp_path):
    _clear_registry_for_tests()
    register_recipe(Recipe(
        name="probe_hist",
        hook_points=[HookPoint(source=TensorSource.WEIGHT, pattern=r"linear\.weight$")],
        weight_diagnostics=["effective_rank", "sv_histogram"],
    ))
    model = nn.Sequential()
    model.add_module("linear", nn.Linear(8, 8))
    rec = Recorder(model, run_dir=tmp_path, recipe="probe_hist", writer="jsonl", every_n_steps=1)
    rec.attach()
    rec.step(step=0, loss=0.0)
    rec.detach()

    # JsonlWriter stores histograms as ".npy" side files
    art_dir = tmp_path / "circuitry" / "artifacts"
    sv_files = list(art_dir.glob("*sv_histogram*.npy"))
    assert len(sv_files) >= 1
```

- [ ] **Step 2: Run, verify fail**

→ FAIL.

- [ ] **Step 3: Implement**

In `src/circuitry/recorder/live.py`, in the weight-diagnostic dispatch, add:

```python
if "sv_histogram" in recipe.weight_diagnostics and weights:
    from circuitry.core.spectral import singular_values
    for name, w in weights.items():
        sv = singular_values(w)
        self._writer.add_histogram(f"spectral/per_param/{name}/sv_histogram", sv, step)
```

Note that JsonlWriter handles `add_histogram` by storing the tensor as a `.npy` side file (already exists per the M1 design); TensorBoardWriter calls native `add_histogram`. No writer-side changes.

- [ ] **Step 4: Run, verify pass**

→ PASS.

- [ ] **Step 5: Commit**

```bash
git add src/circuitry/recorder/live.py tests/recorder/test_step_extensions.py
git commit -m "feat(recorder): 'sv_histogram' weight diagnostic emits per-param histograms"
```

---

### Task N7: Port `diagnose_ei_bottlenecks.py` kernel as a vision custom diagnostic

**Files:**
- Modify: `src/circuitry/recipes/vision.py`
- Create: `tests/recipes/test_vision_custom.py`

Read `mendu/scripts/diagnose_ei_bottlenecks.py`. The diagnostic kernel computes per-layer E/I (excitatory/inhibitory) channel balance from weight signs. Port as a `(StepContext) → dict[str, float]` function emitting `vision/ei_ratio/<layer>`.

- [ ] **Step 1: Inspect the source**

```bash
sed -n '1,40p' ~/workspace/mendu/scripts/diagnose_ei_bottlenecks.py
```

Identify the function that computes the ratio (typically `diagnose_ei_checkpoint(ckpt_path)` or a helper). The kernel is roughly: for each conv layer's weight, count positive vs. negative input channels, emit ratio per layer.

- [ ] **Step 2: Add the failing test**

Create `tests/recipes/test_vision_custom.py`:

```python
import torch
import torch.nn as nn

from circuitry.recipes.vision import ei_ratio
from circuitry.recorder.hooks import StepContext


def test_ei_ratio_all_positive():
    w = torch.ones(8, 3, 3, 3)  # (out, in, k, k); all positive
    ctx = StepContext(step=0, model=nn.Identity(), weights={"conv1.weight": w})
    out = ei_ratio(ctx)
    assert "vision/ei_ratio/conv1.weight" in out
    assert abs(out["vision/ei_ratio/conv1.weight"] - 1.0) < 1e-6


def test_ei_ratio_balanced():
    w = torch.cat([torch.ones(8, 3, 3, 3), -torch.ones(8, 3, 3, 3)], dim=1)  # 50/50
    ctx = StepContext(step=0, model=nn.Identity(), weights={"conv1.weight": w})
    out = ei_ratio(ctx)
    assert abs(out["vision/ei_ratio/conv1.weight"] - 0.5) < 1e-6
```

→ FAIL (function not exported from `recipes/vision.py`).

- [ ] **Step 3: Implement**

In `src/circuitry/recipes/vision.py`, add:

```python
import torch

from circuitry.recorder.hooks import StepContext


def ei_ratio(ctx: StepContext) -> dict[str, float]:
    """Per-conv-weight fraction of input channels with positive mean.

    Ports the kernel of mendu/scripts/diagnose_ei_bottlenecks.py: for each
    conv weight (4-D tensor), reduces the spatial + output dims to a per-input-channel
    mean, then returns the fraction of input channels with mean > 0.
    """
    out: dict[str, float] = {}
    for name, w in ctx.weights.items():
        if w.dim() != 4:
            continue
        # (out, in, kh, kw) → mean over (out, kh, kw) → (in,)
        per_in = w.detach().to(torch.float32).mean(dim=(0, 2, 3))
        out[f"vision/ei_ratio/{name}"] = float((per_in > 0).float().mean().item())
    return out
```

Then add it to the recipe's `custom` list. Find the `RECIPE = Recipe(...)` block in the same file and update:

```python
RECIPE = Recipe(
    name="vision",
    hook_points=[...],
    weight_diagnostics=["effective_rank", "stable_rank"],
    activation_diagnostics=["dead_fraction", "participation_ratio"],
    gradient_diagnostics=["layer_norm"],
    custom=[ei_ratio],   # NEW
)
```

- [ ] **Step 4: Run, verify pass**

`venv/bin/pytest tests/recipes/test_vision_custom.py -v` → 2 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/circuitry/recipes/vision.py tests/recipes/test_vision_custom.py
git commit -m "feat(recipes): port diagnose_ei_bottlenecks kernel as vision custom diag"
```

---

### Task N8: Port `diagnose_signal_prop.py` kernel

**Files:**
- Modify: `src/circuitry/recipes/vision.py`
- Modify: `tests/recipes/test_vision_custom.py`

The diagnostic computes activation-norm decay across vision-stack depth. Kernel: per matched activation, compute mean L2 norm; emit one scalar per layer index.

- [ ] **Step 1: Inspect**

```bash
sed -n '1,50p' ~/workspace/mendu/scripts/diagnose_signal_prop.py
```

- [ ] **Step 2: Add test**

Append to `tests/recipes/test_vision_custom.py`:

```python
def test_signal_prop_depth_emits_per_activation():
    from circuitry.recipes.vision import signal_prop_depth
    act1 = torch.randn(2, 8, 16, 16)
    act2 = torch.randn(2, 8, 16, 16) * 2.0
    ctx = StepContext(step=0, model=nn.Identity(),
                      activations={"blocks.0.attn": act1, "blocks.1.attn": act2})
    out = signal_prop_depth(ctx)
    assert "vision/signal_prop/blocks.0.attn/mean_l2" in out
    assert "vision/signal_prop/blocks.1.attn/mean_l2" in out
    # blocks.1.attn was scaled 2x → ~2x larger
    assert out["vision/signal_prop/blocks.1.attn/mean_l2"] > \
           out["vision/signal_prop/blocks.0.attn/mean_l2"]
```

- [ ] **Step 3: Implement**

Append to `src/circuitry/recipes/vision.py`:

```python
def signal_prop_depth(ctx: StepContext) -> dict[str, float]:
    """Per-activation mean L2 norm — depth-of-signal-propagation diagnostic.

    Ports the kernel of mendu/scripts/diagnose_signal_prop.py: each hooked
    activation reduces to its mean L2 norm across the batch.
    """
    out: dict[str, float] = {}
    for name, act in ctx.activations.items():
        flat = act.detach().to(torch.float32).flatten(1)
        out[f"vision/signal_prop/{name}/mean_l2"] = float(flat.norm(dim=1).mean().item())
    return out
```

Add to `RECIPE.custom`: `custom=[ei_ratio, signal_prop_depth]`.

- [ ] **Step 4: Run, verify pass**

→ PASS.

- [ ] **Step 5: Commit**

```bash
git add src/circuitry/recipes/vision.py tests/recipes/test_vision_custom.py
git commit -m "feat(recipes): port diagnose_signal_prop kernel as vision custom diag"
```

---

### Task N9: Port `diagnose_trained_pc.py` kernel

**Files:**
- Modify: `src/circuitry/recipes/vision.py`
- Modify: `tests/recipes/test_vision_custom.py`

Kernel: per-channel mean L2 norm of weights, plus participation ratio across channels — a "trained predictive-coding signature."

- [ ] **Step 1: Inspect**

```bash
sed -n '1,40p' ~/workspace/mendu/scripts/diagnose_trained_pc.py
```

- [ ] **Step 2: Add test**

Append:

```python
def test_trained_pc_stats_emits_per_conv():
    from circuitry.recipes.vision import trained_pc_stats
    w = torch.randn(16, 8, 3, 3)
    ctx = StepContext(step=0, model=nn.Identity(), weights={"conv1.weight": w})
    out = trained_pc_stats(ctx)
    assert "vision/trained_pc/conv1.weight/channel_mean_l2" in out
    assert "vision/trained_pc/conv1.weight/channel_pr" in out
```

- [ ] **Step 3: Implement**

Append to `vision.py`:

```python
def trained_pc_stats(ctx: StepContext) -> dict[str, float]:
    """Per-conv-weight channel statistics: mean L2 norm + participation ratio.

    Ports the kernel of mendu/scripts/diagnose_trained_pc.py — a 'trained
    predictive-coding signature' measured as the spread of channel-wise L2
    norms (participation ratio of the squared-norms distribution).
    """
    out: dict[str, float] = {}
    for name, w in ctx.weights.items():
        if w.dim() != 4:
            continue
        # (out, in, kh, kw) → channel-norm = norm over (in, kh, kw) → (out,)
        ch_norms = w.detach().to(torch.float32).flatten(1).norm(dim=1)  # (out,)
        out[f"vision/trained_pc/{name}/channel_mean_l2"] = float(ch_norms.mean().item())
        sq = ch_norms ** 2
        # Participation ratio: (sum sq)^2 / sum(sq^2)
        pr = float((sq.sum() ** 2 / (sq ** 2).sum()).item()) if sq.numel() > 0 else 0.0
        out[f"vision/trained_pc/{name}/channel_pr"] = pr
    return out
```

Add to `RECIPE.custom`: `custom=[ei_ratio, signal_prop_depth, trained_pc_stats]`.

- [ ] **Step 4: Run, verify pass**

→ PASS.

- [ ] **Step 5: Commit**

```bash
git add src/circuitry/recipes/vision.py tests/recipes/test_vision_custom.py
git commit -m "feat(recipes): port diagnose_trained_pc kernel as vision custom diag"
```

---

### Task N10: Wire `norms_per_param` + `sv_histogram` into the stock LLM recipe

**Files:**
- Modify: `src/circuitry/recipes/llm.py`
- Test: `tests/recipes/test_llm.py` (extend)

After the universal extensions, the stock LLM recipe should emit these by default — mendu's cutover depends on it.

- [ ] **Step 1: Add the failing test**

Append to `tests/recipes/test_llm.py`:

```python
def test_llm_recipe_has_sv_histogram_and_per_param_grad():
    from circuitry.recipes import get_recipe
    r = get_recipe("llm")
    assert "sv_histogram" in r.weight_diagnostics
    assert "norms_per_param" in r.gradient_diagnostics
```

- [ ] **Step 2: Run, verify fail**

→ FAIL.

- [ ] **Step 3: Implement**

In `src/circuitry/recipes/llm.py`, append to `weight_diagnostics` and `gradient_diagnostics`:

```python
RECIPE = Recipe(
    name="llm",
    hook_points=[...],
    weight_diagnostics=["effective_rank", "stable_rank", "heavy_tail_alpha", "sv_histogram"],
    activation_diagnostics=["dead_fraction", "participation_ratio"],
    gradient_diagnostics=["layer_norm", "norms_per_param"],
)
```

- [ ] **Step 4: Run, verify pass**

→ PASS.

- [ ] **Step 5: Commit**

```bash
git add src/circuitry/recipes/llm.py tests/recipes/test_llm.py
git commit -m "feat(recipes): LLM recipe emits sv_histogram + per-param grad norms"
```

---

### Task N11: Version bump 0.1.0 → 0.2.0a0, integration test

**Files:**
- Modify: `src/circuitry/__init__.py`
- Modify: `pyproject.toml`
- Test: `tests/test_public_api.py` (extend) + run full suite

- [ ] **Step 1: Bump version**

In `src/circuitry/__init__.py`: `__version__ = "0.2.0a0"`.
In `pyproject.toml`: `version = "0.2.0a0"`.

Also add to `__init__.py` re-exports:

```python
from circuitry.core.activation import token_similarity
from circuitry.core.weight import update_delta, direction_cosine
from circuitry.recipes._discovery import discover
```

And extend the `__all__` list.

- [ ] **Step 2: Add the failing test**

Append to `tests/test_public_api.py`:

```python
def test_v02_surface_exports():
    import circuitry
    assert circuitry.__version__ == "0.2.0a0"
    assert hasattr(circuitry, "token_similarity")
    assert hasattr(circuitry, "update_delta")
    assert hasattr(circuitry, "direction_cosine")
    assert hasattr(circuitry, "discover")
```

- [ ] **Step 3: Run full suite**

```bash
cd ~/workspace/circuitry && venv/bin/ruff check src tests
cd ~/workspace/circuitry && venv/bin/pytest tests/ -q
```

Both must pass. Expect ~95 tests total (82 from v0.1.0 + new ones).

- [ ] **Step 4: Verify wheel still builds**

```bash
cd ~/workspace/circuitry && venv/bin/python -m build --wheel
```

Inspect `dist/circuitry-0.2.0a0-py3-none-any.whl` exists; clean up.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml src/circuitry/__init__.py tests/test_public_api.py
git commit -m "chore: bump version to 0.2.0a0 and re-export new surface"
```

---

## Phase 2 — Parity harness (~3 tasks)

Wire the body of `scripts/parity_check.py` against `~/workspace/mendu`'s pre-cutover state. Compare TB scalars from a tiny canonical run under both pipelines.

### Task P1: Parity-script body — dual-pipeline canonical run

**Files:**
- Modify: `scripts/parity_check.py`

The harness should:
1. Build a tiny LLaMA-shaped model (8-dim, 2-layer) with fixed seed.
2. Run N training steps under mendu's `InspectionRecorder` (capturing its TB scalars).
3. Reset model + RNG state, run N steps under circuitry's `Recorder(recipe="llm")`.
4. Extract the **universal-feature** scalar streams from both event-file dirs.
5. Compare by metric with per-bucket tolerances.

- [ ] **Step 1: Write the harness body**

Replace `scripts/parity_check.py` body. Key sketch (full code in the file):

```python
def _build_tiny_llama():
    import torch
    torch.manual_seed(0)
    # ... tiny LLaMA: 64-dim, 2 layers, 16-token sequence
    ...

def _run_mendu(mendu_root, model, optimizer, batch, steps, out_dir):
    import sys
    sys.path.insert(0, str(mendu_root))
    from tools.inspect_checkpoint.live import Cadence, InspectionRecorder
    rec = InspectionRecorder(
        run_dir=out_dir, model=model, optimizer=optimizer,
        cadence=Cadence(step_scalars=1, weight_cheap=1),
    )
    for step in range(steps):
        out = model(batch)
        loss = out.sum()
        loss.backward()
        rec.on_step_pre_optim(step, total=loss.item(), lm_loss=loss.item(),
                              aux_loss=0.0, lr=1e-3)
        if step % 5 == 0:
            rec.on_checkpoint(step)
        optimizer.step()
        optimizer.zero_grad()
    rec.close()

def _run_circuitry(model, optimizer, batch, steps, out_dir):
    from circuitry import Recorder
    rec = Recorder(model, run_dir=out_dir, recipe="llm",
                   writer="tensorboard", every_n_steps=1)
    rec.attach()
    for step in range(steps):
        out = model(batch)
        loss = out.sum()
        loss.backward()
        rec.step(step, loss=loss.item(),
                 loss_components={"lm_loss": loss.item(), "lr": 1e-3})
        optimizer.step()
        optimizer.zero_grad()
    rec.detach()
```

(Use `_build_tiny_llama` to produce a model whose `state_dict` naming matches mendu's `arch_discovery` patterns — `layers.N.attention.wq.weight` etc. — otherwise discovery returns null roles and per-role aggregation drops out of the comparison.)

- [ ] **Step 2: Sanity-run with no comparison**

```bash
cd ~/workspace/circuitry && venv/bin/python scripts/parity_check.py \
  --mendu-root ~/workspace/mendu --steps 10
```

Expect both runs to complete without error; no parity assertion yet.

- [ ] **Step 3: Commit**

```bash
git add scripts/parity_check.py
git commit -m "feat(parity): wire dual-pipeline canonical run in parity_check.py"
```

---

### Task P2: Scalar extractor + tolerance comparator

**Files:**
- Modify: `scripts/parity_check.py`
- Create: `tests/parity/test_smoke.py`

- [ ] **Step 1: Add the comparator**

In `scripts/parity_check.py`, after the dual runs, extract TB scalars from both event-file dirs and compare:

```python
from tensorboard.backend.event_processing import event_accumulator

def _load_scalars(run_dir):
    ea = event_accumulator.EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    ea.Reload()
    return {tag: ea.Scalars(tag) for tag in ea.Tags()["scalars"]}

UNIVERSAL_TAGS = {
    # mendu tag → circuitry tag (post-cutover naming)
    "train/lm_loss": "train/lm_loss",
    "train/lr": "train/lr",
    "grad/global/total_norm": "grad/global/total_norm",
    # spectral per-param: mendu uses tag_for_param_scalar()
    # which produces "spectral/per_param/<name>/effective_rank"
    # circuitry emits the same — see N10
}

def _compare(mendu_scalars, circuitry_scalars):
    failures = []
    for mendu_tag, circuitry_tag in UNIVERSAL_TAGS.items():
        if mendu_tag not in mendu_scalars or circuitry_tag not in circuitry_scalars:
            failures.append(f"missing tag: {mendu_tag} / {circuitry_tag}")
            continue
        bucket = "svd" if any(s in mendu_tag for s in ("effective_rank",
                                                       "condition_number",
                                                       "heavy_tail_alpha",
                                                       "stable_rank",
                                                       "singular_values"))
                 else "default"
        tol = DEFAULT_TOLERANCES[bucket]
        for m, c in zip(mendu_scalars[mendu_tag], circuitry_scalars[circuitry_tag]):
            if not _close(m.value, c.value, tol):
                failures.append(f"{mendu_tag}: {m.value} != {c.value} at step {m.step}")
    return failures

def _close(a, b, tol):
    return abs(a - b) <= tol["atol"] + tol["rtol"] * abs(b)
```

- [ ] **Step 2: Smoke-test the comparator**

Create `tests/parity/test_smoke.py` — a non-mendu test that just exercises `_compare` against synthetic data:

```python
def test_compare_identical_scalars_passes():
    from scripts.parity_check import _compare, DEFAULT_TOLERANCES
    from collections import namedtuple
    Scalar = namedtuple("Scalar", "step value")
    same = {"train/lm_loss": [Scalar(0, 0.5), Scalar(1, 0.4)]}
    out = _compare(same, same)
    assert out == [] or all("missing tag" in f for f in out)
```

(The "missing tag" allowance covers tags in `UNIVERSAL_TAGS` that aren't in the synthetic dicts — fine for a smoke test.)

- [ ] **Step 3: Verify scripts importable**

```bash
cd ~/workspace/circuitry && venv/bin/pytest tests/parity/test_smoke.py -v
```

- [ ] **Step 4: Commit**

```bash
git add scripts/parity_check.py tests/parity/test_smoke.py
git commit -m "feat(parity): scalar extractor + tolerance comparator"
```

---

### Task P3: Run parity, ratchet tolerances, commit numbers

**Files:**
- Create: `parity_results.md`

- [ ] **Step 1: Run against mendu**

```bash
cd ~/workspace/circuitry && venv/bin/python scripts/parity_check.py \
  --mendu-root ~/workspace/mendu --steps 20 2>&1 | tee /tmp/parity.out
```

Read `/tmp/parity.out`. For each scalar that fails: is the divergence within `1e-4`? `1e-3`? Larger? The tolerance bucket may need adjustment if our defaults were too tight.

- [ ] **Step 2: Iterate**

If a failure is a real bug (wrong primitive port): fix it in circuitry, commit a new task, re-run.
If a failure is a tolerance gap (LAPACK / GPU vs. CPU determinism), update `DEFAULT_TOLERANCES` in `scripts/parity_check.py`.

Continue until parity passes for the universal-feature subset.

- [ ] **Step 3: Document numbers**

Create `parity_results.md` at repo root:

```markdown
# circuitry parity vs. mendu (M2)

Date: 2026-MM-DD
Hardware: <CPU model / GPU / RAM>
Steps: 20
Model: tiny LLaMA (64-dim, 2 layers, 16-token batch)

## Universal-feature scalar parity

| Metric | bucket | max rel diff | tolerance | status |
|---|---|---|---|---|
| train/lm_loss | default | <fill in> | rtol=1e-5 | ✓ |
| train/lr | default | <fill in> | rtol=1e-5 | ✓ |
| grad/global/total_norm | default | <fill in> | rtol=1e-5 | ✓ |
| spectral/per_param/.../effective_rank | svd | <fill in> | rtol=1e-4 | ✓ |
| spectral/per_param/.../stable_rank | svd | <fill in> | rtol=1e-4 | ✓ |
...

## Mendu-only scalars (paper2 Recipe, no parity)

These are emitted by mendu's pre-cutover pipeline but not by circuitry's
stock LLM recipe. Post-cutover, mendu's paper2 Recipe re-emits them; the
parity check excludes them by design.

- train/route/{k}_frac
- eval/clean_ppl
- eval/ei_balance/per_layer_*
- optim/per_param/*/adam_{m,v}_norm
- direction/per_param/*/cos_consecutive_updates
- weight/per_param/*/update_delta
```

- [ ] **Step 4: Commit**

```bash
git add parity_results.md scripts/parity_check.py  # if tolerances changed
git commit -m "feat(parity): record parity numbers vs. mendu pre-cutover"
```

---

## Phase 3 — mendu-side cutover (~7 tasks)

These tasks land in `~/workspace/mendu` (different git repo). Use that repo's venv: `~/workspace/mendu/venv/bin/...`. Mendu's own CI / tests need to stay green.

### Task Q1: Install circuitry into mendu's venv

**Files:** mendu venv only.

- [ ] **Step 1: Editable install**

```bash
cd ~/workspace/mendu && venv/bin/pip install -e ~/workspace/circuitry
```

- [ ] **Step 2: Verify importable**

```bash
cd ~/workspace/mendu && venv/bin/python -c "import circuitry; print(circuitry.__version__)"
```

Expect `0.2.0a0`.

- [ ] **Step 3: Note: do not uninstall `latent_inspect_checkpoint` yet** — it stays installed until Q7. Mendu's existing code still imports it.

- [ ] **Step 4: Commit (no files; document in mendu's own git log)**

```bash
cd ~/workspace/mendu && git commit --allow-empty -m "chore: install circuitry editable into mendu venv (M2 cutover begins)"
```

---

### Task Q2: Write mendu's paper2 custom Recipe

**Files:**
- Create: `~/workspace/mendu/paper2/circuitry_recipe.py`
- Test: `~/workspace/mendu/paper2/tests/test_circuitry_recipe.py`

A single Recipe that re-emits everything paper2 needs that circuitry doesn't ship: MoE route fractions, eval-batch perplexity, EI balance per layer, Adam moment norms, direction cosine, attention capture (if `capture_attn=True`).

Per the hybrid plan, mendu owns this; the user supplies state via `Recorder.step(**kwargs)` → `ctx.user`.

- [ ] **Step 1: Write the failing test**

Create `paper2/tests/test_circuitry_recipe.py`:

```python
import torch
import torch.nn as nn

from paper2.circuitry_recipe import PAPER2_RECIPE, eval_ppl, ei_balance


def test_paper2_recipe_registered():
    from circuitry.recipes import get_recipe
    assert get_recipe("paper2").name == "paper2"


def test_eval_ppl_uses_eval_batch_from_user():
    # Provide eval_batch via ctx.user
    model = nn.Linear(8, 8)
    from circuitry.recorder.hooks import StepContext
    eb = torch.randn(2, 8)
    ctx = StepContext(step=0, model=model, user={"eval_batch": eb})
    out = eval_ppl(ctx)
    assert "eval/clean_ppl" in out


def test_ei_balance_per_layer():
    # Provide activations keyed by paper2 naming
    from circuitry.recorder.hooks import StepContext
    act = {"blocks.0": torch.cat([torch.ones(2, 4), -torch.ones(2, 4)], dim=1)}
    ctx = StepContext(step=0, model=nn.Identity(), activations=act)
    out = ei_balance(ctx)
    assert "eval/ei_balance/per_layer_0" in out
```

- [ ] **Step 2: Run, verify fail**

```bash
cd ~/workspace/mendu && venv/bin/pytest paper2/tests/test_circuitry_recipe.py -v
```

→ FAIL (module missing).

- [ ] **Step 3: Implement**

Create `~/workspace/mendu/paper2/circuitry_recipe.py`:

```python
"""Paper2 custom Recipe — bridges mendu-specific diagnostics to circuitry.

Registers a "paper2" recipe that emits everything mendu's old
InspectionRecorder did beyond circuitry's stock LLM recipe surface:

- eval/clean_ppl from a held-out eval batch (requires ``eval_batch`` in
  Recorder.step(**kwargs); thread it through ctx.user).
- eval/ei_balance/per_layer_<idx> from block activations.
- train/route/<k>_frac from MoE expert routing fractions (requires
  ``route_fractions: dict[str, float]`` via ctx.user).
- optim/per_param/<name>/adam_{m,v}_norm from Adam optimizer state (requires
  ``optimizer`` via ctx.user).
- direction/per_param/<name>/cos_consecutive_updates from weight snapshots
  (requires ``prev_state`` and ``prev_prev_state`` via ctx.user).
- weight/per_param/<name>/update_delta from same snapshots.

The hook_points and stock diagnostics inherit from the LLM recipe.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from circuitry.core.weight import direction_cosine, update_delta
from circuitry.recipes import Recipe, register_recipe
from circuitry.recipes.llm import RECIPE as _LLM_RECIPE
from circuitry.recorder.hooks import StepContext


def eval_ppl(ctx: StepContext) -> dict[str, float]:
    eb = ctx.user.get("eval_batch")
    if eb is None:
        return {}
    with torch.no_grad():
        logits = ctx.model(eb)
        if logits.dim() == 3:
            logits = logits.view(-1, logits.shape[-1])
            targets = eb.view(-1) if eb.dim() == 2 else eb.flatten()
            loss = F.cross_entropy(logits, targets)
        else:
            loss = (logits ** 2).mean()
    return {"eval/clean_ppl": float(loss.exp().item())}


def ei_balance(ctx: StepContext) -> dict[str, float]:
    out: dict[str, float] = {}
    for name, act in ctx.activations.items():
        # name like "blocks.0", "blocks.1.attn"; pull layer idx
        import re
        m = re.match(r"blocks\.(\d+)", name)
        if not m:
            continue
        idx = int(m.group(1))
        flat = act.detach().to(torch.float32).flatten()
        out[f"eval/ei_balance/per_layer_{idx}"] = float((flat > 0).float().mean().item())
    return out


def route_fractions(ctx: StepContext) -> dict[str, float]:
    rf = ctx.user.get("route_fractions")
    if not rf:
        return {}
    return {f"train/route/{k}_frac": float(v) for k, v in rf.items()}


def adam_moments(ctx: StepContext) -> dict[str, float]:
    optimizer = ctx.user.get("optimizer")
    if optimizer is None:
        return {}
    out: dict[str, float] = {}
    for name, p in ctx.model.named_parameters():
        state = optimizer.state.get(p)
        if state is None:
            continue
        m = state.get("exp_avg")
        v = state.get("exp_avg_sq")
        if m is not None:
            out[f"optim/per_param/{name}/adam_m_norm"] = float(m.norm().item())
        if v is not None:
            out[f"optim/per_param/{name}/adam_v_norm"] = float(v.norm().item())
    return out


def weight_dynamics(ctx: StepContext) -> dict[str, float]:
    prev = ctx.user.get("prev_state")
    prev_prev = ctx.user.get("prev_prev_state")
    if prev is None:
        return {}
    sd_now = {n: p.detach().cpu() for n, p in ctx.model.named_parameters()}
    out: dict[str, float] = {}
    for name, val in update_delta(sd_now, prev).items():
        out[f"weight/per_param/{name}/update_delta"] = val
    if prev_prev is not None:
        for name, val in direction_cosine(sd_now, prev, prev_prev).items():
            out[f"direction/per_param/{name}/cos_consecutive_updates"] = val
    return out


PAPER2_RECIPE = Recipe(
    name="paper2",
    hook_points=_LLM_RECIPE.hook_points,
    weight_diagnostics=_LLM_RECIPE.weight_diagnostics,
    activation_diagnostics=_LLM_RECIPE.activation_diagnostics,
    gradient_diagnostics=_LLM_RECIPE.gradient_diagnostics,
    custom=[eval_ppl, ei_balance, route_fractions, adam_moments, weight_dynamics],
)


def register() -> None:
    register_recipe(PAPER2_RECIPE)


register()
```

- [ ] **Step 4: Run, verify pass**

```bash
cd ~/workspace/mendu && venv/bin/pytest paper2/tests/test_circuitry_recipe.py -v
```

→ PASS.

- [ ] **Step 5: Commit (in mendu)**

```bash
cd ~/workspace/mendu && git add paper2/circuitry_recipe.py paper2/tests/test_circuitry_recipe.py
cd ~/workspace/mendu && git commit -m "feat(paper2): add custom circuitry Recipe for paper2-specific diagnostics"
```

---

### Task Q3: Rewrite `paper2/bet1_surprise/train/train_350m.py` call-site

**Files:**
- Modify: `~/workspace/mendu/paper2/bet1_surprise/train/train_350m.py`

Original (around L18 + L817-826) imports `InspectionRecorder` and `Cadence` and constructs the recorder. Replace with `Recorder(recipe="paper2", ...)` and thread paper2-specific state via `**kwargs`.

- [ ] **Step 1: Inspect**

```bash
sed -n '15,30p' ~/workspace/mendu/paper2/bet1_surprise/train/train_350m.py
# Find the line where InspectionRecorder is constructed (similar to bet2_daleian L817-826).
grep -n "InspectionRecorder\|Cadence" ~/workspace/mendu/paper2/bet1_surprise/train/train_350m.py
```

- [ ] **Step 2: Replace the import**

Replace `from tools.inspect_checkpoint.live import InspectionRecorder, Cadence` (and the `try/except` wrapper around it) with:

```python
try:
    from circuitry import Recorder
    import paper2.circuitry_recipe  # noqa: F401  registers "paper2" recipe
except Exception:  # pragma: no cover - optional dep
    Recorder = None
```

- [ ] **Step 3: Replace the construction site**

Find the `InspectionRecorder(...)` construction (uses keyword args `run_dir`, `model`, `optimizer`, `eval_batch`, `cadence`, ...). Replace with:

```python
if Recorder is not None:
    inspector = Recorder(
        model=model,
        run_dir=run_dir,
        recipe="paper2",
        writer="tensorboard",
        every_n_steps=log_interval,
    )
    inspector.attach()
    # paper2 state passed through ctx.user in each step()
```

- [ ] **Step 4: Replace `on_step_pre_optim` + `on_checkpoint` calls**

mendu's training loop calls `inspector.on_step_pre_optim(step, total=...)` and `inspector.on_checkpoint(step)` at different cadences. Replace both with a single `inspector.step(step, loss=..., loss_components=..., **paper2_kwargs)` per training step:

```python
inspector.step(
    step,
    loss=total.item(),
    loss_components={"lm_loss": lm_loss.item(),
                     "aux_loss": aux.item(), "lr": lr,
                     "total_loss": total.item()},
    eval_batch=eval_batch if step % ckpt_interval == 0 else None,
    optimizer=optimizer if step % ckpt_interval == 0 else None,
    prev_state=resume_state.inspector_prev_state,
    prev_prev_state=resume_state.inspector_prev_prev_state,
    route_fractions=routes if "routes" in locals() else None,
)
```

The `eval_batch` / `optimizer` keys are only populated at checkpoint cadence — this preserves mendu's two-tier behavior in a single call signature.

- [ ] **Step 5: Run mendu's training-loop smoke test**

```bash
cd ~/workspace/mendu && venv/bin/pytest paper2/tests/ -k "smoke or train_350m" -v
```

Expect existing tests to fail until Q6 (test rewrites). Note the failure pattern for cross-check; proceed.

- [ ] **Step 6: Commit (in mendu)**

```bash
cd ~/workspace/mendu && git add paper2/bet1_surprise/train/train_350m.py
cd ~/workspace/mendu && git commit -m "feat(paper2): cut bet1 train_350m.py over to circuitry.Recorder"
```

---

### Task Q4: Rewrite `paper2/bet1_surprise/train/train_350m_ste.py`

Same pattern as Q3. Different file. Same edit shape.

- [ ] **Step 1-6:** Apply the Q3 edits to `~/workspace/mendu/paper2/bet1_surprise/train/train_350m_ste.py`. Commit message: `feat(paper2): cut bet1 train_350m_ste.py over to circuitry.Recorder`.

---

### Task Q5: Rewrite `paper2/bet2_daleian/train/train_350m.py`

Same pattern. Note that bet2_daleian uses `_InspectionModelView(model)` (a model adapter); the circuitry call should pass the bare `model`, not the adapter, since circuitry traverses `named_modules()` directly.

- [ ] **Step 1-6:** Apply Q3 pattern. Replace `model=_InspectionModelView(model)` with `model=model`. Commit message: `feat(paper2): cut bet2 train_350m.py over to circuitry.Recorder`.

---

### Task Q6: Migrate mendu's 5 affected tests

**Files:**
- `~/workspace/mendu/paper2/tests/inspect_checkpoint/test_live_recorder_smoke.py`
- `~/workspace/mendu/paper2/tests/inspect_checkpoint/test_eval_batch_pinning.py`
- `~/workspace/mendu/paper2/tests/inspect_checkpoint/test_arch_detect.py`
- `~/workspace/mendu/paper2/bet2_daleian/tests/test_spectral_diagnostics.py`
- `~/workspace/mendu/paper2/bet2_daleian/tests/test_spectral_at_depth.py`

- [ ] **Step 1: test_live_recorder_smoke.py**

Replace `InspectionRecorder` construction with `Recorder(model, run_dir, recipe="paper2", writer="null", every_n_steps=1)`. Replace `rec.on_step_pre_optim` / `rec.on_checkpoint` with `rec.step(...)`. Verify the recorder emits scalars (use `RecordingWriter` test double or `writer="jsonl"` and inspect `metrics.jsonl`).

- [ ] **Step 2: test_eval_batch_pinning.py**

This test verifies the eval-batch SHA is logged. Update to verify the paper2 recipe's `eval_ppl` custom diag is invoked when `eval_batch` is in `ctx.user`. (The SHA logging in `InspectionRecorder.__init__` was an artifact of the old API; if mendu still needs it, add an `eval_batch_sha=` ctx.user passthrough and a separate paper2 custom diag function.)

- [ ] **Step 3: test_arch_detect.py**

Replace `from tools.inspect_checkpoint.arch_hooks import detect_arch` with `from circuitry.recipes._discovery import discover` (note the renamed concept). Update assertions accordingly. mendu's old `detect_arch` returned a string like "llama" / "diff" / "daleian"; circuitry's `discover()` returns a `Discovery` object — the test should switch from "arch kind string" assertions to "expected param roles present" assertions.

- [ ] **Step 4: test_spectral_diagnostics.py**

Replace `from paper2.bet2_daleian.analysis.spectral_diagnostics import effective_rank, ...` with `from circuitry.core.spectral import effective_rank` and `from circuitry.core.activation import token_similarity`. Adjust API shape (mendu's `effective_rank` returned a 0-d tensor; circuitry's returns a Python float — see core/spectral.py).

- [ ] **Step 5: test_spectral_at_depth.py**

This test exercises paper2-specific analysis. The `spectral_at_depth.py` file itself is paper2-internal and stays — only its **imports** from `spectral_diagnostics` need redirecting to `circuitry.core.spectral`. Update the imports in both the test and the source.

- [ ] **Step 6: Run mendu's test suite**

```bash
cd ~/workspace/mendu && venv/bin/pytest paper2/ -q
```

All affected tests must pass. Investigate any failures before proceeding.

- [ ] **Step 7: Commit (in mendu)**

```bash
cd ~/workspace/mendu && git add paper2/tests/inspect_checkpoint/*.py paper2/bet2_daleian/tests/test_spectral_*.py paper2/bet2_daleian/analysis/spectral_at_depth.py
cd ~/workspace/mendu && git commit -m "test(paper2): migrate inspect_checkpoint + spectral tests to circuitry"
```

---

### Task Q7: Delete mendu's in-tree inspector + spectral_diagnostics; uninstall latent_inspect_checkpoint

**Files:**
- Delete: `~/workspace/mendu/tools/inspect_checkpoint/` (whole dir)
- Delete: `~/workspace/mendu/paper2/bet2_daleian/analysis/spectral_diagnostics.py`
- Modify: `~/workspace/mendu/CLAUDE.md` (install line)

- [ ] **Step 1: Confirm no remaining importers**

```bash
cd ~/workspace/mendu && grep -rn "tools.inspect_checkpoint\|paper2.bet2_daleian.analysis.spectral_diagnostics\|latent_inspect_checkpoint" --include="*.py" 2>/dev/null
```

Must return zero lines. If anything remains, redirect those imports before deleting.

- [ ] **Step 2: Delete files**

```bash
cd ~/workspace/mendu && rm -rf tools/inspect_checkpoint/
cd ~/workspace/mendu && rm paper2/bet2_daleian/analysis/spectral_diagnostics.py
```

(Keep `mamba_diff_diagnostics.py` — only its docstring mentions spectral_diagnostics, no runtime import.)

- [ ] **Step 3: Uninstall latent_inspect_checkpoint**

```bash
cd ~/workspace/mendu && venv/bin/pip uninstall -y latent_inspect_checkpoint
```

- [ ] **Step 4: Verify mendu's tests still pass**

```bash
cd ~/workspace/mendu && venv/bin/pytest paper2/ -q
```

All green.

- [ ] **Step 5: Update mendu's CLAUDE.md**

Find the line referencing `latent-superpowers/core/inspect-checkpoint` (or similar pip-install instruction). Replace with:

```
pip install -e ~/workspace/circuitry
```

Adjust surrounding context if needed (e.g. note that the recorder is now `circuitry.Recorder` not `InspectionRecorder`).

- [ ] **Step 6: Commit (in mendu)**

```bash
cd ~/workspace/mendu && git add -u && git rm -rf tools/inspect_checkpoint/
cd ~/workspace/mendu && git commit -m "chore: drop in-tree inspector + spectral_diagnostics; circuitry owns these now"
```

---

## Phase 4 — latent-superpowers-inspect cleanup (~2 tasks)

The repo at `~/workspace/latent-superpowers-inspect` has 14 subsystems; only the `inspect-checkpoint` portion is what circuitry replaces. The 13 other subsystems (mlflow, slurm, hydra, wandb, ablation-analysis, ...) are unrelated and must stay untouched.

### Task R1: Tag pre-archival snapshot, then delete inspect-checkpoint subdir + siblings

**Files:**
- Delete: `~/workspace/latent-superpowers-inspect/core/inspect-checkpoint/`
- Delete: `~/workspace/latent-superpowers-inspect/tests/inspect-checkpoint/`
- Delete: `~/workspace/latent-superpowers-inspect/adapters/*/inspect-checkpoint/` (one per adapter: claude-code, codex, gemini, opencode)

- [ ] **Step 1: Confirm scope**

```bash
cd ~/workspace/latent-superpowers-inspect && find . -type d -name "inspect-checkpoint" 2>/dev/null
```

Expect 5-6 hits (core, tests, and one per adapter dir).

- [ ] **Step 2: Tag pre-archival state for recoverability**

```bash
cd ~/workspace/latent-superpowers-inspect && git tag pre-circuitry-extraction
```

- [ ] **Step 3: Delete**

```bash
cd ~/workspace/latent-superpowers-inspect
rm -rf core/inspect-checkpoint/ tests/inspect-checkpoint/
for adapter in adapters/*/inspect-checkpoint; do rm -rf "$adapter"; done
```

- [ ] **Step 4: Verify the other 13 subsystems untouched**

```bash
cd ~/workspace/latent-superpowers-inspect && ls core/ tests/
```

Expect: still 13 core/ entries and 13 tests/ entries (everything except inspect-checkpoint).

- [ ] **Step 5: Commit**

```bash
cd ~/workspace/latent-superpowers-inspect && git add -A
cd ~/workspace/latent-superpowers-inspect && git commit -m "chore: extract inspect-checkpoint to ~/workspace/circuitry (see tag pre-circuitry-extraction)"
```

---

### Task R2: Add forwarding note to latent-superpowers-inspect README

**Files:**
- Modify: `~/workspace/latent-superpowers-inspect/README.md`

- [ ] **Step 1: Read existing README**

```bash
head -30 ~/workspace/latent-superpowers-inspect/README.md
```

- [ ] **Step 2: Add a top note**

Near the top, before the existing content, add:

```markdown
> **Note:** the `inspect-checkpoint` subsystem has been extracted into a
> standalone PyTorch library at https://github.com/vishsangale/circuitry.
> The pre-extraction snapshot is preserved at the `pre-circuitry-extraction`
> tag. The other 13 subsystems in this repo are unaffected.
```

- [ ] **Step 3: Commit**

```bash
cd ~/workspace/latent-superpowers-inspect && git add README.md
cd ~/workspace/latent-superpowers-inspect && git commit -m "docs: note inspect-checkpoint extraction to circuitry"
```

---

## Phase 5 — Release (~3 tasks)

Back in `~/workspace/circuitry`.

### Task S1: Benchmark on M2 hardware, commit numbers to README

**Files:**
- Modify: `README.md` (Performance section)

The M1 README left numbers as TBD. M2 fills them in.

- [ ] **Step 1: Run the benchmark**

```bash
cd ~/workspace/circuitry && venv/bin/python scripts/bench_50m.py \
  --n-layers 8 --d-model 768 --steps 100 2>&1 | tee /tmp/bench.out
```

The script reports per-step overhead at default settings (every_n_steps=200). Record:
- baseline wall-clock (no recorder)
- with-recorder wall-clock
- overhead %

- [ ] **Step 2: Update README**

Find the Performance section (currently says "Benchmark numbers will land alongside the M2 mendu cutover"). Replace with concrete numbers:

```markdown
## Performance

Measured on <hardware> (2026-MM-DD): at `every_n_steps=200` on a 50M-param
8-layer 768-dim decoder transformer, circuitry adds <X>% to per-step
wall-clock (baseline <B>ms → with-recorder <R>ms). The ≤10% budget from
`docs/design.md` §10 holds at default settings.
```

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: commit M2 benchmark numbers (50M decoder, every_n_steps=200)"
```

---

### Task S2: Rewrite `docs/design.md` §7 to reflect actual M2

**Files:**
- Modify: `docs/design.md` §7

The original §7 described a "replace imports" cutover that turned out to be an API rewrite + scalar-surface delta. Rewrite §7 to reflect what M2 actually did.

- [ ] **Step 1: Replace §7 body**

In `docs/design.md`, replace the body of `## 7. Migration plan — bringing mendu over` with:

```markdown
## 7. Migration plan — bringing mendu over

### Phase M1 — extract & publish (shipped 2026-05-21)

`v0.1.0` shipped: primitives, Recorder, LLM/vision/two-tower recipes,
public CLI, MIT license, GitHub release.

### Phase M2 — mendu cutover (shipped 2026-MM-DD)

`v0.2.0` shipped. The cutover was a hybrid: circuitry grew for universal
features (`loss_components` on `Recorder.step`, per-param gradient norms,
sv-histogram weight diagnostic, weight-dynamics primitives `update_delta`
and `direction_cosine`, LLaMA-family `arch_discovery` helper). Paper2-
specific features (MoE route fractions, eval-batch perplexity, EI balance,
Adam moment norms, attention capture) live in a mendu-local custom Recipe
registered via `register_recipe`, threading state through `ctx.user`.

Parity check: tiny canonical LLaMA-shaped run under both pipelines,
universal-feature TB scalars agree within `rtol=1e-5, atol=1e-7` (most
metrics) / `rtol=1e-4, atol=1e-6` (SVD-derived metrics). Numbers in
`parity_results.md`.

TB scalar names: post-cutover runs use circuitry-native naming; pre-cutover
mendu runs are not directly comparable in TensorBoard. Re-run baselines
if cross-comparison is needed.

Archival scope for `latent-superpowers-inspect`: only the `core/inspect-checkpoint/`
subdir + its tests/adapters siblings (not the whole repo — it hosts 13
other unrelated subsystems). Tag `pre-circuitry-extraction` preserves the
pre-archival state.

### Phase M3 — siblings adopt (opportunistic, no timeline)

When `rl-recsys` / `bumblebee` / `plum` / `bonsai-*` / `gpt-2` / `llm-council`
next touch their training loops, they pick up `circuitry` and the relevant
recipe. `circuitry` itself must never depend on any of these projects.
```

- [ ] **Step 2: Commit**

```bash
git add docs/design.md
git commit -m "docs: rewrite §7 migration plan to reflect actual M2 cutover"
```

---

### Task S3: Version bump to 0.2.0, tag, push, GitHub Release

**Files:**
- Modify: `pyproject.toml`, `src/circuitry/__init__.py`, `CHANGELOG.md`

- [ ] **Step 1: Drop the `a0` suffix**

`pyproject.toml`: `version = "0.2.0"`.
`src/circuitry/__init__.py`: `__version__ = "0.2.0"`.

- [ ] **Step 2: Update CHANGELOG.md**

Prepend a new entry:

```markdown
## 0.2.0 — 2026-MM-DD

Mendu cutover release. Adds universal features needed for mendu migration;
paper-specific features stay in caller-side custom Recipes.

### Added

- `circuitry.core.activation.token_similarity` — mean off-diagonal token cosine.
- `circuitry.core.weight.update_delta`, `direction_cosine` — weight dynamics primitives.
- `circuitry.recipes._discovery.discover` — LLaMA-family role/layer classifier.
- `Recorder.step(loss_components=)` — per-component scalar emission as `train/<name>`.
- `"norms_per_param"` gradient diagnostic — per-param `grad/<name>/norm` + global.
- `"sv_histogram"` weight diagnostic — per-param singular-value histograms.
- Vision recipe gains 3 custom diagnostics: `ei_ratio`, `signal_prop_depth`, `trained_pc_stats`
  (ports of mendu's `diagnose_*.py` scripts).
- LLM recipe now emits `sv_histogram` + `norms_per_param` by default.
- Benchmark numbers in README's Performance section.
- Parity-check script body (`scripts/parity_check.py`) wired against mendu.

### Changed

- `docs/design.md` §7 rewritten to reflect actual M2 cutover (hybrid approach,
  not search-and-replace).

### Migration notes (for downstream)

- Pre-cutover mendu TB scalar names (`train/lm_loss`, etc.) are now also
  emitted by circuitry directly via `loss_components=`. Some paper2-only
  scalars are caller-supplied (`eval_batch`, `optimizer`, `prev_state` go
  through `Recorder.step(**kwargs)` → `ctx.user`).
- mendu's pre-cutover and post-cutover TB runs are NOT directly comparable
  by TensorBoard tag (user-accepted naming break).
```

- [ ] **Step 3: Run full circuitry CI gates**

```bash
cd ~/workspace/circuitry && venv/bin/ruff check src tests
cd ~/workspace/circuitry && venv/bin/pytest tests/ -q
cd ~/workspace/circuitry && venv/bin/lint-imports
```

All green.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml src/circuitry/__init__.py CHANGELOG.md
git commit -m "chore: bump version to 0.2.0"
```

- [ ] **Step 5: Tag and push**

```bash
git tag -a v0.2.0 -m "v0.2.0 — mendu cutover"
git push origin main
git push origin v0.2.0
```

- [ ] **Step 6: GitHub Release**

```bash
gh release create v0.2.0 --title "v0.2.0 — mendu cutover" \
  --notes-file <(awk '/^## 0\.2\.0/,/^## 0\.1\.0/' CHANGELOG.md | sed '$d')
```

Verify the release page exists at https://github.com/vishsangale/circuitry/releases/tag/v0.2.0.

---

## Self-review (post-write, pre-execution)

**Spec coverage** vs. design.md §7 + the 4 user decisions:
- ✓ Hybrid strategy: Phase 1 grows circuitry; Phase 3 Q2 builds the paper2 Recipe.
- ✓ Scalar name break: handled by circuitry-native naming; documented in §7 rewrite.
- ✓ diagnose_*.py ports: tasks N7, N8, N9.
- ✓ token_similarity: task N1.
- ✓ Parity check: Phase 2.
- ✓ mendu cutover: Phase 3.
- ✓ latent-superpowers-inspect scope-narrowed archival: Phase 4.
- ✓ design.md §7 rewrite: Phase 5 S2.
- ✓ Bench numbers + Release: Phase 5 S1 + S3.

**Placeholder scan:**
- "fill in" appears once in Phase 2 P3 (`parity_results.md` table). Intentional — the actual numbers come from the run; the table structure is concrete. Accept.
- "TBD" / "TODO" / "implement later": none.

**Type consistency:**
- `discover()` returns `Discovery`; tests verify `Discovery.params` is a list of `ParamInfo`. ✓
- `update_delta`, `direction_cosine` return `dict[str, float]`. ✓
- `Recorder.step(loss_components: dict[str, float] | None)`. ✓
- StepContext.user used as the passthrough for paper2 state — consistent across all uses (Q2's recipe + Q3-Q5 call sites). ✓

**Cross-repo discipline:**
- Each task names its repo and uses the right venv. ✓
- mendu commits happen in `~/workspace/mendu`, not in circuitry. ✓
- latent-superpowers-inspect commits happen in that repo. ✓

**Total task count:** 23 (Phase 1: 11, Phase 2: 3, Phase 3: 7, Phase 4: 2, Phase 5: 3).

---
