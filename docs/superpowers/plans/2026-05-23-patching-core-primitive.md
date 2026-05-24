# v1.0 Core Intervention Primitive Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the core activation-patching primitive to circuitry — a new `patching/` subsystem with hook/replace/restore context manager, prompt-pair runner, pure metrics, and dual-path site resolution (HF + TransformerLens).

**Architecture:** New top-level `patching/` orchestration parallel to `recorder/` and `scan/`, with pure metric helpers in `core/patching.py`. Site resolution via `HFSiteResolver` (recipe-declared layout from HF config) + `TLSiteResolver` (lazy `transformer_lens` import). Intervention is a context manager guaranteeing restore-on-exit.

**Tech Stack:** Python 3.12, PyTorch, HF transformers (optional, for config), TransformerLens (optional, lazy).

---

## File structure

**Create:**
- `src/circuitry/core/patching.py` — pure metrics: `logit_diff`, `kl_divergence`, `ce_loss`
- `src/circuitry/patching/__init__.py` — public API re-exports
- `src/circuitry/patching/sites.py` — `Site` dataclass, `ResolvedSite`, `HFSiteResolver`, `TLSiteResolver`
- `src/circuitry/patching/intervene.py` — `patch_site()` context manager
- `src/circuitry/patching/runner.py` — `PatchRunner` + `PatchResult`
- `tests/patching/__init__.py` — test package marker
- `tests/patching/conftest.py` — shared toy models for patching tests
- `tests/patching/test_sites.py` — Site dataclass + resolution tests
- `tests/patching/test_intervene.py` — intervention context manager tests
- `tests/patching/test_runner.py` — PatchRunner tests
- `tests/core/test_patching.py` — pure metric tests

**Modify:**
- `tests/test_layering.py` — add `patching/` layering rules + `transformer_lens` to allowlist
- `docs/design.md` — contract amendment (intervention mode, repo structure, layering)

---

## Scene-setting context (for all tasks)

**CI invariants:**
- `core/` MUST NOT import from `recorder/`, `recipes/`, `writers/`, `cli/`, or `patching/`
- `patching/` may import `core/` and `recipes/`; MUST NOT import `cli/`
- No `.cuda()` in `core/`; primitives are device-deterministic
- `transformer_lens` import must be lazy (guarded); circuitry must work without it
- Use `venv/bin/pytest` and `venv/bin/python`, never `source venv/bin/activate`

**Existing patterns to follow:**
- Core primitives (see `core/lens.py`): `_as_tensor()` helper, `float()` scalar returns, `from __future__ import annotations`, f32 upcast for numerical stability
- Test patterns (see `tests/core/test_lens.py`): `pytest.approx()` for float comparison, seed with `torch.manual_seed()`, use `pytest.raises` for error cases
- Layering test (see `tests/test_layering.py`): AST-based import scanning, `FORBIDDEN` dict + `ALLOWED_ROOTS` frozenset

---

### Task 1: Pure patching metrics (`core/patching.py`)

**Files:**
- Create: `src/circuitry/core/patching.py`
- Create: `tests/core/test_patching.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/core/test_patching.py
"""Tests for patching metric primitives. Design spec §5."""
from __future__ import annotations

import pytest
import torch

from circuitry.core.patching import ce_loss, kl_divergence, logit_diff


def test_logit_diff_1d():
    logits = torch.tensor([0.0, 1.0, 3.0, 2.0])
    assert logit_diff(logits, correct=2, incorrect=1) == pytest.approx(2.0)


def test_logit_diff_2d_batch():
    logits = torch.tensor([[0.0, 1.0, 3.0], [0.0, 2.0, 4.0]])
    result = logit_diff(logits, correct=2, incorrect=1)
    assert result == pytest.approx(2.0)  # mean of (3-1, 4-2)


def test_logit_diff_3d_uses_last_token():
    logits = torch.zeros(1, 5, 4)
    logits[0, -1, 2] = 10.0
    logits[0, -1, 0] = 3.0
    assert logit_diff(logits, correct=2, incorrect=0) == pytest.approx(7.0)


def test_kl_zero_for_identical_distributions():
    torch.manual_seed(0)
    logits = torch.randn(2, 5, 16)
    kl = kl_divergence(logits, logits)
    assert kl == pytest.approx(0.0, abs=1e-5)


def test_kl_positive_for_different_distributions():
    torch.manual_seed(1)
    p = torch.randn(2, 5, 16)
    q = torch.randn(2, 5, 16)
    assert kl_divergence(p, q) > 0.0


def test_kl_chunking_matches_single_shot():
    torch.manual_seed(2)
    p = torch.randn(2, 9, 32)
    q = torch.randn(2, 9, 32)
    ref = kl_divergence(p, q, chunk_size=100_000)
    for cs in (1, 3, 7, 18):
        got = kl_divergence(p, q, chunk_size=cs)
        assert got == pytest.approx(ref, abs=1e-5), f"chunk_size={cs}"


def test_kl_1d_input():
    torch.manual_seed(3)
    p = torch.randn(16)
    assert kl_divergence(p, p) == pytest.approx(0.0, abs=1e-5)


def test_ce_loss_matches_pytorch():
    torch.manual_seed(4)
    logits = torch.randn(4, 10)
    targets = torch.randint(0, 10, (4,))
    expected = float(torch.nn.functional.cross_entropy(logits, targets).item())
    assert ce_loss(logits, targets) == pytest.approx(expected, abs=1e-5)


def test_ce_loss_3d_last_token():
    torch.manual_seed(5)
    logits = torch.randn(2, 5, 10)
    targets = torch.randint(0, 10, (2,))
    result = ce_loss(logits, targets)
    expected = float(
        torch.nn.functional.cross_entropy(logits[:, -1, :].float(), targets).item()
    )
    assert result == pytest.approx(expected, abs=1e-5)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `venv/bin/pytest tests/core/test_patching.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'circuitry.core.patching'`

- [ ] **Step 3: Write implementation**

```python
# src/circuitry/core/patching.py
"""Pure metrics for activation patching. See design spec §5.

Tensor→float functions. No model execution, no I/O, no .cuda().
"""
from __future__ import annotations

import torch
from torch import Tensor


def logit_diff(logits: Tensor, correct: int, incorrect: int) -> float:
    """Difference in logits at correct vs incorrect token.

    Accepts (vocab,), (batch, vocab), or (batch, seq, vocab). For 3-D input,
    uses the last sequence position.
    """
    x = logits.detach().float()
    if x.ndim == 3:
        x = x[:, -1, :]
    if x.ndim == 1:
        return float((x[correct] - x[incorrect]).item())
    return float((x[:, correct] - x[:, incorrect]).mean().item())


def kl_divergence(
    p_logits: Tensor,
    q_logits: Tensor,
    *,
    chunk_size: int = 256,
) -> float:
    """KL(softmax(p) || softmax(q)), mean over leading dims. Chunked."""
    p = p_logits.detach().to(torch.float32)
    q = q_logits.detach().to(torch.float32)
    if p.ndim == 1:
        p = p.unsqueeze(0)
        q = q.unsqueeze(0)
    if p.ndim == 3:
        p = p.reshape(-1, p.shape[-1])
        q = q.reshape(-1, q.shape[-1])
    n = p.shape[0]
    if n == 0:
        return 0.0
    kl_sum = p.new_zeros(())
    for start in range(0, n, max(1, chunk_size)):
        pc = p[start : start + chunk_size]
        qc = q[start : start + chunk_size]
        log_p = torch.log_softmax(pc, dim=-1)
        log_q = torch.log_softmax(qc, dim=-1)
        kl_sum = kl_sum + (log_p.exp() * (log_p - log_q)).sum()
    return float((kl_sum / n).item())


def ce_loss(logits: Tensor, targets: Tensor) -> float:
    """Cross-entropy loss, mean over batch.

    For 3-D logits (batch, seq, vocab), uses the last sequence position.
    """
    x = logits.detach().float()
    if x.ndim == 3:
        x = x[:, -1, :]
    return float(torch.nn.functional.cross_entropy(x, targets).item())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv/bin/pytest tests/core/test_patching.py -v`
Expected: 9 passed

- [ ] **Step 5: Run layering tests to confirm core invariant isn't violated**

Run: `venv/bin/pytest tests/test_layering.py -v`
Expected: All pass (core/patching.py imports only torch — no layering violation)

- [ ] **Step 6: Commit**

```bash
git add src/circuitry/core/patching.py tests/core/test_patching.py
git commit -m "feat(core): add patching metric primitives — logit_diff, kl_divergence, ce_loss"
```

---

### Task 2: Site dataclass + shared test fixtures

**Files:**
- Create: `src/circuitry/patching/__init__.py`
- Create: `src/circuitry/patching/sites.py`
- Create: `tests/patching/__init__.py`
- Create: `tests/patching/conftest.py`
- Create: `tests/patching/test_sites.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/patching/test_sites.py
"""Tests for Site dataclass + validation."""
from __future__ import annotations

import pytest

from circuitry.patching.sites import VALID_COMPONENTS, Site


def test_site_valid_components():
    for comp in VALID_COMPONENTS:
        kwargs = {"component": comp, "layer": 0}
        if comp == "attn_head_out":
            kwargs["head"] = 0
        if comp == "mlp_neuron":
            kwargs["neuron"] = 0
        site = Site(**kwargs)
        assert site.component == comp


def test_site_rejects_unknown_component():
    with pytest.raises(ValueError, match="Unknown component"):
        Site(component="bogus", layer=0)


def test_attn_head_out_requires_head():
    with pytest.raises(ValueError, match="requires head"):
        Site(component="attn_head_out", layer=0)


def test_attn_head_out_accepts_head():
    s = Site(component="attn_head_out", layer=0, head=3)
    assert s.head == 3


def test_mlp_neuron_requires_neuron():
    with pytest.raises(ValueError, match="requires neuron"):
        Site(component="mlp_neuron", layer=0)


def test_mlp_neuron_accepts_neuron():
    s = Site(component="mlp_neuron", layer=0, neuron=42)
    assert s.neuron == 42


def test_site_frozen():
    s = Site(component="resid_post", layer=0)
    with pytest.raises(AttributeError):
        s.layer = 1  # type: ignore[misc]


def test_site_hashable():
    s1 = Site(component="resid_post", layer=0)
    s2 = Site(component="resid_post", layer=0)
    assert s1 == s2
    assert hash(s1) == hash(s2)
    assert len({s1, s2}) == 1


def test_site_with_position():
    s = Site(component="resid_post", layer=0, position=3)
    assert s.position == 3
    s2 = Site(component="resid_post", layer=0, position=slice(1, 4))
    assert s2.position == slice(1, 4)
```

Create the shared test fixtures:

```python
# tests/patching/__init__.py
```

```python
# tests/patching/conftest.py
"""Shared toy models for patching tests."""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn


class ToyPatchModel(nn.Module):
    """Two-layer identity model. Patching layer 0 output → output changes to
    layer1(patched_value). With identity weights: output == input normally,
    output == patched_value when layer 0 output is patched."""

    def __init__(self, d: int = 4):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(d, d, bias=False),
            nn.Linear(d, d, bias=False),
        ])
        for layer in self.layers:
            nn.init.eye_(layer.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class FakeAttention(nn.Module):
    """Identity attention with o_proj for head-slicing tests."""

    def __init__(self, d_model: int):
        super().__init__()
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        nn.init.eye_(self.o_proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.o_proj(x)


class FakeMLP(nn.Module):
    """Identity MLP with gate_proj + down_proj for neuron-slicing tests."""

    def __init__(self, d_model: int, d_mlp: int):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_mlp, bias=False)
        self.down_proj = nn.Linear(d_mlp, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.gate_proj(x))


class FakeTransformerLayer(nn.Module):
    def __init__(self, d_model: int, d_mlp: int):
        super().__init__()
        self.self_attn = FakeAttention(d_model)
        self.mlp = FakeMLP(d_model, d_mlp)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(x)
        x = x + self.mlp(x)
        return x


class FakeTransformerModel(nn.Module):
    """Llama-like module structure: model.layers.{L}.self_attn.o_proj,
    model.layers.{L}.mlp.down_proj. NOT an HF model — just has matching names."""

    def __init__(self, n_layers: int = 2, d_model: int = 8, n_heads: int = 2,
                 d_mlp: int = 16):
        super().__init__()
        self.layers = nn.ModuleList([
            FakeTransformerLayer(d_model, d_mlp) for _ in range(n_layers)
        ])
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_mlp = d_mlp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


@pytest.fixture
def toy_model():
    return ToyPatchModel(d=4)


@pytest.fixture
def transformer_model():
    return FakeTransformerModel(n_layers=2, d_model=8, n_heads=2, d_mlp=16)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `venv/bin/pytest tests/patching/test_sites.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'circuitry.patching'`

- [ ] **Step 3: Write implementation**

```python
# src/circuitry/patching/__init__.py
"""Activation patching — interventional diagnostics. See design spec §2.

This subsystem is opt-in and isolated: every intervention is scoped to a
context manager, model state is restored on exit (including on exception),
and the model stays frozen throughout.
"""
from __future__ import annotations

from circuitry.patching.sites import Site

__all__ = ["Site"]
```

```python
# src/circuitry/patching/sites.py
"""Site dataclass + resolution for activation patching. Design spec §3."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch.nn as nn
from torch import Tensor

VALID_COMPONENTS = frozenset({
    "resid_pre",
    "resid_post",
    "attn_head_out",
    "mlp_out",
    "mlp_neuron",
})


@dataclass(frozen=True)
class Site:
    """A named intervention point in the model's computation graph."""

    component: str
    layer: int
    head: int | None = None
    neuron: int | None = None
    position: int | slice | None = None

    def __post_init__(self) -> None:
        if self.component not in VALID_COMPONENTS:
            raise ValueError(
                f"Unknown component {self.component!r}; "
                f"valid: {sorted(VALID_COMPONENTS)}"
            )
        if self.component == "attn_head_out" and self.head is None:
            raise ValueError("attn_head_out requires head index")
        if self.component == "mlp_neuron" and self.neuron is None:
            raise ValueError("mlp_neuron requires neuron index")


@dataclass
class ResolvedSite:
    """A Site resolved to a concrete module + hook functions."""

    module: nn.Module
    is_input_hook: bool
    extract: Callable[[Tensor], Tensor]
    inject: Callable[[Tensor, Tensor], Tensor]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv/bin/pytest tests/patching/test_sites.py -v`
Expected: 9 passed

- [ ] **Step 5: Run layering tests**

Run: `venv/bin/pytest tests/test_layering.py -v`
Expected: All pass (patching/sites.py imports only torch.nn and dataclasses)

- [ ] **Step 6: Commit**

```bash
git add src/circuitry/patching/__init__.py src/circuitry/patching/sites.py \
        tests/patching/__init__.py tests/patching/conftest.py tests/patching/test_sites.py
git commit -m "feat(patching): add Site dataclass with validation + shared test fixtures"
```

---

### Task 3: HF site resolution

**Files:**
- Modify: `src/circuitry/patching/sites.py`
- Create: `tests/patching/test_hf_resolution.py`

**Context:** The `HFSiteResolver` maps a `Site` to a concrete `(module, hook_type, extract_fn, inject_fn)` on an HF-style model. It reads `n_heads`, `d_model`, `d_mlp` from config (or explicit args) and resolves module paths like `layers.{L}.self_attn.o_proj`.

For `attn_head_out`: hooks `o_proj` INPUT (pre-hook), reshapes `(batch, seq, d_model)` → `(batch, seq, n_heads, head_dim)`, slices/injects at head index. Requires eager attention.

For `mlp_neuron`: hooks `down_proj` INPUT (pre-hook), indexes neuron on the last dim of the intermediate activation. Architecture-specific: Llama-family first; unsupported architectures raise a clean error.

- [ ] **Step 1: Write failing tests**

```python
# tests/patching/test_hf_resolution.py
"""Tests for HFSiteResolver."""
from __future__ import annotations

import pytest
import torch

from circuitry.patching.sites import HFSiteResolver, Site


@pytest.fixture
def resolver():
    return HFSiteResolver(
        n_heads=2, d_model=8, d_mlp=16,
        layer_pattern="layers.{L}",
        attn_module="self_attn.o_proj",
        mlp_module="mlp",
        mlp_intermediate="mlp.down_proj",
    )


def test_resid_post_resolves_to_layer_output(resolver, transformer_model):
    site = Site(component="resid_post", layer=0)
    resolved = resolver.resolve(transformer_model, site)
    assert resolved.module is transformer_model.layers[0]
    assert resolved.is_input_hook is False


def test_resid_pre_resolves_to_layer_input(resolver, transformer_model):
    site = Site(component="resid_pre", layer=1)
    resolved = resolver.resolve(transformer_model, site)
    assert resolved.module is transformer_model.layers[1]
    assert resolved.is_input_hook is True


def test_attn_head_out_resolves_to_o_proj_input(resolver, transformer_model):
    site = Site(component="attn_head_out", layer=0, head=1)
    resolved = resolver.resolve(transformer_model, site)
    assert resolved.module is transformer_model.layers[0].self_attn.o_proj
    assert resolved.is_input_hook is True


def test_mlp_out_resolves_to_mlp_output(resolver, transformer_model):
    site = Site(component="mlp_out", layer=0)
    resolved = resolver.resolve(transformer_model, site)
    assert resolved.module is transformer_model.layers[0].mlp
    assert resolved.is_input_hook is False


def test_mlp_neuron_resolves_to_down_proj_input(resolver, transformer_model):
    site = Site(component="mlp_neuron", layer=0, neuron=5)
    resolved = resolver.resolve(transformer_model, site)
    assert resolved.module is transformer_model.layers[0].mlp.down_proj
    assert resolved.is_input_hook is True


def test_extract_head_slice_correct(resolver, transformer_model):
    site = Site(component="attn_head_out", layer=0, head=1)
    resolved = resolver.resolve(transformer_model, site)
    torch.manual_seed(0)
    x = torch.randn(2, 3, 8)  # (batch, seq, d_model=8), n_heads=2, head_dim=4
    extracted = resolved.extract(x)
    expected = x.reshape(2, 3, 2, 4)[:, :, 1, :]  # head 1
    assert torch.equal(extracted, expected)


def test_inject_head_slice_correct(resolver, transformer_model):
    site = Site(component="attn_head_out", layer=0, head=0)
    resolved = resolver.resolve(transformer_model, site)
    torch.manual_seed(1)
    x = torch.randn(2, 3, 8)
    new_val = torch.ones(2, 3, 4)
    injected = resolved.inject(x, new_val)
    result_heads = injected.reshape(2, 3, 2, 4)
    assert torch.equal(result_heads[:, :, 0, :], new_val)
    assert torch.equal(result_heads[:, :, 1, :], x.reshape(2, 3, 2, 4)[:, :, 1, :])


def test_extract_neuron_correct(resolver, transformer_model):
    site = Site(component="mlp_neuron", layer=0, neuron=5)
    resolved = resolver.resolve(transformer_model, site)
    torch.manual_seed(2)
    x = torch.randn(2, 3, 16)  # (batch, seq, d_mlp=16)
    extracted = resolved.extract(x)
    assert torch.equal(extracted, x[:, :, 5])


def test_inject_neuron_correct(resolver, transformer_model):
    site = Site(component="mlp_neuron", layer=0, neuron=5)
    resolved = resolver.resolve(transformer_model, site)
    torch.manual_seed(3)
    x = torch.randn(2, 3, 16)
    new_val = torch.ones(2, 3)
    injected = resolved.inject(x, new_val)
    assert torch.equal(injected[:, :, 5], new_val)
    mask = torch.ones(16, dtype=torch.bool)
    mask[5] = False
    assert torch.equal(injected[:, :, mask], x[:, :, mask])


def test_position_slicing(resolver, transformer_model):
    site = Site(component="resid_post", layer=0, position=2)
    resolved = resolver.resolve(transformer_model, site)
    torch.manual_seed(4)
    x = torch.randn(2, 5, 8)
    extracted = resolved.extract(x)
    assert torch.equal(extracted, x[:, 2])


def test_position_inject(resolver, transformer_model):
    site = Site(component="resid_post", layer=0, position=2)
    resolved = resolver.resolve(transformer_model, site)
    torch.manual_seed(5)
    x = torch.randn(2, 5, 8)
    new_val = torch.ones(2, 8)
    injected = resolved.inject(x, new_val)
    assert torch.equal(injected[:, 2], new_val)
    assert torch.equal(injected[:, 0], x[:, 0])


def test_from_config():
    class FakeConfig:
        num_attention_heads = 4
        hidden_size = 32
        intermediate_size = 64

    resolver = HFSiteResolver.from_config(FakeConfig())
    assert resolver.n_heads == 4
    assert resolver.d_model == 32
    assert resolver.d_mlp == 64


def test_from_config_missing_fields():
    class BadConfig:
        pass

    with pytest.raises(ValueError, match="num_attention_heads"):
        HFSiteResolver.from_config(BadConfig())


def test_mlp_neuron_without_d_mlp():
    resolver = HFSiteResolver(n_heads=2, d_model=8, d_mlp=None)
    site = Site(component="mlp_neuron", layer=0, neuron=0)
    from tests.patching.conftest import FakeTransformerModel

    model = FakeTransformerModel()
    with pytest.raises(ValueError, match="d_mlp"):
        resolver.resolve(model, site)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `venv/bin/pytest tests/patching/test_hf_resolution.py -v`
Expected: FAIL — `ImportError: cannot import name 'HFSiteResolver'`

- [ ] **Step 3: Write implementation**

Add to `src/circuitry/patching/sites.py` after the `ResolvedSite` class:

```python
# --------------- position helpers ---------------

def _pos_extract(x: Tensor, pos: int | slice | None) -> Tensor:
    if pos is None:
        return x
    return x[:, pos]


def _pos_inject(full: Tensor, val: Tensor, pos: int | slice | None) -> Tensor:
    if pos is None:
        return val
    out = full.clone()
    out[:, pos] = val
    return out


# --------------- head helpers ---------------

def _extract_head(
    x: Tensor, head: int, n_heads: int, head_dim: int,
    pos: int | slice | None,
) -> Tensor:
    b = x.shape[0]
    s = x.shape[1] if x.ndim == 3 else 1
    reshaped = x.reshape(b, s, n_heads, head_dim) if x.ndim == 3 else x.reshape(b, n_heads, head_dim)
    sliced = reshaped[:, :, head, :] if x.ndim == 3 else reshaped[:, head, :]
    if pos is not None and x.ndim == 3:
        return sliced[:, pos] if isinstance(pos, int) else sliced[:, pos]
    return sliced


def _inject_head(
    full: Tensor, val: Tensor, head: int, n_heads: int, head_dim: int,
    pos: int | slice | None,
) -> Tensor:
    b, s, d = full.shape
    out = full.clone().reshape(b, s, n_heads, head_dim)
    if pos is None:
        out[:, :, head, :] = val
    else:
        out[:, pos, head, :] = val
    return out.reshape(b, s, d)


# --------------- neuron helpers ---------------

def _extract_neuron(x: Tensor, neuron: int, pos: int | slice | None) -> Tensor:
    sliced = _pos_extract(x, pos)
    return sliced[..., neuron]


def _inject_neuron(
    full: Tensor, val: Tensor, neuron: int, pos: int | slice | None,
) -> Tensor:
    out = full.clone()
    if pos is None:
        out[..., neuron] = val
    else:
        out[:, pos, neuron] = val if val.ndim > 0 else val
    return out


# --------------- module traversal ---------------

def _get_submodule(model: nn.Module, path: str) -> nn.Module:
    parts = path.split(".")
    m = model
    for p in parts:
        m = getattr(m, p)
    return m


# --------------- HF site resolver ---------------

class HFSiteResolver:
    """Resolve Sites to HF model modules using config-declared layout."""

    def __init__(
        self,
        n_heads: int,
        d_model: int,
        d_mlp: int | None = None,
        *,
        layer_pattern: str = "model.layers.{L}",
        attn_module: str = "self_attn.o_proj",
        mlp_module: str = "mlp",
        mlp_intermediate: str = "mlp.down_proj",
    ) -> None:
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_mlp = d_mlp
        self.head_dim = d_model // n_heads
        self.layer_pattern = layer_pattern
        self.attn_module = attn_module
        self.mlp_module = mlp_module
        self.mlp_intermediate = mlp_intermediate

    @classmethod
    def from_config(cls, config: Any) -> HFSiteResolver:
        n_heads = getattr(config, "num_attention_heads", None)
        d_model = getattr(config, "hidden_size", None)
        if n_heads is None or d_model is None:
            raise ValueError(
                "Config must have num_attention_heads and hidden_size"
            )
        d_mlp = getattr(config, "intermediate_size", None)
        return cls(n_heads=n_heads, d_model=d_model, d_mlp=d_mlp)

    def _layer_module(self, model: nn.Module, layer: int) -> nn.Module:
        path = self.layer_pattern.replace("{L}", str(layer))
        return _get_submodule(model, path)

    def resolve(self, model: nn.Module, site: Site) -> ResolvedSite:
        layer_mod = self._layer_module(model, site.layer)
        pos = site.position

        if site.component == "resid_pre":
            return ResolvedSite(
                module=layer_mod,
                is_input_hook=True,
                extract=lambda x, _pos=pos: _pos_extract(x, _pos),
                inject=lambda full, val, _pos=pos: _pos_inject(full, val, _pos),
            )

        if site.component == "resid_post":
            return ResolvedSite(
                module=layer_mod,
                is_input_hook=False,
                extract=lambda x, _pos=pos: _pos_extract(x, _pos),
                inject=lambda full, val, _pos=pos: _pos_inject(full, val, _pos),
            )

        if site.component == "attn_head_out":
            attn_mod = _get_submodule(layer_mod, self.attn_module)
            h, nh, hd = site.head, self.n_heads, self.head_dim
            return ResolvedSite(
                module=attn_mod,
                is_input_hook=True,
                extract=lambda x, _h=h, _nh=nh, _hd=hd, _pos=pos: _extract_head(x, _h, _nh, _hd, _pos),
                inject=lambda full, val, _h=h, _nh=nh, _hd=hd, _pos=pos: _inject_head(full, val, _h, _nh, _hd, _pos),
            )

        if site.component == "mlp_out":
            mlp_mod = _get_submodule(layer_mod, self.mlp_module)
            return ResolvedSite(
                module=mlp_mod,
                is_input_hook=False,
                extract=lambda x, _pos=pos: _pos_extract(x, _pos),
                inject=lambda full, val, _pos=pos: _pos_inject(full, val, _pos),
            )

        if site.component == "mlp_neuron":
            if self.d_mlp is None:
                raise ValueError(
                    "mlp_neuron resolution requires d_mlp (intermediate_size) in config"
                )
            intermediate_mod = _get_submodule(layer_mod, self.mlp_intermediate)
            n = site.neuron
            return ResolvedSite(
                module=intermediate_mod,
                is_input_hook=True,
                extract=lambda x, _n=n, _pos=pos: _extract_neuron(x, _n, _pos),
                inject=lambda full, val, _n=n, _pos=pos: _inject_neuron(full, val, _n, _pos),
            )

        raise ValueError(f"Unresolved component: {site.component}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv/bin/pytest tests/patching/test_hf_resolution.py -v`
Expected: All passed

- [ ] **Step 5: Run layering tests**

Run: `venv/bin/pytest tests/test_layering.py -v`
Expected: All pass

- [ ] **Step 6: Commit**

```bash
git add src/circuitry/patching/sites.py tests/patching/test_hf_resolution.py
git commit -m "feat(patching): add HFSiteResolver with head/neuron slicing for Llama-family"
```

---

### Task 4: TL site resolution + lazy import

**Files:**
- Modify: `src/circuitry/patching/sites.py`
- Modify: `tests/test_layering.py`
- Create: `tests/patching/test_tl_resolution.py`

**Context:** The `TLSiteResolver` maps `Site` → TransformerLens hook names. The `transformer_lens` import MUST be lazy (guarded behind `try/except`). The layering test must verify:
1. `transformer_lens` is in `ALLOWED_ROOTS` (so AST scanning accepts the import)
2. `import circuitry` does NOT actually import `transformer_lens` at package-import time
3. `patching/` does not import `cli/`
4. `core/` does not import `patching/`

- [ ] **Step 1: Write failing tests**

```python
# tests/patching/test_tl_resolution.py
"""Tests for TLSiteResolver + lazy transformer_lens import."""
from __future__ import annotations

import importlib
import sys

import pytest

from circuitry.patching.sites import Site, TLSiteResolver


def test_tl_hook_name_resid_pre():
    r = TLSiteResolver()
    assert r.hook_name(Site(component="resid_pre", layer=3)) == "blocks.3.hook_resid_pre"


def test_tl_hook_name_resid_post():
    r = TLSiteResolver()
    assert r.hook_name(Site(component="resid_post", layer=5)) == "blocks.5.hook_resid_post"


def test_tl_hook_name_attn_head_out():
    r = TLSiteResolver()
    assert r.hook_name(Site(component="attn_head_out", layer=2, head=3)) == "blocks.2.attn.hook_z"


def test_tl_hook_name_mlp_out():
    r = TLSiteResolver()
    assert r.hook_name(Site(component="mlp_out", layer=1)) == "blocks.1.mlp.hook_post"


def test_tl_hook_name_mlp_neuron():
    r = TLSiteResolver()
    assert r.hook_name(Site(component="mlp_neuron", layer=0, neuron=42)) == "blocks.0.mlp.hook_post"


def test_lazy_import_does_not_import_transformer_lens():
    """Importing circuitry.patching.sites must NOT import transformer_lens."""
    was_loaded = "transformer_lens" in sys.modules
    if was_loaded:
        pytest.skip("transformer_lens already loaded in this process")
    importlib.reload(importlib.import_module("circuitry.patching.sites"))
    assert "transformer_lens" not in sys.modules


def test_tl_resolver_resolve_requires_transformer_lens():
    r = TLSiteResolver()
    site = Site(component="resid_post", layer=0)

    class FakeModel:
        pass

    try:
        import transformer_lens  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match="transformer_lens"):
            r.resolve(FakeModel(), site)  # type: ignore[arg-type]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `venv/bin/pytest tests/patching/test_tl_resolution.py -v`
Expected: FAIL — `ImportError: cannot import name 'TLSiteResolver'`

- [ ] **Step 3: Update layering test**

Add to `tests/test_layering.py`:

In the `FORBIDDEN` dict, add:
```python
"core": ("circuitry.recorder", "circuitry.recipes", "circuitry.writers", "circuitry.cli", "circuitry.patching"),
"recipes": ("circuitry.cli",),
"patching": ("circuitry.cli",),
```

In `ALLOWED_ROOTS`, add `"transformer_lens"`:
```python
ALLOWED_ROOTS = frozenset({"circuitry", "torch", "numpy", "tensorboard", "sae_lens", "transformer_lens"}) | sys.stdlib_module_names
```

Add a new test:
```python
def test_patching_does_not_import_cli():
    patching_dir = SRC / "patching"
    if not patching_dir.exists():
        pytest.skip("patching/ not yet created")
    for py in patching_dir.rglob("*.py"):
        for imp in _imports(py):
            for forbidden in FORBIDDEN["patching"]:
                assert not imp.startswith(forbidden), (
                    f"patching/{py.relative_to(patching_dir)} imports {imp}, "
                    f"violating layering rule"
                )
```

Update `test_core_does_not_import_higher_layers` — no code change needed, but verify it covers `circuitry.patching` since it's now in `FORBIDDEN["core"]`.

- [ ] **Step 4: Write TLSiteResolver implementation**

Add to `src/circuitry/patching/sites.py`:

```python
# --------------- TL site resolver ---------------

_TL_HOOK_MAP = {
    "resid_pre": "blocks.{L}.hook_resid_pre",
    "resid_post": "blocks.{L}.hook_resid_post",
    "attn_head_out": "blocks.{L}.attn.hook_z",
    "mlp_out": "blocks.{L}.mlp.hook_post",
    "mlp_neuron": "blocks.{L}.mlp.hook_post",
}


class TLSiteResolver:
    """Resolve Sites to TransformerLens hook names. Lazy import."""

    def hook_name(self, site: Site) -> str:
        template = _TL_HOOK_MAP.get(site.component)
        if template is None:
            raise ValueError(f"No TL hook mapping for {site.component}")
        return template.replace("{L}", str(site.layer))

    def resolve(self, model: nn.Module, site: Site) -> ResolvedSite:
        try:
            import transformer_lens  # noqa: F401
        except ImportError:
            raise ImportError(
                "transformer_lens is required for TLSiteResolver.resolve(). "
                "Install it with: pip install transformer_lens"
            ) from None

        hook_name = self.hook_name(site)
        hook_point = model.hook_dict[hook_name]  # type: ignore[attr-defined]

        if site.component == "attn_head_out":
            head = site.head
            pos = site.position
            return ResolvedSite(
                module=hook_point,
                is_input_hook=False,
                extract=lambda x, _h=head, _pos=pos: (
                    _pos_extract(x[:, :, _h, :], _pos) if x.ndim == 4
                    else _pos_extract(x, _pos)
                ),
                inject=lambda full, val, _h=head, _pos=pos: (
                    _inject_tl_head(full, val, _h, _pos)
                ),
            )

        if site.component == "mlp_neuron":
            neuron = site.neuron
            pos = site.position
            return ResolvedSite(
                module=hook_point,
                is_input_hook=False,
                extract=lambda x, _n=neuron, _pos=pos: _extract_neuron(x, _n, _pos),
                inject=lambda full, val, _n=neuron, _pos=pos: _inject_neuron(full, val, _n, _pos),
            )

        pos = site.position
        return ResolvedSite(
            module=hook_point,
            is_input_hook=False,
            extract=lambda x, _pos=pos: _pos_extract(x, _pos),
            inject=lambda full, val, _pos=pos: _pos_inject(full, val, _pos),
        )


def _inject_tl_head(
    full: Tensor, val: Tensor, head: int, pos: int | slice | None,
) -> Tensor:
    out = full.clone()
    if pos is None:
        out[:, :, head, :] = val
    else:
        out[:, pos, head, :] = val
    return out
```

- [ ] **Step 5: Run all tests**

Run: `venv/bin/pytest tests/patching/test_tl_resolution.py tests/test_layering.py -v`
Expected: All pass

- [ ] **Step 6: Commit**

```bash
git add src/circuitry/patching/sites.py tests/patching/test_tl_resolution.py tests/test_layering.py
git commit -m "feat(patching): add TLSiteResolver + layering rules for patching/"
```

---

### Task 5: Intervention context manager (`intervene.py`)

**Files:**
- Create: `src/circuitry/patching/intervene.py`
- Create: `tests/patching/test_intervene.py`
- Modify: `src/circuitry/patching/__init__.py` (re-export `patch_site`)

**Context:** The `patch_site()` context manager is the core intervention primitive. It installs a forward hook that replaces the site activation with a supplied value, runs the forward inside the context, then guarantees restore-on-exit via `try/finally`. It also manages eval mode and param `requires_grad` (frozen model discipline).

The `ToyPatchModel` from `conftest.py` has identity weights so `output == input`. Patching `layers.0` output with a different value `z` → `output == layers.1(z) == z` (identity). This is the deterministic test signal.

- [ ] **Step 1: Write failing tests**

```python
# tests/patching/test_intervene.py
"""Tests for patch_site() context manager."""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from circuitry.patching.intervene import patch_site
from circuitry.patching.sites import HFSiteResolver, ResolvedSite, Site


@pytest.fixture
def resolver():
    return HFSiteResolver(
        n_heads=1, d_model=4, d_mlp=8,
        layer_pattern="layers.{L}",
        attn_module="self_attn.o_proj",
        mlp_module="mlp",
        mlp_intermediate="mlp.down_proj",
    )


def test_patch_changes_output(toy_model, resolver):
    """Patching layer 0 output with z → final output becomes z (identity weights)."""
    torch.manual_seed(0)
    x = torch.randn(1, 4)
    z = torch.ones(1, 4) * 99.0
    site = Site(component="resid_post", layer=0)
    normal_out = toy_model(x)

    with patch_site(toy_model, site, z, resolver):
        patched_out = toy_model(x)

    assert not torch.equal(patched_out, normal_out)
    assert torch.allclose(patched_out, z, atol=1e-6)


def test_restore_after_normal_exit(toy_model, resolver):
    """After context exits normally, model output is identical to before."""
    torch.manual_seed(1)
    x = torch.randn(1, 4)
    before = toy_model(x).clone()
    site = Site(component="resid_post", layer=0)

    with patch_site(toy_model, site, torch.zeros(1, 4), resolver):
        toy_model(x)

    after = toy_model(x)
    assert torch.equal(before, after)


def test_restore_after_exception(toy_model, resolver):
    """After context exits via exception, model output is identical to before."""
    torch.manual_seed(2)
    x = torch.randn(1, 4)
    before = toy_model(x).clone()
    site = Site(component="resid_post", layer=0)

    with pytest.raises(RuntimeError, match="intentional"):
        with patch_site(toy_model, site, torch.zeros(1, 4), resolver):
            toy_model(x)
            raise RuntimeError("intentional")

    after = toy_model(x)
    assert torch.equal(before, after)


def test_params_frozen_during_patch(toy_model, resolver):
    """All params have requires_grad=False inside the context."""
    site = Site(component="resid_post", layer=0)
    with patch_site(toy_model, site, torch.zeros(1, 4), resolver):
        for p in toy_model.parameters():
            assert not p.requires_grad


def test_params_restored_after_patch(toy_model, resolver):
    """Params that had requires_grad=True before patching get it restored."""
    for p in toy_model.parameters():
        p.requires_grad_(True)
    site = Site(component="resid_post", layer=0)

    with patch_site(toy_model, site, torch.zeros(1, 4), resolver):
        pass

    for p in toy_model.parameters():
        assert p.requires_grad


def test_eval_mode_set_and_restored_from_train(toy_model, resolver):
    toy_model.train()
    site = Site(component="resid_post", layer=0)

    with patch_site(toy_model, site, torch.zeros(1, 4), resolver):
        assert not toy_model.training

    assert toy_model.training


def test_eval_mode_stays_eval_if_already_eval(toy_model, resolver):
    toy_model.eval()
    site = Site(component="resid_post", layer=0)

    with patch_site(toy_model, site, torch.zeros(1, 4), resolver):
        assert not toy_model.training

    assert not toy_model.training


def test_no_hooks_remain_after_exit(toy_model, resolver):
    """No forward hooks left on any module after context exits."""
    site = Site(component="resid_post", layer=0)

    def count_hooks(model):
        return sum(
            len(m._forward_hooks) + len(m._forward_pre_hooks)
            for m in model.modules()
        )

    before_hooks = count_hooks(toy_model)
    with patch_site(toy_model, site, torch.zeros(1, 4), resolver):
        pass
    after_hooks = count_hooks(toy_model)
    assert after_hooks == before_hooks


def test_param_values_unchanged(toy_model, resolver):
    """Parameter values are bit-identical before and after patching."""
    params_before = {n: p.clone() for n, p in toy_model.named_parameters()}
    site = Site(component="resid_post", layer=0)

    with patch_site(toy_model, site, torch.zeros(1, 4), resolver):
        toy_model(torch.randn(1, 4))

    for n, p in toy_model.named_parameters():
        assert torch.equal(p, params_before[n]), f"param {n} changed"


def test_activation_grad_enabled(toy_model, resolver):
    """With enable_activation_grad=True, activation grads are available."""
    torch.manual_seed(3)
    x = torch.randn(1, 4)
    z = torch.randn(1, 4)
    site = Site(component="resid_post", layer=0)
    grad_holder = {}

    with patch_site(toy_model, site, z, resolver, enable_activation_grad=True) as handle:
        out = toy_model(x)
        out.sum().backward()
        grad_holder["grad"] = handle.activation_grad

    assert grad_holder["grad"] is not None
    for p in toy_model.parameters():
        assert p.grad is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `venv/bin/pytest tests/patching/test_intervene.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'circuitry.patching.intervene'`

- [ ] **Step 3: Write implementation**

```python
# src/circuitry/patching/intervene.py
"""Activation patching context manager. Design spec §4.

Guarantees: hook removed on exit, eval mode restored, param requires_grad
restored, even on exception. Mutation-last discipline.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Generator

import torch
import torch.nn as nn
from torch import Tensor

from circuitry.patching.sites import ResolvedSite, Site


@dataclass
class PatchHandle:
    """Handle returned by patch_site context manager."""

    activation_grad: Tensor | None = field(default=None, init=False)
    _grad_tensor: Tensor | None = field(default=None, init=False, repr=False)


@contextmanager
def patch_site(
    model: nn.Module,
    site: Site,
    value: Tensor,
    resolver: object,
    *,
    enable_activation_grad: bool = False,
) -> Generator[PatchHandle, None, None]:
    """Patch a site's activation with ``value`` for the duration of the context.

    Guarantees:
    - Hook is removed on exit (including on exception).
    - Model eval mode is set on entry, restored on exit.
    - All param requires_grad are set to False, restored on exit.
    - Param values are never modified.
    """
    resolved: ResolvedSite = resolver.resolve(model, site)  # type: ignore[union-attr]
    handle = PatchHandle()
    hook_handle = None
    was_training = model.training
    original_requires_grad: dict[str, bool] = {}

    for name, p in model.named_parameters():
        original_requires_grad[name] = p.requires_grad
        p.requires_grad_(False)

    try:
        model.eval()

        if resolved.is_input_hook:
            def pre_hook(module: nn.Module, args: tuple) -> tuple:
                x = args[0]
                modified = resolved.inject(x, value)
                if enable_activation_grad:
                    modified = modified.detach().requires_grad_(True)
                    modified.retain_grad()
                    handle._grad_tensor = modified
                return (modified,) + args[1:]

            hook_handle = resolved.module.register_forward_pre_hook(pre_hook)
        else:
            def post_hook(module: nn.Module, input: tuple, output: Tensor) -> Tensor:
                if isinstance(output, tuple):
                    first = output[0]
                    modified = resolved.inject(first, value)
                    if enable_activation_grad:
                        modified = modified.detach().requires_grad_(True)
                        modified.retain_grad()
                        handle._grad_tensor = modified
                    return (modified,) + output[1:]
                modified = resolved.inject(output, value)
                if enable_activation_grad:
                    modified = modified.detach().requires_grad_(True)
                    modified.retain_grad()
                    handle._grad_tensor = modified
                return modified

            hook_handle = resolved.module.register_forward_hook(post_hook)

        yield handle

    finally:
        if hook_handle is not None:
            hook_handle.remove()
        if was_training:
            model.train()
        else:
            model.eval()
        for name, p in model.named_parameters():
            if name in original_requires_grad:
                p.requires_grad_(original_requires_grad[name])
        if handle._grad_tensor is not None and handle._grad_tensor.grad is not None:
            handle.activation_grad = handle._grad_tensor.grad.clone()
```

Update `src/circuitry/patching/__init__.py`:

```python
"""Activation patching — interventional diagnostics. See design spec §2."""
from __future__ import annotations

from circuitry.patching.intervene import PatchHandle, patch_site
from circuitry.patching.sites import Site

__all__ = ["PatchHandle", "Site", "patch_site"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv/bin/pytest tests/patching/test_intervene.py -v`
Expected: All passed

- [ ] **Step 5: Run full patching + layering tests**

Run: `venv/bin/pytest tests/patching/ tests/test_layering.py -v`
Expected: All pass

- [ ] **Step 6: Commit**

```bash
git add src/circuitry/patching/intervene.py src/circuitry/patching/__init__.py \
        tests/patching/test_intervene.py
git commit -m "feat(patching): add patch_site() context manager with restore-on-exit guarantee"
```

---

### Task 6: Prompt-pair runner (`runner.py`)

**Files:**
- Create: `src/circuitry/patching/runner.py`
- Create: `tests/patching/test_runner.py`
- Modify: `src/circuitry/patching/__init__.py` (re-export `PatchRunner`, `PatchResult`)

**Context:** The `PatchRunner` orchestrates clean/corrupted prompt-pair activation patching. Workflow:
1. Run the source prompt (corrupted for denoise, clean for noise); cache activations at all requested sites.
2. For each site: run the target prompt with the cached activation substituted at that site.
3. Evaluate the metric on the patched output vs. the baseline.
4. Return per-site metric deltas.

Uses the `ToyPatchModel` from conftest (identity weights). For denoising:
- Corrupted input `c` → output `c` (identity). Cache activation at layer 0: `c`.
- Clean input `x` with layer 0 patched to `c` → output `c` (instead of `x`).
- `logit_diff` on patched output vs clean output shows the expected delta.

- [ ] **Step 1: Write failing tests**

```python
# tests/patching/test_runner.py
"""Tests for PatchRunner."""
from __future__ import annotations

import pytest
import torch

from circuitry.core.patching import logit_diff
from circuitry.patching.runner import PatchResult, PatchRunner
from circuitry.patching.sites import HFSiteResolver, Site


@pytest.fixture
def resolver():
    return HFSiteResolver(
        n_heads=1, d_model=4, d_mlp=8,
        layer_pattern="layers.{L}",
        attn_module="self_attn.o_proj",
        mlp_module="mlp",
        mlp_intermediate="mlp.down_proj",
    )


def test_run_patching_denoise(toy_model, resolver):
    """Denoising: patching clean activation into corrupted run recovers clean output."""
    torch.manual_seed(0)
    clean = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    corrupted = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    site = Site(component="resid_post", layer=0)

    runner = PatchRunner(toy_model, resolver)
    result = runner.run_patching(
        clean_inputs=clean,
        corrupted_inputs=corrupted,
        sites=[site],
        metric=lambda logits: logit_diff(logits, correct=0, incorrect=3),
        direction="denoise",
    )

    assert isinstance(result, PatchResult)
    assert site in result.metric_values
    clean_metric = logit_diff(toy_model(clean), correct=0, incorrect=3)
    assert result.metric_values[site] == pytest.approx(clean_metric, abs=1e-5)


def test_run_patching_noise(toy_model, resolver):
    """Noising: patching corrupted activation into clean run yields corrupted output."""
    torch.manual_seed(1)
    clean = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    corrupted = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    site = Site(component="resid_post", layer=0)

    runner = PatchRunner(toy_model, resolver)
    result = runner.run_patching(
        clean_inputs=clean,
        corrupted_inputs=corrupted,
        sites=[site],
        metric=lambda logits: logit_diff(logits, correct=0, incorrect=3),
        direction="noise",
    )

    corrupted_metric = logit_diff(toy_model(corrupted), correct=0, incorrect=3)
    assert result.metric_values[site] == pytest.approx(corrupted_metric, abs=1e-5)


def test_multiple_sites(toy_model, resolver):
    """Runner handles multiple sites independently."""
    torch.manual_seed(2)
    clean = torch.randn(1, 4)
    corrupted = torch.randn(1, 4)
    sites = [
        Site(component="resid_post", layer=0),
        Site(component="resid_post", layer=1),
    ]

    runner = PatchRunner(toy_model, resolver)
    result = runner.run_patching(
        clean_inputs=clean,
        corrupted_inputs=corrupted,
        sites=sites,
        metric=lambda logits: float(logits.sum().item()),
        direction="denoise",
    )

    assert len(result.metric_values) == 2
    for s in sites:
        assert s in result.metric_values


def test_custom_metric(toy_model, resolver):
    """Runner accepts arbitrary Callable[[Tensor], float]."""
    clean = torch.ones(1, 4)
    corrupted = torch.zeros(1, 4)
    site = Site(component="resid_post", layer=0)

    def my_metric(logits: torch.Tensor) -> float:
        return float(logits.max().item())

    runner = PatchRunner(toy_model, resolver)
    result = runner.run_patching(
        clean_inputs=clean,
        corrupted_inputs=corrupted,
        sites=[site],
        metric=my_metric,
        direction="denoise",
    )

    assert isinstance(result.metric_values[site], float)


def test_model_clean_after_runner(toy_model, resolver):
    """Model state is clean after runner completes (no leftover hooks)."""
    torch.manual_seed(3)
    clean = torch.randn(1, 4)
    corrupted = torch.randn(1, 4)
    before = toy_model(clean).clone()

    runner = PatchRunner(toy_model, resolver)
    runner.run_patching(
        clean_inputs=clean,
        corrupted_inputs=corrupted,
        sites=[Site(component="resid_post", layer=0)],
        metric=lambda logits: float(logits.sum().item()),
        direction="denoise",
    )

    after = toy_model(clean)
    assert torch.equal(before, after)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `venv/bin/pytest tests/patching/test_runner.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'circuitry.patching.runner'`

- [ ] **Step 3: Write implementation**

```python
# src/circuitry/patching/runner.py
"""Prompt-pair activation patching runner. Design spec §4."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

import torch
import torch.nn as nn
from torch import Tensor

from circuitry.patching.intervene import patch_site
from circuitry.patching.sites import ResolvedSite, Site


@dataclass
class PatchResult:
    """Result of a patching run."""

    metric_values: dict[Site, float] = field(default_factory=dict)
    cached_activations: dict[Site, Tensor] = field(default_factory=dict)


class PatchRunner:
    """Orchestrates clean/corrupted prompt-pair activation patching."""

    def __init__(self, model: nn.Module, resolver: object) -> None:
        self.model = model
        self.resolver = resolver

    @torch.no_grad()
    def _cache_activations(
        self,
        inputs: Tensor,
        sites: list[Site],
    ) -> dict[Site, Tensor]:
        """Run a forward pass and cache activations at all requested sites."""
        cache: dict[Site, Tensor] = {}
        handles = []

        for site in sites:
            resolved: ResolvedSite = self.resolver.resolve(self.model, site)  # type: ignore[union-attr]

            if resolved.is_input_hook:
                def make_pre_hook(s: Site, r: ResolvedSite):
                    def hook_fn(module: nn.Module, args: tuple) -> None:
                        cache[s] = r.extract(args[0]).detach().clone()
                    return hook_fn

                h = resolved.module.register_forward_pre_hook(make_pre_hook(site, resolved))
            else:
                def make_post_hook(s: Site, r: ResolvedSite):
                    def hook_fn(module: nn.Module, input: tuple, output: Tensor) -> None:
                        out = output[0] if isinstance(output, tuple) else output
                        cache[s] = r.extract(out).detach().clone()
                    return hook_fn

                h = resolved.module.register_forward_hook(make_post_hook(site, resolved))

            handles.append(h)

        try:
            was_training = self.model.training
            self.model.eval()
            self.model(inputs)
        finally:
            for h in handles:
                h.remove()
            if was_training:
                self.model.train()

        return cache

    def run_patching(
        self,
        clean_inputs: Tensor,
        corrupted_inputs: Tensor,
        sites: list[Site],
        metric: Callable[[Tensor], float],
        direction: Literal["denoise", "noise"] = "denoise",
    ) -> PatchResult:
        """Run activation patching over prompt pairs.

        For denoise: cache clean activations, patch each into corrupted run.
        For noise: cache corrupted activations, patch each into clean run.
        """
        if direction == "denoise":
            source_inputs = clean_inputs
            target_inputs = corrupted_inputs
        else:
            source_inputs = corrupted_inputs
            target_inputs = clean_inputs

        cached = self._cache_activations(source_inputs, sites)
        result = PatchResult(cached_activations=cached)

        for site in sites:
            with patch_site(self.model, site, cached[site], self.resolver):
                patched_out = self.model(target_inputs)
            result.metric_values[site] = metric(patched_out)

        return result
```

Update `src/circuitry/patching/__init__.py`:

```python
"""Activation patching — interventional diagnostics. See design spec §2."""
from __future__ import annotations

from circuitry.patching.intervene import PatchHandle, patch_site
from circuitry.patching.runner import PatchResult, PatchRunner
from circuitry.patching.sites import Site

__all__ = ["PatchHandle", "PatchResult", "PatchRunner", "Site", "patch_site"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv/bin/pytest tests/patching/test_runner.py -v`
Expected: All passed

- [ ] **Step 5: Run full test suite**

Run: `venv/bin/pytest tests/patching/ tests/test_layering.py tests/core/test_patching.py -v`
Expected: All pass

- [ ] **Step 6: Commit**

```bash
git add src/circuitry/patching/runner.py src/circuitry/patching/__init__.py \
        tests/patching/test_runner.py
git commit -m "feat(patching): add PatchRunner for prompt-pair activation patching"
```

---

### Task 7: Design.md contract amendment + full integration

**Files:**
- Modify: `docs/design.md`
- Modify: `src/circuitry/__init__.py` (optional: no re-export needed — patching is a separate import path)
- Modify: `tests/test_public_api.py` (if adding patching to public surface)

**Context:** This task adds the contract amendment to `docs/design.md` and runs the full test suite as a final integration check. The amendment adds:
1. `patching/` to the repo structure tree
2. Updated layering rules (core must not import patching; patching must not import cli)
3. A new section on the sanctioned intervention mode
4. `core/patching.py` to the §4.1 API listing

- [ ] **Step 1: Update docs/design.md repo structure (§3)**

In the repo structure tree (around line 44-88), add `patching/` after `sae/`:

```
│   ├── patching/           # v1.0: activation patching (interventional)
│   │   ├── sites.py        # Site dataclass + HF/TL resolution
│   │   ├── intervene.py    # patch_site() context manager
│   │   └── runner.py       # PatchRunner prompt-pair runner
```

- [ ] **Step 2: Update layering rules (§3)**

After the existing layering rules, update to:

```markdown
### Layering rules (enforced in CI)

- `core/` MUST NOT import from `recorder/`, `recipes/`, `writers/`, `cli/`, or `patching/`.
- `recipes/` MUST NOT import from `cli/`.
- `patching/` may import from `core/` and `recipes/`; MUST NOT import from `cli/`.
- The package MUST NOT import from any downstream user codebase. `circuitry` is the consumed dependency, never the consumer.
- `transformer_lens` is an approved optional dependency (lazy import only; `circuitry` must work without it).

A simple `import-linter` config or hand-rolled AST test enforces this.
```

- [ ] **Step 3: Add patching metrics to §4.1 API listing**

After the SAE section in §4.1, add:

```markdown
# patching metrics (v1.0)
from circuitry.core import patching
patching.logit_diff(logits: Tensor, correct: int, incorrect: int) -> float
patching.kl_divergence(p_logits: Tensor, q_logits: Tensor, *, chunk_size: int = 256) -> float
patching.ce_loss(logits: Tensor, targets: Tensor) -> float
```

- [ ] **Step 4: Add intervention mode section**

After §4 (Public API), add a new section:

```markdown
## 4.5 Intervention mode (v1.0)

The `patching/` subsystem adds an opt-in **intervention mode** for causal analysis (activation patching, attribution methods). Compared to the observation-only `Recorder` and `scan`:

- **Opt-in**: interventions require explicit use of the `circuitry.patching` API. Recorder and scan remain observation-only.
- **Isolated**: every intervention is scoped to a context manager (`patch_site`). Hooks are removed and model state is restored on exit, including on exception.
- **Frozen model**: parameter `requires_grad` stays off; eval mode is managed and restored; no optimizer or parameter updates.
- **Activation-grad-only**: the only gradient flow permitted is on activation tensors at intervention sites (for attribution methods). Parameter gradients are never enabled.

```python
from circuitry.patching import Site, patch_site, PatchRunner

site = Site(component="attn_head_out", layer=5, head=3)

# Low-level: single intervention
with patch_site(model, site, value=cached_act, resolver=resolver):
    output = model(**inputs)

# High-level: prompt-pair runner
runner = PatchRunner(model, resolver)
result = runner.run_patching(
    clean_inputs=clean_ids,
    corrupted_inputs=corrupted_ids,
    sites=[site],
    metric=logit_diff,
    direction="denoise",
)
```
```

- [ ] **Step 5: Run the full test suite**

Run: `venv/bin/pytest tests/ -v --tb=short`
Expected: All tests pass (existing 219 + new patching tests)

- [ ] **Step 6: Commit**

```bash
git add docs/design.md
git commit -m "docs(design): add intervention mode contract amendment + patching/ layering rules"
```

---

## Self-review checklist

**1. Spec coverage:**
- §1 Taxonomy → not code, documented in spec
- §2 Module layout → Tasks 1-6 create all files
- §3 Site model → Task 2 (Site), Task 3 (HF), Task 4 (TL)
- §4 Intervention mechanism → Task 5 (context manager), Task 5 tests (gradient discipline, frozen model)
- §5 Metrics → Task 1
- §6 Contract amendment → Task 4 (layering tests), Task 7 (design.md)
- §7 Testing → covered across Tasks 1-6 tests
- §9 Public API → Task 5-6 update `__init__.py` re-exports

**2. Placeholder scan:** No TBD/TODO/vague references found.

**3. Type consistency:**
- `Site` used consistently across all tasks (same dataclass from `sites.py`)
- `ResolvedSite` used in Tasks 3, 4, 5 with consistent fields
- `PatchHandle` defined in Task 5, used in tests
- `PatchResult` defined in Task 6, used in tests
- `HFSiteResolver` / `TLSiteResolver` consistent between definition (Task 3/4) and usage (Task 5/6)
- `logit_diff` / `kl_divergence` / `ce_loss` signatures match between Task 1 definition and Task 6 usage
