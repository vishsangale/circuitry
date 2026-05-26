# HF Patching-Backend Generalization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the v1.0 patching pillar's HF backend honor `config.head_dim` (fixing Gemma-2/3) and add a TransformerLens bridge so non-Llama HF models (GPT-2, …) are usable for patching, with a clear error pointing there.

**Architecture:** (A) `head_dim` becomes a single source of truth on `HFSiteResolver`, read from `config.head_dim`; EAP/AtP\* trust it; ACDC inherits. (B) a lazy `to_hooked_transformer()` helper wraps a loaded HF model as a `HookedTransformer` for the existing TL backend. (C) a shared `_layout.py` locator raises a clear unsupported-layout error from both EAP and AtP\*.

**Tech Stack:** Python 3.12, PyTorch, `transformers` (lazy optional), `transformer_lens` (lazy optional), pytest. Spec: `docs/superpowers/specs/2026-05-25-hf-backend-generalization-design.md`.

**Environment:** use full venv paths — `venv/bin/python`, `venv/bin/pytest`, `venv/bin/ruff`. Never `source venv/bin/activate`.

**CI invariants (must not break):** `core/` must not import `patching/`; `patching/` must not import `cli/`; `transformer_lens`/`transformers` imported lazily only; no `.cuda()` in `core/`. Do not regress existing patching tests (toy-Llama exact anchors, Qwen GQA, TL gpt2).

---

## File Structure

- **Create** `src/circuitry/patching/_layout.py` — `locate_layers(model)` + `locate_embed(model)` shared by EAP/AtP\*; raises the clear unsupported-layout `ValueError`.
- **Modify** `src/circuitry/patching/sites.py` — `HFSiteResolver.__init__` gains `head_dim` kwarg; `from_config` reads `config.head_dim`.
- **Modify** `src/circuitry/patching/eap.py` — HF path uses `resolver.head_dim`; TL path prefers `cfg.d_head`; locator → `_layout`.
- **Modify** `src/circuitry/patching/atp.py` — same as eap.py (duplicate constructor + locator).
- **Create** `src/circuitry/patching/tl_bridge.py` — `to_hooked_transformer(hf_model, model_name, ...)`.
- **Modify** `src/circuitry/patching/__init__.py` — export `to_hooked_transformer`.
- **Create** `tests/patching/test_head_dim_generalization.py`, `tests/patching/test_unsupported_layout_error.py`, `tests/patching/test_tl_bridge.py`.
- **Modify** `docs/design.md` (§3, §4.6) and `CHANGELOG.md`.

---

## Task 1: `head_dim` plumbing in `HFSiteResolver`

**Files:**
- Modify: `src/circuitry/patching/sites.py` (`__init__` ~143-161, `from_config` ~163-172)
- Test: `tests/patching/test_head_dim_generalization.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/patching/test_head_dim_generalization.py
import pytest
from circuitry.patching.sites import HFSiteResolver


class _Cfg:
    """Minimal config stub: head_dim independent of hidden_size/num_attention_heads."""
    num_attention_heads = 8
    hidden_size = 64          # d_model/n_heads == 8
    head_dim = 16             # explicit, != 8  -> the Gemma-2 condition
    intermediate_size = 128
    num_key_value_heads = 8


def test_resolver_honors_explicit_head_dim():
    r = HFSiteResolver.from_config(_Cfg())
    assert r.head_dim == 16            # not 64 // 8 == 8


def test_resolver_falls_back_when_no_head_dim():
    class C:
        num_attention_heads = 4
        hidden_size = 64               # 64 // 4 == 16
    r = HFSiteResolver.from_config(C())
    assert r.head_dim == 16            # fallback d_model // n_heads


def test_explicit_head_dim_kwarg():
    r = HFSiteResolver(n_heads=8, d_model=64, head_dim=16)
    assert r.head_dim == 16
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/bin/pytest tests/patching/test_head_dim_generalization.py -v`
Expected: `test_resolver_honors_explicit_head_dim` FAILS (`r.head_dim == 8`); `test_explicit_head_dim_kwarg` FAILS (`__init__` has no `head_dim` param → TypeError).

- [ ] **Step 3: Add the `head_dim` kwarg to `HFSiteResolver.__init__`**

In `src/circuitry/patching/sites.py`, change the signature/body. Current:

```python
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
```

New (add `head_dim` kwarg, keep everything else):

```python
    def __init__(
        self,
        n_heads: int,
        d_model: int,
        d_mlp: int | None = None,
        *,
        head_dim: int | None = None,
        layer_pattern: str = "model.layers.{L}",
        attn_module: str = "self_attn.o_proj",
        mlp_module: str = "mlp",
        mlp_intermediate: str = "mlp.down_proj",
    ) -> None:
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_mlp = d_mlp
        self.head_dim = head_dim if head_dim is not None else d_model // n_heads
```

- [ ] **Step 4: Make `from_config` read `config.head_dim`**

Current:

```python
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
```

New (read optional `head_dim`):

```python
    @classmethod
    def from_config(cls, config: Any) -> HFSiteResolver:
        n_heads = getattr(config, "num_attention_heads", None)
        d_model = getattr(config, "hidden_size", None)
        if n_heads is None or d_model is None:
            raise ValueError(
                "Config must have num_attention_heads and hidden_size"
            )
        d_mlp = getattr(config, "intermediate_size", None)
        head_dim = getattr(config, "head_dim", None)
        return cls(n_heads=n_heads, d_model=d_model, d_mlp=d_mlp, head_dim=head_dim)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `venv/bin/pytest tests/patching/test_head_dim_generalization.py -v`
Expected: all 3 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/circuitry/patching/sites.py tests/patching/test_head_dim_generalization.py
git commit -m "fix(patching): HFSiteResolver honors config.head_dim (Gemma-2/3)

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 2: EAP & AtP\* runners trust `resolver.head_dim` (HF) / `cfg.d_head` (TL)

**Files:**
- Modify: `src/circuitry/patching/eap.py` (`__init__` TL path line 91, HF path line 106)
- Modify: `src/circuitry/patching/atp.py` (`__init__` TL path line 129, HF path line 141)
- Test: `tests/patching/test_head_dim_generalization.py` (append)

- [ ] **Step 1: Write the failing test (append to the file from Task 1)**

```python
def _explicit_head_dim_model():
    """Tiny real Llama whose head_dim (16) != hidden_size/n_heads (64/8=8)."""
    from transformers import LlamaConfig, LlamaForCausalLM
    cfg = LlamaConfig(
        vocab_size=64, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=8, num_key_value_heads=8,
        head_dim=16,
    )
    # Guard: the transformers version must actually honor explicit head_dim.
    assert cfg.head_dim == 16
    model = LlamaForCausalLM(cfg).eval()
    assert model.model.layers[0].self_attn.q_proj.out_features == 8 * 16  # 128, not 64
    return model, cfg


def test_eap_runner_uses_config_head_dim_and_runs():
    import torch
    from circuitry.patching import EAPRunner
    from circuitry.patching.sites import HFSiteResolver
    from circuitry.core.patching import logit_diff_t

    model, cfg = _explicit_head_dim_model()
    resolver = HFSiteResolver.from_config(cfg)
    runner = EAPRunner(model, resolver=resolver)
    assert runner.head_dim == 16   # was 8 before the fix

    torch.manual_seed(0)
    clean = {"input_ids": torch.randint(0, 64, (1, 6))}
    corrupt = {"input_ids": torch.randint(0, 64, (1, 6))}
    metric = lambda out: logit_diff_t(out.logits if hasattr(out, "logits") else out, 1, 2)
    res = runner.run(clean, corrupt, metric)   # must not raise the reshape error
    assert len(res.scores) > 0


def test_atp_runner_uses_config_head_dim_and_runs():
    import torch
    from circuitry.patching import AtPRunner
    from circuitry.patching.sites import HFSiteResolver
    from circuitry.core.patching import logit_diff_t

    model, cfg = _explicit_head_dim_model()
    resolver = HFSiteResolver.from_config(cfg)
    runner = AtPRunner(model, resolver=resolver)
    assert runner.head_dim == 16

    torch.manual_seed(0)
    clean = {"input_ids": torch.randint(0, 64, (1, 6))}
    corrupt = {"input_ids": torch.randint(0, 64, (1, 6))}
    metric = lambda out: logit_diff_t(out.logits if hasattr(out, "logits") else out, 1, 2)
    res = runner.run(clean, corrupt, metric, qk_fix=True)
    assert len(res.scores) > 0


def test_acdc_anchors_hold_with_explicit_head_dim():
    """ACDC inherits head_dim from EAP; empty anchor KL==0, full anchor==corrupted run."""
    import torch
    from circuitry.patching import ACDCRunner
    from circuitry.patching.sites import HFSiteResolver
    from circuitry.patching.graph import edge_sort_key  # noqa: F401  (ensure import ok)

    model, cfg = _explicit_head_dim_model()
    resolver = HFSiteResolver.from_config(cfg)
    acdc = ACDCRunner(model, resolver=resolver)
    assert acdc.head_dim == 16

    torch.manual_seed(0)
    clean = {"input_ids": torch.randint(0, 64, (1, 6))}
    corrupt = {"input_ids": torch.randint(0, 64, (1, 6))}
    corr_act = acdc._cache_corrupted_acts(corrupt)
    with torch.no_grad():
        clean_logits = acdc._eap._call_model(clean)
        clean_logits = clean_logits.logits if hasattr(clean_logits, "logits") else clean_logits
        corr_logits = acdc._eap._call_model(corrupt)
        corr_logits = corr_logits.logits if hasattr(corr_logits, "logits") else corr_logits
    # empty anchor: nothing removed -> circuit logits == clean run
    empty = acdc._forward(clean, set(), corr_act)
    assert torch.allclose(empty[:, -1], clean_logits[:, -1], atol=1e-4)
    # full anchor: all edges removed -> circuit logits == corrupted run
    full = acdc._forward(clean, set(acdc.graph.edges), corr_act)
    assert torch.allclose(full[:, -1], corr_logits[:, -1], atol=1e-4)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/bin/pytest tests/patching/test_head_dim_generalization.py -v`
Expected: the three new tests FAIL — `runner.head_dim == 8` (assertion) and/or `RuntimeError: shape ... invalid` inside `run`.

- [ ] **Step 3: Fix `EAPRunner.__init__` (eap.py)**

TL path — current line 91:
```python
            self.head_dim = cfg.d_model // n_heads
```
New:
```python
            self.head_dim = getattr(cfg, "d_head", None) or (cfg.d_model // n_heads)
```

HF path — current line 106:
```python
            self.head_dim = (resolver.d_model // resolver.n_heads) if resolver is not None and n_heads > 0 else None
```
New (preserve the guard, swap the computation):
```python
            self.head_dim = resolver.head_dim if (resolver is not None and n_heads > 0) else None
```

- [ ] **Step 4: Fix `AtPRunner.__init__` (atp.py)**

TL path — current line 129:
```python
            self.head_dim: int | None = cfg.d_model // n_heads
```
New:
```python
            self.head_dim: int | None = getattr(cfg, "d_head", None) or (cfg.d_model // n_heads)
```

HF path — current line 141:
```python
            self.head_dim = (d_model // n_heads) if (d_model is not None and n_heads > 0) else None
```
New (use the resolver's head_dim; keep the same guard variables):
```python
            self.head_dim = resolver.head_dim if (resolver is not None and n_heads > 0) else None
```

- [ ] **Step 5: Run tests to verify they pass + no regression**

Run: `venv/bin/pytest tests/patching/test_head_dim_generalization.py -v`
Expected: all PASS.
Run: `venv/bin/pytest tests/patching/ -q`
Expected: full patching suite still green (toy anchors, Qwen, TL).

- [ ] **Step 6: Commit**

```bash
git add src/circuitry/patching/eap.py src/circuitry/patching/atp.py tests/patching/test_head_dim_generalization.py
git commit -m "fix(patching): EAP/AtP* use resolver.head_dim (HF) and cfg.d_head (TL)

ACDC inherits via EAPRunner. Closes the Gemma-2/3 head_dim!=d_model/n_heads crash.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 3: Shared `_layout.py` locator + clear unsupported-layout error

**Files:**
- Create: `src/circuitry/patching/_layout.py`
- Modify: `src/circuitry/patching/eap.py` (`_locate_layers` 118-123, `_embed` 125-130)
- Modify: `src/circuitry/patching/atp.py` (`_locate_layers` 159-164, `_embed` 166-171)
- Test: `tests/patching/test_unsupported_layout_error.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/patching/test_unsupported_layout_error.py
import pytest
import torch
import torch.nn as nn

from circuitry.patching import AtPRunner, EAPRunner
from circuitry.patching.sites import HFSiteResolver


class _NoLayers(nn.Module):
    """A non-Llama-layout model: has neither model.layers nor model.model.layers."""
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([nn.Linear(8, 8)])
        self.config = type("C", (), {"num_attention_heads": 2, "hidden_size": 8})()

    def forward(self, **kw):
        raise AssertionError("should fail at layer location, before forward")


@pytest.mark.parametrize("Runner", [EAPRunner, AtPRunner])
def test_unsupported_layout_raises_clear_error(Runner):
    resolver = HFSiteResolver(n_heads=2, d_model=8)
    with pytest.raises(ValueError, match="to_hooked_transformer"):
        Runner(_NoLayers(), resolver=resolver)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/bin/pytest tests/patching/test_unsupported_layout_error.py -v`
Expected: FAIL — raises `AttributeError: '_NoLayers' object has no attribute 'layers'`, not the `ValueError` matching `to_hooked_transformer`.

- [ ] **Step 3: Create the shared locator**

```python
# src/circuitry/patching/_layout.py
"""Shared HF model-layout helpers for the patching backend (EAP/AtP*).

The HF-eager patching backend targets Llama-family layouts
(``model.model.layers`` + ``self_attn.{q,k,v,o}_proj``). Non-Llama models
(GPT-2, etc.) should be routed through the TransformerLens backend via
``circuitry.patching.to_hooked_transformer``.
"""

from __future__ import annotations

import torch.nn as nn

_UNSUPPORTED_MSG = (
    "circuitry's HF patching backend supports Llama-family layouts "
    "(model.model.layers + self_attn.{{q,k,v,o}}_proj). This model ({cls}) is not "
    "supported directly. For GPT-2 and other architectures, convert it with "
    "circuitry.patching.to_hooked_transformer(model, \"<name>\") and use the "
    "TransformerLens backend (TLSiteResolver). See docs/design.md §4.6."
)


def locate_layers(model: nn.Module) -> nn.ModuleList:
    """Return the transformer layers list (model.model.layers or model.layers)."""
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "layers"):
        return inner.layers  # type: ignore[return-value]
    if hasattr(model, "layers"):
        return model.layers  # type: ignore[return-value]
    raise ValueError(_UNSUPPORTED_MSG.format(cls=type(model).__name__))


def locate_embed(model: nn.Module) -> nn.Module:
    """Return the token-embedding module (model.model.embed_tokens or model.embed_tokens)."""
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "embed_tokens"):
        return inner.embed_tokens
    if hasattr(model, "embed_tokens"):
        return model.embed_tokens  # type: ignore[return-value]
    raise ValueError(_UNSUPPORTED_MSG.format(cls=type(model).__name__))
```

- [ ] **Step 4: Wire `EAPRunner` to the shared locator (eap.py)**

Replace the `_locate_layers` staticmethod (118-123) and `_embed` body (125-130). Current:
```python
    @staticmethod
    def _locate_layers(model: nn.Module) -> nn.ModuleList:
        """Return the transformer layers list: tries model.model.layers then model.layers."""
        inner = getattr(model, "model", None)
        if inner is not None and hasattr(inner, "layers"):
            return inner.layers  # type: ignore[return-value]
        return model.layers  # type: ignore[return-value]

    def _embed(self) -> nn.Module:
        """Return the embedding module."""
        inner = getattr(self.model, "model", None)
        if inner is not None and hasattr(inner, "embed_tokens"):
            return inner.embed_tokens
        return self.model.embed_tokens  # type: ignore[return-value]
```
New:
```python
    @staticmethod
    def _locate_layers(model: nn.Module) -> nn.ModuleList:
        from circuitry.patching._layout import locate_layers
        return locate_layers(model)

    def _embed(self) -> nn.Module:
        from circuitry.patching._layout import locate_embed
        return locate_embed(self.model)
```

- [ ] **Step 5: Wire `AtPRunner` to the shared locator (atp.py)**

Replace the duplicated `_locate_layers` (159-164) and `_embed` (166-171) in atp.py with the identical bodies:
```python
    @staticmethod
    def _locate_layers(model: nn.Module) -> nn.ModuleList:
        from circuitry.patching._layout import locate_layers
        return locate_layers(model)

    def _embed(self) -> nn.Module:
        from circuitry.patching._layout import locate_embed
        return locate_embed(self.model)
```

- [ ] **Step 6: Run tests to verify they pass + no regression**

Run: `venv/bin/pytest tests/patching/test_unsupported_layout_error.py -v`
Expected: both parametrized cases (EAPRunner, AtPRunner) PASS.
Run: `venv/bin/pytest tests/patching/ -q`
Expected: full patching suite green (toy models still locate `model.layers`).

- [ ] **Step 7: Commit**

```bash
git add src/circuitry/patching/_layout.py src/circuitry/patching/eap.py src/circuitry/patching/atp.py tests/patching/test_unsupported_layout_error.py
git commit -m "feat(patching): shared layout locator + clear non-Llama error

Both EAPRunner and AtPRunner now raise an actionable ValueError pointing to
to_hooked_transformer instead of a cryptic AttributeError.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 4: `to_hooked_transformer` TL bridge

**Files:**
- Create: `src/circuitry/patching/tl_bridge.py`
- Modify: `src/circuitry/patching/__init__.py`
- Test: `tests/patching/test_tl_bridge.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/patching/test_tl_bridge.py
import importlib.util

import pytest
import torch

tl_missing = importlib.util.find_spec("transformer_lens") is None


def test_export_present():
    import circuitry.patching as p
    assert hasattr(p, "to_hooked_transformer")


@pytest.mark.skipif(tl_missing, reason="transformer_lens not installed")
def test_bridge_gpt2_eap_recovers_name_mover():
    transformers = pytest.importorskip("transformers")
    try:
        hf = transformers.GPT2LMHeadModel.from_pretrained("gpt2")
    except Exception as e:  # pragma: no cover - network/cache gated
        pytest.skip(f"gpt2 unavailable: {e}")

    from circuitry.patching import EAPRunner, to_hooked_transformer
    from circuitry.patching.sites import TLSiteResolver
    from circuitry.core.patching import logit_diff_t

    tl = to_hooked_transformer(hf, "gpt2", device="cpu")
    io = tl.to_single_token(" Mary")
    s = tl.to_single_token(" John")
    clean = tl.to_tokens("When John and Mary went to the store, John gave a drink to")
    corrupt = tl.to_tokens("When John and Mary went to the store, Mary gave a drink to")
    metric = lambda out: logit_diff_t(out.logits if hasattr(out, "logits") else out, io, s)

    res = EAPRunner(tl, resolver=TLSiteResolver()).run(clean, corrupt, metric)
    top_heads = [(e.writer.layer, e.writer.head) for e, _ in res.top_k(40)
                 if e.writer.kind == "attn_head"]
    assert (9, 9) in top_heads[:6]   # the canonical IOI name-mover
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/bin/pytest tests/patching/test_tl_bridge.py -v`
Expected: `test_export_present` FAILS (no `to_hooked_transformer`).

- [ ] **Step 3: Create the bridge module**

```python
# src/circuitry/patching/tl_bridge.py
"""Bridge a loaded HuggingFace model into TransformerLens for the TL patching backend.

The HF-eager patching backend targets Llama-family layouts. For GPT-2 and the other
architectures TransformerLens supports, wrap the loaded HF model as a
``HookedTransformer`` and use it with ``TLSiteResolver``.
"""

from __future__ import annotations

from typing import Any


def to_hooked_transformer(
    hf_model: Any,
    model_name: str,
    *,
    device: str | None = None,
    dtype: Any = None,
    **tl_kwargs: Any,
):
    """Wrap a loaded HF causal-LM as a TransformerLens ``HookedTransformer``.

    Args:
        hf_model: an already-loaded HF ``*ForCausalLM`` model (its weights are reused).
        model_name: the TransformerLens architecture name, e.g. ``"gpt2"``.
        device / dtype: forwarded to ``from_pretrained``.
        **tl_kwargs: forwarded to ``HookedTransformer.from_pretrained`` (e.g.
            ``fold_ln=False``). Defaults apply TL's standard processing.

    Returns:
        A ``HookedTransformer`` usable with ``TLSiteResolver`` and the patching runners.

    Note:
        TransformerLens folds LayerNorm and centers writing/unembed weights, so the
        wrapped model's *activations* differ from the raw HF model's (logits are
        equivalent). Patching runs on the TL-processed model.

    Raises:
        ImportError: if ``transformer_lens`` is not installed.
    """
    try:
        from transformer_lens import HookedTransformer
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "to_hooked_transformer requires transformer_lens. "
            "Install it with: pip install transformer_lens"
        ) from e

    return HookedTransformer.from_pretrained(
        model_name, hf_model=hf_model, device=device, dtype=dtype, **tl_kwargs
    )
```

- [ ] **Step 4: Export it from the package**

In `src/circuitry/patching/__init__.py`, add the import (alphabetically near the others) and the `__all__` entry:
```python
from circuitry.patching.tl_bridge import to_hooked_transformer
```
and add `"to_hooked_transformer",` to the `__all__` list.

- [ ] **Step 5: Run tests to verify they pass**

Run: `venv/bin/pytest tests/patching/test_tl_bridge.py -v`
Expected: `test_export_present` PASS; `test_bridge_gpt2_eap_recovers_name_mover` PASS (or SKIP if gpt2/TL unavailable).

- [ ] **Step 6: Verify layering invariant still holds**

Run: `venv/bin/pytest tests/test_layering.py -q`
Expected: PASS (transformer_lens import is lazy inside the function; not a module-level import).

- [ ] **Step 7: Commit**

```bash
git add src/circuitry/patching/tl_bridge.py src/circuitry/patching/__init__.py tests/patching/test_tl_bridge.py
git commit -m "feat(patching): to_hooked_transformer bridge for non-Llama HF models

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 5: Docs + CHANGELOG

**Files:**
- Modify: `docs/design.md` (§3 backend support, §4.6 intervention mode)
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Update `docs/design.md` §4.6**

Add a paragraph after the intervention-mode backend description:

```markdown
**Backend scope (v1.1):** the HF-eager patching backend (EAP / AtP* / ACDC)
targets Llama-family layouts (`model.model.layers` + `self_attn.{q,k,v,o}_proj`)
and honors an explicit `config.head_dim` (so Gemma-2/3, where `head_dim !=
hidden_size/num_attention_heads`, work). For GPT-2 and other architectures,
wrap the loaded model with `circuitry.patching.to_hooked_transformer(model,
"<tl-name>")` and use the TransformerLens backend (`TLSiteResolver`); pointing
the HF backend at an unsupported layout raises a `ValueError` directing you
there. TransformerLens folds LayerNorm / centers weights, so patching runs on
the TL-processed (logit-equivalent) model.
```

- [ ] **Step 2: Update `docs/design.md` §3**

In the backend/layering notes, add one line: `transformer_lens` remains a lazy optional dependency, now also used by `patching/tl_bridge.py` (imported inside the function body; `test_layering` allowlist unchanged).

- [ ] **Step 3: Update `CHANGELOG.md`**

Add under a new `## [1.1.0] — 2026-05-25` section (Keep-a-Changelog, em-dash):
```markdown
### Fixed
- HF patching backend now honors `config.head_dim`; EAP/AtP*/ACDC run on Gemma-2/3
  (previously crashed when `head_dim != hidden_size / num_attention_heads`).

### Added
- `circuitry.patching.to_hooked_transformer(hf_model, model_name, ...)` — bridge a
  loaded HF model into TransformerLens so non-Llama architectures (GPT-2, …) are
  usable with the TL patching backend.
- Clear `ValueError` (pointing to `to_hooked_transformer`) when the HF backend is
  given an unsupported (non-Llama) layout, replacing a cryptic `AttributeError`.
```

- [ ] **Step 4: Run the full suite + ruff**

Run: `venv/bin/pytest -q`
Expected: all green.
Run: `venv/bin/ruff check src/circuitry/patching/`
Expected: clean.

- [ ] **Step 5: Commit**

```bash
git add docs/design.md CHANGELOG.md
git commit -m "docs(patching): v1.1 HF-backend scope + head_dim + TL bridge

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Self-Review checklist (run after implementing)

- **Spec coverage:** Part A (head_dim) → Tasks 1-2; Part B (TL bridge) → Task 4; Part C (clear error + shared locator) → Task 3; tests → embedded per task; docs/layering → Task 5. No gaps.
- **No regression:** every task ends with `venv/bin/pytest tests/patching/ -q` (Tasks 2-3) or full suite (Task 5).
- **Type/name consistency:** `head_dim` kwarg (Task 1) is read by `from_config` (Task 1) and `EAPRunner`/`AtPRunner` via `resolver.head_dim` (Task 2); `_layout.locate_layers`/`locate_embed` (Task 3) called by both runners; `to_hooked_transformer` (Task 4) name matches the error message (Task 3) and docs (Task 5).
