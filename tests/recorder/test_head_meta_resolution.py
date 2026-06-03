"""fingerprint #1: attention_head_rank head-metadata resolution.

Pre-fix the recorder read head metadata only from ``model.config`` (+
``text_config``), so HF-wrapped models (metadata on ``model.model.config``) and
config-less custom models resolved to nothing and emitted *zero* head-rank
output. These tests cover the broadened resolution: submodule ``.config`` walk,
config-less attention-submodule attributes, and an explicit recipe override.
"""
from __future__ import annotations

import torch.nn as nn

from circuitry.recipes import Recipe
from circuitry.recorder.live import Recorder


def _recipe(**kw) -> Recipe:
    return Recipe(
        name="head-meta-test",
        hook_points=[],
        weight_diagnostics=["attention_head_rank"],
        **kw,
    )


class _Cfg:
    """Minimal HF-style config object (plain attrs, not an nn.Module)."""

    def __init__(self, *, num_attention_heads, hidden_size=None, head_dim=None,
                 num_key_value_heads=None):
        self.num_attention_heads = num_attention_heads
        if hidden_size is not None:
            self.hidden_size = hidden_size
        if head_dim is not None:
            self.head_dim = head_dim
        if num_key_value_heads is not None:
            self.num_key_value_heads = num_key_value_heads


class _Attn(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)
        self.o_proj = nn.Linear(d, d)


class _AttnWithAttrs(_Attn):
    """Config-less attention module that exposes head counts as attributes."""

    def __init__(self, d=16, h=4):
        super().__init__(d, h)
        self.num_heads = h
        self.head_dim = d // h


class _InnerWithConfig(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = _Cfg(num_attention_heads=4, hidden_size=16)
        self.self_attn = _Attn(16, 4)


class _HFWrapped(nn.Module):
    """Top level has NO .config; the inner ``.model`` carries it (HF style)."""

    def __init__(self):
        super().__init__()
        self.model = _InnerWithConfig()


class _ConfigLess(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _AttnWithAttrs(16, 4)


class _TopLevelConfig(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = _Cfg(num_attention_heads=4, hidden_size=16)
        self.self_attn = _Attn(16, 4)


class _Bare(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _Attn(16, 4)  # no config, no head attrs


def test_hf_wrapped_config_resolved_through_attach(tmp_path):
    """Metadata on model.model.config must resolve. Pre-fix: only model.config
    was read, so _attn_meta stayed None after attach()."""
    rec = Recorder(_HFWrapped(), run_dir=tmp_path, recipe=_recipe(),
                   writer="null", strict=False)
    rec.attach()
    try:
        assert rec._attn_meta is not None
        assert rec._attn_meta.n_heads == 4
        assert rec._attn_meta.head_dim == 4  # 16 // 4
    finally:
        rec.detach()


def test_top_level_config_still_resolves(tmp_path):
    """Regression guard: the original model.config path keeps working."""
    rec = Recorder(_TopLevelConfig(), run_dir=tmp_path, recipe=_recipe(),
                   writer="null", strict=False)
    meta = rec._resolve_attn_meta()
    assert meta is not None
    assert (meta.n_heads, meta.head_dim) == (4, 4)


def test_config_less_module_attributes_resolved(tmp_path):
    """A config-less custom model exposing num_heads/head_dim on its attention
    submodule must resolve."""
    rec = Recorder(_ConfigLess(), run_dir=tmp_path, recipe=_recipe(),
                   writer="null", strict=False)
    meta = rec._resolve_attn_meta()
    assert meta is not None
    assert (meta.n_heads, meta.n_kv_heads, meta.head_dim) == (4, 4, 4)


def test_explicit_recipe_override_wins(tmp_path):
    """recipe.attn_head_meta overrides everything (incl. a present config)."""
    rec = Recorder(
        _TopLevelConfig(), run_dir=tmp_path,
        recipe=_recipe(attn_head_meta={"n_heads": 8, "head_dim": 32}),
        writer="null", strict=False,
    )
    meta = rec._resolve_attn_meta()
    assert (meta.n_heads, meta.head_dim) == (8, 32)


def test_override_derives_head_dim_from_hidden_size(tmp_path):
    rec = Recorder(
        _Bare(), run_dir=tmp_path,
        recipe=_recipe(attn_head_meta={"n_heads": 8, "hidden_size": 64}),
        writer="null", strict=False,
    )
    meta = rec._resolve_attn_meta()
    assert (meta.n_heads, meta.head_dim) == (8, 8)  # 64 // 8


def test_unresolvable_returns_none_and_warns(tmp_path, caplog):
    """No config, no head attrs, no override -> None, and attach() warns once
    naming what was searched."""
    import logging

    rec = Recorder(_Bare(), run_dir=tmp_path, recipe=_recipe(),
                   writer="null", strict=False)
    assert rec._resolve_attn_meta() is None
    with caplog.at_level(logging.WARNING):
        rec.attach()
        rec.detach()
    msgs = [r.message for r in caplog.records]
    assert any("attention_head_rank requested but head metadata" in m for m in msgs)
    assert any("attn_head_meta" in m for m in msgs)
