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


def test_llm_recipe_has_sv_histogram_and_per_param_grad():
    from circuitry.recipes import get_recipe
    r = get_recipe("llm")
    assert "sv_histogram" in r.weight_diagnostics
    assert "norms_per_param" in r.gradient_diagnostics


def test_llm_recipe_has_grad_hook():
    """Without a GRAD HookPoint, norms_per_param is a silent no-op."""
    from circuitry.recipes import get_recipe
    from circuitry.recorder.hooks import TensorSource
    r = get_recipe("llm")
    assert any(hp.source == TensorSource.GRAD for hp in r.hook_points)


def test_llm_recipe_includes_attention_head_rank():
    r = get_recipe("llm")
    assert "attention_head_rank" in r.weight_diagnostics


def test_llm_recipe_includes_gate_stats():
    r = get_recipe("llm")
    assert "gate_stats" in r.activation_diagnostics


def test_llm_recipe_has_down_proj_input_hook():
    from circuitry.recorder.hooks import TensorSource
    r = get_recipe("llm")
    matches = [
        hp for hp in r.hook_points
        if hp.source is TensorSource.INPUT
        and hp.pattern is not None
        and "down_proj" in hp.pattern
    ]
    assert len(matches) == 1, [hp.pattern for hp in r.hook_points]
