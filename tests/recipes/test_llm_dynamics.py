"""End-to-end test: llm recipe emits v1.3 training-dynamics tags on a toy transformer."""
from __future__ import annotations

import torch
import torch.nn as nn

from circuitry.recipes.llm import RECIPE
from circuitry.recorder.live import Recorder


class _RecordingWriter:
    def __init__(self):
        self.scalars: list[tuple[str, float, int]] = []
    def add_scalar(self, tag, value, step): self.scalars.append((tag, value, step))
    def add_histogram(self, *a, **k): pass
    def add_image(self, *a, **k): pass
    def add_text(self, *a, **k): pass
    def flush(self): pass
    def close(self): pass


class _TinyAttn(nn.Module):
    """Minimal attention layer with qkv / o projections."""
    def __init__(self, d=16):
        super().__init__()
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.o_proj = nn.Linear(d, d, bias=False)
    def forward(self, x):
        return self.o_proj(self.v_proj(x) + self.q_proj(x) + self.k_proj(x))


class _TinyMLP(nn.Module):
    def __init__(self, d=16):
        super().__init__()
        self.gate_proj = nn.Linear(d, d * 4, bias=False)
        self.down_proj = nn.Linear(d * 4, d, bias=False)
    def forward(self, x):
        return self.down_proj(torch.relu(self.gate_proj(x)))


class _TinyLayer(nn.Module):
    def __init__(self, d=16):
        super().__init__()
        self.self_attn = _TinyAttn(d)
        self.mlp = _TinyMLP(d)
    def forward(self, x):
        return self.mlp(self.self_attn(x) + x)


class _TinyLM(nn.Module):
    def __init__(self, d=16, n_layers=2):
        super().__init__()
        self.layers = nn.ModuleList([_TinyLayer(d) for _ in range(n_layers)])
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def _run_recorder(tmp_path, n_steps=3):
    model = _TinyLM()
    writer = _RecordingWriter()
    rec = Recorder(
        model, run_dir=tmp_path, recipe=RECIPE,
        writer=writer, every_n_steps=1, strict=False,
    )
    rec.attach()
    for s in range(n_steps):
        with torch.no_grad():
            _ = model(torch.randn(2, 8, 16))
        rec.step(s)
    rec.detach()
    return writer


def test_llm_recipe_emits_update_delta(tmp_path):
    writer = _run_recorder(tmp_path)
    tags = {t for t, _, _ in writer.scalars}
    assert any("weight/update_delta/" in t for t in tags), \
        f"Expected update_delta tags; got: {sorted(tags)[:10]}"


def test_llm_recipe_emits_rank_trajectory(tmp_path):
    writer = _run_recorder(tmp_path)
    tags = {t for t, _, _ in writer.scalars}
    assert any("weight/rank_trajectory/" in t for t in tags)


def test_llm_recipe_emits_direction_cosine(tmp_path):
    # direction_cosine needs 3 emit steps (prev_prev populated after step 1)
    writer = _run_recorder(tmp_path, n_steps=3)
    tags = {t for t, _, _ in writer.scalars}
    assert any("weight/direction_cosine/" in t for t in tags)


def test_update_delta_nonneg(tmp_path):
    writer = _run_recorder(tmp_path)
    for t, v, _ in writer.scalars:
        if "weight/update_delta/" in t:
            assert v >= 0.0


def test_direction_cosine_in_range(tmp_path):
    writer = _run_recorder(tmp_path, n_steps=3)
    for t, v, _ in writer.scalars:
        if "weight/direction_cosine/" in t:
            assert -1.0 - 1e-5 <= v <= 1.0 + 1e-5


def _run_recorder_training(tmp_path, n_steps=3):
    """Like _run_recorder but drives a real optimizer step between emits."""
    model = _TinyLM()
    writer = _RecordingWriter()
    rec = Recorder(
        model, run_dir=tmp_path, recipe=RECIPE,
        writer=writer, every_n_steps=1, strict=False,
    )
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    rec.attach()
    for s in range(n_steps):
        opt.zero_grad()
        loss = model(torch.randn(2, 8, 16)).pow(2).sum()
        loss.backward()
        opt.step()
        rec.step(s)
    rec.detach()
    return writer


def test_llm_update_delta_positive_under_training(tmp_path):
    # End-to-end regression for the snapshot-alias bug: with a real optimizer
    # step between emits, update_delta must be strictly positive. An aliasing
    # snapshot would report exactly 0.0 on every module.
    writer = _run_recorder_training(tmp_path)
    deltas = [v for t, v, _ in writer.scalars if "weight/update_delta/" in t]
    assert deltas
    assert max(deltas) > 0.0
