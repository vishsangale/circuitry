"""Recorder integration for the v1.11 copy_suppression_score diagnostic."""
from __future__ import annotations

import torch
import torch.nn as nn

from circuitry import HookPoint, Recipe, Recorder, TensorSource
from circuitry.writers.base import RecordingWriter

D_MODEL = 8
N_HEADS = 2
VOCAB = 32


class _Attn(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.k_proj = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.v_proj = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.o_proj = nn.Linear(D_MODEL, D_MODEL, bias=False)

    def forward(self, x, output_attentions: bool = False):
        B, T, D = x.shape
        H = N_HEADS
        HD = D // H
        q = self.q_proj(x).view(B, T, H, HD).transpose(1, 2)
        k = self.k_proj(x).view(B, T, H, HD).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, HD).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / (HD ** 0.5)
        attn = scores.softmax(dim=-1)
        out = self.o_proj((attn @ v).transpose(1, 2).reshape(B, T, D))
        if output_attentions:
            return out, attn
        return out


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = _Attn()

    def forward(self, x, output_attentions: bool = False):
        return self.self_attn(x, output_attentions=output_attentions)


class _TinyLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.tok_embed = nn.Embedding(VOCAB, D_MODEL)
        self.layers = nn.ModuleList([_Block()])

    def get_output_embeddings(self):
        return None

    def forward(self, input_ids, output_attentions: bool = False):
        x = self.tok_embed(input_ids)
        for layer in self.layers:
            x = layer(x, output_attentions=output_attentions)
        return x


def _recipe(name: str, diagnostics: list[str]) -> Recipe:
    return Recipe(
        name=name,
        hook_points=[
            HookPoint(source=TensorSource.OUTPUT,
                      pattern=r"layers\.\d+\.self_attn$"),
        ],
        activation_diagnostics=diagnostics,
        induction_probe_seq_len=10,
    )


def test_copy_suppression_score_emits_per_head_tags(tmp_path):
    """Recorder emits activation/copy_suppression_score/<module>/head_N tags."""
    model = _TinyLM()
    rec = Recorder(model, tmp_path,
                   _recipe("css", ["copy_suppression_score"]),
                   writer="jsonl", every_n_steps=1, strict=False)
    rec.attach()
    rec.step(0)
    rec.detach()

    out = (tmp_path / "metrics.jsonl").read_text()
    assert "activation/copy_suppression_score/layers.0.self_attn/head_0" in out
    assert "activation/copy_suppression_score/layers.0.self_attn/head_1" in out


def test_probe_shared_when_both_probe_diagnostics_enabled(tmp_path):
    """When both induction_score and copy_suppression_score are enabled, a
    single probe forward pass is shared (the attn module's forward is called
    exactly once for the probe, not twice)."""
    model = _TinyLM()
    probe_call_count = 0
    original_forward = model.layers[0].self_attn.forward

    def _counting_forward(*args, **kwargs):
        nonlocal probe_call_count
        probe_call_count += 1
        return original_forward(*args, **kwargs)

    model.layers[0].self_attn.forward = _counting_forward

    rec = Recorder(model, tmp_path,
                   _recipe("both", ["induction_score", "copy_suppression_score"]),
                   writer="jsonl", every_n_steps=1, strict=False)
    rec.attach()
    probe_call_count = 0  # reset after attach() (may do its own probe runs)
    rec.step(0)
    rec.detach()

    # The probe forward fires the self_attn once (shared cache). The training
    # forward (which may also fire the self_attn) is part of setup — we reset
    # the counter right before step(0), so we're counting only the probe calls
    # that happen inside step(0).
    assert probe_call_count == 1


def test_copy_suppression_no_hook_leak_after_detach(tmp_path):
    """No forward hooks remain on the self_attn module after detach()."""
    model = _TinyLM()
    rec = Recorder(model, tmp_path,
                   _recipe("leak", ["copy_suppression_score"]),
                   writer=RecordingWriter(), every_n_steps=1, strict=False)
    rec.attach()
    rec.step(0)
    rec.detach()

    assert not model.layers[0].self_attn._forward_hooks


def test_copy_suppression_and_induction_coexist(tmp_path):
    """Both diagnostics emit tags when enabled together."""
    model = _TinyLM()
    writer = RecordingWriter()
    rec = Recorder(model, tmp_path,
                   _recipe("both_rw", ["induction_score", "copy_suppression_score"]),
                   writer=writer, every_n_steps=1, strict=False)
    rec.attach()
    rec.step(0)
    rec.detach()

    tags = {t for t, _v, _s in writer.scalars}
    assert any("induction_score" in t for t in tags)
    assert any("copy_suppression_score" in t for t in tags)
