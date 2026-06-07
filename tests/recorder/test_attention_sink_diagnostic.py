"""Recorder integration for the v1.12 attention_sink_score diagnostic."""
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


def _recipe(name: str) -> Recipe:
    return Recipe(
        name=name,
        hook_points=[
            HookPoint(source=TensorSource.OUTPUT,
                      pattern=r"layers\.\d+\.self_attn$"),
        ],
        activation_diagnostics=["attention_sink_score"],
    )


def test_attention_sink_score_emits_per_head_tags(tmp_path):
    """Recorder emits activation/attention_sink_score/<module>/head_N tags."""
    model = _TinyLM()
    rec = Recorder(model, tmp_path, _recipe("sink"),
                   writer="jsonl", every_n_steps=1, strict=False)
    rec.attach()
    model(torch.randint(0, VOCAB, (1, 8)), output_attentions=True)
    rec.step(0)
    rec.detach()

    out = (tmp_path / "metrics.jsonl").read_text()
    assert "activation/attention_sink_score/layers.0.self_attn/head_0" in out
    assert "activation/attention_sink_score/layers.0.self_attn/head_1" in out


def test_attention_sink_score_no_hook_leak_after_detach(tmp_path):
    """No forward hooks remain on the self_attn module after detach()."""
    model = _TinyLM()
    rec = Recorder(model, tmp_path, _recipe("sink_leak"),
                   writer=RecordingWriter(), every_n_steps=1, strict=False)
    rec.attach()
    model(torch.randint(0, VOCAB, (1, 8)), output_attentions=True)
    rec.step(0)
    rec.detach()

    assert not model.layers[0].self_attn._forward_hooks


def test_attention_sink_score_uses_training_forward_not_probe(tmp_path):
    """attention_sink_score reads _main_pass_attn (training forward), so it
    emits tags even with no induction probe (no vocab / embedding layer)."""
    # Model with no Embedding — induction/copy_suppression probes can't fire.
    class _EmbFree(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.self_attn = _Attn()

        def forward(self, x, output_attentions: bool = False):
            return self.self_attn(x, output_attentions=output_attentions)

    model = _EmbFree()
    recipe = Recipe(
        name="sink_no_emb",
        hook_points=[
            HookPoint(source=TensorSource.OUTPUT, pattern=r"self_attn$"),
        ],
        activation_diagnostics=["attention_sink_score"],
    )
    writer = RecordingWriter()
    rec = Recorder(model, tmp_path, recipe, writer=writer,
                   every_n_steps=1, strict=False)
    rec.attach()
    model(torch.randn(1, 6, D_MODEL), output_attentions=True)
    rec.step(0)
    rec.detach()

    tags = {t for t, _v, _s in writer.scalars}
    assert any("attention_sink_score" in t for t in tags)


def test_attention_sink_all_three_attention_diagnostics_coexist(tmp_path):
    """induction_score, copy_suppression_score, and attention_sink_score can
    all be enabled together without error."""
    model = _TinyLM()
    recipe = Recipe(
        name="all_three",
        hook_points=[
            HookPoint(source=TensorSource.OUTPUT,
                      pattern=r"layers\.\d+\.self_attn$"),
        ],
        activation_diagnostics=[
            "induction_score",
            "copy_suppression_score",
            "attention_sink_score",
        ],
        induction_probe_seq_len=10,
    )
    writer = RecordingWriter()
    rec = Recorder(model, tmp_path, recipe, writer=writer,
                   every_n_steps=1, strict=False)
    rec.attach()
    model(torch.randint(0, VOCAB, (1, 8)), output_attentions=True)
    rec.step(0)
    rec.detach()

    tags = {t for t, _v, _s in writer.scalars}
    assert any("induction_score" in t for t in tags)
    assert any("copy_suppression_score" in t for t in tags)
    assert any("attention_sink_score" in t for t in tags)
