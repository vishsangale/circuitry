"""Recorder integration for the v1.10 tuned-lens diagnostic.

Spec: docs/superpowers/specs/2026-06-07-v1.10-tuned-lens-design.md §5.
"""
from __future__ import annotations

import dataclasses
import logging

import torch
from torch import nn

from circuitry import HookPoint, Recipe, Recorder, TensorSource
from circuitry.tuned_lens import fit_tuned_lens
from circuitry.writers.base import RecordingWriter

D_MODEL, VOCAB = 8, 16


class _Block(nn.Module):
    def __init__(self, seed: int) -> None:
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.lin = nn.Linear(D_MODEL, D_MODEL, bias=False)
        with torch.no_grad():
            self.lin.weight.copy_(0.2 * torch.randn(D_MODEL, D_MODEL, generator=g))

    def forward(self, x):
        return x + torch.tanh(self.lin(x))


class _Tiny(nn.Module):
    def __init__(self, n_layers: int = 3) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_Block(i) for i in range(n_layers)])
        self.ln_f = nn.LayerNorm(D_MODEL)
        self.lm_head = nn.Linear(D_MODEL, VOCAB, bias=False)

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, x):
        for b in self.layers:
            x = b(x)
        return self.lm_head(self.ln_f(x))


def _batches(n=3, B=2, T=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(B, T, D_MODEL, generator=g) for _ in range(n)]


def _recipe(name, tuned_lens=None):
    return Recipe(
        name=name,
        hook_points=[HookPoint(source=TensorSource.OUTPUT, pattern=r"layers\.\d+$")],
        activation_diagnostics=["tuned_lens_kl"],
        tuned_lens=tuned_lens,
    )


def test_emits_tuned_lens_kl_per_fitted_layer(tmp_path):
    model = _Tiny(n_layers=3)
    lens = fit_tuned_lens(model, _batches(), steps=20)
    writer = RecordingWriter()
    rec = Recorder(model, tmp_path, _recipe("tl", tuned_lens=lens),
                   writer=writer, every_n_steps=1, strict=False)
    rec.attach()
    model(torch.randn(1, 4, D_MODEL))
    rec.step(0)
    rec.detach()

    tags = {t for t, _v, _s in writer.scalars}
    # layers 0 and 1 are fitted; layer 2 is the final frame (not fitted).
    assert "activation/tuned_lens_kl/layers.0" in tags
    assert "activation/tuned_lens_kl/layers.1" in tags
    assert "activation/tuned_lens_kl/layers.2" not in tags


def test_warns_and_skips_when_no_lens_supplied(tmp_path, caplog):
    model = _Tiny(n_layers=3)
    writer = RecordingWriter()
    rec = Recorder(model, tmp_path, _recipe("tl_nolens", tuned_lens=None),
                   writer=writer, every_n_steps=1, strict=False)
    with caplog.at_level(logging.WARNING, logger="circuitry"):
        rec.attach()
    model(torch.randn(1, 4, D_MODEL))
    rec.step(0)
    rec.detach()

    assert any("no fitted Recipe.tuned_lens" in r.getMessage() for r in caplog.records)
    assert not any(t.startswith("activation/tuned_lens_kl/")
                   for t, _v, _s in writer.scalars)


def test_warns_and_skips_on_fingerprint_mismatch(tmp_path, caplog):
    # Fit on one architecture, attach to a deeper one.
    lens = fit_tuned_lens(_Tiny(n_layers=3), _batches(), steps=10)
    other = _Tiny(n_layers=5)
    writer = RecordingWriter()
    rec = Recorder(other, tmp_path, _recipe("tl_mismatch", tuned_lens=lens),
                   writer=writer, every_n_steps=1, strict=False)
    with caplog.at_level(logging.WARNING, logger="circuitry"):
        rec.attach()
    other(torch.randn(1, 4, D_MODEL))
    rec.step(0)
    rec.detach()

    assert any("fitted on a different model" in r.getMessage()
               for r in caplog.records)
    assert not any(t.startswith("activation/tuned_lens_kl/")
                   for t, _v, _s in writer.scalars)


def test_no_hook_leak_after_detach(tmp_path):
    model = _Tiny(n_layers=3)
    lens = fit_tuned_lens(model, _batches(), steps=5)
    rec = Recorder(model, tmp_path, _recipe("tl_leak", tuned_lens=lens),
                   writer=RecordingWriter(), every_n_steps=1, strict=False)
    rec.attach()
    model(torch.randn(1, 4, D_MODEL))
    rec.step(0)
    rec.detach()
    # No forward hooks should remain on any block.
    for blk in model.layers:
        assert not blk._forward_hooks


def test_logit_and_tuned_lens_coexist(tmp_path):
    """Both lens diagnostics can be enabled together; _lens_meta is shared."""
    model = _Tiny(n_layers=3)
    lens = fit_tuned_lens(model, _batches(), steps=10)
    recipe = dataclasses.replace(
        _recipe("tl_both", tuned_lens=lens),
        activation_diagnostics=["logit_lens_kl", "tuned_lens_kl"],
    )
    writer = RecordingWriter()
    rec = Recorder(model, tmp_path, recipe, writer=writer, every_n_steps=1,
                   strict=False)
    rec.attach()
    model(torch.randn(1, 4, D_MODEL))
    rec.step(0)
    rec.detach()
    tags = {t for t, _v, _s in writer.scalars}
    assert "activation/logit_lens_kl/layers.0" in tags
    assert "activation/tuned_lens_kl/layers.0" in tags
