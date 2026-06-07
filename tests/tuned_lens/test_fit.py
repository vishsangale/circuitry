"""Tests for the tuned-lens workflow (v1.10): fit_tuned_lens + TunedLens.

Spec: docs/superpowers/specs/2026-06-07-v1.10-tuned-lens-design.md §5.
"""
from __future__ import annotations

import re

import pytest
import torch
from torch import nn

from circuitry.core.lens import logit_lens_kl, tuned_lens_kl
from circuitry.tuned_lens import TunedLens, fit_tuned_lens, model_fingerprint


class _Block(nn.Module):
    """A residual block that rotates + shifts the stream — i.e. each layer lives
    in a progressively different basis than the final frame, which is exactly
    what a tuned lens is supposed to undo."""

    def __init__(self, d_model: int, seed: int) -> None:
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.lin = nn.Linear(d_model, d_model)
        with torch.no_grad():
            self.lin.weight.copy_(torch.eye(d_model) + 0.1 * torch.randn(
                d_model, d_model, generator=g))
            self.lin.bias.copy_(0.1 * torch.randn(d_model, generator=g))

    def forward(self, x):
        return x + torch.tanh(self.lin(x))


class _TinyLM(nn.Module):
    """Minimal decoder: embed -> N residual blocks (named `.layers.N`) ->
    final LayerNorm -> tied unembed. Exposes get_output_embeddings()."""

    def __init__(self, vocab: int = 24, d_model: int = 8, n_layers: int = 4) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model)
        self.layers = nn.ModuleList([_Block(d_model, seed=i) for i in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab, bias=False)
        self.lm_head.weight = self.embed.weight  # tied

    def forward(self, input_ids):
        x = self.embed(input_ids)
        for blk in self.layers:
            x = blk(x)
        return self.lm_head(self.norm(x))

    def get_output_embeddings(self):
        return self.lm_head


def _batches(model, n=3, B=2, T=5, vocab=24, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [torch.randint(0, vocab, (B, T), generator=g) for _ in range(n)]


def test_fit_returns_translators_for_all_but_last_layer():
    torch.manual_seed(0)
    model = _TinyLM(n_layers=4)
    lens = fit_tuned_lens(model, _batches(model), steps=50)
    assert lens.layers == [0, 1, 2]  # last block is the target frame
    assert lens.d_model == 8
    assert all(a.shape == (8, 8) and b.shape == (8,) for a, b in lens.translators)
    assert lens.model_fingerprint == model_fingerprint(model)


def test_fitted_lens_beats_logit_lens_on_a_mid_layer():
    torch.manual_seed(1)
    model = _TinyLM(n_layers=4)
    train = _batches(model, n=4, seed=1)
    lens = fit_tuned_lens(model, train, steps=400, lr=5e-2)

    # Recompute residuals + target on held-out data the way the recorder does.
    held = _batches(model, n=1, seed=99)[0]
    res: dict[int, torch.Tensor] = {}
    handles = []
    for name, mod in model.named_modules():
        m = re.search(r"(?:^|\.)layers\.(\d+)$", name)
        if m:
            idx = int(m.group(1))
            handles.append(mod.register_forward_hook(
                lambda _m, _i, out, idx=idx: res.__setitem__(idx, out)))
    model.eval()
    with torch.no_grad():
        model(held)
    for h in handles:
        h.remove()

    W = model.get_output_embeddings().weight.detach()
    ln = model.norm
    last = res[3]
    final_logits = ln(last) @ W.t()

    layer = 1
    A, b = lens.translator_for(layer)
    logit_kl = logit_lens_kl(res[layer], W, final_logits, layer_norm=ln)
    tuned_kl = tuned_lens_kl(res[layer], (A, b), W, final_logits, layer_norm=ln)
    assert tuned_kl < logit_kl


def test_save_load_round_trips(tmp_path):
    torch.manual_seed(2)
    model = _TinyLM(n_layers=3)
    lens = fit_tuned_lens(model, _batches(model), steps=10)
    p = tmp_path / "lens.pt"
    lens.save(p)
    loaded = TunedLens.load(p)
    assert loaded.layers == lens.layers
    assert loaded.d_model == lens.d_model
    assert loaded.model_fingerprint == lens.model_fingerprint
    for (a0, b0), (a1, b1) in zip(lens.translators, loaded.translators, strict=True):
        assert torch.allclose(a0, a1)
        assert torch.allclose(b0, b1)


def test_fingerprint_distinguishes_architectures():
    fp_a = model_fingerprint(_TinyLM(n_layers=4))
    fp_b = model_fingerprint(_TinyLM(n_layers=4))
    fp_c = model_fingerprint(_TinyLM(n_layers=6))
    assert fp_a == fp_b          # same architecture, different weights
    assert fp_a != fp_c          # different depth


def test_explicit_layers_subset():
    torch.manual_seed(3)
    model = _TinyLM(n_layers=4)
    lens = fit_tuned_lens(model, _batches(model), layers=[0, 2], steps=10)
    assert lens.layers == [0, 2]


def test_container_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="translators but"):
        TunedLens(translators=[(torch.eye(4), torch.zeros(4))],
                  layers=[0, 1], d_model=4, model_fingerprint="x")
    with pytest.raises(ValueError, match="A has shape"):
        TunedLens(translators=[(torch.eye(3), torch.zeros(4))],
                  layers=[0], d_model=4, model_fingerprint="x")


def test_load_rejects_foreign_file(tmp_path):
    p = tmp_path / "junk.pt"
    torch.save({"format": "something-else"}, p)
    with pytest.raises(ValueError, match="not a circuitry.TunedLens"):
        TunedLens.load(p)


def test_raises_without_output_embedding():
    class _NoUnembed(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([nn.Linear(4, 4)])

        def forward(self, x):
            return self.layers[0](x)

    with pytest.raises(ValueError, match="no resolvable output embedding"):
        fit_tuned_lens(_NoUnembed(), [torch.randn(2, 4)], steps=1)
