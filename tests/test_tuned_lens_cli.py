"""CLI tests for `circuitry fit-tuned-lens` (v1.10)."""
from __future__ import annotations

import sys
import textwrap

import pytest

from circuitry.cli.main import _load_entrypoint, main
from circuitry.tuned_lens import TunedLens

_LENS_MOD = textwrap.dedent(
    """
    import torch
    from torch import nn

    D, V = 8, 16

    class _Block(nn.Module):
        def __init__(self, s):
            super().__init__()
            g = torch.Generator().manual_seed(s)
            self.lin = nn.Linear(D, D, bias=False)
            with torch.no_grad():
                self.lin.weight.copy_(0.2 * torch.randn(D, D, generator=g))
        def forward(self, x):
            return x + torch.tanh(self.lin(x))

    class _Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([_Block(0), _Block(1), _Block(2)])
            self.norm = nn.LayerNorm(D)
            self.lm_head = nn.Linear(D, V, bias=False)
        def get_output_embeddings(self):
            return self.lm_head
        def forward(self, x):
            for b in self.layers:
                x = b(x)
            return self.lm_head(self.norm(x))

    def make_model():
        return _Tiny()

    def make_batches():
        g = torch.Generator().manual_seed(7)
        return [torch.randn(2, 5, D, generator=g) for _ in range(3)]
    """
)


@pytest.fixture()
def lens_module(tmp_path, monkeypatch):
    (tmp_path / "_lensmod.py").write_text(_LENS_MOD)
    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop("_lensmod", None)
    yield
    sys.modules.pop("_lensmod", None)


def test_fit_tuned_lens_writes_loadable_lens(tmp_path, lens_module, capsys):
    out = tmp_path / "lens.pt"
    rc = main([
        "fit-tuned-lens",
        "--model", "_lensmod:make_model",
        "--batches", "_lensmod:make_batches",
        "--out", str(out),
        "--steps", "5",
    ])
    assert rc == 0
    assert out.exists()
    lens = TunedLens.load(out)
    assert lens.layers == [0, 1]          # last block is the target frame
    assert lens.d_model == 8
    assert "wrote" in capsys.readouterr().out


def test_fit_tuned_lens_explicit_layers(tmp_path, lens_module):
    out = tmp_path / "lens.pt"
    rc = main([
        "fit-tuned-lens",
        "--model", "_lensmod:make_model",
        "--batches", "_lensmod:make_batches",
        "--out", str(out),
        "--layers", "0", "2",
        "--steps", "3",
    ])
    assert rc == 0
    assert TunedLens.load(out).layers == [0, 2]


def test_load_entrypoint_rejects_malformed_spec():
    with pytest.raises(ValueError, match="package.module:attr"):
        _load_entrypoint("no_colon_here")


def test_load_entrypoint_missing_attr(lens_module):
    with pytest.raises(ValueError, match="no attribute"):
        _load_entrypoint("_lensmod:does_not_exist")
