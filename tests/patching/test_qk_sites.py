"""attn_head_q_out / attn_head_k_out per-head sites (q_proj/k_proj output)."""
from __future__ import annotations

import pytest
import torch

from circuitry.patching.sites import VALID_COMPONENTS, HFSiteResolver, Site


def test_qk_components_valid():
    assert "attn_head_q_out" in VALID_COMPONENTS
    assert "attn_head_k_out" in VALID_COMPONENTS
    Site(component="attn_head_q_out", layer=0, head=1)  # requires head: ok
    with pytest.raises(ValueError):
        Site(component="attn_head_q_out", layer=0)  # missing head


def test_q_out_resolves_and_slices(transformer_model):
    r = HFSiteResolver(n_heads=2, d_model=8, d_mlp=16, layer_pattern="layers.{L}")
    site = Site(component="attn_head_q_out", layer=0, head=1)
    resolved = r.resolve(transformer_model, site)
    assert resolved.module is transformer_model.layers[0].self_attn.q_proj
    assert resolved.is_input_hook is False  # OUTPUT hook
    x = torch.randn(2, 3, 8)  # q_proj output (b,s,d_model), n_heads=2, head_dim=4
    assert torch.equal(resolved.extract(x), x.reshape(2, 3, 2, 4)[:, :, 1, :])


def test_k_out_resolves(transformer_model):
    r = HFSiteResolver(n_heads=2, d_model=8, d_mlp=16, layer_pattern="layers.{L}")
    resolved = r.resolve(transformer_model, Site(component="attn_head_k_out", layer=1, head=0))
    assert resolved.module is transformer_model.layers[1].self_attn.k_proj
    assert resolved.is_input_hook is False


def test_inject_head_q_out(transformer_model):
    r = HFSiteResolver(n_heads=2, d_model=8, d_mlp=16, layer_pattern="layers.{L}")
    resolved = r.resolve(transformer_model, Site(component="attn_head_q_out", layer=0, head=0))
    x = torch.randn(2, 3, 8)
    new = torch.ones(2, 3, 4)
    injected = resolved.inject(x, new)
    heads = injected.reshape(2, 3, 2, 4)
    assert torch.equal(heads[:, :, 0, :], new)
    assert torch.equal(heads[:, :, 1, :], x.reshape(2, 3, 2, 4)[:, :, 1, :])
