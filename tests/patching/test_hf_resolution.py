"""Tests for HFSiteResolver."""
from __future__ import annotations

import pytest
import torch

from circuitry.patching.sites import HFSiteResolver, Site


@pytest.fixture
def resolver():
    return HFSiteResolver(
        n_heads=2, d_model=8, d_mlp=16,
        layer_pattern="layers.{L}",
        attn_module="self_attn.o_proj",
        mlp_module="mlp",
        mlp_intermediate="mlp.down_proj",
    )


def test_resid_post_resolves_to_layer_output(resolver, transformer_model):
    site = Site(component="resid_post", layer=0)
    resolved = resolver.resolve(transformer_model, site)
    assert resolved.module is transformer_model.layers[0]
    assert resolved.is_input_hook is False


def test_resid_pre_resolves_to_layer_input(resolver, transformer_model):
    site = Site(component="resid_pre", layer=1)
    resolved = resolver.resolve(transformer_model, site)
    assert resolved.module is transformer_model.layers[1]
    assert resolved.is_input_hook is True


def test_attn_head_out_resolves_to_o_proj_input(resolver, transformer_model):
    site = Site(component="attn_head_out", layer=0, head=1)
    resolved = resolver.resolve(transformer_model, site)
    assert resolved.module is transformer_model.layers[0].self_attn.o_proj
    assert resolved.is_input_hook is True


def test_mlp_out_resolves_to_mlp_output(resolver, transformer_model):
    site = Site(component="mlp_out", layer=0)
    resolved = resolver.resolve(transformer_model, site)
    assert resolved.module is transformer_model.layers[0].mlp
    assert resolved.is_input_hook is False


def test_mlp_neuron_resolves_to_down_proj_input(resolver, transformer_model):
    site = Site(component="mlp_neuron", layer=0, neuron=5)
    resolved = resolver.resolve(transformer_model, site)
    assert resolved.module is transformer_model.layers[0].mlp.down_proj
    assert resolved.is_input_hook is True


def test_extract_head_slice_correct(resolver, transformer_model):
    site = Site(component="attn_head_out", layer=0, head=1)
    resolved = resolver.resolve(transformer_model, site)
    torch.manual_seed(0)
    x = torch.randn(2, 3, 8)  # (batch, seq, d_model=8), n_heads=2, head_dim=4
    extracted = resolved.extract(x)
    expected = x.reshape(2, 3, 2, 4)[:, :, 1, :]  # head 1
    assert torch.equal(extracted, expected)


def test_inject_head_slice_correct(resolver, transformer_model):
    site = Site(component="attn_head_out", layer=0, head=0)
    resolved = resolver.resolve(transformer_model, site)
    torch.manual_seed(1)
    x = torch.randn(2, 3, 8)
    new_val = torch.ones(2, 3, 4)
    injected = resolved.inject(x, new_val)
    result_heads = injected.reshape(2, 3, 2, 4)
    assert torch.equal(result_heads[:, :, 0, :], new_val)
    assert torch.equal(result_heads[:, :, 1, :], x.reshape(2, 3, 2, 4)[:, :, 1, :])


def test_extract_neuron_correct(resolver, transformer_model):
    site = Site(component="mlp_neuron", layer=0, neuron=5)
    resolved = resolver.resolve(transformer_model, site)
    torch.manual_seed(2)
    x = torch.randn(2, 3, 16)  # (batch, seq, d_mlp=16)
    extracted = resolved.extract(x)
    assert torch.equal(extracted, x[:, :, 5])


def test_inject_neuron_correct(resolver, transformer_model):
    site = Site(component="mlp_neuron", layer=0, neuron=5)
    resolved = resolver.resolve(transformer_model, site)
    torch.manual_seed(3)
    x = torch.randn(2, 3, 16)
    new_val = torch.ones(2, 3)
    injected = resolved.inject(x, new_val)
    assert torch.equal(injected[:, :, 5], new_val)
    mask = torch.ones(16, dtype=torch.bool)
    mask[5] = False
    assert torch.equal(injected[:, :, mask], x[:, :, mask])


def test_position_slicing(resolver, transformer_model):
    site = Site(component="resid_post", layer=0, position=2)
    resolved = resolver.resolve(transformer_model, site)
    torch.manual_seed(4)
    x = torch.randn(2, 5, 8)
    extracted = resolved.extract(x)
    assert torch.equal(extracted, x[:, 2])


def test_position_inject(resolver, transformer_model):
    site = Site(component="resid_post", layer=0, position=2)
    resolved = resolver.resolve(transformer_model, site)
    torch.manual_seed(5)
    x = torch.randn(2, 5, 8)
    new_val = torch.ones(2, 8)
    injected = resolved.inject(x, new_val)
    assert torch.equal(injected[:, 2], new_val)
    assert torch.equal(injected[:, 0], x[:, 0])


def test_from_config():
    class FakeConfig:
        num_attention_heads = 4
        hidden_size = 32
        intermediate_size = 64

    resolver = HFSiteResolver.from_config(FakeConfig())
    assert resolver.n_heads == 4
    assert resolver.d_model == 32
    assert resolver.d_mlp == 64


def test_from_config_missing_fields():
    class BadConfig:
        pass

    with pytest.raises(ValueError, match="num_attention_heads"):
        HFSiteResolver.from_config(BadConfig())


def test_mlp_neuron_without_d_mlp(transformer_model):
    resolver = HFSiteResolver(n_heads=2, d_model=8, d_mlp=None)
    site = Site(component="mlp_neuron", layer=0, neuron=0)
    with pytest.raises(ValueError, match="d_mlp"):
        resolver.resolve(transformer_model, site)
