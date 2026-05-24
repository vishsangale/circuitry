"""Tests for patch_site() context manager."""
from __future__ import annotations

import pytest
import torch

from circuitry.patching.intervene import patch_site
from circuitry.patching.sites import HFSiteResolver, Site


@pytest.fixture
def resolver():
    return HFSiteResolver(
        n_heads=1, d_model=4, d_mlp=8,
        layer_pattern="layers.{L}",
        attn_module="self_attn.o_proj",
        mlp_module="mlp",
        mlp_intermediate="mlp.down_proj",
    )


def test_patch_changes_output(toy_model, resolver):
    """Patching layer 0 output with z → final output becomes z (identity weights)."""
    torch.manual_seed(0)
    x = torch.randn(1, 4)
    z = torch.ones(1, 4) * 99.0
    site = Site(component="resid_post", layer=0)
    normal_out = toy_model(x)

    with patch_site(toy_model, site, z, resolver):
        patched_out = toy_model(x)

    assert not torch.equal(patched_out, normal_out)
    assert torch.allclose(patched_out, z, atol=1e-6)


def test_restore_after_normal_exit(toy_model, resolver):
    torch.manual_seed(1)
    x = torch.randn(1, 4)
    before = toy_model(x).clone()
    site = Site(component="resid_post", layer=0)

    with patch_site(toy_model, site, torch.zeros(1, 4), resolver):
        toy_model(x)

    after = toy_model(x)
    assert torch.equal(before, after)


def test_restore_after_exception(toy_model, resolver):
    torch.manual_seed(2)
    x = torch.randn(1, 4)
    before = toy_model(x).clone()
    site = Site(component="resid_post", layer=0)

    with pytest.raises(RuntimeError, match="intentional"):
        with patch_site(toy_model, site, torch.zeros(1, 4), resolver):
            toy_model(x)
            raise RuntimeError("intentional")

    after = toy_model(x)
    assert torch.equal(before, after)


def test_params_frozen_during_patch(toy_model, resolver):
    site = Site(component="resid_post", layer=0)
    with patch_site(toy_model, site, torch.zeros(1, 4), resolver):
        for p in toy_model.parameters():
            assert not p.requires_grad


def test_params_restored_after_patch(toy_model, resolver):
    for p in toy_model.parameters():
        p.requires_grad_(True)
    site = Site(component="resid_post", layer=0)

    with patch_site(toy_model, site, torch.zeros(1, 4), resolver):
        pass

    for p in toy_model.parameters():
        assert p.requires_grad


def test_eval_mode_set_and_restored_from_train(toy_model, resolver):
    toy_model.train()
    site = Site(component="resid_post", layer=0)

    with patch_site(toy_model, site, torch.zeros(1, 4), resolver):
        assert not toy_model.training

    assert toy_model.training


def test_eval_mode_stays_eval_if_already_eval(toy_model, resolver):
    toy_model.eval()
    site = Site(component="resid_post", layer=0)

    with patch_site(toy_model, site, torch.zeros(1, 4), resolver):
        assert not toy_model.training

    assert not toy_model.training


def test_no_hooks_remain_after_exit(toy_model, resolver):
    site = Site(component="resid_post", layer=0)

    def count_hooks(model):
        return sum(
            len(m._forward_hooks) + len(m._forward_pre_hooks)
            for m in model.modules()
        )

    before_hooks = count_hooks(toy_model)
    with patch_site(toy_model, site, torch.zeros(1, 4), resolver):
        pass
    after_hooks = count_hooks(toy_model)
    assert after_hooks == before_hooks


def test_param_values_unchanged(toy_model, resolver):
    params_before = {n: p.clone() for n, p in toy_model.named_parameters()}
    site = Site(component="resid_post", layer=0)

    with patch_site(toy_model, site, torch.zeros(1, 4), resolver):
        toy_model(torch.randn(1, 4))

    for n, p in toy_model.named_parameters():
        assert torch.equal(p, params_before[n]), f"param {n} changed"


def test_activation_grad_enabled(toy_model, resolver):
    """With enable_activation_grad=True, activation grads are available."""
    torch.manual_seed(3)
    x = torch.randn(1, 4)
    z = torch.randn(1, 4)
    site = Site(component="resid_post", layer=0)
    grad_holder = {}

    with patch_site(toy_model, site, z, resolver, enable_activation_grad=True) as handle:
        out = toy_model(x)
        out.sum().backward()
        grad_holder["grad"] = handle.activation_grad

    assert grad_holder["grad"] is not None
    for p in toy_model.parameters():
        assert p.grad is None
