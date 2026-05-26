import pytest
from circuitry.patching.sites import HFSiteResolver


class _Cfg:
    """Minimal config stub: head_dim independent of hidden_size/num_attention_heads."""
    num_attention_heads = 8
    hidden_size = 64          # d_model/n_heads == 8
    head_dim = 16             # explicit, != 8  -> the Gemma-2 condition
    intermediate_size = 128
    num_key_value_heads = 8


def test_resolver_honors_explicit_head_dim():
    r = HFSiteResolver.from_config(_Cfg())
    assert r.head_dim == 16            # not 64 // 8 == 8


def test_resolver_falls_back_when_no_head_dim():
    class C:
        num_attention_heads = 4
        hidden_size = 64               # 64 // 4 == 16
    r = HFSiteResolver.from_config(C())
    assert r.head_dim == 16            # fallback d_model // n_heads


def test_explicit_head_dim_kwarg():
    r = HFSiteResolver(n_heads=8, d_model=64, head_dim=16)
    assert r.head_dim == 16
