"""Tests for Site dataclass + validation."""
from __future__ import annotations

import pytest

from circuitry.patching.sites import VALID_COMPONENTS, Site


def test_site_valid_components():
    for comp in VALID_COMPONENTS:
        kwargs = {"component": comp, "layer": 0}
        if comp in ("attn_head_out", "attn_head_q_out", "attn_head_k_out"):
            kwargs["head"] = 0
        if comp == "mlp_neuron":
            kwargs["neuron"] = 0
        site = Site(**kwargs)
        assert site.component == comp


def test_site_rejects_unknown_component():
    with pytest.raises(ValueError, match="Unknown component"):
        Site(component="bogus", layer=0)


def test_attn_head_out_requires_head():
    with pytest.raises(ValueError, match="requires head"):
        Site(component="attn_head_out", layer=0)


def test_attn_head_out_accepts_head():
    s = Site(component="attn_head_out", layer=0, head=3)
    assert s.head == 3


def test_attn_head_q_out_requires_head():
    with pytest.raises(ValueError, match="requires head"):
        Site(component="attn_head_q_out", layer=0)


def test_attn_head_q_out_accepts_head():
    s = Site(component="attn_head_q_out", layer=0, head=1)
    assert s.head == 1


def test_attn_head_k_out_requires_head():
    with pytest.raises(ValueError, match="requires head"):
        Site(component="attn_head_k_out", layer=0)


def test_attn_head_k_out_accepts_head():
    s = Site(component="attn_head_k_out", layer=0, head=2)
    assert s.head == 2


def test_mlp_neuron_requires_neuron():
    with pytest.raises(ValueError, match="requires neuron"):
        Site(component="mlp_neuron", layer=0)


def test_mlp_neuron_accepts_neuron():
    s = Site(component="mlp_neuron", layer=0, neuron=42)
    assert s.neuron == 42


def test_site_frozen():
    s = Site(component="resid_post", layer=0)
    with pytest.raises(AttributeError):
        s.layer = 1  # type: ignore[misc]


def test_site_hashable():
    s1 = Site(component="resid_post", layer=0)
    s2 = Site(component="resid_post", layer=0)
    assert s1 == s2
    assert hash(s1) == hash(s2)
    assert len({s1, s2}) == 1


def test_site_with_position():
    s = Site(component="resid_post", layer=0, position=3)
    assert s.position == 3
    s2 = Site(component="resid_post", layer=0, position=slice(1, 4))
    assert s2.position == slice(1, 4)
