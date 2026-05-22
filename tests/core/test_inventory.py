"""Tests for circuitry.core.inventory."""

from __future__ import annotations

import json

import torch
import torch.nn as nn

from circuitry.core.inventory import ModelInventory, ParameterRecord


class _Wrapper(nn.Module):
    """Stand-in for HF ``Gemma4ClippableLinear``: holds the actual weight on
    a child ``nn.Linear`` rather than directly on itself."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        # Non-Parameter buffer — mirrors Gemma4ClippableLinear's clip ranges.
        self.register_buffer("input_min", torch.zeros(()))


class _DoubleLinearWrapper(nn.Module):
    """Wrapper with TWO Linear children — primary-weight resolution should
    return None (ambiguous)."""

    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Linear(4, 8, bias=False)
        self.b = nn.Linear(8, 4, bias=False)


def test_build_lists_every_named_parameter():
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 4))
    inv = ModelInventory.build(model)
    names = {r.name for r in inv.parameters}
    assert names == {"0.weight", "0.bias", "2.weight", "2.bias"}


def test_record_fields_are_populated():
    model = nn.Linear(4, 8)
    inv = ModelInventory.build(model)
    by_name = {r.name: r for r in inv.parameters}
    w = by_name["weight"]
    assert isinstance(w, ParameterRecord)
    assert w.shape == (8, 4)
    assert w.numel == 32
    assert w.dtype == torch.float32
    assert w.requires_grad is True
    assert w.owning_module_name == ""  # parameter is on the root model
    assert w.owning_module_class == "Linear"
    assert w.leaf_attr == "weight"


def test_find_primary_weight_direct_attribute():
    """For a plain Linear, find_primary_weight returns the .weight Parameter."""
    model = nn.Linear(4, 8, bias=False)
    inv = ModelInventory.build(model)
    rec = inv.find_primary_weight("")  # root model is the Linear itself
    assert rec is not None
    assert rec.name == "weight" and rec.shape == (8, 4)


def test_find_primary_weight_recurses_into_wrapper():
    """Gemma4ClippableLinear-style: wrapper has no ``.weight`` directly, but
    a single Linear child does. Inventory resolves through the child."""
    model = nn.Sequential(_Wrapper(4, 8))
    inv = ModelInventory.build(model)
    rec = inv.find_primary_weight("0")  # the wrapper module
    assert rec is not None
    assert rec.name == "0.linear.weight"
    assert rec.shape == (8, 4)


def test_find_primary_weight_returns_none_when_ambiguous():
    """Wrapper with two Linear children — caller should WARN and skip."""
    model = nn.Sequential(_DoubleLinearWrapper())
    inv = ModelInventory.build(model)
    rec = inv.find_primary_weight("0")
    assert rec is None


def test_find_primary_weight_returns_none_when_no_2d_param():
    """Module with only 1-D parameters (e.g. LayerNorm-only) is not weight-
    diagnostic-able. Caller will skip."""
    model = nn.Sequential(nn.LayerNorm(8))
    inv = ModelInventory.build(model)
    rec = inv.find_primary_weight("0")
    assert rec is None  # LayerNorm.weight is 1-D


def test_with_prefix_subsets_inventory():
    """Modality scoping: filter parameters to a subtree by name prefix."""

    class Two(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lang = nn.Linear(4, 8)
            self.vision = nn.Linear(4, 8)

    inv = ModelInventory.build(Two())
    lang = inv.with_prefix("lang")
    names = {r.name for r in lang.parameters}
    assert names == {"lang.weight", "lang.bias"}


def test_match_pattern_returns_subset():
    model = nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 4))
    inv = ModelInventory.build(model)
    hits = inv.match_pattern(r"^\d+\.weight$")
    names = {r.name for r in hits}
    assert names == {"0.weight", "1.weight"}


def test_tied_weights_list_both_names():
    """When two Parameters share the same Tensor (e.g.
    ``tie_word_embeddings=True``), the inventory must list both names so a
    recipe hooking either side resolves correctly."""

    class TiedModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = nn.Embedding(10, 8)
            self.lm_head = nn.Linear(8, 10, bias=False)
            self.lm_head.weight = self.embed.weight  # tie

    inv = ModelInventory.build(TiedModel())
    names = {r.name for r in inv.parameters}
    assert "embed.weight" in names and "lm_head.weight" in names


def test_to_json_is_valid_and_round_trippable_in_structure():
    model = nn.Linear(4, 8)
    inv = ModelInventory.build(model)
    parsed = json.loads(inv.to_json())
    assert isinstance(parsed, list) and len(parsed) == 2
    by_name = {r["name"]: r for r in parsed}
    assert by_name["weight"]["shape"] == [8, 4]
    assert by_name["weight"]["leaf_attr"] == "weight"
    assert by_name["weight"]["owning_module_class"] == "Linear"
