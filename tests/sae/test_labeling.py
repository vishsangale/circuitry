"""Tests for sae/labeling.py — pluggable auto-interp labeling (v1.42)."""
from __future__ import annotations

from circuitry.sae.labeling import FeatureEvidence, describe_features


def test_to_prompt_includes_all_evidence():
    ev = FeatureEvidence(
        layer=3,
        feature=42,
        top_tokens=("Paris", "London"),
        top_logit_tokens=("capital",),
        activation_stats={"max": 3.25, "freq": 0.01},
        notes="fires on city names",
    )
    prompt = ev.to_prompt()
    assert "layer 3, index 42" in prompt
    assert "'Paris'" in prompt and "'London'" in prompt
    assert "'capital'" in prompt
    assert "freq=0.01" in prompt and "max=3.25" in prompt
    assert "fires on city names" in prompt
    assert prompt.endswith("Label:")


def test_to_prompt_omits_absent_evidence():
    prompt = FeatureEvidence(layer=0, feature=1).to_prompt()
    assert "activating tokens" not in prompt
    assert "promoted" not in prompt
    assert "Activation stats" not in prompt
    assert "Notes" not in prompt


def test_describe_features_keys_and_values():
    evidence = [
        FeatureEvidence(layer=0, feature=3, top_tokens=("cat",)),
        FeatureEvidence(layer=1, feature=7, top_tokens=("dog",)),
    ]
    labels = describe_features(
        evidence, lambda prompt: "animal: cat" if "'cat'" in prompt else "animal: dog",
    )
    assert labels == {(0, 3): "animal: cat", (1, 7): "animal: dog"}


def test_describe_features_strips_and_drops_empty():
    evidence = [
        FeatureEvidence(layer=0, feature=0),
        FeatureEvidence(layer=0, feature=1),
    ]
    labels = describe_features(
        evidence,
        lambda prompt: "  padded  " if "index 0" in prompt else "   ",
    )
    assert labels == {(0, 0): "padded"}


def test_labels_plug_into_export():
    from circuitry.patching.clt import CLTEdge, CLTGraphResult, CLTNode
    from circuitry.patching.export import to_neuronpedia_graph

    src, dst = CLTNode(layer=0, feature=3), CLTNode(layer=1, feature=5)
    result = CLTGraphResult(
        scores={CLTEdge(src, dst): 1.0},
        node_scores={src: 1.0, dst: 0.5},
        n_layers=2,
        n_features=[8, 8],
        layer_order=[0, 1],
    )
    labels = describe_features(
        [FeatureEvidence(layer=0, feature=3, top_tokens=("x",))],
        lambda _prompt: "my label",
    )
    g = to_neuronpedia_graph(result, slug="s", scan="m", labels=labels)
    by_id = {n["node_id"]: n for n in g["nodes"]}
    assert by_id["0_3_0"]["clerp"] == "my label"


def test_exports():
    import circuitry
    from circuitry import sae

    assert circuitry.describe_features is describe_features
    assert sae.FeatureEvidence is FeatureEvidence
    assert "FeatureEvidence" in circuitry.__all__
