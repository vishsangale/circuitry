"""Tests for AtP node enumeration + result container. Spec §2."""
from __future__ import annotations

from circuitry.patching.atp import AtPNode, AtPResult, enumerate_nodes
from circuitry.patching.graph import Node


def test_enumerate_without_neurons():
    nodes = enumerate_nodes(n_layers=2, n_heads=2)
    kinds = {(n.node.kind, n.node.layer, n.node.head, n.slot) for n in nodes}
    assert (AtPNode(Node("embed"), None)) in nodes
    assert ("attn_head", 0, 0, "q") in kinds
    assert ("attn_head", 0, 0, "k") in kinds
    assert ("attn_head", 0, 0, "v") in kinds
    assert ("mlp", 0, None, None) in kinds
    assert all(n.node.kind != "mlp_neuron" for n in nodes)
    # embed(1) + layer0(2 heads*3 slots + 1 mlp = 7) + layer1(7) = 15
    assert len(nodes) == 15


def test_enumerate_with_neurons():
    nodes = enumerate_nodes(n_layers=1, n_heads=1, d_mlp=4)
    neuron_nodes = [n for n in nodes if n.node.kind == "mlp_neuron"]
    assert len(neuron_nodes) == 4
    assert {n.node.neuron for n in neuron_nodes} == {0, 1, 2, 3}


def test_atpnode_frozen_hashable():
    a = AtPNode(Node("attn_head", 0, 1), "q")
    b = AtPNode(Node("attn_head", 0, 1), "q")
    assert a == b and hash(a) == hash(b)
    assert len({a, b}) == 1


def test_result_ranked_topk_threshold():
    n1 = AtPNode(Node("mlp", 0), None)
    n2 = AtPNode(Node("embed"), None)
    res = AtPResult({n1: -5.0, n2: 1.0})
    assert res.ranked()[0] == (n1, -5.0)          # by |score|
    assert res.top_k(1) == [(n1, -5.0)]
    assert set(res.threshold(2.0)) == {n1}          # |−5| ≥ 2, |1| < 2
