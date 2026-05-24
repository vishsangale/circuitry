"""Tests for the EAP edge graph. Spec §2."""
from __future__ import annotations

import torch

from circuitry.patching.graph import Edge, EdgeGraph, Node, build_graph


def test_nodes_enumerated() -> None:
    g: EdgeGraph = build_graph(n_layers=2, n_heads=2)
    kinds = {(n.kind, n.layer, n.head) for n in g.writers}
    # writers: embed + 2 heads*2 layers + 1 mlp*2 layers = 1 + 4 + 2 = 7
    assert ("embed", None, None) in kinds
    assert ("attn_head", 0, 0) in kinds
    assert ("attn_head", 1, 1) in kinds
    assert ("mlp", 0, None) in kinds
    assert len(g.writers) == 7
    # logits is NOT a writer
    assert all(n.kind != "logits" for n in g.writers)


def test_readers_have_slots():
    g = build_graph(n_layers=2, n_heads=2)
    slots = {(node.kind, node.layer, node.head, slot) for node, slot in g.readers}
    assert ("attn_head", 0, 0, "q") in slots
    assert ("attn_head", 0, 0, "k") in slots
    assert ("attn_head", 0, 0, "v") in slots
    assert ("mlp", 0, None, "mlp_in") in slots
    assert ("logits", None, None, "logits_in") in slots
    # readers: 2 heads*2 layers*3 slots + 2 mlp*1 + 1 logits = 12 + 2 + 1 = 15
    assert len(g.readers) == 15
    # embed is NOT a reader
    assert all(node.kind != "embed" for node, _ in g.readers)


def test_causal_validity():
    g = build_graph(n_layers=2, n_heads=2)
    edge_set = {(e.writer, e.reader, e.slot) for e in g.edges}
    embed = Node("embed")
    h00 = Node("attn_head", 0, 0)
    h01 = Node("attn_head", 0, 1)
    mlp0 = Node("mlp", 0)
    h10 = Node("attn_head", 1, 0)
    logits = Node("logits")
    # embed feeds everything downstream
    assert (embed, h00, "q") in edge_set
    assert (embed, logits, "logits_in") in edge_set
    # same-layer attn head -> attn head does NOT exist (parallel)
    assert (h00, h01, "q") not in edge_set
    # same-layer attn -> mlp DOES exist
    assert (h00, mlp0, "mlp_in") in edge_set
    # mlp0 -> downstream layer-1 head exists
    assert (mlp0, h10, "q") in edge_set
    # downstream -> upstream does NOT exist
    assert (h10, h00, "q") not in edge_set
    # logits is never a writer
    assert all(e.writer != logits for e in g.edges)


def test_valid_mask_matches_edges():
    g = build_graph(n_layers=2, n_heads=2)
    mask = g.valid_mask()
    assert mask.shape == (len(g.writers), len(g.readers))
    assert mask.dtype == torch.bool
    # number of True entries == number of edges
    assert int(mask.sum().item()) == len(g.edges)
    # each edge maps to a True cell
    for e in g.edges:
        w = g.writer_index(e.writer)
        r = g.reader_index(e.reader, e.slot)
        assert mask[w, r]


def test_node_edge_frozen_hashable():
    n1, n2 = Node("attn_head", 0, 1), Node("attn_head", 0, 1)
    assert n1 == n2 and hash(n1) == hash(n2)
    e1 = Edge(Node("embed"), Node("mlp", 0), "mlp_in")
    assert e1 in {e1}
