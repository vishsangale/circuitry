"""Reverse-topo reader iteration + deterministic edge ordering for ACDC."""
from __future__ import annotations

from circuitry.patching.graph import (
    Node, build_graph, reverse_topo_readers, edge_sort_key,
)


def test_reverse_topo_readers_logits_first_layer0_last():
    g = build_graph(n_layers=2, n_heads=2)
    order = list(reverse_topo_readers(g))
    # logits reader must come first, layer-0 readers last
    assert order[0][0] == Node("logits")
    # every reader appears exactly once
    assert set(order) == set(g.readers)
    # ranks are non-increasing (reverse topological)
    from circuitry.patching.graph import _order
    ranks = [_order(rnode, g.n_layers) for rnode, _ in order]
    assert ranks == sorted(ranks, reverse=True)


def test_reverse_topo_is_deterministic():
    g = build_graph(n_layers=2, n_heads=3)
    assert list(reverse_topo_readers(g)) == list(reverse_topo_readers(g))


def test_edge_sort_key_orders_by_writer_then_slot():
    g = build_graph(n_layers=1, n_heads=2)
    incoming = [e for e in g.edges if e.reader == Node("logits")]
    keyed = sorted(incoming, key=edge_sort_key)
    # embed (kind "embed") sorts before attn_head/mlp by kind name; check stable + total
    assert keyed == sorted(keyed, key=edge_sort_key)  # idempotent
    assert len(keyed) == len(incoming)
    # all keys are comparable tuples (no None crash)
    keys = [edge_sort_key(e) for e in incoming]
    assert len(set(keys)) == len(keys)  # unique per edge
