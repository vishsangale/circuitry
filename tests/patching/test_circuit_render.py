"""Tests for to_markdown(), to_json(), from_json() on circuit result types."""
from __future__ import annotations

import json

import pytest

from circuitry.patching.acdc import ACDCResult
from circuitry.patching.atp import AtPNode, AtPResult
from circuitry.patching.eap import EAPResult
from circuitry.patching.graph import (
    Edge,
    Node,
    _node_from_dict,
    _node_str,
    _node_to_dict,
    build_graph,
)


# ---------------------------------------------------------------------------
# _node_str
# ---------------------------------------------------------------------------

def test_node_str_embed():
    assert _node_str(Node("embed")) == "embed"


def test_node_str_logits():
    assert _node_str(Node("logits")) == "logits"


def test_node_str_attn_head():
    assert _node_str(Node("attn_head", layer=2, head=5)) == "L2H5"


def test_node_str_mlp():
    assert _node_str(Node("mlp", layer=3)) == "mlp.L3"


def test_node_str_mlp_with_component():
    assert _node_str(Node("mlp", layer=1, component="mlp_out")) == "mlp.L1.mlp_out"


def test_node_str_mlp_neuron():
    assert _node_str(Node("mlp_neuron", layer=0, neuron=42)) == "neuron.L0.42"


# ---------------------------------------------------------------------------
# _node_to_dict / _node_from_dict roundtrip
# ---------------------------------------------------------------------------

def test_node_roundtrip_embed():
    n = Node("embed")
    assert _node_from_dict(_node_to_dict(n)) == n


def test_node_roundtrip_attn_head():
    n = Node("attn_head", layer=1, head=3)
    assert _node_from_dict(_node_to_dict(n)) == n


def test_node_roundtrip_mlp_with_component():
    n = Node("mlp", layer=2, component="attn_out")
    assert _node_from_dict(_node_to_dict(n)) == n


# ---------------------------------------------------------------------------
# EAPResult.to_markdown
# ---------------------------------------------------------------------------

def _make_eap(n_layers: int = 2, n_heads: int = 2) -> EAPResult:
    graph = build_graph(n_layers, n_heads)
    scores = {e: float(i) * 0.1 for i, e in enumerate(graph.edges)}
    return EAPResult(graph=graph, scores=scores)


def test_eap_to_markdown_has_header():
    md = _make_eap().to_markdown()
    assert "## EAP Circuit" in md


def test_eap_to_markdown_shows_layer_count():
    md = _make_eap(n_layers=3, n_heads=4).to_markdown()
    assert "3 layers" in md
    assert "4 heads" in md


def test_eap_to_markdown_table_rows():
    md = _make_eap().to_markdown()
    assert "| rank |" in md
    assert "| writer |" in md or "writer" in md


def test_eap_to_markdown_top_k_limits_rows():
    eap = _make_eap(n_layers=6, n_heads=4)
    md = eap.to_markdown(top_k=5)
    # Table has header + separator + 5 data rows → at most 7 pipe-starting lines with |
    data_rows = [l for l in md.splitlines() if l.startswith("| ") and "---" not in l and "rank" not in l]
    assert len(data_rows) <= 5


def test_eap_to_markdown_node_labels_present():
    eap = _make_eap()
    md = eap.to_markdown(top_k=3)
    # embed and mlp nodes should appear
    assert "embed" in md or "L0" in md or "mlp" in md


# ---------------------------------------------------------------------------
# EAPResult.to_json / from_json roundtrip
# ---------------------------------------------------------------------------

def test_eap_json_roundtrip_kind():
    eap = _make_eap()
    data = json.loads(eap.to_json())
    assert data["kind"] == "eap"


def test_eap_json_roundtrip_scores_count():
    eap = _make_eap(n_layers=2, n_heads=2)
    data = json.loads(eap.to_json())
    assert len(data["scores"]) == len(eap.scores)


def test_eap_json_roundtrip_values():
    eap = _make_eap()
    eap2 = EAPResult.from_json(eap.to_json())
    # All scores present and values match within floating-point tolerance
    assert len(eap2.scores) == len(eap.scores)
    for edge, score in eap.scores.items():
        matched = any(
            abs(v - score) < 1e-9 and e.writer == edge.writer and e.reader == edge.reader
            for e, v in eap2.scores.items()
        )
        assert matched, f"score for {edge} not found in roundtripped result"


def test_eap_json_graph_metadata():
    eap = _make_eap(n_layers=3, n_heads=4)
    data = json.loads(eap.to_json())
    assert data["n_layers"] == 3
    assert data["n_heads"] == 4


# ---------------------------------------------------------------------------
# AtPResult.to_markdown
# ---------------------------------------------------------------------------

def _make_atp(n_layers: int = 2, n_heads: int = 2) -> AtPResult:
    nodes = [
        AtPNode(Node("embed"), None),
        AtPNode(Node("attn_head", 0, 0), "q"),
        AtPNode(Node("attn_head", 0, 1), "v"),
        AtPNode(Node("mlp", 0), None),
    ]
    scores = {n: float(i) * 0.5 for i, n in enumerate(nodes)}
    return AtPResult(scores=scores)


def test_atp_to_markdown_has_header():
    md = _make_atp().to_markdown()
    assert "## AtP Node Attribution" in md


def test_atp_to_markdown_shows_total_nodes():
    atp = _make_atp()
    md = atp.to_markdown()
    assert f"{len(atp.scores)}" in md


def test_atp_to_markdown_table_rows():
    md = _make_atp().to_markdown()
    assert "| rank |" in md
    assert "slot" in md


def test_atp_to_markdown_slot_label():
    md = _make_atp().to_markdown(top_k=4)
    assert "q" in md or "v" in md  # attention slots appear
    assert "—" in md               # no-slot nodes show dash


# ---------------------------------------------------------------------------
# ACDCResult.to_markdown
# ---------------------------------------------------------------------------

def _make_acdc(n_layers: int = 2, n_heads: int = 2, keep_n: int = 3) -> ACDCResult:
    graph = build_graph(n_layers, n_heads)
    kept = graph.edges[:keep_n]
    removed = graph.edges[keep_n:]
    return ACDCResult(kept_edges=kept, removed_edges=removed, final_kl=0.042, graph=graph)


def test_acdc_to_markdown_has_header():
    md = _make_acdc().to_markdown()
    assert "## ACDC Circuit" in md


def test_acdc_to_markdown_shows_kept_count():
    acdc = _make_acdc(keep_n=3)
    md = acdc.to_markdown()
    assert "3" in md


def test_acdc_to_markdown_shows_final_kl():
    md = _make_acdc().to_markdown()
    assert "0.042" in md


def test_acdc_to_markdown_top_k_elision():
    acdc = _make_acdc(n_layers=4, n_heads=2, keep_n=20)
    md = acdc.to_markdown(top_k=5)
    assert "more" in md


def test_acdc_to_markdown_no_elision_when_top_k_none():
    acdc = _make_acdc(n_layers=2, n_heads=2, keep_n=3)
    md = acdc.to_markdown()
    assert "more" not in md


# ---------------------------------------------------------------------------
# ACDCResult.to_json / from_json roundtrip
# ---------------------------------------------------------------------------

def test_acdc_json_roundtrip_kind():
    data = json.loads(_make_acdc().to_json())
    assert data["kind"] == "acdc"


def test_acdc_json_roundtrip_kept_count():
    acdc = _make_acdc(keep_n=3)
    data = json.loads(acdc.to_json())
    assert len(data["kept_edges"]) == 3


def test_acdc_json_roundtrip_final_kl():
    acdc = _make_acdc()
    acdc2 = ACDCResult.from_json(acdc.to_json())
    assert abs(acdc2.final_kl - acdc.final_kl) < 1e-9


def test_acdc_json_roundtrip_kept_edges_match():
    acdc = _make_acdc(n_layers=3, n_heads=2, keep_n=5)
    acdc2 = ACDCResult.from_json(acdc.to_json())
    kept_orig = {(e.writer, e.reader, e.slot) for e in acdc.kept_edges}
    kept_rt = {(e.writer, e.reader, e.slot) for e in acdc2.kept_edges}
    assert kept_orig == kept_rt


def test_acdc_json_removed_edges_reconstructed():
    acdc = _make_acdc(n_layers=2, n_heads=2, keep_n=3)
    acdc2 = ACDCResult.from_json(acdc.to_json())
    total_orig = len(acdc.kept_edges) + len(acdc.removed_edges)
    total_rt = len(acdc2.kept_edges) + len(acdc2.removed_edges)
    assert total_orig == total_rt


# ---------------------------------------------------------------------------
# EAPResult.save / load
# ---------------------------------------------------------------------------

def test_eap_save_and_load_roundtrip(tmp_path):
    eap = _make_eap(n_layers=2, n_heads=2)
    path = tmp_path / "circuit.json"
    eap.save(path)
    assert path.exists()
    eap2 = EAPResult.load(path)
    assert len(eap2.scores) == len(eap.scores)


def test_eap_save_produces_valid_json(tmp_path):
    eap = _make_eap()
    path = tmp_path / "circuit.json"
    eap.save(path)
    data = json.loads(path.read_text())
    assert data["kind"] == "eap"


# ---------------------------------------------------------------------------
# ACDCResult.save / load
# ---------------------------------------------------------------------------

def test_acdc_save_and_load_roundtrip(tmp_path):
    acdc = _make_acdc(n_layers=2, n_heads=2, keep_n=3)
    path = tmp_path / "acdc.json"
    acdc.save(path)
    assert path.exists()
    acdc2 = ACDCResult.load(path)
    assert acdc2.n_kept() == acdc.n_kept()
    assert abs(acdc2.final_kl - acdc.final_kl) < 1e-9


def test_acdc_save_produces_valid_json(tmp_path):
    acdc = _make_acdc()
    path = tmp_path / "acdc.json"
    acdc.save(path)
    data = json.loads(path.read_text())
    assert data["kind"] == "acdc"
    assert "kept_edges" in data
