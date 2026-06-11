"""Tests for patching/export.py — Neuronpedia JSON + self-contained HTML (v1.41)."""
from __future__ import annotations

import json
import re

import pytest

from circuitry.patching.atp import AtPNode, AtPResult
from circuitry.patching.clt import CLTEdge, CLTGraphResult, CLTNode
from circuitry.patching.eap import EAPResult
from circuitry.patching.export import (
    save_html,
    save_neuronpedia_graph,
    to_html,
    to_neuronpedia_graph,
)
from circuitry.patching.graph import Node, build_graph
from circuitry.patching.sae_edges import (
    SAEFeatureCircuit,
    SAEFeatureEdge,
    SAEFeatureEdgeGraph,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def clt_result() -> CLTGraphResult:
    n0, n1, n2, n3 = (
        CLTNode(layer=0, feature=3),
        CLTNode(layer=0, feature=7),
        CLTNode(layer=1, feature=1),
        CLTNode(layer=1, feature=4),
    )
    return CLTGraphResult(
        scores={
            CLTEdge(n0, n2): 0.9,
            CLTEdge(n0, n3): -0.5,
            CLTEdge(n1, n2): 0.1,
            CLTEdge(n1, n3): 0.02,
        },
        node_scores={n0: 1.5, n1: 0.2, n2: 0.8, n3: -0.3},
        n_layers=2,
        n_features=[8, 8],
        layer_order=[0, 1],
    )


@pytest.fixture
def eap_result() -> EAPResult:
    graph = build_graph(n_layers=2, n_heads=2)
    scores = {e: 0.01 * (i + 1) * (-1) ** i for i, e in enumerate(graph.edges)}
    return EAPResult(graph=graph, scores=scores)


@pytest.fixture
def sae_circuit() -> SAEFeatureCircuit:
    w_feat = AtPNode(Node("sae_feature", layer=0, neuron=12), None)
    w_err = AtPNode(Node("sae_error", layer=0, neuron=0), None)
    r_feat = AtPNode(Node("sae_feature", layer=1, neuron=5), None)
    edges = {
        SAEFeatureEdge(w_feat, r_feat): 0.7,
        SAEFeatureEdge(w_err, r_feat): -0.2,
    }
    nodes = AtPResult(scores={w_feat: 0.9, w_err: 0.1, r_feat: 0.6})
    graph = SAEFeatureEdgeGraph(sites=[], survivors={}, edges=sorted(edges, key=str))
    return SAEFeatureCircuit(nodes, edges, graph)


# ---------------------------------------------------------------------------
# to_neuronpedia_graph — schema shape
# ---------------------------------------------------------------------------


def test_neuronpedia_top_level_keys(clt_result):
    g = to_neuronpedia_graph(clt_result, slug="test", scan="tiny")
    assert set(g) == {"metadata", "qParams", "nodes", "links"}


def test_neuronpedia_metadata_fields(clt_result):
    g = to_neuronpedia_graph(
        clt_result, slug="my-slug", scan="gemma-2-2b", prompt="a b",
        prompt_tokens=["a", "b"], node_threshold=0.05,
    )
    md = g["metadata"]
    assert md["slug"] == "my-slug"
    assert md["scan"] == "gemma-2-2b"
    assert md["prompt"] == "a b"
    assert md["prompt_tokens"] == ["a", "b"]
    assert md["node_threshold"] == 0.05
    assert md["schema_version"] == 1
    assert md["transcoder_list"] == []


def test_neuronpedia_qparams_fields(clt_result):
    q = to_neuronpedia_graph(clt_result, slug="s", scan="m")["qParams"]
    assert set(q) == {"pinnedIds", "supernodes", "linkType", "clickedId", "sg_pos"}
    assert q["pinnedIds"] == [] and q["linkType"] == "both"


def test_neuronpedia_node_fields(clt_result):
    g = to_neuronpedia_graph(clt_result, slug="s", scan="m")
    required = {
        "node_id", "feature", "layer", "ctx_idx", "feature_type", "token_prob",
        "is_target_logit", "run_idx", "reverse_ctx_idx", "jsNodeId", "clerp",
        "influence", "activation",
    }
    for node in g["nodes"]:
        assert required <= set(node)
        assert isinstance(node["layer"], str)
        assert isinstance(node["feature"], int)


def test_neuronpedia_link_fields(clt_result):
    g = to_neuronpedia_graph(clt_result, slug="s", scan="m")
    for link in g["links"]:
        assert set(link) == {"source", "target", "weight"}


def test_json_serializable(clt_result, eap_result, sae_circuit):
    for result in (clt_result, eap_result, sae_circuit):
        text = json.dumps(to_neuronpedia_graph(result, slug="s", scan="m"))
        assert json.loads(text)


# ---------------------------------------------------------------------------
# to_neuronpedia_graph — content per result type
# ---------------------------------------------------------------------------


def test_clt_counts(clt_result):
    g = to_neuronpedia_graph(clt_result, slug="s", scan="m")
    assert len(g["links"]) == 4
    assert len(g["nodes"]) == 4
    ids = {n["node_id"] for n in g["nodes"]}
    assert ids == {"0_3_0", "0_7_0", "1_1_0", "1_4_0"}


def test_clt_influence_from_node_scores(clt_result):
    g = to_neuronpedia_graph(clt_result, slug="s", scan="m")
    by_id = {n["node_id"]: n for n in g["nodes"]}
    assert by_id["0_3_0"]["influence"] == 1.5
    assert by_id["1_4_0"]["influence"] == -0.3


def test_clt_link_weights_match_scores(clt_result):
    g = to_neuronpedia_graph(clt_result, slug="s", scan="m")
    weights = {(l["source"], l["target"]): l["weight"] for l in g["links"]}
    assert weights[("0_3_0", "1_1_0")] == 0.9
    assert weights[("0_7_0", "1_4_0")] == 0.02


def test_eap_feature_types(eap_result):
    g = to_neuronpedia_graph(eap_result, slug="s", scan="m")
    types = {n["feature_type"] for n in g["nodes"]}
    assert types == {"embedding", "feature", "logit"}
    embed = [n for n in g["nodes"] if n["feature_type"] == "embedding"]
    assert embed[0]["layer"] == "E"
    logit = [n for n in g["nodes"] if n["feature_type"] == "logit"]
    assert logit[0]["layer"] == str(eap_result.graph.n_layers)


def test_eap_link_count_matches_scores(eap_result):
    g = to_neuronpedia_graph(eap_result, slug="s", scan="m")
    assert len(g["links"]) == len(eap_result.scores)


def test_eap_mlp_and_head_ids_distinct(eap_result):
    g = to_neuronpedia_graph(eap_result, slug="s", scan="m")
    ids = [n["node_id"] for n in g["nodes"]]
    assert len(ids) == len(set(ids))
    # heads 0..n_heads-1, mlp gets the synthetic index n_heads
    assert "0_0_0" in ids and "0_2_0" in ids  # L0H0 and L0 mlp (n_heads=2)


def test_sae_circuit_error_node(sae_circuit):
    g = to_neuronpedia_graph(sae_circuit, slug="s", scan="m")
    by_type = {}
    for n in g["nodes"]:
        by_type.setdefault(n["feature_type"], []).append(n)
    assert len(by_type["error"]) == 1
    assert len(by_type["feature"]) == 2
    assert by_type["error"][0]["node_id"].startswith("0_e")


def test_sae_circuit_influence_from_atp_nodes(sae_circuit):
    g = to_neuronpedia_graph(sae_circuit, slug="s", scan="m")
    by_id = {n["node_id"]: n for n in g["nodes"]}
    assert by_id["0_12_0"]["influence"] == 0.9
    assert by_id["1_5_0"]["influence"] == 0.6


def test_unsupported_type_raises():
    with pytest.raises(TypeError, match="unsupported result type"):
        to_neuronpedia_graph(object(), slug="s", scan="m")


# ---------------------------------------------------------------------------
# Filtering and labels
# ---------------------------------------------------------------------------


def test_top_k_filters_edges(clt_result):
    g = to_neuronpedia_graph(clt_result, slug="s", scan="m", top_k=2)
    assert len(g["links"]) == 2
    assert {abs(l["weight"]) for l in g["links"]} == {0.9, 0.5}


def test_node_threshold_filters_edges(clt_result):
    g = to_neuronpedia_graph(clt_result, slug="s", scan="m", node_threshold=0.4)
    assert len(g["links"]) == 2
    # nodes are restricted to kept-edge endpoints
    ids = {n["node_id"] for n in g["nodes"]}
    assert "0_7_0" not in ids


def test_labels_written_to_clerp(clt_result):
    g = to_neuronpedia_graph(
        clt_result, slug="s", scan="m", labels={(0, 3): "name mover"},
    )
    by_id = {n["node_id"]: n for n in g["nodes"]}
    assert by_id["0_3_0"]["clerp"] == "name mover"
    assert by_id["0_7_0"]["clerp"] == "L0/f7"  # default label preserved


def test_deterministic_output(clt_result):
    a = json.dumps(to_neuronpedia_graph(clt_result, slug="s", scan="m"))
    b = json.dumps(to_neuronpedia_graph(clt_result, slug="s", scan="m"))
    assert a == b


# ---------------------------------------------------------------------------
# save_neuronpedia_graph
# ---------------------------------------------------------------------------


def test_save_neuronpedia_graph(tmp_path, clt_result):
    out = save_neuronpedia_graph(
        clt_result, str(tmp_path / "g.json"), slug="s", scan="m",
    )
    assert out == str(tmp_path / "g.json")
    data = json.loads((tmp_path / "g.json").read_text())
    assert data["metadata"]["slug"] == "s"


# ---------------------------------------------------------------------------
# HTML export
# ---------------------------------------------------------------------------


def test_html_is_self_contained(clt_result):
    doc = to_html(clt_result)
    assert doc.startswith("<!DOCTYPE html>")
    # no external fetches: no http(s) URLs anywhere in the document
    assert not re.search(r"https?://", doc)
    assert "<svg" in doc and "</svg>" in doc


def test_html_embeds_graph_json(clt_result):
    doc = to_html(clt_result)
    m = re.search(
        r'<script type="application/json" id="graph-data">(.*?)</script>',
        doc, re.S,
    )
    assert m, "embedded graph JSON missing"
    data = json.loads(m.group(1))
    assert len(data["nodes"]) == 4 and len(data["links"]) == 4


def test_html_node_and_edge_elements(clt_result):
    doc = to_html(clt_result)
    assert doc.count('class="edge"') == 4
    assert doc.count('class="node ') == 4


def test_html_top_k_default_limits_edges():
    graph = build_graph(n_layers=3, n_heads=4)
    assert len(graph.edges) > 50
    result = EAPResult(
        graph=graph,
        scores={e: 0.01 * (i + 1) for i, e in enumerate(graph.edges)},
    )
    doc = to_html(result)  # default top_k=50
    assert doc.count('class="edge"') == 50
    full = to_html(result, top_k=None)
    assert full.count('class="edge"') == len(result.scores)


def test_html_escapes_labels(clt_result):
    doc = to_html(clt_result, labels={(0, 3): "<b>injected & raw</b>"})
    assert "<b>injected" not in doc
    assert "&lt;b&gt;injected &amp; raw&lt;/b&gt;" in doc


def test_html_title(clt_result):
    doc = to_html(clt_result, title="My Circuit")
    assert "<title>My Circuit</title>" in doc


def test_save_html(tmp_path, sae_circuit):
    out = save_html(sae_circuit, str(tmp_path / "g.html"))
    assert out == str(tmp_path / "g.html")
    assert (tmp_path / "g.html").read_text().startswith("<!DOCTYPE html>")


# ---------------------------------------------------------------------------
# Top-level exports
# ---------------------------------------------------------------------------


def test_top_level_exports():
    import circuitry

    for name in (
        "to_neuronpedia_graph", "save_neuronpedia_graph", "to_html", "save_html",
    ):
        assert hasattr(circuitry, name)
        assert name in circuitry.__all__
