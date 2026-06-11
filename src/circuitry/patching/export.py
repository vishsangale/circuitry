"""Graph export — circuit-tracer / Neuronpedia JSON and self-contained HTML.

Serializes circuitry graph results to interchange formats so they can be
explored in the existing visualization ecosystem instead of terminating at
``to_markdown()``:

- :func:`to_neuronpedia_graph` / :func:`save_neuronpedia_graph` — the
  circuit-tracer attribution-graph JSON schema (schema_version 1;
  github.com/decoderesearch/circuit-tracer, ``frontend/graph_models.py``),
  uploadable to neuronpedia.org for interactive exploration.
- :func:`to_html` / :func:`save_html` — a dependency-free single-file HTML
  report (inline SVG, vanilla JS; no CDN fetches).

Supported results: ``CLTGraphResult``, ``EAPResult``, and
``SAEFeatureCircuit`` (duck-typed to avoid importing sae_lens-adjacent
modules at import time).

Pure serialization: no forward passes, no tensor math, stdlib ``json`` only.
"""
from __future__ import annotations

import html as _html
import json
import os
from dataclasses import dataclass
from typing import Any

from circuitry.patching.clt import CLTGraphResult
from circuitry.patching.eap import EAPResult
from circuitry.patching.graph import Node, _node_str

__all__ = [
    "to_neuronpedia_graph",
    "save_neuronpedia_graph",
    "to_html",
    "save_html",
]

# Labels keyed by (layer, feature) — layer as it appears in the source node
# (int for CLT/SAE feature nodes).  Applied to the ``clerp`` field.
Labels = dict[tuple[Any, int], str]


@dataclass(frozen=True)
class _GNode:
    """Normalized graph node (intermediate form shared by both exporters)."""

    node_id: str
    layer: str          # "E" for embedding, str(int) for blocks, str(n) for logits
    feature: int
    ctx_idx: int
    feature_type: str   # "feature" | "error" | "embedding" | "logit"
    label: str
    influence: float | None = None
    activation: float | None = None


_GLink = tuple[str, str, float]  # (source node_id, target node_id, weight)


# ---------------------------------------------------------------------------
# Normalization: result object -> (nodes, links)
# ---------------------------------------------------------------------------


def _select_edges(
    scores: dict[Any, float], *, top_k: int | None, node_threshold: float | None
) -> list[tuple[Any, float]]:
    ranked = sorted(scores.items(), key=lambda kv: abs(kv[1]), reverse=True)
    if node_threshold is not None:
        ranked = [(e, s) for e, s in ranked if abs(s) >= node_threshold]
    if top_k is not None:
        ranked = ranked[:top_k]
    return ranked


def _label_for(labels: Labels | None, layer: Any, feature: int, default: str) -> str:
    if labels is not None and (layer, feature) in labels:
        return labels[(layer, feature)]
    return default


def _normalize_clt(
    result: CLTGraphResult,
    *,
    top_k: int | None,
    node_threshold: float | None,
    labels: Labels | None,
) -> tuple[list[_GNode], list[_GLink]]:
    kept = _select_edges(result.scores, top_k=top_k, node_threshold=node_threshold)
    nodes: dict[str, _GNode] = {}
    links: list[_GLink] = []
    for edge, score in kept:
        ids = []
        for cn in (edge.src, edge.dst):
            nid = f"{cn.layer}_{cn.feature}_0"
            if nid not in nodes:
                nodes[nid] = _GNode(
                    node_id=nid,
                    layer=str(cn.layer),
                    feature=cn.feature,
                    ctx_idx=0,
                    feature_type="feature",
                    label=_label_for(labels, cn.layer, cn.feature, f"L{cn.layer}/f{cn.feature}"),
                    influence=result.node_scores.get(cn),
                )
            ids.append(nid)
        links.append((ids[0], ids[1], float(score)))
    return list(nodes.values()), links


def _eap_node_identity(node: Node, n_layers: int, n_heads: int) -> tuple[str, str, int, str]:
    """(node_id, layer, feature, feature_type) for an EAP graph Node."""
    if node.kind == "embed":
        return ("E_0_0", "E", 0, "embedding")
    if node.kind == "logits":
        return (f"{n_layers}_0_0", str(n_layers), 0, "logit")
    if node.kind == "attn_head":
        return (f"{node.layer}_{node.head}_0", str(node.layer), int(node.head), "feature")
    if node.kind == "mlp":
        # Synthetic feature index one past the last head — unique within the layer.
        return (f"{node.layer}_{n_heads}_0", str(node.layer), n_heads, "feature")
    if node.kind == "mlp_neuron":
        return (
            f"{node.layer}_n{node.neuron}_0",
            str(node.layer),
            int(node.neuron),
            "feature",
        )
    raise ValueError(f"unsupported EAP node kind {node.kind!r}")


def _normalize_eap(
    result: EAPResult,
    *,
    top_k: int | None,
    node_threshold: float | None,
    labels: Labels | None,
) -> tuple[list[_GNode], list[_GLink]]:
    n_layers = result.graph.n_layers
    n_heads = result.graph.n_heads
    kept = _select_edges(result.scores, top_k=top_k, node_threshold=node_threshold)
    nodes: dict[str, _GNode] = {}
    links: list[_GLink] = []
    for edge, score in kept:
        ids = []
        for gn in (edge.writer, edge.reader):
            nid, layer, feature, ftype = _eap_node_identity(gn, n_layers, n_heads)
            if nid not in nodes:
                nodes[nid] = _GNode(
                    node_id=nid,
                    layer=layer,
                    feature=feature,
                    ctx_idx=0,
                    feature_type=ftype,
                    label=_label_for(labels, gn.layer, feature, _node_str(gn)),
                )
            ids.append(nid)
        links.append((ids[0], ids[1], float(score)))
    return list(nodes.values()), links


def _normalize_sae_circuit(
    result: Any,  # SAEFeatureCircuit (duck-typed)
    *,
    top_k: int | None,
    node_threshold: float | None,
    labels: Labels | None,
) -> tuple[list[_GNode], list[_GLink]]:
    kept = _select_edges(result.edges, top_k=top_k, node_threshold=node_threshold)
    influence: dict[Any, float] = {}
    node_result = getattr(result, "nodes", None)
    if node_result is not None and hasattr(node_result, "scores"):
        influence = {ap.node: s for ap, s in node_result.scores.items()}
    nodes: dict[str, _GNode] = {}
    links: list[_GLink] = []
    for edge, score in kept:
        ids = []
        for ap in (edge.writer, edge.reader):
            gn = ap.node
            layer = gn.layer if gn.layer is not None else 0
            feature = int(gn.neuron) if gn.neuron is not None else 0
            if gn.kind == "sae_error":
                ftype = "error"
                nid = f"{layer}_e{feature}_0"
                default = f"L{layer}/error"
            else:
                ftype = "feature"
                nid = f"{layer}_{feature}_0"
                default = f"L{layer}/f{feature}"
            if gn.component:
                nid = f"{nid}_{gn.component}"
                default = f"{default}@{gn.component}"
            if nid not in nodes:
                nodes[nid] = _GNode(
                    node_id=nid,
                    layer=str(layer),
                    feature=feature,
                    ctx_idx=0,
                    feature_type=ftype,
                    label=_label_for(labels, layer, feature, default),
                    influence=influence.get(gn),
                )
            ids.append(nid)
        links.append((ids[0], ids[1], float(score)))
    return list(nodes.values()), links


def _normalize(
    result: Any,
    *,
    top_k: int | None = None,
    node_threshold: float | None = None,
    labels: Labels | None = None,
) -> tuple[list[_GNode], list[_GLink]]:
    if isinstance(result, CLTGraphResult):
        return _normalize_clt(result, top_k=top_k, node_threshold=node_threshold, labels=labels)
    if isinstance(result, EAPResult):
        return _normalize_eap(result, top_k=top_k, node_threshold=node_threshold, labels=labels)
    if type(result).__name__ == "SAEFeatureCircuit":
        return _normalize_sae_circuit(
            result, top_k=top_k, node_threshold=node_threshold, labels=labels
        )
    raise TypeError(
        "unsupported result type for graph export: "
        f"{type(result).__name__} (expected CLTGraphResult, EAPResult, or SAEFeatureCircuit)"
    )


# ---------------------------------------------------------------------------
# Neuronpedia / circuit-tracer JSON
# ---------------------------------------------------------------------------


def to_neuronpedia_graph(
    result: Any,
    *,
    slug: str,
    scan: str,
    prompt: str = "",
    prompt_tokens: list[str] | None = None,
    node_threshold: float | None = None,
    top_k: int | None = None,
    labels: Labels | None = None,
) -> dict:
    """Serialize a graph result to the circuit-tracer / Neuronpedia JSON schema.

    Args:
        result: ``CLTGraphResult``, ``EAPResult``, or ``SAEFeatureCircuit``.
        slug: URL slug identifying the graph on Neuronpedia.
        scan: model identifier (Neuronpedia "scan" id, e.g. ``"gemma-2-2b"``).
        prompt: the prompt the graph was computed on (display only).
        prompt_tokens: tokenized prompt strings (display only).
        node_threshold: keep only edges with ``|score| >= node_threshold``.
        top_k: keep only the top-k edges by ``|score|`` (applied after
            *node_threshold*).
        labels: optional ``{(layer, feature): str}`` human labels written to
            each feature node's ``clerp`` field.

    Returns:
        A JSON-serializable dict with ``metadata`` / ``qParams`` / ``nodes`` /
        ``links`` keys (circuit-tracer ``schema_version`` 1).
    """
    nodes, links = _normalize(
        result, top_k=top_k, node_threshold=node_threshold, labels=labels
    )
    return {
        "metadata": {
            "slug": slug,
            "scan": scan,
            "transcoder_list": [],
            "prompt_tokens": list(prompt_tokens or []),
            "prompt": prompt,
            "node_threshold": node_threshold,
            "schema_version": 1,
        },
        "qParams": {
            "pinnedIds": [],
            "supernodes": [],
            "linkType": "both",
            "clickedId": "",
            "sg_pos": "",
        },
        "nodes": [
            {
                "node_id": n.node_id,
                "feature": n.feature,
                "layer": n.layer,
                "ctx_idx": n.ctx_idx,
                "feature_type": n.feature_type,
                "token_prob": 0.0,
                "is_target_logit": False,
                "run_idx": 0,
                "reverse_ctx_idx": 0,
                "jsNodeId": f"{n.layer}_{n.feature}-0",
                "clerp": n.label,
                "influence": n.influence,
                "activation": n.activation,
            }
            for n in nodes
        ],
        "links": [
            {"source": s, "target": t, "weight": w} for s, t, w in links
        ],
    }


def save_neuronpedia_graph(result: Any, path: str, **kwargs: Any) -> str:
    """:func:`to_neuronpedia_graph` + ``json.dump``. Returns the absolute path."""
    graph = to_neuronpedia_graph(result, **kwargs)
    path = os.path.abspath(path)
    with open(path, "w") as f:
        json.dump(graph, f, indent=2)
    return path


# ---------------------------------------------------------------------------
# Self-contained HTML
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 1.5rem; }}
  svg {{ border: 1px solid #ddd; background: #fff; }}
  .edge {{ stroke-linecap: round; }}
  .edge.dim {{ opacity: 0.06; }}
  .node circle {{ fill: #4a7dbd; stroke: #234; stroke-width: 1; cursor: pointer; }}
  .node.error circle {{ fill: #c46a4a; }}
  .node.embedding circle, .node.logit circle {{ fill: #6aa56a; }}
  .node text {{ font-size: 11px; fill: #222; }}
  #info {{ margin-top: 0.75rem; color: #444; font-size: 13px; min-height: 1.2em; }}
</style>
</head>
<body>
<h2>{title}</h2>
<p>{n_nodes} nodes, {n_links} edges. Blue edges: positive score; red: negative.
Hover a node to isolate its edges; click empty space to reset.</p>
{svg}
<div id="info"></div>
<script type="application/json" id="graph-data">{data_json}</script>
<script>
(function () {{
  var info = document.getElementById("info");
  var edges = Array.prototype.slice.call(document.querySelectorAll(".edge"));
  function reset() {{
    edges.forEach(function (e) {{ e.classList.remove("dim"); }});
    info.textContent = "";
  }}
  Array.prototype.forEach.call(document.querySelectorAll(".node"), function (g) {{
    g.addEventListener("mouseenter", function () {{
      var id = g.getAttribute("data-id");
      edges.forEach(function (e) {{
        var hit = e.getAttribute("data-src") === id || e.getAttribute("data-dst") === id;
        e.classList.toggle("dim", !hit);
      }});
      info.textContent = g.getAttribute("data-label");
    }});
    g.addEventListener("mouseleave", reset);
  }});
  document.querySelector("svg").addEventListener("click", reset);
}})();
</script>
</body>
</html>
"""


def _layer_sort_key(layer: str) -> tuple[int, int]:
    if layer == "E":
        return (0, 0)
    return (1, int(layer))


def _layout(
    nodes: list[_GNode],
) -> tuple[dict[str, tuple[float, float]], int, int]:
    """Deterministic layered layout: x by layer column, y by index in column."""
    columns: dict[str, list[_GNode]] = {}
    for n in nodes:
        columns.setdefault(n.layer, []).append(n)
    ordered_layers = sorted(columns, key=_layer_sort_key)
    col_w, row_h, margin = 180, 46, 60
    pos: dict[str, tuple[float, float]] = {}
    max_rows = 1
    for ci, layer in enumerate(ordered_layers):
        col = sorted(columns[layer], key=lambda n: (n.feature_type, n.feature))
        max_rows = max(max_rows, len(col))
        for ri, n in enumerate(col):
            pos[n.node_id] = (margin + ci * col_w, margin + ri * row_h)
    width = 2 * margin + (len(ordered_layers) - 1) * col_w + 120
    height = 2 * margin + (max_rows - 1) * row_h
    return pos, width, height


def to_html(
    result: Any,
    *,
    title: str | None = None,
    top_k: int | None = 50,
    node_threshold: float | None = None,
    labels: Labels | None = None,
) -> str:
    """Render a graph result as a self-contained HTML document.

    Single file, zero runtime dependencies, no external URL fetches: inline
    SVG (layered left-to-right DAG) plus a small vanilla-JS hover handler.
    The normalized graph is embedded as ``<script type="application/json">``
    for downstream tooling.

    *top_k* defaults to 50 — full edge sets are typically too dense to read;
    pass ``top_k=None`` to render everything.
    """
    nodes, links = _normalize(
        result, top_k=top_k, node_threshold=node_threshold, labels=labels
    )
    pos, width, height = _layout(nodes)
    max_w = max((abs(w) for _, _, w in links), default=1.0) or 1.0

    parts: list[str] = [
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
    ]
    for src, dst, w in links:
        (x1, y1), (x2, y2) = pos[src], pos[dst]
        color = "#3a6ea5" if w >= 0 else "#b03a3a"
        sw = 0.5 + 3.0 * abs(w) / max_w
        opacity = 0.25 + 0.75 * abs(w) / max_w
        parts.append(
            f'<line class="edge" data-src="{_html.escape(src)}" data-dst="{_html.escape(dst)}"'
            f' x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}"'
            f' stroke="{color}" stroke-width="{sw:.2f}" opacity="{opacity:.2f}">'
            f"<title>{_html.escape(src)} → {_html.escape(dst)}: {w:.4g}</title></line>"
        )
    for n in nodes:
        x, y = pos[n.node_id]
        detail = n.label
        if n.influence is not None:
            detail += f" (influence {n.influence:.4g})"
        parts.append(
            f'<g class="node {n.feature_type}" data-id="{_html.escape(n.node_id)}"'
            f' data-label="{_html.escape(detail)}">'
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="7"><title>{_html.escape(detail)}</title></circle>'
            f'<text x="{x + 10:.1f}" y="{y + 4:.1f}">{_html.escape(n.label)}</text></g>'
        )
    parts.append("</svg>")

    # "<" is escaped so a hostile label (e.g. "</script>") cannot terminate
    # the embedding <script> block.
    data_json = json.dumps(
        {
            "nodes": [
                {
                    "node_id": n.node_id,
                    "layer": n.layer,
                    "feature": n.feature,
                    "feature_type": n.feature_type,
                    "clerp": n.label,
                    "influence": n.influence,
                }
                for n in nodes
            ],
            "links": [{"source": s, "target": t, "weight": w} for s, t, w in links],
        }
    ).replace("<", "\\u003c")
    return _HTML_TEMPLATE.format(
        title=_html.escape(title or "circuitry attribution graph"),
        n_nodes=len(nodes),
        n_links=len(links),
        svg="\n".join(parts),
        data_json=data_json,
    )


def save_html(result: Any, path: str, **kwargs: Any) -> str:
    """:func:`to_html` + write. Returns the absolute path."""
    doc = to_html(result, **kwargs)
    path = os.path.abspath(path)
    with open(path, "w") as f:
        f.write(doc)
    return path
