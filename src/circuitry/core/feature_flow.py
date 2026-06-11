"""Cross-layer feature flow — data-free SAE feature matching across layers.

Tracks how an SAE/transcoder feature persists, transforms, or first appears
across layers by matching dictionary features between layers on decoder-row
cosine similarity (no forward passes, no data): feature ``i`` at layer A
matches the feature at layer B whose decoder direction has the highest
cosine with ``W_dec_A[i]``.  Chaining adjacent-layer matches yields a flow
graph of feature evolution; following a chain gives a feature's "lifetime",
and features with no upstream match are "born" at their layer.

Reference: Laptev et al., "Analyze Feature Flow to Enhance Interpretation
and Steering in Language Models" (arXiv:2502.03032).

All functions are pure: decoder matrices in, matches/graph out.  To steer
along a discovered flow path, pass each layer's decoder row to the existing
``patching.apply_steer`` / ``patching.generation.apply_steer_steps``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

__all__ = ["match_features", "FlowEdge", "FeatureFlowGraph", "feature_flow_graph"]


def _normalize_rows(W: Any) -> Tensor:
    t = torch.as_tensor(W).detach().to(torch.float32)
    if t.ndim != 2:
        raise ValueError(f"decoder matrix must be 2-D (n_features, d_model), got ndim={t.ndim}")
    return t / t.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def match_features(
    W_dec_a: Any,
    W_dec_b: Any,
    *,
    k: int = 1,
) -> tuple[Tensor, Tensor]:
    """Top-k cosine matches for every feature of dictionary A in dictionary B.

    Args:
        W_dec_a: ``(n_features_a, d_model)`` decoder matrix (rows = feature
            directions), e.g. an SAE ``W_dec``.
        W_dec_b: ``(n_features_b, d_model)`` decoder matrix at another layer.
        k: matches per feature (default 1 — the paper's argmax matching).

    Returns:
        ``(indices, sims)`` — int64 / float32 tensors of shape
        ``(n_features_a, k)``; row ``i`` holds feature ``i``'s best matches
        in B, sorted by cosine descending.
    """
    a = _normalize_rows(W_dec_a)
    b = _normalize_rows(W_dec_b)
    if a.shape[1] != b.shape[1]:
        raise ValueError(
            f"d_model mismatch: {a.shape[1]} (A) vs {b.shape[1]} (B)"
        )
    sims = a @ b.T                                  # (f_a, f_b)
    topk = sims.topk(min(k, sims.shape[1]), dim=-1)
    return topk.indices, topk.values


@dataclass(frozen=True, order=True)
class FlowEdge:
    """A matched feature pair between two (typically adjacent) layers."""

    src_layer: int
    src_feature: int
    dst_layer: int
    dst_feature: int


@dataclass
class FeatureFlowGraph:
    """Cross-layer feature flow from :func:`feature_flow_graph`.

    Attributes:
        scores: ``{FlowEdge: float}`` — cosine similarity per kept edge.
        layer_ids: layer index per dictionary, in order.
        n_features: dictionary size per layer.
        threshold: the similarity cutoff edges were kept at.
    """

    scores: dict[FlowEdge, float]
    layer_ids: list[int]
    n_features: list[int]
    threshold: float

    def ranked(self) -> list[tuple[FlowEdge, float]]:
        """All edges sorted by similarity descending."""
        return sorted(self.scores.items(), key=lambda kv: kv[1], reverse=True)

    def top_k(self, k: int) -> list[tuple[FlowEdge, float]]:
        return self.ranked()[:k]

    def path_from(self, layer: int, feature: int) -> list[tuple[int, int, float]]:
        """Greedy argmax chain downstream from ``(layer, feature)``.

        Returns ``[(layer, feature, sim_to_next), ...]`` starting at the
        query feature; the last entry's similarity is ``nan`` (no successor
        kept above the threshold, or the final layer was reached).
        """
        out_edges: dict[tuple[int, int], tuple[FlowEdge, float]] = {}
        for edge, sim in self.scores.items():
            key = (edge.src_layer, edge.src_feature)
            if key not in out_edges or sim > out_edges[key][1]:
                out_edges[key] = (edge, sim)

        path: list[tuple[int, int, float]] = []
        cur = (layer, feature)
        visited = set()
        while cur not in visited:
            visited.add(cur)
            hop = out_edges.get(cur)
            if hop is None:
                path.append((cur[0], cur[1], float("nan")))
                break
            edge, sim = hop
            path.append((cur[0], cur[1], sim))
            cur = (edge.dst_layer, edge.dst_feature)
        else:  # pragma: no cover — cycles impossible with strictly later layers
            pass
        return path

    def born_at(self, layer: int) -> list[int]:
        """Features at *layer* with no kept upstream edge ("born" here).

        For the first layer in ``layer_ids`` every feature is born there.
        """
        if layer not in self.layer_ids:
            raise ValueError(f"layer {layer} not in layer_ids {self.layer_ids}")
        pos = self.layer_ids.index(layer)
        n = self.n_features[pos]
        if pos == 0:
            return list(range(n))
        has_upstream = {
            e.dst_feature for e in self.scores if e.dst_layer == layer
        }
        return [f for f in range(n) if f not in has_upstream]

    def to_markdown(self, *, top_k: int = 20) -> str:
        lines = ["## Feature Flow Graph", ""]
        lines.append(f"- Layers: {self.layer_ids}")
        lines.append(f"- Features per layer: {self.n_features}")
        lines.append(f"- Edges kept at cosine >= {self.threshold}: {len(self.scores)}")
        lines.append("")
        ranked = self.top_k(top_k)
        lines.append(f"### Top-{len(ranked)} Matches by Cosine")
        lines.append("")
        lines.append("| rank | src layer | src feat | dst layer | dst feat | cosine |")
        lines.append("| ---: | ---: | ---: | ---: | ---: | ---: |")
        for i, (e, sim) in enumerate(ranked, 1):
            lines.append(
                f"| {i} | {e.src_layer} | {e.src_feature}"
                f" | {e.dst_layer} | {e.dst_feature} | {sim:.4f} |"
            )
        return "\n".join(lines)


def feature_flow_graph(
    decoders: list[Any],
    *,
    layer_ids: list[int] | None = None,
    threshold: float = 0.5,
    k: int = 1,
) -> FeatureFlowGraph:
    """Build a cross-layer feature flow graph from per-layer decoders.

    Matches each adjacent pair of dictionaries with :func:`match_features`
    and keeps edges with cosine ``>= threshold``.

    Args:
        decoders: one ``(n_features, d_model)`` decoder matrix per layer,
            in layer order (≥ 2 required).
        layer_ids: layer index per decoder (default ``0..len-1``).
        threshold: minimum cosine for an edge to be kept.
        k: candidate matches considered per feature (all above-threshold
            candidates become edges; ``path_from`` follows the best).

    Returns:
        :class:`FeatureFlowGraph`.
    """
    if len(decoders) < 2:
        raise ValueError(f"need >= 2 decoders, got {len(decoders)}")
    if layer_ids is None:
        layer_ids = list(range(len(decoders)))
    if len(layer_ids) != len(decoders):
        raise ValueError(
            f"{len(layer_ids)} layer_ids for {len(decoders)} decoders"
        )
    mats = [_normalize_rows(d) for d in decoders]
    scores: dict[FlowEdge, float] = {}
    for i in range(len(mats) - 1):
        indices, sims = match_features(mats[i], mats[i + 1], k=k)
        for f in range(indices.shape[0]):
            for j in range(indices.shape[1]):
                sim = float(sims[f, j])
                if sim >= threshold:
                    scores[FlowEdge(
                        src_layer=layer_ids[i],
                        src_feature=f,
                        dst_layer=layer_ids[i + 1],
                        dst_feature=int(indices[f, j]),
                    )] = sim
    return FeatureFlowGraph(
        scores=scores,
        layer_ids=list(layer_ids),
        n_features=[m.shape[0] for m in mats],
        threshold=threshold,
    )
