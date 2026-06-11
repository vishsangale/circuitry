"""Cross-method circuit consensus — stability ensembles over discovery methods.

An edge found by one attribution method may be an artifact of that method's
approximations (EAP's gradient term, ACDC's greedy order, ...).  Agreement
across *independent methods* on the same task is a stronger stability signal
than within-method data subsampling — `CertifiedCircuitRunner` certifies one
method across data perturbations; `CircuitConsensus` certifies edges across
methods.  Reference: CIRCUS — Circuit Consensus under Uncertainty via
Stability Ensembles (arXiv:2603.00523).

Pure aggregation: takes already-computed circuits (edge sets, or scored
results such as ``EAPResult`` binarized via ``threshold``/``top_k``); runs no
forward passes.
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

__all__ = ["CircuitConsensus"]


def _binarize(result: Any, *, tau: float | None, top_k: int | None) -> set:
    """Turn a scored result (or an edge collection) into an edge set."""
    if tau is not None and hasattr(result, "threshold"):
        return set(result.threshold(tau))
    if top_k is not None and hasattr(result, "top_k"):
        return {edge for edge, _ in result.top_k(top_k)}
    if hasattr(result, "scores") and isinstance(result.scores, dict):
        raise ValueError(
            "scored result passed without a binarization rule — "
            "supply tau= or top_k= to from_results()"
        )
    return set(result)


class CircuitConsensus:
    """Per-edge agreement across N independently-discovered circuits.

    Args:
        circuits: ``{method_name: edge_collection}``.  Edges must be hashable
            and comparable across methods (e.g. ``patching.graph.Edge`` from
            runners sharing one graph layout).

    Attributes:
        circuits: ``{method_name: frozenset_of_edges}`` as ingested.
    """

    def __init__(self, circuits: dict[str, Iterable[Any]]) -> None:
        if len(circuits) < 2:
            raise ValueError(
                f"consensus needs >= 2 circuits, got {len(circuits)}"
            )
        self.circuits: dict[str, frozenset] = {
            name: frozenset(edges) for name, edges in circuits.items()
        }

    @classmethod
    def from_results(
        cls,
        results: list[Any],
        *,
        tau: float | None = None,
        top_k: int | None = None,
        names: list[str] | None = None,
    ) -> CircuitConsensus:
        """Build a consensus from scored results, binarized uniformly.

        Each result is reduced to an edge set via ``result.threshold(tau)``
        (when *tau* is given) or ``result.top_k(top_k)`` (when *top_k* is
        given).  Plain edge collections pass through unchanged.

        Args:
            results: scored results (``EAPResult``, ``ReLPRunner`` output,
                ``ACDCResult``, ...) and/or plain edge collections.
            tau: |score| threshold applied to every scored result.
            top_k: top-k rule applied to every scored result (ignored when
                *tau* is given).
            names: per-result method names (default ``method_0..N-1``).
        """
        if names is None:
            names = [f"method_{i}" for i in range(len(results))]
        if len(names) != len(results):
            raise ValueError(
                f"{len(names)} names for {len(results)} results"
            )
        return cls({
            name: _binarize(r, tau=tau, top_k=top_k)
            for name, r in zip(names, results, strict=True)
        })

    def agreement(self) -> dict[Any, float]:
        """``{edge: fraction of methods whose circuit contains it}``."""
        n = len(self.circuits)
        counts: dict[Any, int] = {}
        for edges in self.circuits.values():
            for e in edges:
                counts[e] = counts.get(e, 0) + 1
        return {e: c / n for e, c in counts.items()}

    def consensus_edges(self, min_agreement: float = 1.0) -> set:
        """Edges whose agreement fraction is ``>= min_agreement``."""
        return {
            e for e, frac in self.agreement().items() if frac >= min_agreement
        }

    def pairwise_jaccard(self) -> dict[tuple[str, str], float]:
        """Jaccard similarity for every unordered method pair.

        An empty-vs-empty pair scores 1.0 (identical circuits).
        """
        names = sorted(self.circuits)
        out: dict[tuple[str, str], float] = {}
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                ea, eb = self.circuits[a], self.circuits[b]
                union = len(ea | eb)
                out[(a, b)] = (len(ea & eb) / union) if union else 1.0
        return out

    def to_markdown(self, *, top_k: int = 20) -> str:
        """Markdown summary: per-method sizes, pairwise Jaccard, agreement bands."""
        lines = ["## Circuit Consensus", ""]
        lines.append(f"- Methods: {len(self.circuits)}")
        for name in sorted(self.circuits):
            lines.append(f"  - `{name}`: {len(self.circuits[name])} edges")
        lines.append("")

        lines.append("### Pairwise Jaccard")
        lines.append("")
        lines.append("| method A | method B | jaccard |")
        lines.append("| --- | --- | ---: |")
        for (a, b), j in sorted(self.pairwise_jaccard().items()):
            lines.append(f"| `{a}` | `{b}` | {j:.3f} |")
        lines.append("")

        agreement = self.agreement()
        ranked = sorted(agreement.items(), key=lambda kv: kv[1], reverse=True)
        lines.append(f"### Top-{min(top_k, len(ranked))} Edges by Agreement")
        lines.append("")
        lines.append("| edge | agreement |")
        lines.append("| --- | ---: |")
        for edge, frac in ranked[:top_k]:
            lines.append(f"| `{edge}` | {frac:.2f} |")
        return "\n".join(lines)
