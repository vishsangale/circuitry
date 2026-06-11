"""Auto-interp feature labeling with a pluggable label function.

circuitry never calls an LLM API (library invariant: no downstream
dependencies) — instead it assembles the per-feature *evidence prompt* and
the user supplies ``label_fn: Callable[[str], str]`` (their own LLM call, a
lookup table, a human-in-the-loop, ...).  The returned labels are keyed
``(layer, feature)`` so they plug directly into the graph exporters
(``to_neuronpedia_graph(..., labels=...)`` / ``to_html(..., labels=...)``)
and show up as node ``clerp`` fields on Neuronpedia.

References: SAGE (arXiv:2511.20820) for feature labeling; ADAG
(arXiv:2604.07615) for describing attribution graphs.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

__all__ = ["FeatureEvidence", "describe_features"]


@dataclass(frozen=True)
class FeatureEvidence:
    """Evidence bundle for labeling one SAE/transcoder feature.

    All fields beyond ``layer``/``feature`` are optional — supply whatever
    evidence is available; :meth:`to_prompt` renders only what is present.

    Attributes:
        layer: layer index of the feature's site.
        feature: feature index within the dictionary.
        top_tokens: token strings on which the feature activates most
            strongly (max-activating examples).
        top_logit_tokens: token strings the feature promotes in the output
            distribution (e.g. via ``core.circuits.feature_token_alignment``).
        activation_stats: scalar statistics, e.g. ``{"max": 3.2,
            "mean": 0.4, "freq": 0.01}``.
        notes: free-form extra context (e.g. example sentences).
    """

    layer: int
    feature: int
    top_tokens: tuple[str, ...] = ()
    top_logit_tokens: tuple[str, ...] = ()
    activation_stats: dict[str, float] = field(default_factory=dict)
    notes: str = ""

    def to_prompt(self) -> str:
        """Render the evidence as a labeling prompt (one feature per call)."""
        lines = [
            "Give a short (<= 8 words) human-readable label for this neural"
            " network feature based on the evidence below.",
            f"Feature: layer {self.layer}, index {self.feature}.",
        ]
        if self.top_tokens:
            lines.append(
                "Top activating tokens: " + ", ".join(repr(t) for t in self.top_tokens)
            )
        if self.top_logit_tokens:
            lines.append(
                "Output tokens promoted: "
                + ", ".join(repr(t) for t in self.top_logit_tokens)
            )
        if self.activation_stats:
            stats = ", ".join(
                f"{k}={v:.4g}" for k, v in sorted(self.activation_stats.items())
            )
            lines.append(f"Activation stats: {stats}")
        if self.notes:
            lines.append(f"Notes: {self.notes}")
        lines.append("Label:")
        return "\n".join(lines)


def describe_features(
    evidence: Iterable[FeatureEvidence],
    label_fn: Callable[[str], str],
) -> dict[tuple[int, int], str]:
    """Label features by applying *label_fn* to each evidence prompt.

    Args:
        evidence: one :class:`FeatureEvidence` per feature to label.
        label_fn: maps an evidence prompt to a short label.  The user brings
            their own implementation (LLM API call, cache, human).  Raised
            exceptions propagate; a feature whose label comes back empty or
            whitespace-only is omitted from the result.

    Returns:
        ``{(layer, feature): label}`` — pass directly as the ``labels=``
        argument of ``to_neuronpedia_graph`` / ``to_html``.
    """
    labels: dict[tuple[int, int], str] = {}
    for ev in evidence:
        label = label_fn(ev.to_prompt()).strip()
        if label:
            labels[(ev.layer, ev.feature)] = label
    return labels
