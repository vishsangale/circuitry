"""SAE feature attribution across independent forward passes. v1.22.0.

SAEFeatureTemporalRunner runs SAEFeatureRunner on each (step_key, clean, corrupted)
triple independently and collects per-step AtPResult objects.  Also computes
attribution deltas between consecutive steps, and provides helpers to identify
stable vs step-specific features.

Scope note: each step is run independently — this does NOT model true recurrent
dependencies where step k's activations depend on step k-1's hidden state.
Temporal SAEs that require stored per-step activations are a separate design
problem (noted in TODO).  This runner tracks feature attribution stability and
change across independent steps (training checkpoints, different input prompts,
or any other per-step contrast).
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from circuitry.patching.atp import AtPNode, AtPResult
from circuitry.patching.sae_features import SAEFeatureRunner
from circuitry.patching.sites import Site

_Inputs = Any


class TemporalAtPResult:
    """Per-step SAE feature attribution from SAEFeatureTemporalRunner.

    Attributes:
        scores:       dict[step_key, AtPResult] — per-step attribution scores.
        step_keys:    ordered list of step keys.
        delta_scores: dict[step_key, AtPResult] — per-step attribution deltas.
                      delta_scores[k] = scores[k] − scores[k−1] (missing → 0).
                      Only populated for steps after the first.
    """

    def __init__(
        self,
        scores: dict[Any, AtPResult],
        step_keys: list[Any],
    ) -> None:
        self.scores = scores
        self.step_keys = list(step_keys)
        self.delta_scores: dict[Any, AtPResult] = self._compute_deltas()

    def _compute_deltas(self) -> dict[Any, AtPResult]:
        if len(self.step_keys) < 2:
            return {}
        deltas: dict[Any, AtPResult] = {}
        for i in range(1, len(self.step_keys)):
            prev_k = self.step_keys[i - 1]
            curr_k = self.step_keys[i]
            prev = self.scores[prev_k].scores
            curr = self.scores[curr_k].scores
            all_nodes = set(prev) | set(curr)
            delta: dict[AtPNode, float] = {
                node: curr.get(node, 0.0) - prev.get(node, 0.0)
                for node in all_nodes
            }
            deltas[curr_k] = AtPResult(delta)
        return deltas

    def stable_features(self, threshold: float = 0.5) -> list[AtPNode]:
        """Return nodes with |score| ≥ threshold at ALL steps, sorted by layer/neuron."""
        if not self.step_keys:
            return []
        stable: set[AtPNode] | None = None
        for key in self.step_keys:
            active = {n for n, s in self.scores[key].scores.items() if abs(s) >= threshold}
            stable = active if stable is None else stable & active
        return _sorted_nodes(stable or set())

    def step_specific_features(
        self,
        step_key: Any,
        threshold: float = 0.5,
    ) -> list[AtPNode]:
        """Return nodes with |score| ≥ threshold at *step_key* but not at any other step."""
        if step_key not in self.scores:
            raise KeyError(f"Unknown step_key: {step_key!r}")
        active_here = {
            n for n, s in self.scores[step_key].scores.items() if abs(s) >= threshold
        }
        for other_key in self.step_keys:
            if other_key == step_key:
                continue
            other_active = {
                n for n, s in self.scores[other_key].scores.items() if abs(s) >= threshold
            }
            active_here -= other_active
            if not active_here:
                break
        return _sorted_nodes(active_here)

    def top_stable(self, k: int = 10) -> list[tuple[AtPNode, float]]:
        """Top-k features by minimum |score| across all steps (most reliably active).

        Returns list of (node, min_abs_score) sorted by min_abs_score descending.
        """
        if not self.step_keys:
            return []
        all_nodes: set[AtPNode] = set()
        for key in self.step_keys:
            all_nodes |= set(self.scores[key].scores)
        ranked: list[tuple[AtPNode, float]] = []
        for node in all_nodes:
            min_abs = min(
                abs(self.scores[sk].scores.get(node, 0.0)) for sk in self.step_keys
            )
            ranked.append((node, min_abs))
        ranked.sort(key=lambda ns: ns[1], reverse=True)
        return ranked[:k]


def _sorted_nodes(nodes: set[AtPNode]) -> list[AtPNode]:
    return sorted(
        nodes,
        key=lambda n: (n.node.layer if n.node.layer is not None else -1,
                       n.node.neuron if n.node.neuron is not None else -1),
    )


class SAEFeatureTemporalRunner:
    """Multi-step SAE feature attribution across independent forward passes.

    Each (step_key, clean_inputs, corrupted_inputs) triple is evaluated
    independently via SAEFeatureRunner.run().  Results are aggregated into a
    TemporalAtPResult that exposes per-step scores, attribution deltas between
    consecutive steps, and helpers for identifying stable vs step-specific features.

    Usage::

        runner = SAEFeatureTemporalRunner(model, sae_sites, resolver)
        result = runner.run(
            steps=[
                ("step_0", clean_0, corrupt_0),
                ("step_100", clean_100, corrupt_100),
                ("step_500", clean_500, corrupt_500),
            ],
            metric=lambda out: logit_diff_t(out, correct=0, incorrect=1),
        )
        print(result.stable_features(threshold=0.3))
        print(result.delta_scores["step_100"].top_k(5))
    """

    def __init__(
        self,
        model: Any,
        sae_sites: dict[Site, Any],
        resolver: Any,
    ) -> None:
        self._runner = SAEFeatureRunner(model, sae_sites, resolver)

    def run(
        self,
        steps: list[tuple[Any, _Inputs, _Inputs]],
        metric: Callable[[Any], Any],
        **runner_kwargs: Any,
    ) -> TemporalAtPResult:
        """Run attribution for each step independently.

        Args:
            steps:          list of (step_key, clean_inputs, corrupted_inputs) triples.
                            step_key may be any hashable — int (training step), str, etc.
                            Keys must be unique and the list must be non-empty.
            metric:         differentiable scalar metric.
            **runner_kwargs: forwarded to SAEFeatureRunner.run() (graddrop,
                            include_error_node, max_features, variant, n_ig_steps).

        Returns:
            TemporalAtPResult with .scores, .delta_scores, .stable_features(),
            .step_specific_features(), and .top_stable().
        """
        if not steps:
            raise ValueError("steps must be non-empty")
        step_keys = [k for k, _, _ in steps]
        if len(set(step_keys)) != len(step_keys):
            raise ValueError(
                "step_keys must be unique; "
                f"got duplicates in {step_keys!r}"
            )

        scores: dict[Any, AtPResult] = {}
        for step_key, clean, corrupted in steps:
            result = self._runner.run(clean, corrupted, metric, **runner_kwargs)
            scores[step_key] = result

        return TemporalAtPResult(scores=scores, step_keys=step_keys)
