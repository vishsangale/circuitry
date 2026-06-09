"""Certified circuit stability via randomised data subsampling.

CertifiedCircuitRunner wraps any attribution runner (EAPRunner, ReLPRunner,
EdgePruningRunner, ACDCRunner, ...) and repeats the attribution on random
subsets of the input batch.  An edge is "certified" if it appears in the
top-K edges of at least ``confidence * n_subsamples`` subsets; otherwise it
is "abstained".

Reference: arXiv:2602.22968 "Certified Circuit Stability via Data Subsampling".
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from circuitry.patching.graph import Edge

_Inputs = Tensor | dict[str, Any]


def _batch_size(inputs: _Inputs) -> int:
    if isinstance(inputs, Tensor):
        return inputs.shape[0]
    for v in inputs.values():
        if isinstance(v, Tensor):
            return v.shape[0]
    raise ValueError("Cannot determine batch size from inputs")


def _index_inputs(inputs: _Inputs, idx: Tensor) -> _Inputs:
    """Index the batch dimension of tensor or dict inputs."""
    if isinstance(inputs, Tensor):
        return inputs[idx]
    return {k: v[idx] if isinstance(v, Tensor) else v for k, v in inputs.items()}


@dataclass
class CertifiedCircuitResult:
    """Output of :class:`CertifiedCircuitRunner`.

    Attributes:
        certified_edges: Edges that appeared in top-K in ≥ ``confidence``
                         fraction of subsamples.  Ordered by vote count
                         (most stable first).
        abstained_edges: Edges that appeared below the confidence threshold.
                         Ordered by vote count (closest to threshold first).
        vote_counts:     Raw vote count per edge across all subsamples.
        n_subsamples:    Number of random subsamples used.
        top_k:           K used for inclusion in each subsample's top edges.
        confidence:      Minimum fraction required for certification.
    """

    certified_edges: list[Edge]
    abstained_edges: list[Edge]
    vote_counts: dict[Edge, int] = field(default_factory=dict)
    n_subsamples: int = 0
    top_k: int = 10
    confidence: float = 0.95

    def certified_set(self) -> set[Edge]:
        return set(self.certified_edges)

    def n_certified(self) -> int:
        return len(self.certified_edges)

    def n_abstained(self) -> int:
        return len(self.abstained_edges)


class CertifiedCircuitRunner:
    """Wraps any attribution runner with randomised data subsampling.

    For each of ``n_subsamples`` random subsets of the input batch, runs the
    wrapped ``base_runner`` and records which edges appear in the top-``top_k``
    results.  An edge is *certified* (stable under data perturbation) if it
    appears in at least ``ceil(confidence * n_subsamples)`` subsets.

    Args:
        base_runner:  Any runner with a ``.run(clean, corrupted, metric)``
                      method returning an :class:`~circuitry.patching.eap.EAPResult`.
        n_subsamples: Number of random subsets (default 20).
        confidence:   Minimum vote fraction for certification (default 0.95).
        subsample_frac: Fraction of the batch to use per subset (default 0.5).
        seed:         Base random seed for reproducibility.

    Example::

        eap = EAPRunner(model, resolver)
        certified_runner = CertifiedCircuitRunner(eap, n_subsamples=20)
        result = certified_runner.run(clean, corrupted, metric, top_k=15)
        stable = result.certified_edges   # list[Edge]

    Reference: arXiv:2602.22968
    """

    def __init__(
        self,
        base_runner: Any,
        *,
        n_subsamples: int = 20,
        confidence: float = 0.95,
        subsample_frac: float = 0.5,
        seed: int = 0,
    ) -> None:
        if not (0.0 < confidence <= 1.0):
            raise ValueError(f"confidence must be in (0, 1], got {confidence}")
        if not (0.0 < subsample_frac <= 1.0):
            raise ValueError(f"subsample_frac must be in (0, 1], got {subsample_frac}")
        if n_subsamples < 1:
            raise ValueError(f"n_subsamples must be >= 1, got {n_subsamples}")

        self.base_runner = base_runner
        self.n_subsamples = n_subsamples
        self.confidence = confidence
        self.subsample_frac = subsample_frac
        self.seed = seed

    def run(
        self,
        clean_inputs: _Inputs,
        corrupted_inputs: _Inputs,
        metric: Any,
        *,
        top_k: int = 10,
    ) -> CertifiedCircuitResult:
        """Run certified attribution with randomised subsampling.

        Args:
            clean_inputs:      Clean model inputs (tensor or dict).
            corrupted_inputs:  Corrupted model inputs, same structure.
            metric:            Metric callable ``(model_out) -> scalar Tensor``.
            top_k:             Number of top edges to consider per subsample.

        Returns:
            :class:`CertifiedCircuitResult` with certified and abstained edges.
        """
        n = _batch_size(clean_inputs)
        sub_n = max(1, int(round(n * self.subsample_frac)))

        vote_counts: dict[Edge, int] = defaultdict(int)
        rng = torch.Generator().manual_seed(self.seed)

        for _ in range(self.n_subsamples):
            idx = torch.randperm(n, generator=rng)[:sub_n]
            sub_clean = _index_inputs(clean_inputs, idx)
            sub_corrupted = _index_inputs(corrupted_inputs, idx)

            result = self.base_runner.run(sub_clean, sub_corrupted, metric)

            k = min(top_k, len(result.scores))
            for edge, _ in result.top_k(k):
                vote_counts[edge] += 1

        min_votes = self.confidence * self.n_subsamples
        all_seen = set(vote_counts.keys())

        certified = sorted(
            [e for e in all_seen if vote_counts[e] >= min_votes],
            key=lambda e: vote_counts[e],
            reverse=True,
        )
        abstained = sorted(
            [e for e in all_seen if vote_counts[e] < min_votes],
            key=lambda e: vote_counts[e],
            reverse=True,
        )

        return CertifiedCircuitResult(
            certified_edges=certified,
            abstained_edges=abstained,
            vote_counts=dict(vote_counts),
            n_subsamples=self.n_subsamples,
            top_k=top_k,
            confidence=self.confidence,
        )
