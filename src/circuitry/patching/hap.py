"""HAP — Hybrid Attribution + Pruning.

Hu et al. 2025. https://arxiv.org/abs/2510.03282

Pre-filters edges via EAP, then runs EdgePruning on the reduced subgraph.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch.nn as nn
from torch import Tensor

from circuitry.patching.eap import EAPRunner
from circuitry.patching.edge_pruning import EdgePruningResult, EdgePruningRunner

_Inputs = Tensor | dict[str, Any]


class HAPRunner:
    """Hybrid Attribution + Pruning.

    Phase 1: EAP pre-scores all edges; keeps top ``top_p`` fraction by |score|.
    Phase 2: EdgePruningRunner optimizes masks on the reduced subgraph.
    """

    def __init__(self, model: nn.Module, resolver: Any = None) -> None:
        self._eap = EAPRunner(model, resolver)
        self._pruner = EdgePruningRunner(model, resolver)

    def run(
        self,
        clean_inputs: _Inputs,
        corrupted_inputs: _Inputs,
        metric: Callable[[Any], Tensor],
        *,
        top_p: float = 0.5,
        **pruning_kwargs: Any,
    ) -> EdgePruningResult:
        """Run HAP.

        Args:
            top_p: fraction of edges to keep after EAP pre-filter (0 < top_p <= 1).
            **pruning_kwargs: forwarded to EdgePruningRunner.run().
        """
        # Phase 1: EAP pre-filter
        eap_result = self._eap.run(clean_inputs, corrupted_inputs, metric)
        all_ranked = eap_result.ranked()
        n_keep = max(1, int(len(all_ranked) * top_p))
        top_edges = [e for e, _ in all_ranked[:n_keep]]

        # Phase 2: Edge pruning on reduced subgraph
        # Pass candidate_edges to restrict the search space
        return self._pruner.run(
            clean_inputs, corrupted_inputs, metric,
            candidate_edges=top_edges,
            **pruning_kwargs,
        )
