"""EAP exact cross-check on a linear MLP-only toy: analytic node-aggregate scores
== brute-force node patching. Spec §8 (the correctness gate)."""
from __future__ import annotations

from collections import defaultdict

import pytest
import torch

from circuitry.core.patching import logit_diff
from circuitry.patching.eap import EAPRunner


def _metric(logits):
    return logit_diff(logits, correct=0, incorrect=1)


def test_eap_node_aggregate_matches_node_patching(linear_mlp_toy):
    """On a fully linear model, brute-force NODE patching of writer u changes the
    metric by EXACTLY the sum of EAP edge scores leaving u."""
    torch.manual_seed(0)
    clean = torch.tensor([[0, 1, 2]])
    corrupted = torch.tensor([[2, 0, 1]])

    runner = EAPRunner(linear_mlp_toy)
    result = runner.run(clean_inputs=clean, corrupted_inputs=corrupted, metric=_metric)

    node_delta = runner.bruteforce_node_scores(
        clean_inputs=clean, corrupted_inputs=corrupted, metric=_metric
    )
    agg = defaultdict(float)
    for edge, s in result.scores.items():
        agg[edge.writer] += s
    assert set(agg) == set(node_delta)
    for node, delta in node_delta.items():
        assert agg[node] == pytest.approx(delta, abs=1e-4), node


def test_model_clean_after_eap(linear_mlp_toy):
    clean = torch.tensor([[0, 1, 2]])
    before = linear_mlp_toy(clean).clone()
    runner = EAPRunner(linear_mlp_toy)
    runner.run(clean_inputs=clean, corrupted_inputs=torch.tensor([[2, 0, 1]]), metric=_metric)
    after = linear_mlp_toy(clean)
    assert torch.equal(before, after)
