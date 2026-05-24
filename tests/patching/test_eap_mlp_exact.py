"""EAP exact cross-check on a linear MLP-only toy: analytic per-edge scores
== brute-force per-edge patching. Spec §8 (the correctness gate)."""
from __future__ import annotations

import pytest
import torch

from circuitry.core.patching import logit_diff_t
from circuitry.patching.eap import EAPRunner


def _metric(logits):
    return logit_diff_t(logits, correct=0, incorrect=1)


def test_eap_matches_bruteforce_per_edge(linear_mlp_toy):
    """On a fully linear model, EAP's first-order EDGE score is EXACT: patching a
    single edge (u->v) — adding Δact_u to v's residual input — changes the metric
    by exactly grad_v · Δact_u == EAP(u->v)."""
    torch.manual_seed(0)
    clean = torch.tensor([[0, 1, 2]])
    corrupted = torch.tensor([[2, 0, 1]])
    runner = EAPRunner(linear_mlp_toy)
    result = runner.run(clean_inputs=clean, corrupted_inputs=corrupted, metric=_metric)
    bruteforce = runner.bruteforce_edge_scores(
        clean_inputs=clean, corrupted_inputs=corrupted, metric=_metric)
    assert set(result.scores) == set(bruteforce)
    for edge, analytic in result.scores.items():
        assert analytic == pytest.approx(bruteforce[edge], abs=1e-4), edge


def test_model_clean_after_eap(linear_mlp_toy):
    clean = torch.tensor([[0, 1, 2]])
    before = linear_mlp_toy(clean).clone()
    runner = EAPRunner(linear_mlp_toy)
    runner.run(clean_inputs=clean, corrupted_inputs=torch.tensor([[2, 0, 1]]), metric=_metric)
    after = linear_mlp_toy(clean)
    assert torch.equal(before, after)
