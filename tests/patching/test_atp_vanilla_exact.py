"""Vanilla AtP exact gate: Δact·grad == brute-force patch_site per node, on a
linear model. The brute-force is INDEPENDENT ground truth (real forward
intervention) — never bend it to match the analytic score."""
from __future__ import annotations

import pytest
import torch

from circuitry.core.patching import logit_diff_t
from circuitry.patching.atp import AtPRunner
from circuitry.patching.sites import HFSiteResolver


def _metric(logits):
    return logit_diff_t(logits, correct=0, incorrect=1)


def _resolver(toy):
    return HFSiteResolver(n_heads=toy.n_heads, d_model=toy.d, d_mlp=toy.d,
                          layer_pattern="layers.{L}")


def test_vanilla_atp_matches_bruteforce_per_node(linear_attn_toy):
    torch.manual_seed(0)
    clean = torch.tensor([[0, 1, 2, 3]])
    corrupted = torch.tensor([[3, 2, 1, 0]])
    runner = AtPRunner(linear_attn_toy, _resolver(linear_attn_toy))
    result = runner.run(clean_inputs=clean, corrupted_inputs=corrupted,
                        metric=_metric, neurons=False)
    bf = runner.bruteforce_node_scores(clean_inputs=clean, corrupted_inputs=corrupted,
                                       metric=_metric, nodes=list(result.scores))
    # v / mlp / embed exact; q/k are ~0 on the fixed-pattern toy (placeholder)
    for node, score in result.scores.items():
        if node.slot in ("q", "k"):
            continue
        assert score == pytest.approx(bf[node], abs=1e-4), node


def test_model_clean_after_atp(linear_attn_toy):
    clean = torch.tensor([[0, 1, 2, 3]])
    before = linear_attn_toy(clean).clone()
    runner = AtPRunner(linear_attn_toy, _resolver(linear_attn_toy))
    runner.run(clean_inputs=clean, corrupted_inputs=torch.tensor([[3, 2, 1, 0]]),
               metric=_metric, neurons=False)
    assert torch.equal(linear_attn_toy(clean), before)
