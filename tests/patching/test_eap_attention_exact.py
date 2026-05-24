"""EAP exact per-edge cross-check with attention edges on a linear fixed-pattern toy."""
from __future__ import annotations

import pytest
import torch

from circuitry.core.patching import logit_diff_t
from circuitry.patching.eap import EAPRunner
from circuitry.patching.sites import HFSiteResolver


def _metric(logits):
    return logit_diff_t(logits, correct=0, incorrect=1)


def _resolver(toy):
    return HFSiteResolver(n_heads=toy.n_heads, d_model=toy.d, d_mlp=toy.d,
                          layer_pattern="layers.{L}")


def test_eap_matches_bruteforce_per_edge_with_attention(linear_attn_toy):
    torch.manual_seed(0)
    clean = torch.tensor([[0, 1, 2, 3]])
    corrupted = torch.tensor([[3, 2, 1, 0]])
    runner = EAPRunner(linear_attn_toy, _resolver(linear_attn_toy))
    result = runner.run(clean_inputs=clean, corrupted_inputs=corrupted, metric=_metric)
    bruteforce = runner.bruteforce_edge_scores(
        clean_inputs=clean, corrupted_inputs=corrupted, metric=_metric)
    assert set(result.scores) == set(bruteforce)
    for edge, analytic in result.scores.items():
        assert analytic == pytest.approx(bruteforce[edge], abs=1e-4), edge


def test_attn_head_writer_is_d_model(linear_attn_toy):
    runner = EAPRunner(linear_attn_toy, _resolver(linear_attn_toy))
    acts = runner.writer_activations(torch.tensor([[0, 1, 2, 3]]))
    for node, act in acts.items():
        assert act.shape[-1] == linear_attn_toy.d, node
