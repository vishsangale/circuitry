"""Neuron-level AtP nodes: exact vs brute-force patch_site per neuron (linear toy)."""
from __future__ import annotations

import pytest
import torch

from circuitry.core.patching import logit_diff_t
from circuitry.patching.atp import AtPRunner
from circuitry.patching.sites import HFSiteResolver


def _metric(logits):
    return logit_diff_t(logits, correct=0, incorrect=1)


def test_neuron_atp_matches_bruteforce(linear_mlp_toy):
    torch.manual_seed(0)
    clean = torch.tensor([[0, 1, 2]])
    corrupted = torch.tensor([[2, 0, 1]])
    resolver = HFSiteResolver(n_heads=1, d_model=linear_mlp_toy.d, d_mlp=linear_mlp_toy.d,
                              layer_pattern="layers.{L}")
    runner = AtPRunner(linear_mlp_toy, resolver)
    result = runner.run(clean_inputs=clean, corrupted_inputs=corrupted,
                        metric=_metric, neurons=True)
    neuron_nodes = [n for n in result.scores if n.node.kind == "mlp_neuron"]
    assert neuron_nodes, "no neuron nodes produced"
    bf = runner.bruteforce_node_scores(clean_inputs=clean, corrupted_inputs=corrupted,
                                       metric=_metric, nodes=neuron_nodes)
    for node in neuron_nodes:
        assert result.scores[node] == pytest.approx(bf[node], abs=1e-4), node


def test_no_param_grad_leak_with_neurons(linear_mlp_toy):
    for p in linear_mlp_toy.parameters():
        p.requires_grad_(True)
        p.grad = None
    resolver = HFSiteResolver(n_heads=1, d_model=linear_mlp_toy.d, d_mlp=linear_mlp_toy.d,
                              layer_pattern="layers.{L}")
    AtPRunner(linear_mlp_toy, resolver).run(
        clean_inputs=torch.tensor([[0, 1, 2]]), corrupted_inputs=torch.tensor([[2, 0, 1]]),
        metric=_metric, neurons=True)
    assert not [n for n, p in linear_mlp_toy.named_parameters() if p.grad is not None]
