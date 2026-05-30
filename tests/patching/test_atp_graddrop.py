"""GradDrop reduces sign-cancellation undercounting (sum of |per-position score|)."""
from __future__ import annotations

import torch

from circuitry.core.patching import logit_diff_t
from circuitry.patching.atp import AtPRunner
from circuitry.patching.sites import HFSiteResolver


def _metric(logits):
    return logit_diff_t(logits, correct=0, incorrect=1)


def test_graddrop_geq_plain_magnitude(linear_mlp_toy):
    # GradDrop = Σ|per-pos|, so its magnitude is >= |Σ per-pos| (plain) for every
    # node — strictly greater where per-position contributions flip sign.
    clean = torch.tensor([[0, 1, 2]])
    corrupted = torch.tensor([[2, 0, 1]])
    resolver = HFSiteResolver(n_heads=1, d_model=linear_mlp_toy.d, d_mlp=linear_mlp_toy.d,
                              layer_pattern="layers.{L}")
    runner = AtPRunner(linear_mlp_toy, resolver)
    plain = runner.run(clean_inputs=clean, corrupted_inputs=corrupted, metric=_metric, neurons=False)
    gd = runner.run(clean_inputs=clean, corrupted_inputs=corrupted, metric=_metric,
                    neurons=False, graddrop=True)
    for node in plain.scores:
        if node.slot in ("q", "k"):
            continue
        assert abs(gd.scores[node]) >= abs(plain.scores[node]) - 1e-5, node


def test_no_param_grad_leak_graddrop(linear_mlp_toy):
    for p in linear_mlp_toy.parameters():
        p.requires_grad_(True)
        p.grad = None
    resolver = HFSiteResolver(n_heads=1, d_model=linear_mlp_toy.d, d_mlp=linear_mlp_toy.d,
                              layer_pattern="layers.{L}")
    AtPRunner(linear_mlp_toy, resolver).run(
        clean_inputs=torch.tensor([[0, 1, 2]]), corrupted_inputs=torch.tensor([[2, 0, 1]]),
        metric=_metric, neurons=False, graddrop=True)
    assert not [n for n, p in linear_mlp_toy.named_parameters() if p.grad is not None]
