"""ACDC recovery metric + greedy loop + ordering + sweep + custom metric."""
from __future__ import annotations

import torch

from circuitry.patching.acdc import ACDCRunner


def test_recovery_metric_last_token_kl_zero_for_identical(linear_mlp_toy):
    runner = ACDCRunner(linear_mlp_toy)
    logits = torch.randn(1, 4, 5)
    assert runner._recovery_kl(logits, logits, position=-1) == 0.0


def test_recovery_metric_positive_for_different(linear_mlp_toy):
    runner = ACDCRunner(linear_mlp_toy)
    a = torch.randn(1, 4, 5)
    b = torch.randn(1, 4, 5)
    assert runner._recovery_kl(a, b, position=-1) > 0.0


def test_recovery_metric_position_none_averages_all(linear_mlp_toy):
    runner = ACDCRunner(linear_mlp_toy)
    a = torch.randn(1, 4, 5)
    b = torch.randn(1, 4, 5)
    last = runner._recovery_kl(a, b, position=-1)
    allpos = runner._recovery_kl(a, b, position=None)
    assert last != allpos  # different reductions
