"""Tests for patching metric primitives. Design spec §5."""
from __future__ import annotations

import pytest
import torch

from circuitry.core.patching import ce_loss, kl_divergence, logit_diff


def test_logit_diff_1d():
    logits = torch.tensor([0.0, 1.0, 3.0, 2.0])
    assert logit_diff(logits, correct=2, incorrect=1) == pytest.approx(2.0)


def test_logit_diff_2d_batch():
    logits = torch.tensor([[0.0, 1.0, 3.0], [0.0, 2.0, 4.0]])
    result = logit_diff(logits, correct=2, incorrect=1)
    assert result == pytest.approx(2.0)  # mean of (3-1, 4-2)


def test_logit_diff_3d_uses_last_token():
    logits = torch.zeros(1, 5, 4)
    logits[0, -1, 2] = 10.0
    logits[0, -1, 0] = 3.0
    assert logit_diff(logits, correct=2, incorrect=0) == pytest.approx(7.0)


def test_kl_zero_for_identical_distributions():
    torch.manual_seed(0)
    logits = torch.randn(2, 5, 16)
    kl = kl_divergence(logits, logits)
    assert kl == pytest.approx(0.0, abs=1e-5)


def test_kl_positive_for_different_distributions():
    torch.manual_seed(1)
    p = torch.randn(2, 5, 16)
    q = torch.randn(2, 5, 16)
    assert kl_divergence(p, q) > 0.0


def test_kl_chunking_matches_single_shot():
    torch.manual_seed(2)
    p = torch.randn(2, 9, 32)
    q = torch.randn(2, 9, 32)
    ref = kl_divergence(p, q, chunk_size=100_000)
    for cs in (1, 3, 7, 18):
        got = kl_divergence(p, q, chunk_size=cs)
        assert got == pytest.approx(ref, abs=1e-5), f"chunk_size={cs}"


def test_kl_1d_input():
    torch.manual_seed(3)
    p = torch.randn(16)
    assert kl_divergence(p, p) == pytest.approx(0.0, abs=1e-5)


def test_ce_loss_matches_pytorch():
    torch.manual_seed(4)
    logits = torch.randn(4, 10)
    targets = torch.randint(0, 10, (4,))
    expected = float(torch.nn.functional.cross_entropy(logits, targets).item())
    assert ce_loss(logits, targets) == pytest.approx(expected, abs=1e-5)


def test_ce_loss_3d_last_token():
    torch.manual_seed(5)
    logits = torch.randn(2, 5, 10)
    targets = torch.randint(0, 10, (2,))
    result = ce_loss(logits, targets)
    expected = float(
        torch.nn.functional.cross_entropy(logits[:, -1, :].float(), targets).item()
    )
    assert result == pytest.approx(expected, abs=1e-5)
