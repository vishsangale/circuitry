"""Tests for patching metric primitives. Design spec §5."""
from __future__ import annotations

import pytest
import torch

from circuitry.core.patching import (
    ce_loss,
    ce_loss_t,
    kl_divergence,
    kl_divergence_t,
    logit_diff,
    logit_diff_t,
)


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


# ---------------------------------------------------------------------------
# Differentiable _t variant tests
# ---------------------------------------------------------------------------


def test_logit_diff_t_matches_float_and_is_differentiable():
    torch.manual_seed(10)
    logits = torch.randn(2, 5, 8, requires_grad=True)
    t = logit_diff_t(logits, correct=0, incorrect=1)
    assert t.requires_grad
    assert float(t.item()) == pytest.approx(logit_diff(logits, correct=0, incorrect=1), abs=1e-5)
    t.backward()  # no error; grad flows
    assert logits.grad is not None


def test_kl_divergence_t_returns_differentiable_tensor():
    torch.manual_seed(11)
    p = torch.randn(2, 5, 8, requires_grad=True)
    q = torch.randn(2, 5, 8)
    t = kl_divergence_t(p, q)
    assert isinstance(t, torch.Tensor)
    assert t.requires_grad
    t.backward()
    assert p.grad is not None


def test_ce_loss_t_matches_float_and_is_differentiable():
    torch.manual_seed(12)
    logits = torch.randn(4, 10, requires_grad=True)
    targets = torch.randint(0, 10, (4,))
    t = ce_loss_t(logits, targets)
    assert t.requires_grad
    assert float(t.item()) == pytest.approx(ce_loss(logits, targets), abs=1e-5)
    t.backward()
    assert logits.grad is not None
