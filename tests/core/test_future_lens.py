"""Tests for future_lens_kl in circuitry.core.lens."""
from __future__ import annotations

import math

import pytest
import torch

from circuitry.core.lens import future_lens_kl, logit_lens_kl


def test_future_lens_kl_horizon0_equals_logit_lens():
    """horizon=0 should be position-for-position equal to logit_lens_kl."""
    torch.manual_seed(10)
    d_model, vocab, seq = 8, 16, 6
    W = torch.randn(d_model, vocab)
    residual = torch.randn(seq, d_model)
    final_logits = torch.randn(seq, vocab)

    kl_future = future_lens_kl(residual, W, final_logits, horizon=0)
    kl_logit = logit_lens_kl(residual, W, final_logits)

    assert kl_future == pytest.approx(kl_logit, abs=1e-5)


def test_future_lens_kl_horizon_ge_seq_returns_zero():
    """horizon >= seq → 0.0 (no valid positions)."""
    torch.manual_seed(11)
    d_model, vocab, seq = 8, 16, 3
    W = torch.randn(d_model, vocab)
    residual = torch.randn(seq, d_model)
    target_logits = torch.randn(seq, vocab)

    assert future_lens_kl(residual, W, target_logits, horizon=5) == 0.0
    assert future_lens_kl(residual, W, target_logits, horizon=3) == 0.0


def test_future_lens_kl_horizon1_finite():
    """seq=4, horizon=1 → a finite float (not nan/inf)."""
    torch.manual_seed(12)
    d_model, vocab, seq = 8, 16, 4
    W = torch.randn(d_model, vocab)
    residual = torch.randn(seq, d_model)
    target_logits = torch.randn(seq, vocab)

    result = future_lens_kl(residual, W, target_logits, horizon=1)
    assert isinstance(result, float)
    assert math.isfinite(result)
    assert result >= 0.0


def test_future_lens_kl_with_layer_norm():
    """Passing a LayerNorm produces a finite result that differs from the no-norm case."""
    torch.manual_seed(13)
    d_model, vocab, seq = 8, 16, 5
    W = torch.randn(d_model, vocab)
    residual = torch.randn(seq, d_model)
    target_logits = torch.randn(seq, vocab)
    ln = torch.nn.LayerNorm(d_model)

    result_with_ln = future_lens_kl(residual, W, target_logits, horizon=1, layer_norm=ln)
    result_no_ln = future_lens_kl(residual, W, target_logits, horizon=1)

    assert math.isfinite(result_with_ln)
    assert result_with_ln >= 0.0
    # The two values should differ (LayerNorm changes the distribution)
    assert result_with_ln != pytest.approx(result_no_ln, abs=1e-4)


def test_future_lens_kl_returns_float():
    """Return type is a plain Python float."""
    torch.manual_seed(14)
    d_model, vocab, seq = 8, 16, 4
    W = torch.randn(d_model, vocab)
    residual = torch.randn(seq, d_model)
    target_logits = torch.randn(seq, vocab)

    result = future_lens_kl(residual, W, target_logits, horizon=1)
    assert isinstance(result, float)
