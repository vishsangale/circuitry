"""Tests for logit_lens_kl. Spec §4.1."""
from __future__ import annotations

import logging

import pytest
import torch

from circuitry.core.lens import logit_lens_kl


def test_kl_is_zero_when_projection_equals_final_logits():
    torch.manual_seed(0)
    d_model, vocab = 8, 16
    W = torch.randn(d_model, vocab)
    residual = torch.randn(2, 5, d_model)
    final_logits = residual @ W  # by construction, lens == final
    kl = logit_lens_kl(residual, W, final_logits)
    assert kl == pytest.approx(0.0, abs=1e-5)


def test_kl_positive_for_independent_inputs():
    torch.manual_seed(1)
    d_model, vocab = 8, 16
    residual = torch.randn(2, 5, d_model)
    W = torch.randn(d_model, vocab)
    final_logits = torch.randn(2, 5, vocab)
    kl = logit_lens_kl(residual, W, final_logits)
    assert kl > 0.0


def test_transpose_orientation_autodetected():
    torch.manual_seed(2)
    d_model, vocab = 4, 7
    W = torch.randn(d_model, vocab)
    residual = torch.randn(1, 3, d_model)
    final_logits = residual @ W
    kl_a = logit_lens_kl(residual, W, final_logits)
    kl_b = logit_lens_kl(residual, W.t(), final_logits)  # (vocab, d_model)
    assert kl_a == pytest.approx(kl_b, abs=1e-5)


def test_layer_norm_callable_is_applied():
    torch.manual_seed(3)
    d_model, vocab = 6, 10
    W = torch.randn(d_model, vocab)
    residual = torch.randn(2, 4, d_model) * 100.0  # large magnitude
    ln = torch.nn.LayerNorm(d_model)
    final_logits = ln(residual) @ W
    kl_with_ln = logit_lens_kl(residual, W, final_logits, layer_norm=ln)
    kl_without_ln = logit_lens_kl(residual, W, final_logits)
    assert kl_with_ln == pytest.approx(0.0, abs=1e-4)
    assert kl_without_ln > 1e-3  # mismatch when LN is skipped


def test_rejects_non_2d_unembed():
    with pytest.raises(ValueError, match="must be 2-D"):
        logit_lens_kl(torch.randn(2, 4), torch.randn(4, 5, 6), torch.randn(2, 5))


def test_warns_when_dmodel_equals_vocab(caplog):
    """d_model == vocab_size makes shape-based orientation impossible.
    Function must emit a warning."""
    torch.manual_seed(4)
    n = 5
    W = torch.randn(n, n)
    residual = torch.randn(1, 2, n)
    final_logits = residual @ W
    with caplog.at_level(logging.WARNING, logger="circuitry.core.lens"):
        logit_lens_kl(residual, W, final_logits)
    assert any("d_model == vocab" in r.getMessage() for r in caplog.records)
