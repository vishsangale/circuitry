"""Tests for logit_lens_distributions and LayerPrediction."""
from __future__ import annotations

import torch
import pytest

from circuitry.core.lens import logit_lens_distributions, LayerPrediction


D_MODEL = 8
VOCAB = 16


def _make_unembed(d_model: int = D_MODEL, vocab: int = VOCAB) -> torch.Tensor:
    torch.manual_seed(42)
    return torch.randn(d_model, vocab)


def test_logit_lens_distributions_basic():
    """List input of 1-D tensors; check shapes and probs contract."""
    torch.manual_seed(0)
    unembed = _make_unembed()
    residuals = [torch.randn(D_MODEL) for _ in range(3)]
    result = logit_lens_distributions(residuals, unembed, top_k=3)

    assert len(result) == 3
    for i, lp in enumerate(result):
        assert isinstance(lp, LayerPrediction)
        assert lp.layer_idx == i
        assert lp.token_ids.shape == (3,)
        assert lp.probs.shape == (3,)
        # Top-k probs are a subset of softmax → sum ≤ 1
        assert float(lp.probs.sum()) <= 1.0 + 1e-6
        # All probs are non-negative
        assert (lp.probs >= 0).all()


def test_logit_lens_distributions_dict_input():
    """Dict with non-contiguous layer indices is sorted correctly."""
    torch.manual_seed(1)
    unembed = _make_unembed()
    t0 = torch.randn(D_MODEL)
    t2 = torch.randn(D_MODEL)
    result = logit_lens_distributions({0: t0, 2: t2}, unembed, top_k=5)

    assert len(result) == 2
    assert result[0].layer_idx == 0
    assert result[1].layer_idx == 2


def test_logit_lens_distributions_with_layer_norm():
    """Applying a LayerNorm changes the output."""
    torch.manual_seed(2)
    unembed = _make_unembed()
    residuals = [torch.randn(D_MODEL) * 10.0 for _ in range(2)]
    ln = torch.nn.LayerNorm(D_MODEL)

    result_no_ln = logit_lens_distributions(residuals, unembed, top_k=5)
    result_with_ln = logit_lens_distributions(residuals, unembed, layer_norm=ln, top_k=5)

    # The probabilities should differ when LayerNorm is applied to large-magnitude residuals
    for no_ln, with_ln in zip(result_no_ln, result_with_ln):
        # At least the probabilities should differ
        assert not torch.allclose(no_ln.probs, with_ln.probs), (
            "Expected layer_norm to change the output probabilities"
        )


def test_logit_lens_distributions_2d_input():
    """2-D (seq, d_model) input collapses correctly; returns LayerPrediction."""
    torch.manual_seed(3)
    unembed = _make_unembed()
    seq_len = 4
    residual_2d = torch.randn(seq_len, D_MODEL)
    result = logit_lens_distributions([residual_2d], unembed, top_k=5)

    assert len(result) == 1
    lp = result[0]
    assert lp.layer_idx == 0
    assert lp.token_ids.shape == (5,)
    assert lp.probs.shape == (5,)

    # Verify that the collapse is indeed mean over all seq*1 tokens
    expected_vec = residual_2d.reshape(-1, D_MODEL).mean(0).float()
    logits = expected_vec @ unembed.float()
    probs_full = torch.softmax(logits, dim=-1)
    top_probs, top_ids = torch.topk(probs_full, 5)
    assert torch.equal(lp.token_ids, top_ids)
    assert torch.allclose(lp.probs, top_probs, atol=1e-6)


def test_logit_lens_distributions_top_k_1():
    """top_k=1: returned prob is the argmax probability."""
    torch.manual_seed(4)
    unembed = _make_unembed()
    residual = torch.randn(D_MODEL)
    result = logit_lens_distributions([residual], unembed, top_k=1)

    assert len(result) == 1
    lp = result[0]
    assert lp.token_ids.shape == (1,)
    assert lp.probs.shape == (1,)

    # Verify it's the argmax
    logits = residual.float() @ unembed.float()
    probs_full = torch.softmax(logits, dim=-1)
    assert int(lp.token_ids[0]) == int(probs_full.argmax())
    assert float(lp.probs[0]) == pytest.approx(float(probs_full.max()), abs=1e-6)


def test_logit_lens_distributions_empty_returns_empty():
    """Empty residuals returns empty list without error."""
    unembed = _make_unembed()
    result = logit_lens_distributions([], unembed, top_k=5)
    assert result == []


def test_logit_lens_distributions_3d_input():
    """3-D (batch, seq, d_model) input collapses via reshape(-1, d)."""
    torch.manual_seed(5)
    unembed = _make_unembed()
    residual_3d = torch.randn(2, 4, D_MODEL)  # batch=2, seq=4
    result = logit_lens_distributions([residual_3d], unembed, top_k=3)

    assert len(result) == 1
    lp = result[0]
    assert lp.token_ids.shape == (3,)
    assert lp.probs.shape == (3,)
