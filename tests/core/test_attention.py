"""Tests for induction_score and attention_pattern_entropy. Spec §4.2."""
from __future__ import annotations

import math

import pytest
import torch

from circuitry.core.attention import (
    attention_pattern_entropy,
    induction_score,
)


def test_perfect_induction_head_scores_one():
    """An attention pattern where every query at t+n attends to key t+1
    perfectly should score 1.0."""
    n = 8  # seq_len_repeat
    seq = 2 * n
    n_heads = 1
    attn = torch.zeros(1, n_heads, seq, seq)
    for t in range(n - 1):
        attn[0, 0, t + n, t + 1] = 1.0
    scores = induction_score(attn, seq_len_repeat=n)
    assert scores == pytest.approx([1.0], abs=1e-6)


def test_random_head_scores_near_one_over_seq():
    """A uniformly random head should score ~1/seq on the induction probe."""
    torch.manual_seed(0)
    n = 16
    seq = 2 * n
    n_heads = 4
    # Uniform attention over keys (each row sums to 1).
    attn = torch.full((1, n_heads, seq, seq), 1.0 / seq)
    scores = induction_score(attn, seq_len_repeat=n)
    expected = 1.0 / seq
    for s in scores:
        assert s == pytest.approx(expected, abs=1e-6)


def test_accepts_3d_input():
    """Should accept (n_heads, seq, seq) without a batch dim."""
    n = 4
    seq = 2 * n
    attn = torch.zeros(2, seq, seq)
    attn[0, n + 0, 1] = 1.0
    attn[0, n + 1, 2] = 1.0
    attn[0, n + 2, 3] = 1.0
    scores = induction_score(attn, seq_len_repeat=n)
    assert len(scores) == 2
    assert scores[0] == pytest.approx(1.0, abs=1e-6)


def test_rejects_seq_too_short():
    n = 8
    attn = torch.zeros(1, 1, n, n)  # seq = n, but we need 2 * n
    with pytest.raises(ValueError, match="seq="):
        induction_score(attn, seq_len_repeat=n)


def test_entropy_of_uniform_attention_is_log_seq():
    """Uniform attention over `seq` keys has entropy ln(seq)."""
    seq = 8
    n_heads = 3
    attn = torch.full((1, n_heads, seq, seq), 1.0 / seq)
    ents = attention_pattern_entropy(attn)
    for e in ents:
        assert e == pytest.approx(math.log(seq), abs=1e-5)


def test_entropy_of_one_hot_attention_is_zero():
    """Deterministic attention has entropy 0."""
    seq = 5
    n_heads = 2
    attn = torch.zeros(1, n_heads, seq, seq)
    for h in range(n_heads):
        for t in range(seq):
            attn[0, h, t, 0] = 1.0
    ents = attention_pattern_entropy(attn)
    for e in ents:
        assert e == pytest.approx(0.0, abs=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_induction_score_on_cuda():
    # Regression (v0.9.1): arange query/key index tensors were on CPU while the
    # attn pattern was on CUDA → device mismatch in advanced indexing.
    attn = torch.rand(2, 4, 50, 50, device="cuda")
    scores = induction_score(attn, seq_len_repeat=25)
    assert len(scores) == 4
    assert all(isinstance(x, float) for x in scores)


def test_entropy_normalizes_unnormalized_rows():
    """Rows that don't sum to 1 (sigmoid / linear attention) must be normalized
    before entropy, so uniform-but-unnormalized weights give ln(seq), not a
    value inflated by the total mass."""
    seq, n_heads = 8, 2
    # Every key weighted 0.5 -> each row sums to 4.0, not 1.0.
    attn = torch.full((1, n_heads, seq, seq), 0.5)
    ents = attention_pattern_entropy(attn)
    for e in ents:
        assert e == pytest.approx(math.log(seq), abs=1e-5)


def test_entropy_handles_fully_masked_zero_rows_without_nan():
    """A fully-masked query row (all zeros, sums to 0) must not produce NaN;
    the eps-clamped divide leaves it all-zero -> entropy 0."""
    seq, n_heads = 5, 1
    attn = torch.zeros(1, n_heads, seq, seq)
    attn[0, 0, 0, 0] = 1.0  # row 0 valid (one-hot), rows 1..4 fully masked
    ents = attention_pattern_entropy(attn)
    assert not math.isnan(ents[0])
    assert ents[0] == pytest.approx(0.0, abs=1e-6)


def test_entropy_left_padded_pad_rows_dropped_by_default():
    """recsys B: LEFT-padded models feed PAD query rows an all-(-inf) key set;
    ``softmax`` returns an all-NaN row that poisons a naive mean. The per-head
    mean must be NaN-aware and average only the valid (non-pad) query rows."""
    seq, n_heads, n_pad = 6, 2, 2  # first n_pad positions are left-padding
    neg_inf = float("-inf")
    scores = torch.zeros(1, n_heads, seq, seq)
    scores[..., :n_pad] = neg_inf       # PAD keys are never attended to
    scores[:, :, :n_pad, :] = neg_inf   # PAD query rows attend to nothing
    attn = torch.softmax(scores, dim=-1)
    assert attn[0, 0, 0].isnan().all()  # sanity: this is the NaN bug source
    ents = attention_pattern_entropy(attn)
    # Pre-fix this returned NaN; valid rows attend uniformly over (seq - n_pad)
    # keys -> entropy ln(seq - n_pad).
    for e in ents:
        assert not math.isnan(e)
        assert e == pytest.approx(math.log(seq - n_pad), abs=1e-5)


def test_entropy_valid_mask_restricts_to_valid_rows():
    """An explicit (B, T) ``valid_mask`` (auto-expanded across heads) must
    average only the marked query rows."""
    seq, n_heads = 5, 1
    attn = torch.full((1, n_heads, seq, seq), 1.0 / seq)  # uniform -> ln(seq)
    attn[0, 0, 0] = 0.0
    attn[0, 0, 0, 0] = 1.0  # row 0 one-hot (entropy 0)
    mask = torch.ones(1, seq, dtype=torch.bool)
    mask[0, 0] = False  # exclude the one-hot row
    ents = attention_pattern_entropy(attn, valid_mask=mask)
    assert ents[0] == pytest.approx(math.log(seq), abs=1e-5)
    # Without the mask, the entropy-0 row pulls the per-head mean down.
    ents_nomask = attention_pattern_entropy(attn)
    assert ents_nomask[0] < ents[0]


def test_entropy_all_valid_mask_matches_no_mask():
    """Backward-compat: an all-True mask on a clean pattern is a no-op."""
    seq, n_heads = 7, 3
    torch.manual_seed(0)
    attn = torch.softmax(torch.randn(1, n_heads, seq, seq), dim=-1)
    mask = torch.ones(1, n_heads, seq, dtype=torch.bool)
    masked = attention_pattern_entropy(attn, valid_mask=mask)
    plain = attention_pattern_entropy(attn)
    assert masked == pytest.approx(plain, abs=1e-6)
