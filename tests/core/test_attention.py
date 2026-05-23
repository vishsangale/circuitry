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
