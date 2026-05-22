"""Tests for attention_head_rank — per-head effective_rank of a projection."""
from __future__ import annotations

import pytest
import torch

from circuitry.core.weight import attention_head_rank, effective_rank


def test_head_rank_q_proj_shape():
    # q_proj-shape: (n_heads * head_dim, d_model).
    n_heads, head_dim, d_model = 4, 8, 32
    W = torch.randn(n_heads * head_dim, d_model)
    ranks = attention_head_rank(W, n_heads=n_heads, head_dim=head_dim, axis=0)
    assert len(ranks) == n_heads
    # Each head's slice is (head_dim, d_model) — effective_rank bounded
    # by min(head_dim, d_model) == head_dim.
    for r in ranks:
        assert 0 < r <= head_dim


def test_head_rank_o_proj_shape():
    # o_proj-shape: (d_model, n_heads * head_dim).
    n_heads, head_dim, d_model = 4, 8, 32
    W = torch.randn(d_model, n_heads * head_dim)
    ranks = attention_head_rank(W, n_heads=n_heads, head_dim=head_dim, axis=1)
    assert len(ranks) == n_heads
    for r in ranks:
        assert 0 < r <= head_dim


def test_head_rank_matches_effective_rank_per_slice():
    # The head-i rank should equal effective_rank on the head-i slice.
    n_heads, head_dim, d_model = 2, 4, 16
    torch.manual_seed(0)
    W = torch.randn(n_heads * head_dim, d_model)
    ranks = attention_head_rank(W, n_heads=n_heads, head_dim=head_dim, axis=0)
    for i in range(n_heads):
        slice_i = W[i * head_dim : (i + 1) * head_dim]
        assert abs(ranks[i] - effective_rank(slice_i)) < 1e-4


def test_head_rank_rejects_mismatched_dim():
    W = torch.randn(31, 16)  # 31 != n_heads * head_dim for any plausible split
    with pytest.raises(ValueError, match="head_dim"):
        attention_head_rank(W, n_heads=4, head_dim=8, axis=0)


def test_head_rank_supports_gqa_via_caller_choice_of_n_heads():
    # GQA: k_proj has fewer heads (num_key_value_heads). Caller passes
    # whichever head count is right for that projection; the primitive
    # does not infer.
    n_kv_heads, head_dim, d_model = 2, 8, 32
    W = torch.randn(n_kv_heads * head_dim, d_model)
    ranks = attention_head_rank(W, n_heads=n_kv_heads, head_dim=head_dim,
                                axis=0)
    assert len(ranks) == n_kv_heads
