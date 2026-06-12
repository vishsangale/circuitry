"""Tests for cd_token_contributions. v1.33."""
from __future__ import annotations

import pytest
import torch

from circuitry.patching.cd import CDResult, cd_token_contributions

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _uniform_attn(seq_len: int, n_heads: int = 1) -> torch.Tensor:
    """Return uniform attention weights (n_heads, seq, seq), rows sum to 1."""
    return torch.ones(n_heads, seq_len, seq_len) / seq_len


def _identity_attn(seq_len: int, n_heads: int = 1) -> torch.Tensor:
    """Return identity attention weights (n_heads, seq, seq)."""
    eye = torch.eye(seq_len)
    return eye.unsqueeze(0).expand(n_heads, -1, -1).clone()


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

def test_cd_returns_cd_result():
    """Result is a CDResult with a .contributions tensor."""
    attn = _uniform_attn(4)
    result = cd_token_contributions([attn])
    assert isinstance(result, CDResult)
    assert isinstance(result.contributions, torch.Tensor)
    assert result.contributions.shape == (4, 4)


def test_cd_single_layer_identity_attention():
    """Uniform attention redistributes all rows to uniform with add_residual=False."""
    seq_len = 5
    attn = _uniform_attn(seq_len)
    result = cd_token_contributions([attn], add_residual=False)
    expected = torch.ones(seq_len, seq_len) / seq_len
    assert torch.allclose(result.contributions, expected, atol=1e-6)


def test_cd_self_attention_preserves_identity():
    """Identity attention with add_residual=False leaves contributions unchanged."""
    seq_len = 6
    attn = _identity_attn(seq_len)
    result = cd_token_contributions([attn], add_residual=False)
    expected = torch.eye(seq_len)
    assert torch.allclose(result.contributions, expected, atol=1e-6)


def test_cd_rows_sum_to_one():
    """contributions rows sum to ~1.0 on random multi-layer inputs."""
    torch.manual_seed(0)
    seq_len = 8
    n_heads = 4

    def rand_attn() -> torch.Tensor:
        raw = torch.rand(n_heads, seq_len, seq_len)
        return raw / raw.sum(dim=-1, keepdim=True)

    layers = [rand_attn() for _ in range(5)]
    result = cd_token_contributions(layers)
    row_sums = result.contributions.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones(seq_len), atol=1e-5)


def test_cd_multi_layer_reduces_identity():
    """3 layers of uniform attention make contributions closer to uniform than 1 layer."""
    seq_len = 6
    attn = _uniform_attn(seq_len)
    uniform = torch.ones(seq_len, seq_len) / seq_len

    result_1 = cd_token_contributions([attn], add_residual=True)
    result_3 = cd_token_contributions([attn, attn, attn], add_residual=True)

    dist_1 = (result_1.contributions - uniform).abs().mean().item()
    dist_3 = (result_3.contributions - uniform).abs().mean().item()
    assert dist_3 < dist_1, (
        f"3 layers should be closer to uniform than 1 layer, "
        f"got dist_1={dist_1:.4f}, dist_3={dist_3:.4f}"
    )


def test_cd_no_residual():
    """add_residual=False: only the attention path contributes, no blending with prior C."""
    seq_len = 4
    attn = _uniform_attn(seq_len)

    result_no_res = cd_token_contributions([attn], add_residual=False)
    result_res = cd_token_contributions([attn], add_residual=True)

    # Without residual: each row is uniform regardless of initial identity.
    expected_no_res = torch.ones(seq_len, seq_len) / seq_len
    assert torch.allclose(result_no_res.contributions, expected_no_res, atol=1e-6)

    # With residual: blended, so NOT equal to pure uniform.
    assert not torch.allclose(result_res.contributions, expected_no_res, atol=1e-3)


def test_cd_head_agg_max_vs_mean():
    """max and mean aggregation give different results on multi-head input."""
    torch.manual_seed(42)
    seq_len = 5
    n_heads = 3
    raw = torch.rand(n_heads, seq_len, seq_len)
    attn = raw / raw.sum(dim=-1, keepdim=True)

    result_mean = cd_token_contributions([attn], head_agg="mean")
    result_max = cd_token_contributions([attn], head_agg="max")
    assert not torch.allclose(result_mean.contributions, result_max.contributions, atol=1e-4)


def test_cd_empty_raises():
    """Empty attn_weights list raises ValueError."""
    with pytest.raises(ValueError, match="non-empty"):
        cd_token_contributions([])


def test_cd_mismatched_seq_len_raises():
    """Layers with different seq_len raise ValueError."""
    attn_4 = _uniform_attn(4)
    attn_6 = _uniform_attn(6)
    with pytest.raises(ValueError, match="seq_len"):
        cd_token_contributions([attn_4, attn_6])


def test_cd_4d_input():
    """Batch-first (batch, heads, seq, seq) tensors are accepted and averaged."""
    batch, n_heads, seq_len = 3, 2, 5
    torch.manual_seed(7)
    raw = torch.rand(batch, n_heads, seq_len, seq_len)
    attn_4d = raw / raw.sum(dim=-1, keepdim=True)

    result = cd_token_contributions([attn_4d])
    assert isinstance(result, CDResult)
    assert result.contributions.shape == (seq_len, seq_len)
    row_sums = result.contributions.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones(seq_len), atol=1e-5)
