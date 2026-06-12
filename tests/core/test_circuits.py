"""Tests for circuitry.core.circuits module."""

import pytest
import torch

from circuitry.core.circuits import (
    composition_scores,
    head_composition_score,
    ov_matrix,
    qk_matrix,
    top_embedding_tokens,
    top_logit_tokens,
)

# ---------------------------------------------------------------------------
# ov_matrix
# ---------------------------------------------------------------------------


def test_ov_matrix_shape():
    """Batched: W_V (4,8,2), W_O (4,2,8) -> (4,8,8)."""
    W_V = torch.randn(4, 8, 2)
    W_O = torch.randn(4, 2, 8)
    out = ov_matrix(W_V, W_O)
    assert out.shape == (4, 8, 8)


def test_ov_matrix_correctness():
    """Single head: W_V (8,2), W_O (2,8) -> equals W_V @ W_O."""
    W_V = torch.randn(8, 2)
    W_O = torch.randn(2, 8)
    out = ov_matrix(W_V, W_O)
    expected = W_V @ W_O
    assert torch.allclose(out, expected)


def test_ov_matrix_unbatched():
    """2-D inputs (d_model, d_head) and (d_head, d_model) -> (d_model, d_model)."""
    d_model, d_head = 6, 3
    W_V = torch.randn(d_model, d_head)
    W_O = torch.randn(d_head, d_model)
    out = ov_matrix(W_V, W_O)
    assert out.shape == (d_model, d_model)


# ---------------------------------------------------------------------------
# qk_matrix
# ---------------------------------------------------------------------------


def test_qk_matrix_shape():
    """Batched: W_Q, W_K (4,8,2) -> (4,8,8)."""
    W_Q = torch.randn(4, 8, 2)
    W_K = torch.randn(4, 8, 2)
    out = qk_matrix(W_Q, W_K)
    assert out.shape == (4, 8, 8)


def test_qk_matrix_symmetric_when_equal():
    """W_Q == W_K => W_QK is symmetric (W_QK == W_QK.T)."""
    W = torch.randn(8, 2)
    out = qk_matrix(W, W)
    assert torch.allclose(out, out.T, atol=1e-5)


def test_qk_matrix_correctness():
    """Single head: should equal W_Q @ W_K.T."""
    W_Q = torch.randn(8, 2)
    W_K = torch.randn(8, 2)
    out = qk_matrix(W_Q, W_K)
    expected = W_Q @ W_K.T
    assert torch.allclose(out, expected)


# ---------------------------------------------------------------------------
# head_composition_score
# ---------------------------------------------------------------------------


def test_head_composition_score_range():
    """Random inputs -> score in [0, 1]."""
    W_OV = torch.randn(8, 8)
    W_dest = torch.randn(8, 2)
    score = head_composition_score(W_OV, W_dest)
    assert 0.0 <= score <= 1.0


def test_head_composition_score_zero():
    """Orthogonal case: W_OV projects onto x-axis, W_dest selects y -> score ~ 0."""
    # d_model=2, d_head=1
    # W_OV maps everything to the x-axis (row [1,0], col [1,0])
    W_OV = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
    # W_dest selects y component only -> column is e_1
    W_dest = torch.tensor([[0.0], [1.0]])
    # W_OV @ W_dest = [[1,0],[0,0]] @ [[0],[1]] = [[0],[0]] => zero matrix
    score = head_composition_score(W_OV, W_dest)
    assert abs(score) < 1e-6


def test_head_composition_score_returns_float():
    """Return type must be Python float."""
    W_OV = torch.randn(8, 8)
    W_dest = torch.randn(8, 2)
    score = head_composition_score(W_OV, W_dest)
    assert isinstance(score, float)


# ---------------------------------------------------------------------------
# composition_scores
# ---------------------------------------------------------------------------


def test_composition_scores_shape():
    """W_OV_src (3,8,8), W_dest (5,8,2) -> (3,5)."""
    W_OV_src = torch.randn(3, 8, 8)
    W_dest = torch.randn(5, 8, 2)
    out = composition_scores(W_OV_src, W_dest)
    assert out.shape == (3, 5)


def test_composition_scores_values_in_range():
    """All values in [0, 1]."""
    W_OV_src = torch.randn(3, 8, 8)
    W_dest = torch.randn(5, 8, 2)
    out = composition_scores(W_OV_src, W_dest)
    assert float(out.min()) >= 0.0
    assert float(out.max()) <= 1.0 + 1e-6


# ---------------------------------------------------------------------------
# top_logit_tokens
# ---------------------------------------------------------------------------


def test_top_logit_tokens_length():
    """Returns tuple of two lists, each length k."""
    direction = torch.randn(8)
    W_U = torch.randn(8, 50)
    k = 5
    token_ids, scores = top_logit_tokens(direction, W_U, k=k)
    assert isinstance(token_ids, list)
    assert isinstance(scores, list)
    assert len(token_ids) == k
    assert len(scores) == k


def test_top_logit_tokens_correctness():
    """direction = e_0, W_U[0, 7] = 5.0, others = 0 -> top token is 7."""
    d_model, vocab_size = 8, 20
    direction = torch.zeros(d_model)
    direction[0] = 1.0
    W_U = torch.zeros(d_model, vocab_size)
    W_U[0, 7] = 5.0
    token_ids, scores = top_logit_tokens(direction, W_U, k=1)
    assert token_ids[0] == 7


# ---------------------------------------------------------------------------
# top_embedding_tokens
# ---------------------------------------------------------------------------


def test_top_embedding_tokens_correctness():
    """direction = e_0, W_E[3, 0] = 5.0, others = 0 -> top token is 3."""
    d_model, vocab_size = 8, 20
    direction = torch.zeros(d_model)
    direction[0] = 1.0
    W_E = torch.zeros(vocab_size, d_model)
    W_E[3, 0] = 5.0
    token_ids, scores = top_embedding_tokens(direction, W_E, k=1)
    assert token_ids[0] == 3


# ---------------------------------------------------------------------------
# v1.42 — weight-based transcoder analysis
# ---------------------------------------------------------------------------

from circuitry.core import circuits  # noqa: E402


class TestTranscoderVirtualWeights:
    def test_shape_and_value(self):
        dec = torch.randn(6, 4)
        enc = torch.randn(4, 8)
        v = circuits.transcoder_virtual_weights(dec, enc)
        assert v.shape == (6, 8)
        torch.testing.assert_close(v, dec @ enc)

    def test_identity_composition(self):
        # decoder rows == encoder columns -> V is the Gram matrix
        w = torch.eye(3)
        v = circuits.transcoder_virtual_weights(w, w)
        torch.testing.assert_close(v, torch.eye(3))

    def test_d_model_mismatch_raises(self):
        with pytest.raises(ValueError, match="d_model mismatch"):
            circuits.transcoder_virtual_weights(torch.randn(6, 4), torch.randn(5, 8))

    def test_float32_output(self):
        v = circuits.transcoder_virtual_weights(
            torch.randn(2, 3, dtype=torch.float64), torch.randn(3, 2)
        )
        assert v.dtype == torch.float32


class TestTopVirtualConnections:
    def test_orders_by_abs_weight(self):
        v = torch.tensor([[0.1, -5.0], [2.0, 0.0]])
        conns = circuits.top_virtual_connections(v, k=3)
        assert conns[0] == (0, 1, -5.0)
        assert conns[1] == (1, 0, 2.0)
        assert conns[2] == (0, 0, pytest.approx(0.1))

    def test_k_capped_at_numel(self):
        v = torch.ones(2, 2)
        assert len(circuits.top_virtual_connections(v, k=100)) == 4

    def test_non_2d_raises(self):
        with pytest.raises(ValueError, match="2-D"):
            circuits.top_virtual_connections(torch.ones(2, 2, 2))


class TestFeatureTokenAlignment:
    def test_shapes(self):
        ids, scores = circuits.feature_token_alignment(
            torch.randn(5, 4), torch.randn(4, 11), k=3
        )
        assert ids.shape == (5, 3) and scores.shape == (5, 3)
        assert ids.dtype == torch.int64

    def test_matches_top_logit_tokens_per_row(self):
        dec = torch.randn(3, 4)
        wu = torch.randn(4, 9)
        ids, scores = circuits.feature_token_alignment(dec, wu, k=4)
        for f in range(3):
            ref_ids, ref_scores = circuits.top_logit_tokens(dec[f], wu, k=4)
            assert ids[f].tolist() == ref_ids
            torch.testing.assert_close(scores[f], torch.tensor(ref_scores))

    def test_k_capped_at_vocab(self):
        ids, _ = circuits.feature_token_alignment(
            torch.randn(2, 4), torch.randn(4, 3), k=10
        )
        assert ids.shape == (2, 3)
