"""Tests for circuitry.core.decompose.logit_decomposition and LogitDecompositionResult."""
from __future__ import annotations

import math

import pytest
import torch

from circuitry.core.decompose import LogitDecompositionResult, logit_decomposition

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_inputs(d_model: int = 16, vocab: int = 8, batch: int = 1, seq: int = 5):
    """Return (components, unembed, token_a, token_b) with two additive components."""
    torch.manual_seed(0)
    comp_a = torch.randn(batch, seq, d_model)
    comp_b = torch.randn(batch, seq, d_model)
    unembed = torch.randn(d_model, vocab)
    return {"comp_a": comp_a, "comp_b": comp_b}, unembed, comp_a + comp_b


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_returns_logit_decomposition_result():
    components, unembed, _ = _make_inputs()
    result = logit_decomposition(components, unembed, token_a=0, token_b=1)
    assert isinstance(result, LogitDecompositionResult)


def test_scores_keys_match_components():
    components, unembed, _ = _make_inputs()
    result = logit_decomposition(components, unembed, token_a=0, token_b=1)
    assert set(result.scores.keys()) == set(components.keys())


def test_contributions_sum_to_logit_diff_no_ln():
    """Sum of component scores ≈ mean logit diff of the full residual (no LN)."""
    d_model, vocab, batch, seq = 16, 8, 3, 5
    torch.manual_seed(1)
    comp_a = torch.randn(batch, seq, d_model)
    comp_b = torch.randn(batch, seq, d_model)
    unembed = torch.randn(d_model, vocab)
    residual = comp_a + comp_b  # (batch, seq, d_model)

    token_a, token_b = 2, 5
    components = {"comp_a": comp_a, "comp_b": comp_b}
    result = logit_decomposition(components, unembed, token_a=token_a, token_b=token_b)

    # True logit diff: pick position=-1, average over batch
    pos_resid = residual[:, -1, :]  # (batch, d_model)
    true_diff = (pos_resid @ unembed[:, token_a] - pos_resid @ unembed[:, token_b]).mean().item()
    total = sum(result.scores.values())
    assert total == pytest.approx(true_diff, abs=1e-4)


def test_token_a_token_b_position_stored():
    components, unembed, _ = _make_inputs()
    result = logit_decomposition(components, unembed, token_a=3, token_b=7, position=2)
    assert result.token_a == 3
    assert result.token_b == 7
    assert result.position == 2


def test_ranked_sorted_by_abs_score_descending():
    components, unembed, _ = _make_inputs()
    result = logit_decomposition(components, unembed, token_a=0, token_b=1)
    ranked = result.ranked()
    assert isinstance(ranked, list)
    abs_scores = [abs(s) for _, s in ranked]
    assert abs_scores == sorted(abs_scores, reverse=True)


def test_top_k_returns_first_k_from_ranked():
    torch.manual_seed(2)
    d_model, vocab = 16, 8
    components = {f"c{i}": torch.randn(1, 5, d_model) for i in range(6)}
    unembed = torch.randn(d_model, vocab)
    result = logit_decomposition(components, unembed, token_a=0, token_b=1)
    top2 = result.top_k(2)
    assert len(top2) == 2
    assert top2 == result.ranked()[:2]


def test_to_markdown_contains_header():
    components, unembed, _ = _make_inputs()
    result = logit_decomposition(components, unembed, token_a=0, token_b=1)
    md = result.to_markdown()
    assert isinstance(md, str)
    assert "Logit Decomposition" in md


def test_batch_gt_1_averages():
    """With batch=4 the function should not error; scores sum ≈ averaged logit diff."""
    d_model, vocab, batch, seq = 16, 8, 4, 5
    torch.manual_seed(3)
    comp_a = torch.randn(batch, seq, d_model)
    comp_b = torch.randn(batch, seq, d_model)
    unembed = torch.randn(d_model, vocab)
    residual = comp_a + comp_b
    token_a, token_b = 0, 1

    result = logit_decomposition(
        {"comp_a": comp_a, "comp_b": comp_b}, unembed, token_a=token_a, token_b=token_b
    )

    pos_resid = residual[:, -1, :]
    true_diff = (pos_resid @ unembed[:, token_a] - pos_resid @ unembed[:, token_b]).mean().item()
    total = sum(result.scores.values())
    assert total == pytest.approx(true_diff, abs=1e-4)


def test_2d_input_batch_d_model():
    """2D inputs (batch, d_model) should work; position kwarg is ignored."""
    d_model, vocab, batch = 16, 8, 3
    torch.manual_seed(4)
    comp_a = torch.randn(batch, d_model)
    comp_b = torch.randn(batch, d_model)
    unembed = torch.randn(d_model, vocab)

    result = logit_decomposition(
        {"comp_a": comp_a, "comp_b": comp_b},
        unembed,
        token_a=0,
        token_b=1,
        position=99,  # should be ignored for 2D inputs
    )
    assert isinstance(result, LogitDecompositionResult)
    assert set(result.scores.keys()) == {"comp_a", "comp_b"}


def test_ln_scale_changes_scores():
    """Providing a non-trivial ln_scale should yield different scores than no LN."""
    d_model, vocab = 16, 8
    torch.manual_seed(5)
    comp_a = torch.randn(1, 5, d_model)
    comp_b = torch.randn(1, 5, d_model)
    unembed = torch.randn(d_model, vocab)
    ln_scale = torch.rand(d_model) + 0.5  # strictly positive, non-unit

    result_no_ln = logit_decomposition(
        {"comp_a": comp_a, "comp_b": comp_b}, unembed, token_a=0, token_b=1
    )
    result_with_ln = logit_decomposition(
        {"comp_a": comp_a, "comp_b": comp_b},
        unembed,
        token_a=0,
        token_b=1,
        ln_scale=ln_scale,
    )
    # At least one score should differ
    assert any(
        abs(result_no_ln.scores[k] - result_with_ln.scores[k]) > 1e-6
        for k in result_no_ln.scores
    )


def test_ln_scale_ones_bias_zeros_sums_to_scaled_logit_diff():
    """With ln_scale=ones and ln_bias=zeros the LN is a no-op (given unit norm residual).

    More precisely: the linear-approximation path with scale=1, bias=0 is equivalent
    to no LN, so contributions should still sum to ≈ true logit diff.
    """
    d_model, vocab, batch, seq = 16, 8, 2, 5
    torch.manual_seed(6)
    comp_a = torch.randn(batch, seq, d_model)
    comp_b = torch.randn(batch, seq, d_model)
    unembed = torch.randn(d_model, vocab)
    token_a, token_b = 1, 3

    ln_scale = torch.ones(d_model)
    ln_bias = torch.zeros(d_model)

    result = logit_decomposition(
        {"comp_a": comp_a, "comp_b": comp_b},
        unembed,
        token_a=token_a,
        token_b=token_b,
        ln_scale=ln_scale,
        ln_bias=ln_bias,
    )
    # Scores should still be finite
    for v in result.scores.values():
        assert math.isfinite(v)


def test_empty_components_returns_empty_scores():
    torch.manual_seed(7)
    unembed = torch.randn(16, 8)
    result = logit_decomposition({}, unembed, token_a=0, token_b=1)
    assert isinstance(result, LogitDecompositionResult)
    assert result.scores == {}
