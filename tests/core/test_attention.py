"""Tests for induction_score, copy_suppression_score, attention_sink_score,
head_specialization, attention_pattern_entropy."""
from __future__ import annotations

import math

import pytest
import torch

from circuitry.core.attention import (
    attention_pattern_entropy,
    attention_rollout,
    attention_sink_score,
    copy_suppression_score,
    head_specialization,
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


# ---- copy_suppression_score --------------------------------------------------

def test_perfect_copy_suppression_head_scores_one():
    """A head that attends 1.0 to the same-token position T+i→i scores 1.0."""
    n = 8
    seq = 2 * n
    n_heads = 1
    attn = torch.zeros(1, n_heads, seq, seq)
    for t in range(n):
        attn[0, 0, t + n, t] = 1.0  # query T+t → key t (same token)
    scores = copy_suppression_score(attn, seq_len_repeat=n)
    assert scores == pytest.approx([1.0], abs=1e-6)


def test_copy_suppression_uniform_scores_near_one_over_seq():
    """Uniform attention should give ~1/seq, same as induction_score."""
    n = 16
    seq = 2 * n
    n_heads = 4
    attn = torch.full((1, n_heads, seq, seq), 1.0 / seq)
    scores = copy_suppression_score(attn, seq_len_repeat=n)
    expected = 1.0 / seq
    for s in scores:
        assert s == pytest.approx(expected, abs=1e-6)


def test_copy_suppression_distinct_from_induction():
    """A pure induction head (T+i → i+1) should score near 0 on
    copy_suppression_score, and a pure copy-suppression head (T+i → i)
    should score near 0 on induction_score."""
    n = 8
    seq = 2 * n
    n_heads = 2

    attn = torch.zeros(1, n_heads, seq, seq)
    # Head 0: induction pattern (T+i → i+1)
    for t in range(n - 1):
        attn[0, 0, t + n, t + 1] = 1.0
    # Head 1: copy-suppression pattern (T+i → i)
    for t in range(n):
        attn[0, 1, t + n, t] = 1.0

    ind = induction_score(attn, seq_len_repeat=n)
    css = copy_suppression_score(attn, seq_len_repeat=n)

    assert ind[0] == pytest.approx(1.0, abs=1e-6)   # head 0 is induction
    assert ind[1] == pytest.approx(0.0, abs=1e-6)   # head 1 is NOT induction
    assert css[0] == pytest.approx(0.0, abs=1e-6)   # head 0 is NOT copy-suppression
    assert css[1] == pytest.approx(1.0, abs=1e-6)   # head 1 is copy-suppression


def test_copy_suppression_uses_all_t_positions():
    """copy_suppression_score averages over all T positions (induction uses T-1)."""
    n = 4
    seq = 2 * n
    n_heads = 1
    attn = torch.zeros(1, n_heads, seq, seq)
    # Set only the last copy-suppression position (T+n-1 → n-1).
    attn[0, 0, seq - 1, n - 1] = 1.0
    scores = copy_suppression_score(attn, seq_len_repeat=n)
    # The 1.0 at position n-1 contributes 1/n to the mean.
    assert scores[0] == pytest.approx(1.0 / n, abs=1e-6)


def test_copy_suppression_rejects_too_short_seq():
    n = 8
    attn = torch.zeros(1, 1, n, n)  # seq = n, but we need 2 * n
    with pytest.raises(ValueError, match="seq="):
        copy_suppression_score(attn, seq_len_repeat=n)


def test_copy_suppression_rejects_non_square():
    with pytest.raises(ValueError, match="square"):
        copy_suppression_score(torch.zeros(1, 1, 6, 4), seq_len_repeat=3)


def test_copy_suppression_accepts_3d_input():
    n = 4
    seq = 2 * n
    attn = torch.zeros(2, seq, seq)  # (n_heads, seq, seq)
    for t in range(n):
        attn[1, t + n, t] = 1.0  # head 1 is perfect copy-suppression
    scores = copy_suppression_score(attn, seq_len_repeat=n)
    assert len(scores) == 2
    assert scores[1] == pytest.approx(1.0, abs=1e-6)


# ---- head_specialization -----------------------------------------------------

def test_pure_induction_head_classified_as_induction():
    types = head_specialization([0.8], [0.05], [0.1])
    assert types == ["induction"]


def test_pure_copy_suppression_classified():
    types = head_specialization([0.05], [0.6], [0.1])
    assert types == ["copy_suppression"]


def test_pure_sink_classified():
    types = head_specialization([0.05], [0.05], [0.9])
    assert types == ["sink"]


def test_uniform_when_all_below_threshold():
    types = head_specialization([0.1], [0.1], [0.1])
    assert types == ["uniform"]


def test_ambiguous_head_takes_strongest_ratio():
    """Head above both induction (ratio=2.0) and sink (ratio=1.2) thresholds
    should be classified as induction (higher ratio)."""
    # ind=0.8 -> ratio 0.8/0.4=2.0; snk=0.6 -> ratio 0.6/0.5=1.2
    types = head_specialization([0.8], [0.1], [0.6])
    assert types == ["induction"]


def test_multiple_heads_mixed_types():
    types = head_specialization(
        [0.8, 0.05, 0.05],   # head 0: induction
        [0.05, 0.5, 0.05],   # head 1: copy_suppression
        [0.05, 0.05, 0.9],   # head 2: sink
    )
    assert types == ["induction", "copy_suppression", "sink"]


def test_mismatched_lengths_raises():
    with pytest.raises(ValueError, match="same length"):
        head_specialization([0.5, 0.5], [0.5], [0.5, 0.5])


def test_custom_thresholds_respected():
    """A score of 0.3 qualifies as induction only when the threshold is <= 0.3."""
    assert head_specialization([0.3], [0.0], [0.0], induction_threshold=0.4) == ["uniform"]
    assert head_specialization([0.3], [0.0], [0.0], induction_threshold=0.3) == ["induction"]


# ---- attention_sink_score ----------------------------------------------------

def test_perfect_sink_head_scores_one():
    """A head that attends 1.0 to position 0 at every query position scores 1.0."""
    seq, n_heads = 8, 2
    attn = torch.zeros(1, n_heads, seq, seq)
    attn[0, 0, :, 0] = 1.0  # head 0: all attention to position 0
    scores = attention_sink_score(attn)
    assert scores[0] == pytest.approx(1.0, abs=1e-6)
    assert scores[1] == pytest.approx(0.0, abs=1e-6)


def test_uniform_head_scores_one_over_seq():
    """Uniform attention gives score 1/seq for any sink position."""
    seq, n_heads = 8, 3
    attn = torch.full((1, n_heads, seq, seq), 1.0 / seq)
    for s in attention_sink_score(attn):
        assert s == pytest.approx(1.0 / seq, abs=1e-6)


def test_sink_pos_negative_index():
    """sink_pos=-1 selects the last key position."""
    seq, n_heads = 6, 1
    attn = torch.zeros(1, n_heads, seq, seq)
    attn[0, 0, :, seq - 1] = 1.0  # all attention to last position
    scores = attention_sink_score(attn, sink_pos=-1)
    assert scores[0] == pytest.approx(1.0, abs=1e-6)


def test_sink_score_independent_of_other_heads():
    """Only the column at sink_pos matters; other columns are irrelevant."""
    seq, n_heads = 5, 2
    attn = torch.rand(1, n_heads, seq, seq)
    attn[0, 0, :, 0] = 0.0  # head 0: no attention to pos 0
    attn[0, 1, :, 0] = 0.5  # head 1: half attention to pos 0
    scores = attention_sink_score(attn)
    assert scores[0] == pytest.approx(0.0, abs=1e-6)
    assert scores[1] == pytest.approx(0.5, abs=1e-6)


def test_sink_score_accepts_3d_input():
    """(n_heads, seq, seq) input (no batch dim) is handled correctly."""
    seq, n_heads = 4, 2
    attn = torch.zeros(n_heads, seq, seq)
    attn[1, :, 0] = 1.0
    scores = attention_sink_score(attn)
    assert len(scores) == n_heads
    assert scores[0] == pytest.approx(0.0, abs=1e-6)
    assert scores[1] == pytest.approx(1.0, abs=1e-6)


def test_sink_score_batched_mean():
    """Score averages over both batch and query dimensions."""
    seq, n_heads = 4, 1
    # batch item 0: all attention to pos 0 → 1.0 per row
    # batch item 1: no attention to pos 0 → 0.0 per row
    # mean = 0.5
    attn = torch.zeros(2, n_heads, seq, seq)
    attn[0, 0, :, 0] = 1.0
    scores = attention_sink_score(attn)
    assert scores[0] == pytest.approx(0.5, abs=1e-6)


# ---- attention_pattern_entropy -----------------------------------------------

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


# ---------------------------------------------------------------------------
# attention_rollout tests (v1.30)
# ---------------------------------------------------------------------------



def test_attention_rollout_output_shape():
    """Output shape is (B, T)."""
    torch.manual_seed(0)
    B, H, T = 2, 4, 8
    attn = [torch.softmax(torch.randn(B, H, T, T), dim=-1) for _ in range(3)]
    out = attention_rollout(attn)
    assert out.shape == (B, T)


def test_attention_rollout_sums_to_one():
    """Each row of the output should sum to approximately 1."""
    torch.manual_seed(1)
    B, H, T = 1, 2, 6
    attn = [torch.softmax(torch.randn(B, H, T, T), dim=-1) for _ in range(4)]
    out = attention_rollout(attn)
    row_sums = out.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones(B), atol=1e-5)


def test_attention_rollout_uniform_symmetric():
    """Uniform attention → all non-CLS patch positions have equal saliency.

    Adding identity for the residual stream biases the first (CLS) position,
    but all other positions must be symmetric under patch permutation.
    """
    B, H, T = 1, 3, 5
    uniform = torch.full((B, H, T, T), 1.0 / T)
    out = attention_rollout([uniform] * 4)
    patch_saliency = out[0, 1:]  # exclude index 0 (CLS)
    assert torch.allclose(
        patch_saliency,
        patch_saliency[0].expand_as(patch_saliency),
        atol=1e-6,
    )


def test_attention_rollout_with_grads_shape():
    """GMAR with grads should return same shape as uniform rollout."""
    torch.manual_seed(2)
    B, H, T = 2, 3, 7
    attn = [torch.softmax(torch.randn(B, H, T, T), dim=-1) for _ in range(3)]
    grads = [torch.randn(B, H, T, T) for _ in range(3)]
    out_plain = attention_rollout(attn)
    out_gmar = attention_rollout(attn, grads=grads)
    assert out_gmar.shape == out_plain.shape


def test_attention_rollout_gmar_differs_from_plain():
    """GMAR should generally differ from uniform rollout when heads have varying magnitudes."""
    torch.manual_seed(3)
    B, H, T = 1, 4, 6
    attn = [torch.softmax(torch.randn(B, H, T, T), dim=-1) for _ in range(3)]
    # Make one head have much larger gradients → GMAR should be dominated by that head
    grads = [torch.ones(B, H, T, T) for _ in range(3)]
    grads[0] = grads[0].clone()
    grads[0][:, 0] *= 100.0  # head 0 has 100× larger gradient
    out_plain = attention_rollout(attn)
    out_gmar = attention_rollout(attn, grads=grads)
    assert not torch.allclose(out_plain, out_gmar, atol=1e-4)


def test_attention_rollout_empty_raises():
    with pytest.raises(ValueError):
        attention_rollout([])


def test_attention_rollout_3d_input():
    """(H, T, T) input (no batch dim) should be handled."""
    torch.manual_seed(4)
    H, T = 4, 8
    attn = [torch.softmax(torch.randn(H, T, T), dim=-1) for _ in range(2)]
    out = attention_rollout(attn)
    assert out.shape == (1, T)
