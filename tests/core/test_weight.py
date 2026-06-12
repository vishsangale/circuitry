from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from circuitry.core import weight
from circuitry.core.weight import FinetuningDeltaResult, finetuning_delta_svd, update_weight_ratio


def test_singular_values_diagonal():
    W = torch.diag(torch.tensor([3.0, 2.0, 1.0]))
    s = weight.singular_values(W)
    assert torch.allclose(s, torch.tensor([3.0, 2.0, 1.0]))


def test_singular_values_k_truncates():
    W = torch.diag(torch.tensor([3.0, 2.0, 1.0, 0.5]))
    s = weight.singular_values(W, k=2)
    assert s.shape == (2,)
    assert torch.allclose(s, torch.tensor([3.0, 2.0]))


def test_singular_values_max_dim_subsamples():
    # Wide matrix; max_dim should cap SVD cost without hanging.
    torch.manual_seed(0)
    W = torch.randn(64, 2048)
    s = weight.singular_values(W, max_dim=256)
    assert s.shape[0] <= 256


def test_singular_values_accepts_numpy():
    W = np.eye(4, dtype=np.float32)
    s = weight.singular_values(W)
    assert torch.allclose(s, torch.ones(4))


def test_effective_rank_identity_is_n():
    n = 5
    assert weight.effective_rank(torch.eye(n)) == pytest.approx(n, rel=1e-6)


def test_effective_rank_rank1_is_one():
    u = torch.randn(8, 1)
    v = torch.randn(1, 8)
    W = u @ v
    assert weight.effective_rank(W) == pytest.approx(1.0, abs=1e-4)


def test_effective_rank_returns_python_float():
    val = weight.effective_rank(torch.eye(3))
    assert isinstance(val, float)


def test_effective_rank_invariant_under_orthogonal():
    torch.manual_seed(0)
    W = torch.randn(16, 16)
    Q, _ = torch.linalg.qr(torch.randn(16, 16))
    assert weight.effective_rank(W) == pytest.approx(
        weight.effective_rank(Q @ W), rel=1e-5
    )


def test_stable_rank_identity():
    n = 5
    assert weight.stable_rank(torch.eye(n)) == pytest.approx(n, rel=1e-6)


def test_stable_rank_rank1_is_one():
    u = torch.randn(8, 1)
    v = torch.randn(1, 8)
    assert weight.stable_rank(u @ v) == pytest.approx(1.0, abs=1e-4)


def test_stable_rank_returns_float():
    assert isinstance(weight.stable_rank(torch.eye(3)), float)


def test_condition_number_orthogonal_is_one():
    Q, _ = torch.linalg.qr(torch.randn(8, 8))
    assert weight.condition_number(Q) == pytest.approx(1.0, abs=1e-4)


def test_condition_number_diag():
    W = torch.diag(torch.tensor([4.0, 1.0]))
    assert weight.condition_number(W) == pytest.approx(4.0, rel=1e-6)


def test_condition_number_returns_float():
    assert isinstance(weight.condition_number(torch.eye(3)), float)


def test_heavy_tail_alpha_random_matrix():
    # Marchenko-Pastur bulk → alpha is finite and positive.
    torch.manual_seed(0)
    W = torch.randn(64, 64)
    alpha = weight.heavy_tail_alpha(W)
    assert isinstance(alpha, float)
    assert math.isfinite(alpha)
    assert alpha > 0


def test_heavy_tail_alpha_low_rank_is_finite():
    # Constructed power-law tail (rank ~10 in 64-dim space).
    torch.manual_seed(0)
    U = torch.randn(64, 10)
    V = torch.randn(10, 64)
    alpha = weight.heavy_tail_alpha(U @ V)
    assert math.isfinite(alpha)


def test_update_delta_zero_when_unchanged():
    sd = {"w": torch.ones(3, 3)}
    out = weight.update_delta(sd, sd)
    assert out["w"] == 0.0


def test_update_delta_l2_when_shifted():
    sd_now = {"w": torch.tensor([[1.0, 0.0], [0.0, 1.0]])}
    sd_prev = {"w": torch.tensor([[0.0, 0.0], [0.0, 0.0]])}
    out = weight.update_delta(sd_now, sd_prev)
    assert abs(out["w"] - (2.0 ** 0.5)) < 1e-6


def test_relative_update_delta_is_scale_invariant():
    """||ΔW||/||W|| is identical for two matrices that differ only by an overall
    scale — unlike the absolute update_delta."""
    base_now = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    base_prev = torch.tensor([[0.9, 0.0], [0.0, 0.9]])
    small = weight.relative_update_delta({"w": base_now}, {"w": base_prev})
    big = weight.relative_update_delta({"w": base_now * 1000},
                                       {"w": base_prev * 1000})
    assert abs(small["w"] - big["w"]) < 1e-6
    # And the absolute deltas differ by ~1000x, confirming the scale problem.
    abs_small = weight.update_delta({"w": base_now}, {"w": base_prev})["w"]
    abs_big = weight.update_delta({"w": base_now * 1000},
                                  {"w": base_prev * 1000})["w"]
    assert abs(abs_big / abs_small - 1000.0) < 1e-3


def test_relative_update_delta_zero_when_unchanged():
    sd = {"w": torch.ones(3, 3)}
    assert weight.relative_update_delta(sd, sd)["w"] == 0.0


def test_direction_cosine_collinear_updates():
    # prev_prev -> prev: +I; prev -> now: +2I (same direction)
    sd_pp = {"w": torch.zeros(2, 2)}
    sd_p = {"w": torch.eye(2)}
    sd_n = {"w": torch.eye(2) * 3.0}
    out = weight.direction_cosine(sd_n, sd_p, sd_pp)
    assert abs(out["w"] - 1.0) < 1e-6


def test_direction_cosine_opposite_updates():
    sd_pp = {"w": torch.zeros(2, 2)}
    sd_p = {"w": torch.eye(2)}
    sd_n = {"w": torch.zeros(2, 2)}  # update reverses
    out = weight.direction_cosine(sd_n, sd_p, sd_pp)
    assert abs(out["w"] - (-1.0)) < 1e-6


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_update_delta_cross_device():
    # Regression (v1.4.1): the live Recorder holds current weights on GPU but the
    # prior snapshot on CPU; subtracting across devices raised. update_delta must
    # align devices and return a correct norm.
    now = {"w": torch.randn(8, 8, device="cuda")}
    prev = {"w": torch.zeros(8, 8)}  # CPU snapshot, as the Recorder stores it
    out = weight.update_delta(now, prev)
    assert out["w"] == pytest.approx(now["w"].float().norm().item(), rel=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_direction_cosine_cross_device():
    # Regression (v1.4.1): same cross-device guard for direction_cosine.
    now = {"w": torch.randn(8, 8, device="cuda")}
    prev = {"w": torch.zeros(8, 8)}       # CPU
    prev_prev = {"w": torch.ones(8, 8)}   # CPU
    out = weight.direction_cosine(now, prev, prev_prev)
    assert -1.0 - 1e-6 <= out["w"] <= 1.0 + 1e-6


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_singular_values_on_cuda():
    # Regression (v0.9.1): the max_dim subsample built a CPU randperm index and
    # index_select'd a CUDA matrix → device mismatch. 768 > max_dim 512 triggers it.
    W = torch.randn(768, 768, device="cuda")
    s = weight.singular_values(W)
    assert s.numel() > 0
    assert torch.isfinite(s).all()


# ---------------------------------------------------------------------------
# A1 — randperm determinism
# ---------------------------------------------------------------------------

def test_singular_values_seed_is_reproducible():
    """A seeded call on a matrix whose smaller dim > 512 must return identical
    results across two independent calls."""
    torch.manual_seed(99)
    W = torch.randn(700, 1500)  # min(shape)=700 > default max_dim=512 → subsample fires
    s1 = weight.singular_values(W, seed=0)
    s2 = weight.singular_values(W, seed=0)
    assert torch.allclose(s1, s2), "seeded singular_values must be bit-exact reproducible"


def test_singular_values_unseeded_path_runs():
    """seed=None preserves the old unseeded path — it must still run and return
    a valid tensor (we do not assert two unseeded calls differ)."""
    torch.manual_seed(7)
    W = torch.randn(700, 1500)
    s = weight.singular_values(W, seed=None)
    assert s.numel() > 0
    assert torch.isfinite(s).all()


def test_singular_values_different_seeds_may_differ():
    """Two different seeds can produce different singular-value vectors because
    the subsample draws a different set of columns."""
    torch.manual_seed(42)
    W = torch.randn(700, 1500)
    s0 = weight.singular_values(W, seed=0)
    s1 = weight.singular_values(W, seed=1)
    # They are likely different (probabilistically true for large random matrices).
    # We only assert shape consistency and finiteness here — correctness, not equality.
    assert s0.shape == s1.shape
    assert torch.isfinite(s0).all()
    assert torch.isfinite(s1).all()


# ---------------------------------------------------------------------------
# A2 — Gram fast path
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("shape,dtype,rtol", [
    # strongly rectangular, float64 — tight tolerance
    ((32, 1024), torch.float64, 1e-5),
    ((16, 2000), torch.float64, 1e-5),
    # strongly rectangular, float32 — looser tolerance for large singular values
    ((32, 1024), torch.float32, 1e-3),
    ((16, 2000), torch.float32, 1e-3),
])
def test_gram_path_top_sv_match_svdvals(shape, dtype, rtol):
    """use_gram=True top singular values must be close to svdvals reference."""
    torch.manual_seed(0)
    W = torch.randn(*shape, dtype=dtype)
    s_ref = weight.singular_values(W, max_dim=None, use_gram=False)
    s_gram = weight.singular_values(W, max_dim=None, use_gram=True)
    # Compare the top-k values (Gram gives no accuracy guarantee for the tail)
    k = min(shape[0], shape[1], 10)
    assert torch.allclose(s_ref[:k], s_gram[:k], rtol=rtol, atol=0), (
        f"Gram path top-{k} sv mismatch for shape={shape} dtype={dtype}: "
        f"max_err={((s_ref[:k] - s_gram[:k]).abs() / s_ref[:k].clamp_min(1e-30)).max()}"
    )


def test_gram_auto_falls_back_for_near_square():
    """auto must use svdvals for near-square matrices (no aspect-ratio win)."""
    torch.manual_seed(0)
    W = torch.randn(64, 65)  # aspect ratio ~1 → Gram buys nothing → auto = svdvals
    s_auto = weight.singular_values(W, max_dim=None, use_gram="auto")
    s_ref = weight.singular_values(W, max_dim=None, use_gram=False)
    assert torch.allclose(s_auto, s_ref), "auto should match svdvals on near-square matrices"


def test_gram_auto_engages_for_wide_matrix():
    """auto should engage the Gram path for a strongly rectangular matrix."""
    torch.manual_seed(0)
    W = torch.randn(20, 200)  # aspect ratio 10:1 → Gram should engage
    s_auto = weight.singular_values(W, max_dim=None, use_gram="auto")
    s_ref = weight.singular_values(W, max_dim=None, use_gram=False)
    # Values should be close (Gram is numerically faithful for large sv)
    assert torch.allclose(s_auto, s_ref, rtol=1e-4, atol=1e-6), (
        "auto Gram path result not close to svdvals for wide matrix"
    )


def test_gram_auto_falls_back_when_subsampling_applied():
    """auto must fall back to svdvals when max_dim subsampling was applied."""
    torch.manual_seed(0)
    # min(shape)=700 > 512 → subsampling fires → auto must use svdvals
    W = torch.randn(700, 1500)
    s_auto = weight.singular_values(W, seed=0, use_gram="auto")
    s_ref = weight.singular_values(W, seed=0, use_gram=False)
    assert torch.allclose(s_auto, s_ref), (
        "auto should fall back to svdvals when subsampling was applied"
    )


def test_condition_number_unchanged_by_gram():
    """condition_number must always use svdvals (use_gram=False), never the Gram
    path — the Gram squares the condition number and wrecks sigma_min.

    Built on an ILL-conditioned wide matrix (cond ~1e7) where the Gram path
    genuinely disagrees with svdvals, so this test goes red if condition_number
    ever silently switches to the Gram path. (A well-conditioned matrix would
    let both paths agree to ~1e-16 and the test would prove nothing.)
    """
    torch.manual_seed(0)
    k, n = 20, 200
    u = torch.linalg.qr(torch.randn(k, k, dtype=torch.float64))[0]      # (k, k) orthonormal
    v = torch.linalg.qr(torch.randn(n, k, dtype=torch.float64))[0]      # (n, k) orthonormal cols
    s = torch.logspace(0, -7, k, dtype=torch.float64)                   # cond = 1 / 1e-7 = 1e7
    w = (u * s) @ v.T                                                   # (k, n), singular values == s

    cn_svd = weight._condition_number_from_sv(
        weight.singular_values(w, max_dim=None, use_gram=False)
    )
    cn_gram = weight._condition_number_from_sv(
        weight.singular_values(w, max_dim=None, use_gram=True)
    )
    # Precondition (teeth): on this ill-conditioned matrix the Gram path must
    # genuinely differ from svdvals — else the assertion below is vacuous.
    assert not math.isclose(cn_gram, cn_svd, rel_tol=1e-6), (
        "test precondition broken: Gram path should diverge from svdvals on cond~1e7"
    )
    # condition_number must equal the svdvals reference, NOT the Gram result.
    assert weight.condition_number(w) == pytest.approx(cn_svd, rel=1e-9)


def test_gram_degenerate_all_zero_no_nan():
    """Degenerate all-zero matrix must yield finite output (no NaN/inf)."""
    W = torch.zeros(10, 100)
    s = weight.singular_values(W, max_dim=None, use_gram=True)
    assert torch.isfinite(s).all(), "all-zero matrix must not produce NaN/inf singular values"
    assert (s == 0.0).all(), "all-zero matrix singular values must be 0"


def test_gram_degenerate_rank_deficient_no_nan():
    """Rank-deficient matrix with some zero singular values: finite output."""
    torch.manual_seed(0)
    # Rank-5 matrix in a 10×100 space
    U = torch.randn(10, 5)
    V = torch.randn(5, 100)
    W = U @ V
    s = weight.singular_values(W, max_dim=None, use_gram=True)
    assert torch.isfinite(s).all()
    assert s.numel() > 0


def test_spectral_entropy_rank_one_is_near_zero():
    """A rank-1 matrix has one nonzero singular value -> entropy ~0 (tiny fp
    residual singular values keep it from being exactly 0)."""
    W = torch.outer(torch.arange(1.0, 5.0), torch.arange(1.0, 4.0))
    assert weight.spectral_entropy(W) == pytest.approx(0.0, abs=1e-4)


def test_spectral_entropy_flat_spectrum_is_log_n():
    """Identity has all singular values equal -> entropy log(n)."""
    n = 6
    assert weight.spectral_entropy(torch.eye(n)) == pytest.approx(math.log(n), abs=1e-6)


def test_spectral_entropy_equals_log_effective_rank():
    torch.manual_seed(0)
    W = torch.randn(12, 8)
    assert weight.spectral_entropy(W) == pytest.approx(
        math.log(weight.effective_rank(W)), abs=1e-6
    )


def test_spectral_entropy_rejects_3d():
    with pytest.raises(ValueError, match="spectral_entropy"):
        weight.spectral_entropy(torch.randn(2, 3, 4))


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")
def test_singular_values_on_mps_matches_cpu():
    """MPS has no float64; singular_values must offload to CPU and match, not
    crash on the float64 Gram path (surfaced by real-model validation)."""
    torch.manual_seed(0)
    W = torch.randn(128, 32)  # rectangular -> Gram path (float64)
    s_cpu = weight.singular_values(W)
    s_mps = weight.singular_values(W.to("mps"))
    assert torch.allclose(s_cpu, s_mps.cpu(), atol=1e-4)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")
def test_weight_diagnostics_run_on_mps():
    W = torch.randn(96, 96, device="mps")
    assert math.isfinite(weight.effective_rank(W))
    assert math.isfinite(weight.condition_number(W))
    assert math.isfinite(weight.stable_rank(W))
    assert math.isfinite(weight.spectral_entropy(W))
    # strongly-rectangular -> Gram float64 path on CPU
    Wr = torch.randn(256, 16, device="mps")
    assert math.isfinite(weight.effective_rank(Wr))


# ---------------------------------------------------------------------------
# update_weight_ratio tests (v1.30)
# ---------------------------------------------------------------------------



def test_update_weight_ratio_small_update():
    """A small update relative to the base should give a small ratio."""
    torch.manual_seed(0)
    W = torch.randn(16, 16)
    dW = 0.01 * torch.randn(16, 16)
    r = update_weight_ratio(W, W + dW)
    assert isinstance(r, float)
    assert r < 0.1


def test_update_weight_ratio_identity_is_zero():
    """Identical matrices → ratio = 0."""
    W = torch.randn(8, 8)
    assert update_weight_ratio(W, W) == pytest.approx(0.0, abs=1e-8)


def test_update_weight_ratio_scale_invariant():
    """Scaling both W_prev and the update by the same factor leaves ratio unchanged."""
    torch.manual_seed(1)
    W = torch.randn(8, 8)
    dW = 0.1 * torch.randn(8, 8)
    r1 = update_weight_ratio(W, W + dW)
    r2 = update_weight_ratio(5.0 * W, 5.0 * (W + dW))
    assert r1 == pytest.approx(r2, rel=1e-5)


def test_update_weight_ratio_positive():
    torch.manual_seed(2)
    W = torch.randn(4, 4)
    r = update_weight_ratio(W, W + torch.randn(4, 4))
    assert r >= 0.0


# ---------------------------------------------------------------------------
# finetuning_delta_svd tests (v1.30)
# ---------------------------------------------------------------------------


def test_finetuning_delta_svd_returns_result():
    torch.manual_seed(3)
    W_base = torch.randn(16, 8)
    W_ft = W_base + 0.1 * torch.randn(16, 8)
    res = finetuning_delta_svd(W_base, W_ft)
    assert isinstance(res, FinetuningDeltaResult)
    assert isinstance(res.sv_scale_factor, float)
    assert isinstance(res.left_rotation_similarity, float)
    assert isinstance(res.right_rotation_similarity, float)


def test_finetuning_delta_svd_zero_update():
    """Identical base and fine-tuned → scale factor ≈ 0."""
    torch.manual_seed(4)
    W = torch.randn(8, 8)
    res = finetuning_delta_svd(W, W)
    assert res.sv_scale_factor == pytest.approx(0.0, abs=1e-4)


def test_finetuning_delta_svd_lora_like_high_rotation():
    """LoRA update preserves the base directions → both rotation similarities should be > 0."""
    torch.manual_seed(5)
    W_base = torch.randn(16, 8)
    # LoRA-style low-rank update that happens to share directions with W_base
    U, _, Vh = torch.linalg.svd(W_base, full_matrices=False)
    lora = 0.1 * (U[:, :2] @ Vh[:2])
    res = finetuning_delta_svd(W_base, W_base + lora)
    assert res.left_rotation_similarity > 0.0
    assert res.right_rotation_similarity > 0.0


def test_finetuning_delta_svd_similarities_in_unit_interval():
    torch.manual_seed(6)
    W_base = torch.randn(8, 6)
    W_ft = W_base + 0.5 * torch.randn(8, 6)
    res = finetuning_delta_svd(W_base, W_ft)
    assert 0.0 <= res.left_rotation_similarity <= 1.0
    assert 0.0 <= res.right_rotation_similarity <= 1.0
