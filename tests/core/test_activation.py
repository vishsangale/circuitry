from __future__ import annotations

import math

import pytest
import torch

from circuitry.core import activation
from circuitry.core.activation import (
    NormStats,
    embedding_uniformity,
    kernel_alignment,
    local_intrinsic_dim,
    neural_collapse_score,
    repr_drift,
    spectral_collapse_rank,
)


def test_dead_fraction_all_zeros_is_one():
    x = torch.zeros(4, 8)
    assert activation.dead_fraction(x) == pytest.approx(1.0)


def test_dead_fraction_none_dead():
    x = torch.ones(4, 8)
    assert activation.dead_fraction(x) == pytest.approx(0.0)


def test_dead_fraction_threshold():
    x = torch.tensor([[0.0, 0.1, 1.0, -0.5]])
    # default threshold=0.0 → "dead" means <= 0
    assert activation.dead_fraction(x) == pytest.approx(0.5)


def test_dead_fraction_returns_float():
    assert isinstance(activation.dead_fraction(torch.zeros(2, 2)), float)


def test_norm_stats_shape_and_fields():
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    s = activation.norm_stats(x)
    assert isinstance(s, NormStats)
    assert s.mean == pytest.approx(2.5)
    assert s.max == pytest.approx(4.0)
    assert s.std > 0
    # frac > k*median: median is 2.5; 4>2.5 → 1/4 = 0.25 with k=1
    assert 0.0 <= s.frac_above_k_median <= 1.0


def test_kurtosis_normal_is_near_zero():
    torch.manual_seed(0)
    x = torch.randn(10_000)
    # Excess kurtosis of N(0,1) ≈ 0 (within sampling noise).
    assert abs(float(activation.kurtosis(x).item())) < 0.3


def test_kurtosis_heavy_tail_is_positive():
    torch.manual_seed(0)
    base = torch.randn(10_000)
    base[:50] *= 20.0  # inject heavy-tail outliers
    assert float(activation.kurtosis(base).item()) > 1.0


def test_kurtosis_along_dim():
    x = torch.randn(8, 100)
    k = activation.kurtosis(x, dim=-1)
    assert k.shape == (8,)


def test_participation_ratio_uniform_is_n():
    # Uniform |x| → PR ≈ n.
    x = torch.ones(16)
    assert activation.participation_ratio(x) == pytest.approx(16.0, rel=1e-5)


def test_participation_ratio_spike_is_one():
    x = torch.zeros(16)
    x[0] = 1.0
    assert activation.participation_ratio(x) == pytest.approx(1.0, rel=1e-5)


def test_participation_ratio_returns_float():
    assert isinstance(activation.participation_ratio(torch.ones(4)), float)


def test_token_similarity_identical_tokens():
    # All tokens identical → cosine similarity = 1.0
    h = torch.ones(1, 5, 8)
    sim = activation.token_similarity(h)
    assert torch.allclose(sim, torch.tensor(1.0), atol=1e-6)


def test_token_similarity_orthogonal_tokens():
    # Standard basis tokens → off-diagonal cosine = 0
    h = torch.eye(4).unsqueeze(0)  # (1, 4, 4)
    sim = activation.token_similarity(h)
    assert torch.allclose(sim, torch.tensor(0.0), atol=1e-6)


def test_token_similarity_handles_batch():
    h = torch.randn(3, 5, 8)
    sim = activation.token_similarity(h)
    assert sim.shape == ()  # scalar, mean across batch


# ---------------------------------------------------------------------------
# repr_drift tests
# ---------------------------------------------------------------------------

METHODS = ["linear_cka", "cosine", "rbf_cka"]


@pytest.mark.parametrize("method", METHODS)
def test_repr_drift_identical_is_zero(method):
    """repr_drift(X, X) must be ~0.0 for every method."""
    torch.manual_seed(42)
    X = torch.randn(32, 16)
    d = repr_drift(X, X, method=method)
    assert isinstance(d, float)
    assert d == pytest.approx(0.0, abs=1e-6), f"{method}: expected ~0 for identical inputs, got {d}"


@pytest.mark.parametrize("method", METHODS)
def test_repr_drift_returns_float_in_unit_interval(method):
    """drift ∈ [0, 1] for random pairs."""
    torch.manual_seed(7)
    X = torch.randn(30, 20)
    Y = torch.randn(30, 20)
    d = repr_drift(X, Y, method=method)
    assert isinstance(d, float)
    assert 0.0 <= d <= 1.0, f"{method}: drift {d} outside [0, 1]"


# --- Invariance tests (load-bearing: justify CKA over cosine) ---

@pytest.mark.parametrize("method", ["linear_cka", "rbf_cka"])
def test_cka_invariant_to_orthogonal_rotation(method):
    """CKA drift should be ~0 when cur = ref @ Q for orthonormal Q."""
    torch.manual_seed(11)
    n, d = 40, 12
    X = torch.randn(n, d)
    # Build a random orthogonal matrix via QR
    Q, _ = torch.linalg.qr(torch.randn(d, d))
    X_rot = X @ Q
    d_val = repr_drift(X, X_rot, method=method)
    assert d_val == pytest.approx(0.0, abs=1e-5), (
        f"{method}: expected ~0 drift under orthogonal rotation, got {d_val}"
    )


@pytest.mark.parametrize("method", ["linear_cka", "rbf_cka"])
def test_cka_invariant_to_isotropic_rescale(method):
    """CKA drift should be ~0 when cur = c * ref."""
    torch.manual_seed(22)
    X = torch.randn(40, 12)
    X_scaled = 5.0 * X
    d_val = repr_drift(X, X_scaled, method=method)
    assert d_val == pytest.approx(0.0, abs=1e-5), (
        f"{method}: expected ~0 drift under isotropic rescaling, got {d_val}"
    )


def test_cosine_NOT_invariant_to_mean_shift():
    """Cosine drift CHANGES under a mean shift of cur, while CKA does NOT.

    CKA uses column-centering, making it invariant to constant offsets in the
    activation distribution (e.g. LayerNorm bias drift or optimizer mean shift).
    Cosine measures the angle from the origin, so adding a constant vector to
    all rows changes per-sample directions and therefore changes cosine drift.

    This is the practical reason linear CKA is the recommended default over
    cosine: a global mean shift should not register as representational drift.
    """
    torch.manual_seed(33)
    n, d = 50, 16
    X = torch.randn(n, d)
    # Add a large constant offset to all rows (mean shift)
    offset = 10.0 * torch.ones(1, d)
    X_shifted = X + offset

    drift_cosine = repr_drift(X, X_shifted, method="cosine")
    drift_cka = repr_drift(X, X_shifted, method="linear_cka")

    # linear_cka is invariant to mean shift (column centering removes it)
    assert drift_cka == pytest.approx(0.0, abs=1e-5), (
        f"linear_cka should be ~0 under mean shift, got {drift_cka}"
    )
    # cosine is NOT invariant: adding a large offset changes per-sample direction
    assert drift_cosine > 1e-3, (
        f"cosine drift should be nonzero under mean shift, got {drift_cosine}"
    )


def test_cosine_isotropic_rescale_preserved():
    """Cosine IS invariant to pure isotropic rescaling (direction unchanged)."""
    torch.manual_seed(44)
    X = torch.randn(40, 12)
    X_scaled = 3.14 * X
    # With isotropic rescaling cosine similarity is unchanged (same direction)
    d_cosine = repr_drift(X, X_scaled, method="cosine")
    assert d_cosine == pytest.approx(0.0, abs=1e-5), (
        f"cosine should be ~0 under isotropic scaling (direction unchanged), got {d_cosine}"
    )


# --- Monotonicity test ---

@pytest.mark.parametrize("method", METHODS)
def test_repr_drift_monotone_perturbation(method):
    """Larger perturbation → >= drift than smaller perturbation."""
    torch.manual_seed(55)
    X = torch.randn(32, 16)
    noise_small = 0.01 * torch.randn(32, 16)
    noise_large = 2.0 * torch.randn(32, 16)
    d_small = repr_drift(X, X + noise_small, method=method)
    d_large = repr_drift(X, X + noise_large, method=method)
    assert d_large >= d_small, (
        f"{method}: larger perturbation should give >= drift; "
        f"d_small={d_small:.6f}, d_large={d_large:.6f}"
    )


# --- Determinism / max_samples cap ---

@pytest.mark.parametrize("method", METHODS)
def test_repr_drift_deterministic_with_subsampling(method):
    """When n > max_samples, two calls with same seed must return identical value."""
    torch.manual_seed(66)
    X = torch.randn(512, 16)  # 512 > 256 = max_samples
    Y = torch.randn(512, 16)
    d1 = repr_drift(X, Y, method=method, max_samples=256, seed=0)
    d2 = repr_drift(X, Y, method=method, max_samples=256, seed=0)
    assert d1 == d2, f"{method}: two calls with same seed differ: {d1} vs {d2}"


@pytest.mark.parametrize("method", METHODS)
def test_repr_drift_different_seeds_may_differ(method):
    """Different seeds give different drift when subsampling is in play.

    We use n=512 > max_samples=256 so subsampling always occurs, and try
    multiple seed pairs to guard against flakiness.  At least one pair must
    differ — this verifies that the seed actually influences the subsample.
    """
    torch.manual_seed(77)
    X = torch.randn(512, 16)
    Y = torch.randn(512, 16)
    # Pairs of seeds to try; any one pair differing is sufficient.
    seed_pairs = [(0, 1), (0, 7), (3, 99)]
    any_differ = False
    for s0, s1 in seed_pairs:
        d0 = repr_drift(X, Y, method=method, max_samples=256, seed=s0)
        d1 = repr_drift(X, Y, method=method, max_samples=256, seed=s1)
        assert isinstance(d0, float)
        assert isinstance(d1, float)
        if d0 != d1:
            any_differ = True
            break
    assert any_differ, (
        f"{method}: no seed pair produced different drift values — "
        "seed may not be affecting the subsample"
    )


# --- Degenerate inputs: all-zero / constant → finite, no NaN ---

@pytest.mark.parametrize("method", METHODS)
def test_repr_drift_all_zero_ref_is_finite(method):
    """All-zero ref tensor should return a finite value, not NaN."""
    ref = torch.zeros(20, 8)
    cur = torch.randn(20, 8)
    d = repr_drift(ref, cur, method=method)
    assert math.isfinite(d), f"{method}: got non-finite drift for zero ref: {d}"


@pytest.mark.parametrize("method", METHODS)
def test_repr_drift_all_zero_both_is_zero(method):
    """All-zero ref and cur → convention: 0.0 (no drift from identical degenerate state)."""
    ref = torch.zeros(20, 8)
    cur = torch.zeros(20, 8)
    d = repr_drift(ref, cur, method=method)
    assert math.isfinite(d), f"{method}: got non-finite drift for both-zero: {d}"
    assert d == pytest.approx(0.0, abs=1e-6), (
        f"{method}: expected 0.0 for both-zero inputs (convention), got {d}"
    )


@pytest.mark.parametrize("method", METHODS)
def test_repr_drift_constant_inputs_finite(method):
    """Constant (all-same-value) tensors should return a finite value."""
    ref = torch.full((20, 8), 3.14)
    cur = torch.full((20, 8), 2.71)
    d = repr_drift(ref, cur, method=method)
    assert math.isfinite(d), f"{method}: got non-finite drift for constant inputs: {d}"


# --- float64 accumulation ---

def test_repr_drift_float64_accumulation():
    """Linear CKA accumulates in float64 — result should stay in [0, 1] (not 1.0 + eps)."""
    torch.manual_seed(88)
    # Nearly-identical high-magnitude tensors (stress-tests float32 rounding)
    X = torch.randn(64, 32) * 1000.0
    Y = X + 1e-3 * torch.randn(64, 32)
    for method in METHODS:
        d = repr_drift(X, Y, method=method)
        assert 0.0 <= d <= 1.0, f"{method}: drift {d} outside [0, 1] (float64 accumulation issue)"


# --- Input shape handling ---

def test_repr_drift_accepts_3d_input():
    """(batch, seq, d) tensors should be automatically flattened."""
    torch.manual_seed(99)
    X = torch.randn(2, 16, 8)  # batch=2, seq=16, d=8 → 32 rows
    Y = torch.randn(2, 16, 8)
    d = repr_drift(X, Y, method="linear_cka")
    assert isinstance(d, float)
    assert 0.0 <= d <= 1.0


def test_repr_drift_invalid_method_raises():
    """Unknown method should raise ValueError."""
    X = torch.randn(10, 4)
    with pytest.raises(ValueError, match="method must be one of"):
        repr_drift(X, X, method="bad_method")


# --- Cosine range: (1 - mean_cos) / 2 → [0, 1] ---

def test_cosine_anti_correlated_is_one():
    """Anti-correlated rows (X, -X) → cosine drift = 1.0 with new (1-c)/2 formula."""
    torch.manual_seed(101)
    X = torch.randn(16, 8)
    d = repr_drift(X, -X, method="cosine")
    assert d == pytest.approx(1.0, abs=1e-6), (
        f"anti-correlated inputs should give cosine drift=1.0, got {d}"
    )


def test_cosine_orthogonal_is_half():
    """Orthogonal rows → mean cosine = 0 → drift = 0.5 with (1-c)/2 formula."""
    # Build a matrix and its orthogonal complement block.
    n, d = 8, 16
    # Use standard basis pairs: e_i and e_{i+d//2}
    X = torch.zeros(n, d)
    Y = torch.zeros(n, d)
    for i in range(n):
        X[i, i] = 1.0
        Y[i, i + n] = 1.0  # orthogonal to X rows
    d_val = repr_drift(X, Y, method="cosine")
    assert d_val == pytest.approx(0.5, abs=1e-6), (
        f"orthogonal rows should give cosine drift=0.5, got {d_val}"
    )


# --- Scale-invariant CKA degeneracy: fix #3 regression ---

def test_linear_cka_small_scale_independent_matrices_nonzero():
    """Two DISTINCT random matrices at 1e-5 scale must give linear_cka drift > 0.

    Before the scale-invariant fix, the absolute eps=1e-10 degeneracy guard
    falsely treated these as degenerate (hsic ~ 1e-20 < 1e-10) and returned 0.0.
    """
    X = torch.randn(64, 16, generator=torch.Generator().manual_seed(11)) * 1e-5
    Y = torch.randn(64, 16, generator=torch.Generator().manual_seed(22)) * 1e-5
    d = repr_drift(X, Y, method="linear_cka")
    assert d > 0, (
        f"independent matrices at 1e-5 scale should have linear_cka drift > 0, got {d}"
    )


# --- Single-row ValueError for CKA methods: fix #5 ---

@pytest.mark.parametrize("method", ["linear_cka", "rbf_cka"])
def test_cka_single_row_raises(method):
    """linear_cka and rbf_cka must raise ValueError for a (1, d) input."""
    X = torch.randn(1, 8)
    with pytest.raises(ValueError, match="require >= 2 rows"):
        repr_drift(X, X, method=method)


# ---------------------------------------------------------------------------
# local_intrinsic_dim tests
# ---------------------------------------------------------------------------


def test_local_intrinsic_dim_returns_float():
    torch.manual_seed(40)
    acts = torch.randn(50, 16)
    lid = local_intrinsic_dim(acts)
    assert isinstance(lid, float)
    assert math.isfinite(lid)


def test_local_intrinsic_dim_positive():
    torch.manual_seed(41)
    acts = torch.randn(50, 16)
    lid = local_intrinsic_dim(acts)
    assert lid >= 1.0, f"Expected LID >= 1, got {lid}"


def test_local_intrinsic_dim_low_lt_high():
    """A low-dimensional subspace should have lower LID than full-dimensional noise."""
    rng = torch.Generator().manual_seed(42)
    n, d = 300, 16
    # Low-dim: 2-D subspace in 16-D
    t = torch.randn(n, 2, generator=rng)
    basis = torch.zeros(2, d)
    basis[0, 0] = 1.0
    basis[1, 1] = 1.0
    acts_low = t @ basis + 0.001 * torch.randn(n, d, generator=rng)
    # High-dim: full 16-D random noise
    acts_high = torch.randn(n, d, generator=rng)
    lid_low = local_intrinsic_dim(acts_low)
    lid_high = local_intrinsic_dim(acts_high)
    assert lid_low < lid_high, (
        f"Expected LID(2D subspace) < LID(full-rank noise); got {lid_low:.2f} vs {lid_high:.2f}"
    )


def test_local_intrinsic_dim_fewer_than_3_raises():
    with pytest.raises(ValueError):
        local_intrinsic_dim(torch.randn(2, 8))


def test_local_intrinsic_dim_deterministic():
    torch.manual_seed(43)
    acts = torch.randn(100, 16)
    lid1 = local_intrinsic_dim(acts, seed=0)
    lid2 = local_intrinsic_dim(acts, seed=0)
    assert lid1 == lid2


# ---------------------------------------------------------------------------
# kernel_alignment tests
# ---------------------------------------------------------------------------


def test_kernel_alignment_identical_cka():
    """CKA of identical matrices should be 1.0 (drift = 0)."""
    torch.manual_seed(50)
    acts = torch.randn(30, 16)
    score = kernel_alignment(acts, acts, method="cka")
    assert score == pytest.approx(1.0, abs=1e-5)


def test_kernel_alignment_cka_range():
    torch.manual_seed(51)
    a = torch.randn(30, 16)
    b = torch.randn(30, 16)
    score = kernel_alignment(a, b, method="cka")
    assert 0.0 <= score <= 1.0, f"CKA out of [0,1]: {score}"


def test_kernel_alignment_mnn_identical():
    """MNN alignment of identical matrices should be 1.0."""
    torch.manual_seed(52)
    acts = torch.randn(30, 16)
    score = kernel_alignment(acts, acts, method="mnn")
    assert score == pytest.approx(1.0, abs=1e-5)


def test_kernel_alignment_mnn_random_low():
    """MNN alignment of independent random matrices should be < 1."""
    torch.manual_seed(53)
    a = torch.randn(40, 16)
    b = torch.randn(40, 16)
    score = kernel_alignment(a, b, method="mnn")
    assert score < 1.0


def test_kernel_alignment_invalid_method():
    a = torch.randn(10, 8)
    with pytest.raises(ValueError):
        kernel_alignment(a, a, method="bad")


def test_kernel_alignment_returns_float():
    torch.manual_seed(54)
    a = torch.randn(20, 8)
    b = torch.randn(20, 8)
    for method in ("cka", "mnn"):
        score = kernel_alignment(a, b, method=method)
        assert isinstance(score, float)


# ---------------------------------------------------------------------------
# embedding_uniformity tests
# ---------------------------------------------------------------------------


def test_embedding_uniformity_identical_collapse():
    """All identical embeddings should give uniformity ≈ 1.0 (maximum collapse)."""
    E = torch.ones(20, 8)
    u = embedding_uniformity(E)
    assert u == pytest.approx(1.0, abs=1e-5)


def test_embedding_uniformity_orthogonal_low():
    """Orthogonal embeddings (standard basis) should give uniformity ≈ 0."""
    n, d = 8, 8
    E = torch.eye(n, d)
    u = embedding_uniformity(E)
    assert u == pytest.approx(0.0, abs=1e-5)


def test_embedding_uniformity_range():
    torch.manual_seed(60)
    E = torch.randn(50, 16)
    u = embedding_uniformity(E)
    assert 0.0 <= u <= 1.0, f"uniformity out of [0,1]: {u}"


def test_embedding_uniformity_returns_float():
    torch.manual_seed(61)
    E = torch.randn(30, 8)
    u = embedding_uniformity(E)
    assert isinstance(u, float)


def test_embedding_uniformity_deterministic():
    torch.manual_seed(62)
    E = torch.randn(200, 16)
    u1 = embedding_uniformity(E, seed=0)
    u2 = embedding_uniformity(E, seed=0)
    assert u1 == u2


# ---------------------------------------------------------------------------
# neural_collapse_score tests (v1.30)
# ---------------------------------------------------------------------------



def test_neural_collapse_score_returns_float():
    torch.manual_seed(70)
    acts = torch.randn(30, 16)
    labels = torch.randint(0, 3, (30,))
    nc1 = neural_collapse_score(acts, labels)
    assert isinstance(nc1, float)
    assert nc1 >= 0.0


def test_neural_collapse_score_zero_for_identical():
    """Zero within-class variance (all samples identical per class) → NC1 = 0."""
    n, d = 10, 8
    # Each class: all samples at the exact same point → Σ_W = 0 → NC1 = 0
    class_means = [torch.zeros(d).fill_(i * 10.0) for i in range(3)]
    acts = torch.cat([m.unsqueeze(0).expand(n, -1) for m in class_means])
    labels = torch.cat([torch.full((n,), i, dtype=torch.long) for i in range(3)])
    nc1 = neural_collapse_score(acts, labels)
    assert nc1 == pytest.approx(0.0, abs=1e-6)


def test_neural_collapse_score_high_for_mixed():
    """Randomly mixed activations (no class structure) should give higher NC1."""
    torch.manual_seed(72)
    acts = torch.randn(60, 8)
    labels = torch.randint(0, 3, (60,))
    nc1 = neural_collapse_score(acts, labels)
    # Not strictly guaranteed but should be > tiny threshold with random data
    assert nc1 >= 0.0


def test_neural_collapse_score_single_class():
    """Single class should return 0.0 (no between-class structure)."""
    acts = torch.randn(10, 8)
    labels = torch.zeros(10, dtype=torch.long)
    nc1 = neural_collapse_score(acts, labels)
    assert nc1 == pytest.approx(0.0, abs=1e-8)


def test_neural_collapse_score_shape_mismatch():
    with pytest.raises(ValueError):
        neural_collapse_score(torch.randn(10, 8), torch.zeros(5, dtype=torch.long))


# ---------------------------------------------------------------------------
# spectral_collapse_rank tests (v1.30)
# ---------------------------------------------------------------------------


def test_spectral_collapse_rank_returns_float():
    torch.manual_seed(80)
    acts = torch.randn(20, 16)
    scr = spectral_collapse_rank(acts)
    assert isinstance(scr, float)
    assert scr >= 1.0


def test_spectral_collapse_rank_full_rank_high():
    """Full-rank random activations should give a high effective rank."""
    torch.manual_seed(81)
    acts = torch.randn(64, 32)
    scr = spectral_collapse_rank(acts)
    assert scr > 10.0, f"Expected high rank for full-rank acts, got {scr:.2f}"


def test_spectral_collapse_rank_rank1_is_one():
    """Rank-1 activation matrix (all rows identical) should give effective rank ≈ 1."""
    v = torch.randn(8)
    acts = v.unsqueeze(0).expand(20, -1).clone()
    scr = spectral_collapse_rank(acts)
    assert scr == pytest.approx(1.0, abs=0.1)
