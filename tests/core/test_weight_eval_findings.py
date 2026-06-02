"""Regression tests for the v1.7 real-model evaluation findings in ``core/weight``.

Source: ``docs/observations/2026-05-31-real-model-evaluation.md`` (F1, F38).

Each test encodes the *correct* behaviour, so it is RED under the current code and
flips GREEN once the finding is fixed. They are marked ``xfail(strict=True)`` so the
suite stays green today while the fix is pending; when a fix lands the test XPASSes,
strict-xfail turns that into a failure, and the marker must be removed.

To watch them actually fail (reproduce the findings), run with ``--runxfail``:

    .venv/bin/pytest tests/core/test_weight_eval_findings.py --runxfail
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from circuitry.core import weight

# ---------------------------------------------------------------------------
# F1 — SVD-derived weight diagnostics are biased + non-deterministic for any
#      matrix whose smaller dimension exceeds the default max_dim=512 (i.e.
#      every layer of every real LLM). singular_values() random-column-
#      subsamples to 512 columns by default; the recorder never overrides it.
# ---------------------------------------------------------------------------


def test_F1_condition_number_matches_numpy_on_wide_matrix():
    torch.manual_seed(0)
    M = torch.randn(1024, 1024)  # min-dim 1024 > default max_dim 512 -> subsample fires
    expected = float(np.linalg.cond(M.numpy()))
    got = weight.condition_number(M)
    assert got == pytest.approx(expected, rel=0.05), (
        f"condition_number={got:.2f} vs numpy {expected:.2f} -- subsampling biases sigma_min"
    )


def test_F1_condition_number_is_deterministic():
    torch.manual_seed(0)
    M = torch.randn(1024, 1024)
    vals = [weight.condition_number(M) for _ in range(4)]
    assert len({round(v, 6) for v in vals}) == 1, (
        f"condition_number varies run-to-run: {vals}"
    )


def test_F1_effective_rank_matches_full_svd_on_wide_matrix():
    torch.manual_seed(0)
    A = torch.randn(768, 3072)  # min-dim 768 > 512
    s = np.linalg.svd(A.numpy(), compute_uv=False)
    p = s / s.sum()
    true_eff_rank = float(np.exp(-(p * np.log(p)).sum()))
    got = weight.effective_rank(A)
    assert got == pytest.approx(true_eff_rank, rel=0.05), (
        f"effective_rank={got:.1f} vs true {true_eff_rank:.1f}"
    )


def test_F1_heavy_tail_alpha_matches_full_svd_on_wide_matrix():
    torch.manual_seed(0)
    A = torch.randn(768, 3072)
    s = torch.from_numpy(np.linalg.svd(A.numpy(), compute_uv=False))
    true_alpha = weight._heavy_tail_alpha_from_sv(s)
    got = weight.heavy_tail_alpha(A)
    assert got == pytest.approx(true_alpha, rel=0.05), (
        f"heavy_tail_alpha={got:.3f} vs true {true_alpha:.3f}"
    )


def test_F1_condition_number_exposes_max_dim_escape_hatch():
    torch.manual_seed(0)
    M = torch.randn(1024, 1024)
    # Must accept max_dim (full-SVD override) without TypeError.
    got = weight.condition_number(M, max_dim=None)
    assert got == pytest.approx(float(np.linalg.cond(M.numpy())), rel=0.05)


# ---------------------------------------------------------------------------
# F38 — effective_rank / stable_rank on a 3-D batched expert tensor silently
#       flatten the leading (expert) axis into rows -> the reported rank is
#       ~n_experts, not the per-expert rank. The danger class is silent-wrong:
#       no error is raised. Fix is either fail-loud (raise on ndim>2) or a
#       batch-aware result; the assertions below accept EITHER (must not
#       silently collapse to ~n_experts).
#
#       Note for the implementer: conv weights (4-D) legitimately reach these
#       via spectral.rank_trajectory, which relies on the [out, in*kh*kw]
#       flatten. A fail-loud fix must keep that path working (e.g. pre-flatten
#       in rank_trajectory), so do not blanket-reject ndim>2 upstream.
# ---------------------------------------------------------------------------


def test_F38_effective_rank_does_not_collapse_batched_expert_axis():
    n_experts = 8
    torch.manual_seed(0)
    W = torch.randn(n_experts, 256, 256)  # 8 experts, each full-rank 256x256
    try:
        er = weight.effective_rank(W)
    except (ValueError, RuntimeError):
        return  # fail-loud fix is acceptable
    assert er > n_experts + 1, (
        f"effective_rank silently collapsed the expert axis: {er:.2f} ~= n_experts={n_experts}"
    )


def test_F38_stable_rank_does_not_collapse_batched_expert_axis():
    n_experts = 8
    torch.manual_seed(0)
    W = torch.randn(n_experts, 256, 256)
    try:
        sr = weight.stable_rank(W)
    except (ValueError, RuntimeError):
        return  # fail-loud fix is acceptable
    assert sr > n_experts + 1, (
        f"stable_rank silently collapsed the expert axis: {sr:.2f} ~= n_experts={n_experts}"
    )
