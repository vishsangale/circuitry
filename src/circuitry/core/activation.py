"""Activation-space diagnostics. Pure; CPU-deterministic.

See docs/design.md §4.1 for the contract.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

ArrayLike = torch.Tensor | np.ndarray


@dataclass(frozen=True)
class NormStats:
    mean: float
    std: float
    max: float
    frac_above_k_median: float


def _as_tensor(x: ArrayLike) -> torch.Tensor:
    return torch.as_tensor(x).to(dtype=torch.float32)


def dead_fraction(x: ArrayLike, threshold: float = 0.0) -> float:
    """Fraction of activations at or below ``threshold``."""
    t = _as_tensor(x)
    if t.numel() == 0:
        return 0.0
    return float((t <= threshold).float().mean().item())


def norm_stats(x: ArrayLike, k: float = 3.0) -> NormStats:
    """Per-element norm statistics. ``frac_above_k_median`` is the fraction of
    elements whose absolute value exceeds ``k * median(|x|)`` — a cheap
    heavy-tail indicator.
    """
    t = _as_tensor(x).flatten()
    if t.numel() == 0:
        return NormStats(0.0, 0.0, 0.0, 0.0)
    abs_t = t.abs()
    med = abs_t.median().item()
    return NormStats(
        mean=float(t.mean().item()),
        std=float(t.std(unbiased=False).item()),
        max=float(abs_t.max().item()),
        frac_above_k_median=float((abs_t > k * med).float().mean().item()) if med > 0 else 0.0,
    )


def kurtosis(x: ArrayLike, dim: int | tuple[int, ...] = -1) -> torch.Tensor:
    """Excess kurtosis along ``dim``. Returns a tensor (not a Python float)
    because callers commonly want per-channel kurtosis."""
    t = _as_tensor(x)
    mean = t.mean(dim=dim, keepdim=True)
    centered = t - mean
    var = centered.pow(2).mean(dim=dim)
    m4 = centered.pow(4).mean(dim=dim)
    # Avoid div-by-zero
    out = m4 / var.clamp_min(1e-30).pow(2) - 3.0
    out = torch.where(var > 0, out, torch.zeros_like(out))
    return out


def participation_ratio(x: ArrayLike) -> float:
    """``(sum |x|)^2 / sum(x^2)`` — soft count of "active" units.

    Equals ``n`` when ``|x|`` is uniform, equals 1 when ``x`` is a one-hot.
    """
    t = _as_tensor(x).flatten()
    if t.numel() == 0:
        return 0.0
    num = t.abs().sum().pow(2)
    den = t.pow(2).sum().clamp_min(1e-30)
    return float((num / den).item())


def token_similarity(h: torch.Tensor) -> torch.Tensor:
    """Mean off-diagonal cosine similarity between token hidden states.

    Args:
        h: ``(batch, seq, dim)`` or ``(seq, dim)`` hidden states.

    Returns:
        Scalar mean off-diagonal cosine similarity (averaged over batch).
    """
    if h.dim() == 2:
        h = h.unsqueeze(0)
    normalized = torch.nn.functional.normalize(h, dim=-1)
    gram = torch.matmul(normalized, normalized.transpose(-2, -1))  # (B, S, S)
    seq = gram.shape[-1]
    if seq < 2:
        return torch.tensor(0.0, dtype=h.dtype, device=h.device)
    off_diag_mask = ~torch.eye(seq, dtype=torch.bool, device=h.device)
    off_diag = gram[..., off_diag_mask].view(gram.shape[0], -1)
    return off_diag.mean()


def repr_drift(
    ref: torch.Tensor,
    cur: torch.Tensor,
    method: str = "linear_cka",
    *,
    max_samples: int = 256,
    eps: float = 1e-10,
    seed: int = 0,
) -> float:
    """Representational drift between reference and current activation tensors.

    Parameters
    ----------
    ref, cur:
        Activation tensors with the same shape semantics. Accepted shapes:
        ``(n, d)``, ``(batch, seq, d)``, or any higher-rank tensor; all but
        the last dimension are flattened into a single row dimension.
    method:
        Similarity kernel used to compute drift.  Three options:

        ``"linear_cka"`` (default)
            Linear CKA via column-centering: ``||Xc^T Yc||_F^2 /
            (||Xc^T Xc||_F * ||Yc^T Yc||_F)``.  Invariant to orthogonal
            rotation and isotropic rescaling — so LayerNorm-scale growth and
            uniform LR-warmup scaling do **not** show up as drift.
            Drift = ``1 - CKA``.

        ``"cosine"``
            Mean per-sample cosine similarity between matching rows of
            ``ref`` and ``cur``.  Cheaper (O(n d)) but **not** invariant to
            mean shifts: adding a constant offset to all rows of ``cur``
            changes per-sample directions and therefore changes drift, even
            though the relative geometry is unchanged.  Linear CKA's column-
            centering makes it robust to such offsets.
            Drift = ``(1 - mean_cosine) / 2`` (ranges in ``[0, 1]``:
            identical→0, orthogonal→0.5, anti-correlated→1.0).

        ``"rbf_cka"``
            RBF-kernel CKA with median-heuristic bandwidth (no tunable
            hyperparameter).  Drift = ``1 - CKA``.  More sensitive to
            nonlinear structure than linear CKA; slightly more expensive.

    max_samples:
        Row cap before Gram computation.  CKA is O(n^2 d); subsampling
        ``n`` rows to ``max_samples`` keeps cost bounded.  Rows are drawn
        with a **seeded** generator (``seed``) so two calls with the same
        inputs return the same value (CPU-deterministic).

    Notes on minimum row counts
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ``"linear_cka"`` and ``"rbf_cka"`` require **at least 2 rows** after
    flattening (column-centering collapses a single row to all-zeros).
    A ``ValueError`` is raised if ``n < 2`` for these methods.
    ``"cosine"`` works with any ``n >= 1``.
    eps:
        Denominator guard added to the CKA normalisation to prevent
        division by zero on degenerate (all-zero / constant) layers.
        When both inputs are degenerate the returned drift is ``0.0``
        (convention: a dead layer has not drifted from itself).
    seed:
        Seed for the subsampling generator.  Has no effect when
        ``n_rows <= max_samples`` (no subsampling occurs).

    Returns
    -------
    float
        Drift score in ``[0.0, 1.0]`` where ``0.0`` means identical
        representation and larger values indicate more drift.  The output
        is always finite (never NaN).

    Notes
    -----
    All Gram products are accumulated in **float64** to prevent CKA values
    from floating outside ``[0, 1]`` due to float32 rounding.  The output
    is cast back to a Python ``float`` at the end.

    This function is **pure**: no hooks, no logging, no ``.cuda()`` calls.
    Device is taken from the input tensors.
    """
    _VALID_METHODS = {"linear_cka", "cosine", "rbf_cka"}
    if method not in _VALID_METHODS:
        raise ValueError(f"method must be one of {_VALID_METHODS!r}, got {method!r}")

    # --- reshape to (n_rows, d_features) ---------------------------------
    def _flatten(t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 1:
            return t.unsqueeze(0)  # (1, d)
        return t.reshape(-1, t.shape[-1])  # (n, d)

    ref_2d = _flatten(ref)
    cur_2d = _flatten(cur)

    n = ref_2d.shape[0]
    if n != cur_2d.shape[0]:
        raise ValueError(
            f"ref and cur must have the same number of rows after flattening; "
            f"got {n} vs {cur_2d.shape[0]}"
        )

    if method in {"linear_cka", "rbf_cka"} and n < 2:
        raise ValueError(
            f"linear_cka/rbf_cka require >= 2 rows after flattening; got {n}"
        )

    # --- subsample rows if n > max_samples --------------------------------
    if n > max_samples:
        g = torch.Generator(device=ref_2d.device).manual_seed(seed)
        idx = torch.randperm(n, device=ref_2d.device, generator=g)[:max_samples]
        ref_2d = ref_2d[idx]
        cur_2d = cur_2d[idx]

    if ref_2d.device.type == "mps":
        # MPS has no float64; compute the CKA Gram products on CPU (float64) so
        # representational-drift stays device-deterministic on Apple Silicon.
        ref_2d = ref_2d.cpu()
        cur_2d = cur_2d.cpu()

    # promote to float64 for numerically stable Gram products
    R = ref_2d.to(dtype=torch.float64)
    C = cur_2d.to(dtype=torch.float64)

    if method == "cosine":
        return _drift_cosine(R, C, eps)
    elif method == "linear_cka":
        return _drift_linear_cka(R, C, eps)
    else:  # rbf_cka
        return _drift_rbf_cka(R, C, eps)


def _center_columns(X: torch.Tensor) -> torch.Tensor:
    """Column-center a matrix (subtract column means)."""
    return X - X.mean(dim=0, keepdim=True)


def _frobenius_inner_product(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """||A^T B||_F^2 computed as sum of element-wise products of A^T B."""
    # Equivalent to trace((A^T B)(A^T B)^T) = ||A^T B||_F^2
    # For (n,d) matrices: A^T B is (d,d); cost O(n d^2).
    # Better to compute as ||A^T B||_F^2 = sum((A^T B)**2)
    AtB = A.T @ B  # (d, d)
    return (AtB * AtB).sum()


def _drift_linear_cka(R: torch.Tensor, C: torch.Tensor, eps: float) -> float:
    """Drift = 1 - linear_CKA(R, C).  R and C are float64 (n, d).

    Convention: if both inputs are degenerate (zero/constant), return 0.0
    (no drift — identical degenerate state).  If only one is degenerate,
    return 1.0 (fully drifted).

    Degeneracy is tested scale-invariantly: hsic_xx <= eps * raw_xx where
    raw_xx = ||R^T R||_F^2 (uncentered), so the test is relative to the
    input magnitude rather than an absolute threshold.
    """
    Rc = _center_columns(R)
    Cc = _center_columns(C)

    hsic_xx = _frobenius_inner_product(Rc, Rc)
    hsic_yy = _frobenius_inner_product(Cc, Cc)

    # Scale-invariant degeneracy test: compare centered HSIC to uncentered magnitude.
    raw_xx = _frobenius_inner_product(R, R)
    raw_yy = _frobenius_inner_product(C, C)
    x_degen = float(hsic_xx.item()) <= eps * float(raw_xx.item())
    y_degen = float(hsic_yy.item()) <= eps * float(raw_yy.item())

    if x_degen and y_degen:
        return 0.0
    if x_degen or y_degen:
        return 1.0

    hsic_xy = _frobenius_inner_product(Rc, Cc)
    denom = (hsic_xx * hsic_yy).sqrt()  # both > 0 here; no absolute eps needed
    cka = float((hsic_xy / denom).clamp(0.0, 1.0).item())
    return float(1.0 - cka)


def _drift_cosine(R: torch.Tensor, C: torch.Tensor, eps: float) -> float:
    """Drift = 1 - mean_per_sample_cosine(R, C).  R and C are float64 (n, d).

    Convention: if both inputs are all-zero (zero-norm rows), return 0.0.
    """
    # Compute per-row norms; rows with near-zero norm get cosine = 1.0 (no drift)
    r_norms = R.norm(dim=-1, keepdim=True)  # (n, 1)
    c_norms = C.norm(dim=-1, keepdim=True)  # (n, 1)

    # Rows where either norm is degenerate: treat as identical → cosine = 1.0
    degenerate = (r_norms.squeeze(-1) < 1e-12) | (c_norms.squeeze(-1) < 1e-12)

    r_unit = R / r_norms.clamp_min(1e-12)
    c_unit = C / c_norms.clamp_min(1e-12)

    cos_sim = (r_unit * c_unit).sum(dim=-1)  # (n,)
    # Degenerate rows contribute 1.0 (no drift)
    cos_sim = torch.where(degenerate, torch.ones_like(cos_sim), cos_sim)
    mean_cos = float(cos_sim.mean().clamp(-1.0, 1.0).item())
    return float((1.0 - mean_cos) / 2.0)


def _drift_rbf_cka(R: torch.Tensor, C: torch.Tensor, eps: float) -> float:
    """Drift = 1 - rbf_CKA(R, C).  Bandwidth via median heuristic.  R and C are float64 (n, d)."""
    n = R.shape[0]

    def _rbf_gram(X: torch.Tensor) -> torch.Tensor:
        """RBF Gram matrix K_{ij} = exp(-||x_i - x_j||^2 / (2 sigma^2))
        with sigma^2 = median of pairwise squared distances / 2."""
        # pairwise squared distances: (n, n)
        sq_dists = torch.cdist(X, X, p=2).pow(2)  # (n, n)
        # median heuristic: sigma^2 = median(off-diagonal squared distances) / 2
        off_diag_mask = ~torch.eye(n, dtype=torch.bool, device=X.device)
        off_diag_dists = sq_dists[off_diag_mask]
        if off_diag_dists.numel() == 0 or float(off_diag_dists.median().item()) < 1e-30:
            # degenerate: all points identical → K = all-ones → centered K = 0
            return torch.zeros(n, n, dtype=X.dtype, device=X.device)
        sigma2 = float(off_diag_dists.median().item()) / 2.0
        K = torch.exp(-sq_dists / (2.0 * sigma2))
        return K

    Kx = _rbf_gram(R)
    Ky = _rbf_gram(C)

    # Double-center the Gram matrices (equivalent to HSIC with RBF kernel)
    def _double_center(K: torch.Tensor) -> torch.Tensor:
        row_mean = K.mean(dim=1, keepdim=True)
        col_mean = K.mean(dim=0, keepdim=True)
        grand_mean = K.mean()
        return K - row_mean - col_mean + grand_mean

    Kxc = _double_center(Kx)
    Kyc = _double_center(Ky)

    # HSIC: (1/(n-1)^2) * ||Kxc * Kyc||_F^2  — constant cancels in CKA ratio
    hsic_xx = (Kxc * Kxc).sum()
    hsic_yy = (Kyc * Kyc).sum()

    # Degeneracy guard: RBF Gram entries are in [0,1] so hsic values are bounded
    # by n^2 — compare directly against eps (absolute scale is fine here).
    x_degen = float(hsic_xx.item()) < eps
    y_degen = float(hsic_yy.item()) < eps

    if x_degen and y_degen:
        return 0.0
    if x_degen or y_degen:
        return 1.0

    hsic_xy = (Kxc * Kyc).sum()
    # Both > 0 here; a small eps floor is fine since RBF values are bounded.
    denom = (hsic_xx * hsic_yy).sqrt().clamp_min(eps)
    cka = float((hsic_xy / denom).clamp(0.0, 1.0).item())
    return float(1.0 - cka)


def local_intrinsic_dim(
    acts: ArrayLike,
    *,
    max_samples: int = 2048,
    seed: int = 0,
) -> float:
    """Local intrinsic dimensionality via Two-NN estimator (Levina & Bickel 2004).

    Estimates the dimensionality of the activation manifold from nearest-
    neighbour distance ratios.  Complements ``effective_rank`` (linear/spectral)
    with a nonlinear manifold geometry measure.

    Args:
        acts:        (n, d) activation tensor; any rank — flattened to 2-D.
        max_samples: subsample rows before computing pairwise distances.
        seed:        for reproducible subsampling.

    Returns:
        Estimated intrinsic dimensionality (float >= 0).

    Reference: Levina & Bickel 2004; applied to LLMs in arXiv:2402.18048,
               arXiv:2601.22722.
    """
    t = _as_tensor(acts)
    if t.ndim == 1:
        t = t.unsqueeze(0)
    t = t.reshape(-1, t.shape[-1])
    n = t.shape[0]
    if n < 3:
        raise ValueError(f"local_intrinsic_dim requires >= 3 samples; got {n}")

    if n > max_samples:
        g = torch.Generator(device="cpu").manual_seed(seed)
        idx = torch.randperm(n, generator=g)[:max_samples]
        t = t[idx]
        n = max_samples

    if t.device.type != "cpu":
        t = t.cpu()
    t = t.to(dtype=torch.float64)

    dists = torch.cdist(t, t, p=2)
    dists.fill_diagonal_(float("inf"))
    knn, _ = dists.topk(2, dim=1, largest=False)
    d1, d2 = knn[:, 0], knn[:, 1]

    valid = d1 > 1e-12
    if valid.sum().item() < 3:
        return 1.0
    ratio_log = (d2[valid] / d1[valid].clamp_min(1e-12)).log()
    mean_log = float(ratio_log.mean().item())
    return float(1.0 / mean_log) if mean_log > 1e-10 else float("inf")


def kernel_alignment(
    acts_a: ArrayLike,
    acts_b: ArrayLike,
    *,
    method: str = "cka",
    max_samples: int = 256,
    seed: int = 0,
) -> float:
    """Cross-model representation alignment score.

    Measures how similarly two models represent the same data.  Unlike
    ``repr_drift`` (single-model change over training), this compares two
    different models on the same inputs.

    Args:
        acts_a, acts_b: (n, d) activation tensors from two models on the same data.
        method:         ``'cka'`` — linear CKA in [0, 1]; or
                        ``'mnn'`` — mutual nearest-neighbour overlap in [0, 1].
        max_samples:    subsample to this many rows before computing.
        seed:           for reproducible subsampling.

    Returns:
        Alignment score in [0, 1]; 1.0 = perfectly aligned.

    Reference: Huh et al. ICML 2024 "Platonic Representation Hypothesis"
               arXiv:2405.07987.
    """
    if method not in {"cka", "mnn"}:
        raise ValueError(f"kernel_alignment: method must be 'cka' or 'mnn'; got {method!r}")

    def _prep(x: ArrayLike) -> torch.Tensor:
        t = _as_tensor(x)
        if t.ndim == 1:
            t = t.unsqueeze(0)
        return t.reshape(-1, t.shape[-1])

    a = _prep(acts_a)
    b = _prep(acts_b)
    if a.shape[0] != b.shape[0]:
        raise ValueError(
            f"kernel_alignment: acts_a and acts_b must have the same number of rows; "
            f"got {a.shape[0]} vs {b.shape[0]}"
        )
    n = a.shape[0]
    if n > max_samples:
        g = torch.Generator(device="cpu").manual_seed(seed)
        idx = torch.randperm(n, generator=g)[:max_samples]
        a, b = a[idx], b[idx]

    if a.device.type != "cpu":
        a = a.cpu()
    if b.device.type != "cpu":
        b = b.cpu()
    a64 = a.to(dtype=torch.float64)
    b64 = b.to(dtype=torch.float64)

    if method == "cka":
        return float(1.0 - _drift_linear_cka(a64, b64, eps=1e-10))

    k = min(10, a64.shape[0] - 1)
    if k < 1:
        return 1.0
    da = torch.cdist(a64, a64, p=2)
    db = torch.cdist(b64, b64, p=2)
    da.fill_diagonal_(float("inf"))
    db.fill_diagonal_(float("inf"))
    _, nn_a = da.topk(k, dim=1, largest=False)
    _, nn_b = db.topk(k, dim=1, largest=False)
    overlap = sum(
        len(set(nn_a[i].tolist()) & set(nn_b[i].tolist()))
        for i in range(a64.shape[0])
    )
    return float(overlap / (a64.shape[0] * k))


def embedding_uniformity(
    E: ArrayLike,
    *,
    n_samples: int = 2048,
    seed: int = 0,
) -> float:
    """Mean pairwise cosine similarity in a random sample of embedding rows.

    High values (→ 1.0) indicate embedding collapse (all items similar).
    Low values (→ 0.0) indicate diverse, well-distributed embeddings.

    Args:
        E:         (vocab_size, d_model) embedding weight matrix.
        n_samples: number of rows to sample for pairwise computation.
        seed:      for reproducible sampling.

    Returns:
        Mean off-diagonal cosine similarity in [−1, 1].

    Reference: Guo et al. ICML 2024 "Embedding Collapse" arXiv:2310.04400.
    """
    t = _as_tensor(E)
    if t.ndim == 1:
        return 1.0
    E_2d = t.reshape(-1, t.shape[-1]).to(torch.float32)
    n = E_2d.shape[0]
    if n < 2:
        return 1.0
    if n > n_samples:
        g = torch.Generator(device="cpu").manual_seed(seed)
        idx = torch.randperm(n, generator=g)[:n_samples]
        E_2d = E_2d[idx]
    if E_2d.device.type != "cpu":
        E_2d = E_2d.cpu()
    norms = E_2d.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    E_norm = E_2d / norms
    gram = E_norm @ E_norm.T
    n_s = gram.shape[0]
    mask = ~torch.eye(n_s, dtype=torch.bool)
    return float(gram[mask].mean().item())


def gate_stats(x: ArrayLike, eps: float = 1e-6) -> dict[str, float]:
    """Statistics on a post-gate MLP activation tensor.

    For Llama-/Gemma-style gated MLPs the input to ``down_proj`` is
    ``act(gate_proj(x)) * up_proj(x)`` — i.e. the activations the MLP
    actually integrates. ``dead_fraction`` on the MLP output can't tell
    you whether a channel was gated off versus computed-and-discarded;
    this primitive reads the gated tensor directly.

    Returns a dict so the recorder can fan out ``gate_stats/<sub>``
    scalars without the primitive needing to know about tag naming.

    Keys:
        ``frac_active`` — fraction of entries with ``|x| > eps``.
        ``mean_abs`` — mean of ``|x|`` over all entries.
        ``std`` — standard deviation of ``x`` (unbiased=False).
    """
    t = _as_tensor(x).detach().to(torch.float32)
    abs_t = t.abs()
    return {
        "frac_active": float((abs_t > eps).float().mean().item()),
        "mean_abs": float(abs_t.mean().item()),
        "std": float(t.std(unbiased=False).item()),
    }
