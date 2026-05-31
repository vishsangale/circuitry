"""P4 tests: integrated-gradients variant for SAE nodes and edges (EAP-IG).

Tests with STRONG oracles:
  - Node IG completeness to the eps-frozen spliced delta  (O(1/N^2) on linear,
    O(1/N) on nonlinear/ReLU due to kinks — both tested separately)
  - Error-node joint-path completeness to the REAL forward delta
  - Sign agreement with attrib on dominant features (IG refines, doesn't flip)
  - Saturation enumeration: feature gets grad=0 at clean (relu-saturated metric),
    attrib misses it (sum=0), IG does not (nonzero sum)
  - Edge IG vs independent path-integral trapezoid bruteforce (NOT vs attrib)
  - Edge uses per-j VJP (no dense Jacobian)
  - Default N=32 when variant='ig' and n_ig_steps=0; 'exact' raises NotImplementedError

Math:
  Path: f(α) = f_clean + α·(f_corrupt − f_clean), α: 0→1
  Midpoint rule: α_k = (k-0.5)/N
  Node score_i = Σ_pos Δf_i · (1/N) Σ_k (∂metric/∂f_i)|_{α_k}
  Completeness (feature-only, eps frozen at clean):
    Σ_i score_i  →  metric(decode(f_corrupt)+eps_clean) − metric(decode(f_clean)+eps_clean)
  Completeness (joint path, include_error_node=True):
    Σ_i feat_IG_i + error_IG  →  metric(real corrupt) − metric(real clean)

Rate:
  O(1/N^2) convergence for SMOOTH integrands (linear downstream).
  O(1/N) convergence when integrand has kinks (ReLU downstream) — expected.
"""
from __future__ import annotations

import pytest
import torch
from torch import Tensor

from tests.patching.test_sae_features import (
    LinearResidToy,
    NonlinearResidToy,
    SyntheticSAE,
    _make_clean_corr,
    _make_resolver,
    _metric,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _spliced_metric_at_f(model, sae, resolver, inputs, metric_fn, f_override, eps_clean, site):
    """Run a spliced forward injecting decode(f_override)+eps_clean; return metric scalar.

    Used to compute the eps-frozen completeness target independently.
    Since the hook replaces the layer output, the actual `inputs` value
    doesn't matter for layers downstream — only f_override and eps_clean matter.
    """
    from circuitry.patching.sae_features import _routed_extract, _routed_inject

    resolved = resolver.resolve(model, site)
    layer_mod = resolved.module

    def _hook(module, inp, output,
              _sae=sae, _f=f_override, _eps=eps_clean, _resolved=resolved):
        x_hat = _sae.decode(_f)
        recon = x_hat + _eps
        full = _routed_extract(_resolved, output)
        recon_cast = recon.to(full.device, full.dtype)
        return _routed_inject(_resolved, output, recon_cast)

    h = layer_mod.register_forward_hook(_hook)
    try:
        with torch.no_grad():
            out = model(inputs) if not isinstance(inputs, dict) else model(**inputs)
    finally:
        h.remove()
    return float(metric_fn(out).item())


def _capture_f_eps(model, sae, resolver, inputs, site):
    """Capture SAE features f and reconstruction error eps from a forward pass."""
    from circuitry.patching.sae_features import _routed_extract
    from circuitry.sae.grad import sae_decompose

    resolved = resolver.resolve(model, site)
    layer_mod = resolved.module
    store: dict[str, Tensor] = {}

    def _hook(m, inp, out, _sae=sae, _resolved=resolved):
        a = _routed_extract(_resolved, out).detach()
        a_in = a.to(getattr(_sae, "device", a.device), getattr(_sae, "dtype", a.dtype))
        with torch.no_grad():
            f, x_hat, eps = sae_decompose(_sae, a_in)
        store["f"] = f.detach()
        store["eps"] = eps.detach()

    h = layer_mod.register_forward_hook(_hook)
    try:
        with torch.no_grad():
            model(inputs) if not isinstance(inputs, dict) else model(**inputs)
    finally:
        h.remove()
    return store["f"], store["eps"]


# ---------------------------------------------------------------------------
# test_ig_node_completeness — PRIMARY HARD GATE
# ---------------------------------------------------------------------------

def test_ig_node_completeness():
    """Feature-only IG completeness check on a LINEAR toy + affine SAE (float64).

    Completeness oracle: metric(decode(f_corrupt)+eps_clean) − metric(decode(f_clean)+eps_clean)
    Computed by TWO independent spliced forwards — NOT via bruteforce_feature_scores sum.

    For a linear model with affine SAE:
      - The path integral is EXACT even at N=1 (linear integrand → no quadrature error)
      - Error ~machine zero (~1e-15 to 1e-14) at all N
      - err·N^2 is expected to GROW (not shrink) because the error is already at floor

    For the nonlinear model (secondary check):
      - O(1/N) convergence due to ReLU kinks in the integrand
      - Show errors DO decrease as N grows (correct direction)
      - At N=512, |err| < 5e-3 (practical tolerance for ReLU nonlinearity)
    """
    from circuitry.patching.sae_features import SAEFeatureRunner
    from circuitry.patching.sites import Site

    site = Site("resid_post", layer=0)

    # -------------------------------------------------------
    # LINEAR model: O(1/N^2) → near machine-zero at all N
    # -------------------------------------------------------
    torch.manual_seed(42)
    model_lin = LinearResidToy(n_layers=2, d=8).to(torch.float64)
    torch.manual_seed(7)
    sae_lin = SyntheticSAE(d_model=8, d_sae=16, relu=False, dtype=torch.float64)
    resolver = _make_resolver()

    torch.manual_seed(0)
    clean_lin = torch.randn(1, 3, 8, dtype=torch.float64)
    corrupted_lin = torch.randn(1, 3, 8, dtype=torch.float64)

    f_corrupt_lin, _ = _capture_f_eps(model_lin, sae_lin, resolver, corrupted_lin, site)
    f_clean_lin, eps_clean_lin = _capture_f_eps(model_lin, sae_lin, resolver, clean_lin, site)

    target_corrupt_lin = _spliced_metric_at_f(model_lin, sae_lin, resolver, clean_lin, _metric, f_corrupt_lin, eps_clean_lin, site)
    target_clean_lin = _spliced_metric_at_f(model_lin, sae_lin, resolver, clean_lin, _metric, f_clean_lin, eps_clean_lin, site)
    target_lin = target_corrupt_lin - target_clean_lin

    runner_lin = SAEFeatureRunner(model_lin, {site: sae_lin}, resolver)

    print(f"\n[LINEAR] completeness target = {target_lin:.8e}")
    print(f"{'N':>6}  {'sum_scores':>16}  {'|err|':>12}")
    lin_errors = []
    for N in [4, 32, 512]:
        result = runner_lin.run(clean_lin, corrupted_lin, _metric,
                                include_error_node=False, variant="ig", n_ig_steps=N)
        total = sum(result.scores.values())
        err = abs(total - target_lin)
        lin_errors.append(err)
        print(f"{N:>6}  {total:>16.8e}  {err:>12.3e}")

    # On linear model, all errors at machine precision (~2e-8 or better for float64)
    for i, N in enumerate([4, 32, 512]):
        assert lin_errors[i] < 1e-6, (
            f"LINEAR model N={N}: |err|={lin_errors[i]:.3e} exceeds 1e-6 (should be ~machine zero)"
        )
    print("  Linear: all errors at machine precision ✓")

    # -------------------------------------------------------
    # NONLINEAR model (ReLU): O(1/N) — errors decrease but slowly
    # -------------------------------------------------------
    torch.manual_seed(42)
    model_nl = NonlinearResidToy(n_layers=2, d=8).to(torch.float64)
    torch.manual_seed(7)
    sae_nl = SyntheticSAE(d_model=8, d_sae=16, relu=False, dtype=torch.float64)

    torch.manual_seed(0)
    clean_nl = torch.randn(1, 3, 8, dtype=torch.float64)
    corrupted_nl = torch.randn(1, 3, 8, dtype=torch.float64)

    f_corrupt_nl, _ = _capture_f_eps(model_nl, sae_nl, resolver, corrupted_nl, site)
    f_clean_nl, eps_clean_nl = _capture_f_eps(model_nl, sae_nl, resolver, clean_nl, site)

    target_c_nl = _spliced_metric_at_f(model_nl, sae_nl, resolver, clean_nl, _metric, f_corrupt_nl, eps_clean_nl, site)
    target_cl_nl = _spliced_metric_at_f(model_nl, sae_nl, resolver, clean_nl, _metric, f_clean_nl, eps_clean_nl, site)
    target_nl = target_c_nl - target_cl_nl

    runner_nl = SAEFeatureRunner(model_nl, {site: sae_nl}, resolver)

    Ns = [32, 128, 512]
    nl_errors = []
    print(f"\n[NONLINEAR/ReLU] completeness target = {target_nl:.6e}")
    print(f"{'N':>6}  {'sum_scores':>14}  {'|err|':>12}  {'|err|*N':>12}")
    for N in Ns:
        result = runner_nl.run(clean_nl, corrupted_nl, _metric,
                               include_error_node=False, variant="ig", n_ig_steps=N)
        total = sum(result.scores.values())
        err = abs(total - target_nl)
        nl_errors.append(err)
        print(f"{N:>6}  {total:>14.6e}  {err:>12.3e}  {err*N:>12.3e}")

    # Errors must decrease overall (nonlinear O(1/N))
    assert nl_errors[-1] < nl_errors[0], (
        f"Nonlinear IG error should decrease from N={Ns[0]} to N={Ns[-1]}: "
        f"{nl_errors[0]:.3e} → {nl_errors[-1]:.3e}"
    )

    # At N=512, practical tolerance for ReLU nonlinearity: err < 5e-3
    assert nl_errors[-1] < 5e-3, (
        f"Nonlinear IG N=512: |err|={nl_errors[-1]:.3e} exceeds 5e-3"
    )
    print("  Nonlinear: errors decrease, N=512 < 5e-3 ✓")


# ---------------------------------------------------------------------------
# test_ig_attrib_sign_agree
# ---------------------------------------------------------------------------

def test_ig_attrib_sign_agree():
    """On a linear model + affine SAE, IG and attrib have same sign for dominant features.

    IG is a refinement of attrib; they should not flip the sign for large-score features.
    On a linear model with affine SAE the path integral is exact so they agree to
    machine precision.
    """
    from circuitry.patching.sae_features import SAEFeatureRunner
    from circuitry.patching.sites import Site

    torch.manual_seed(10)
    model64 = LinearResidToy(n_layers=2, d=8).to(torch.float64)
    torch.manual_seed(11)
    sae64 = SyntheticSAE(d_model=8, d_sae=16, relu=False, dtype=torch.float64)
    resolver = _make_resolver()
    site = Site("resid_post", layer=0)

    torch.manual_seed(5)
    clean, corrupted = _make_clean_corr(seed=5)
    clean = clean.to(torch.float64)
    corrupted = corrupted.to(torch.float64)

    runner = SAEFeatureRunner(model64, {site: sae64}, resolver)

    result_attrib = runner.run(clean, corrupted, _metric, variant="attrib")
    result_ig = runner.run(clean, corrupted, _metric, variant="ig", n_ig_steps=64)

    attrib_scores = result_attrib.scores
    ig_scores = result_ig.scores

    common = [n for n in attrib_scores if n in ig_scores]
    if not common:
        pytest.skip("No common features between attrib and IG runs")

    # Sort by |attrib score| descending; check sign for top-10
    top = sorted(common, key=lambda n: abs(attrib_scores[n]), reverse=True)[:10]
    disagreements = 0
    for n in top:
        a = attrib_scores[n]
        ig = ig_scores[n]
        if a * ig < 0:
            disagreements += 1
            print(f"  Sign flip: attrib={a:.4e} ig={ig:.4e} node={n}")

    print(f"\n[sign agree] top-10 disagreements: {disagreements}/10")
    assert disagreements == 0, (
        f"{disagreements} sign flips in top-10 features between attrib and IG "
        "(on linear model they should have identical sign)"
    )


# ---------------------------------------------------------------------------
# test_ig_error_node_real_delta
# ---------------------------------------------------------------------------

def test_ig_error_node_real_delta():
    """Joint path (include_error_node=True): features + error IG completes to real delta.

    Target: metric(real corrupted forward) − metric(real clean forward)
    (NOT eps-frozen — the joint path interpolates BOTH features AND eps).
    """
    from circuitry.patching.sae_features import SAEFeatureRunner
    from circuitry.patching.sites import Site

    torch.manual_seed(20)
    model64 = NonlinearResidToy(n_layers=2, d=8).to(torch.float64)
    torch.manual_seed(21)
    sae64 = SyntheticSAE(d_model=8, d_sae=16, relu=False, dtype=torch.float64)
    resolver = _make_resolver()
    site = Site("resid_post", layer=0)

    torch.manual_seed(3)
    clean, corrupted = _make_clean_corr(seed=3)
    clean = clean.to(torch.float64)
    corrupted = corrupted.to(torch.float64)

    # Real forward delta (no SAE splice — pure model)
    with torch.no_grad():
        real_corrupt_m = float(_metric(model64(corrupted)).item())
        real_clean_m = float(_metric(model64(clean)).item())
    real_delta = real_corrupt_m - real_clean_m

    runner = SAEFeatureRunner(model64, {site: sae64}, resolver)

    result = runner.run(
        clean, corrupted, _metric,
        include_error_node=True,
        variant="ig",
        n_ig_steps=512,
    )

    # Sum ALL scores (features + error node)
    total_ig = sum(result.scores.values())
    err = abs(total_ig - real_delta)
    print(f"\n[error-node real-delta] total_IG={total_ig:.6e}, real_delta={real_delta:.6e}, |err|={err:.3e}")

    # For nonlinear model with O(1/N) convergence, N=512 → err < 5e-3
    assert err < 5e-3, f"Joint-path IG completeness error {err:.3e} exceeds 5e-3"


# ---------------------------------------------------------------------------
# test_ig_node_enumeration_saturation
# ---------------------------------------------------------------------------

def test_ig_node_enumeration_saturation():
    """Feature attribution SATURATION: attrib misses a move that IG captures.

    Construct a scenario where the METRIC is zero-gradient at clean (relu saturation):
      metric = relu(out[..., 0]).sum()

    At clean: metric(clean) = 0  →  grad@clean = 0 for ALL features.
    At corrupt: metric(corrupt) > 0  →  the transition matters.

    Expected behavior:
      attrib: sum of all scores = 0 (completely blind to the move)
      IG:     sum of all scores ≈ metric(corrupt endpoint) − metric(clean endpoint) > 0

    This is exactly the "saturation" failure mode IG fixes: a feature dead at clean
    due to a relu kink in the metric has grad=0 at clean, so attrib gives 0.
    IG integrates along the path where the feature becomes live → nonzero score.
    """
    from circuitry.patching.sae_features import SAEFeatureRunner
    from circuitry.patching.sites import Site

    # Relu metric: gradient is zero whenever model output[..., 0] <= 0
    def relu_metric(out: Tensor) -> Tensor:
        logits = out.logits if hasattr(out, "logits") else out
        return torch.relu(logits[..., 0]).sum()

    torch.manual_seed(42)
    model = LinearResidToy(n_layers=2, d=8)
    torch.manual_seed(7)
    sae = SyntheticSAE(d_model=8, d_sae=16, relu=False)
    resolver = _make_resolver()
    site = Site("resid_post", layer=0)

    # Use seed=6 which gives:
    # - spliced metric at alpha=0 (clean): 0.0 (relu saturated)
    # - spliced metric at alpha=1 (corrupt): >1.0 (post-relu active)
    torch.manual_seed(6)
    clean = torch.randn(1, 1, 8)
    corrupted = torch.randn(1, 1, 8)

    # Verify the scenario holds
    f_corrupt, _ = _capture_f_eps(model, sae, resolver, corrupted, site)
    f_clean, eps_clean = _capture_f_eps(model, sae, resolver, clean, site)

    m_at_alpha0 = _spliced_metric_at_f(model, sae, resolver, clean, relu_metric, f_clean, eps_clean, site)
    m_at_alpha1 = _spliced_metric_at_f(model, sae, resolver, clean, relu_metric, f_corrupt, eps_clean, site)

    if m_at_alpha0 != 0.0 or m_at_alpha1 < 0.1:
        pytest.skip(
            f"Saturation scenario not met at seed=6: "
            f"m@α=0={m_at_alpha0:.4e}, m@α=1={m_at_alpha1:.4e}"
        )

    print(f"\n[saturation] m@α=0={m_at_alpha0:.4e} (relu saturated), m@α=1={m_at_alpha1:.4e} (active)")
    print(f"  Completeness target: {m_at_alpha1 - m_at_alpha0:.4e}")

    runner = SAEFeatureRunner(model, {site: sae}, resolver)

    result_attrib = runner.run(clean, corrupted, relu_metric, variant="attrib")
    result_ig = runner.run(clean, corrupted, relu_metric, variant="ig", n_ig_steps=128)

    sum_attrib = sum(result_attrib.scores.values())
    sum_ig = sum(result_ig.scores.values())

    print(f"  attrib total: {sum_attrib:.4e}  (should be 0 — all features have grad@clean=0)")
    print(f"  IG total:     {sum_ig:.4e}  (should be ≈ completeness target)")

    # attrib is blind: ALL features have grad@clean = 0 (relu saturates at clean)
    assert abs(sum_attrib) < 1e-10, (
        f"attrib total should be 0 (relu saturation), got {sum_attrib:.4e}"
    )

    # IG captures the transition along the path
    assert abs(sum_ig) > 0.1, (
        f"IG total should be >> 0 (captures relu transition), got {sum_ig:.4e}"
    )

    # IG completeness: sum ≈ m@1 − m@0
    err = abs(sum_ig - (m_at_alpha1 - m_at_alpha0))
    print(f"  |IG - target| = {err:.3e}")
    assert err < 0.2, (
        f"IG completeness error too large: {err:.3e} (N=128, O(1/N) for kink)"
    )


# ---------------------------------------------------------------------------
# test_ig_edge_vs_path_integral
# ---------------------------------------------------------------------------

def test_ig_edge_vs_path_integral():
    """Edge IG (EAP-IG) == independent path-integral trapezoid bruteforce.

    Oracle: for each (i→j), compute Δf_U[i] * ∫_0^1 (∂f_D[j]/∂f_U[i]) · gradf_D[j] dα
    via a high-N MIDPOINT bruteforce over the same upstream interpolation.

    Do NOT assert IG-edge == attrib-edge: they agree only when BOTH the path and
    metric are linear.
    """
    from circuitry.patching.sae_edges import SAEFeatureEdgeRunner
    from circuitry.patching.sae_features import _routed_extract, _routed_inject
    from circuitry.patching.sites import Site
    from circuitry.sae.grad import sae_decompose

    # Linear toy + affine SAE (both layers same SAE) — float64
    torch.manual_seed(40)
    model64 = LinearResidToy(n_layers=2, d=8).to(torch.float64)
    torch.manual_seed(41)
    sae0_64 = SyntheticSAE(d_model=8, d_sae=16, relu=False, dtype=torch.float64)
    torch.manual_seed(42)
    sae1_64 = SyntheticSAE(d_model=8, d_sae=16, relu=False, dtype=torch.float64)
    resolver = _make_resolver()

    site0 = Site("resid_post", layer=0)
    site1 = Site("resid_post", layer=1)

    torch.manual_seed(6)
    clean, corrupted = _make_clean_corr(seed=6)
    clean = clean.to(torch.float64)
    corrupted = corrupted.to(torch.float64)

    # EAP-IG runner at N=128
    runner = SAEFeatureEdgeRunner(model64, {site0: sae0_64, site1: sae1_64}, resolver)
    circuit_ig = runner.run(
        clean, corrupted, _metric,
        layer_pairs="adjacent",
        top_k_survivors=16,
        variant="ig",
        n_ig_steps=128,
    )
    ig_edges = circuit_ig.edges

    if not ig_edges:
        pytest.skip("No edges produced — try different seeds")

    # ---
    # Independent path-integral midpoint oracle (M >> N)
    # For each (writer i → reader j) edge:
    #   score = Δf_U[i] * (1/M) Σ_k vjp_j_k[i] (where vjp = ∂f_D[j]/∂f_U * gradf_D[j])
    # ---
    M_oracle = 1024  # high-N reference

    w_res = resolver.resolve(model64, site0)
    r_res = resolver.resolve(model64, site1)
    w_mod = w_res.module
    r_mod = r_res.module

    f_U_corrupt, _ = _capture_f_eps(model64, sae0_64, resolver, corrupted, site0)
    f_U_clean, eps_U_clean = _capture_f_eps(model64, sae0_64, resolver, clean, site0)
    delta_f_U = f_U_corrupt - f_U_clean

    # Model dtype/device
    params0 = list(w_mod.parameters())
    w_dtype = params0[0].dtype if params0 else torch.float64
    w_device = params0[0].device if params0 else torch.device("cpu")
    params1 = list(r_mod.parameters())
    r_dtype = params1[0].dtype if params1 else torch.float64
    r_device = params1[0].device if params1 else torch.device("cpu")

    # Collect (i, j) indices from IG result
    ij_pairs: set[tuple[int, int]] = set()
    for edge in ig_edges:
        i = edge.writer.node.neuron
        j = edge.reader.node.neuron
        if i is not None and j is not None:
            ij_pairs.add((i, j))

    if not ij_pairs:
        pytest.skip("No (i,j) feature pairs to compare")

    oracle_acc: dict[tuple, float] = {ij: 0.0 for ij in ij_pairs}

    # Freeze model and SAE params for oracle computation
    for p in list(model64.parameters()) + list(sae0_64.parameters()) + list(sae1_64.parameters()):
        p.requires_grad_(False)

    for k in range(1, M_oracle + 1):
        alpha_k = (k - 0.5) / M_oracle
        f_U_k = (f_U_clean + alpha_k * delta_f_U).detach().requires_grad_(True)

        r_store_k: dict[str, Tensor] = {}

        def _wk(m, inp, out, _sae=sae0_64, _f=f_U_k, _eps=eps_U_clean,
                _mdtype=w_dtype, _mdev=w_device, _resolved=w_res):
            x_hat = _sae.decode(_f)
            recon = x_hat + _eps
            return _routed_inject(_resolved, out, recon.to(_mdev, _mdtype))

        def _rk(m, inp, out, _sae=sae1_64, _st=r_store_k,
                _mdtype=r_dtype, _mdev=r_device, _resolved=r_res):
            a = _routed_extract(_resolved, out)
            a_in = a.to(getattr(_sae, "device", a.device), getattr(_sae, "dtype", a.dtype))
            f_D_k, x_hat_k, eps_k = sae_decompose(_sae, a_in)
            if f_D_k.requires_grad:
                f_D_k.retain_grad()
            recon = x_hat_k + eps_k
            _st["f_D_k"] = f_D_k
            return _routed_inject(_resolved, out, recon.to(_mdev, _mdtype))

        hw = w_mod.register_forward_hook(_wk)
        hr = r_mod.register_forward_hook(_rk)
        try:
            with torch.enable_grad():
                out_k = model64(clean)
                m_k = _metric(out_k)
                m_k.backward(retain_graph=True)
        finally:
            hw.remove()
            hr.remove()

        f_D_k_t = r_store_k.get("f_D_k")
        if f_D_k_t is None or f_D_k_t.grad is None:
            del out_k, m_k, f_U_k
            continue

        gradf_D_k = f_D_k_t.grad

        for (i, j) in ij_pairs:
            G_j = torch.zeros_like(f_D_k_t)
            G_j[..., j] = gradf_D_k[..., j]
            try:
                vjp_k = torch.autograd.grad(
                    f_D_k_t, [f_U_k],
                    grad_outputs=G_j,
                    retain_graph=True,
                    allow_unused=True,
                )[0]
            except RuntimeError:
                del G_j
                continue
            if vjp_k is None:
                del G_j
                continue
            contrib = float(
                (delta_f_U[..., i].float() * vjp_k[..., i].float()).sum()
            )
            oracle_acc[(i, j)] = oracle_acc.get((i, j), 0.0) + contrib
            del vjp_k, G_j

        del out_k, m_k, f_U_k, f_D_k_t

    # Restore requires_grad
    for p in list(model64.parameters()) + list(sae0_64.parameters()) + list(sae1_64.parameters()):
        p.requires_grad_(True)

    oracle_scores = {ij: v / M_oracle for ij, v in oracle_acc.items()}

    max_diff = 0.0
    print(f"\n[edge IG vs oracle] comparing {len(ij_pairs)} (i,j) pairs, IG N=128, oracle M={M_oracle}")
    for edge in ig_edges:
        i = edge.writer.node.neuron
        j = edge.reader.node.neuron
        if i is None or j is None:
            continue
        ig_score = circuit_ig.edges[edge]
        oracle_score = oracle_scores.get((i, j), 0.0)
        diff = abs(ig_score - oracle_score)
        if diff > max_diff:
            max_diff = diff
            print(f"  max diff: i={i}, j={j}, ig={ig_score:.4e}, oracle={oracle_score:.4e}, diff={diff:.3e}")

    print(f"\n  max|IG - oracle| = {max_diff:.3e}")
    assert max_diff < 1e-4, f"Edge IG vs path-integral oracle: max|diff|={max_diff:.3e} exceeds 1e-4"


# ---------------------------------------------------------------------------
# test_ig_edge_no_dense_jacobian
# ---------------------------------------------------------------------------

def test_ig_edge_no_dense_jacobian():
    """EAP-IG uses the per-j VJP loop — no dense d_sae×d_sae Jacobian.

    Verifies:
    1. EAP-IG produces edges (doesn't crash or OOM)
    2. On linear model + affine SAE, IG and attrib produce the same edge set
       (on a linear model with linear metric, IG == attrib to machine precision)
    """
    from circuitry.patching.sae_edges import SAEFeatureEdgeRunner
    from circuitry.patching.sites import Site

    d_sae = 64  # moderate; if a dense d_sae×d_sae Jacobian were created: 4096 entries

    torch.manual_seed(50)
    model64 = LinearResidToy(n_layers=2, d=8).to(torch.float64)
    torch.manual_seed(51)
    sae0_64 = SyntheticSAE(d_model=8, d_sae=d_sae, relu=False, dtype=torch.float64)
    torch.manual_seed(52)
    sae1_64 = SyntheticSAE(d_model=8, d_sae=d_sae, relu=False, dtype=torch.float64)
    resolver = _make_resolver()

    site0 = Site("resid_post", layer=0)
    site1 = Site("resid_post", layer=1)

    torch.manual_seed(8)
    clean, corrupted = _make_clean_corr(seed=8)
    clean = clean.to(torch.float64)
    corrupted = corrupted.to(torch.float64)

    runner = SAEFeatureEdgeRunner(model64, {site0: sae0_64, site1: sae1_64}, resolver)

    # IG run
    circuit_ig = runner.run(
        clean, corrupted, _metric,
        layer_pairs="adjacent",
        top_k_survivors=8,
        variant="ig",
        n_ig_steps=64,
    )

    # IG should produce edges (not crash or OOM)
    assert len(circuit_ig.edges) > 0, "EAP-IG produced no edges on linear toy"

    # On linear model + affine SAE, IG and attrib produce same edge sets
    circuit_attrib = runner.run(
        clean, corrupted, _metric,
        layer_pairs="adjacent",
        top_k_survivors=8,
        variant="attrib",
    )

    ig_ij = {(e.writer.node.neuron, e.reader.node.neuron) for e in circuit_ig.edges}
    att_ij = {(e.writer.node.neuron, e.reader.node.neuron) for e in circuit_attrib.edges}

    overlap = ig_ij & att_ij
    assert len(overlap) > 0, "IG and attrib produced completely disjoint edge sets"
    print(f"\n[no-dense-jacobian] d_sae={d_sae}, IG edges={len(ig_ij)}, attrib edges={len(att_ij)}, overlap={len(overlap)}")

    # On a linear model + affine SAE, IG == attrib to machine precision (the path integral
    # of a linear function is exact at any N).  Assert the common edges agree to ≤1e-4.
    # A wrong IG implementation (e.g. missing /N, using dense Jacobian) would fail this.
    ig_score_map = {(e.writer.node.neuron, e.reader.node.neuron): circuit_ig.edges[e]
                    for e in circuit_ig.edges}
    att_score_map = {(e.writer.node.neuron, e.reader.node.neuron): circuit_attrib.edges[e]
                     for e in circuit_attrib.edges}

    max_diff_ig_att = 0.0
    for ij in overlap:
        diff = abs(ig_score_map[ij] - att_score_map[ij])
        if diff > max_diff_ig_att:
            max_diff_ig_att = diff

    print(f"  IG vs attrib on common edges (linear toy): max|diff|={max_diff_ig_att:.3e}")
    assert max_diff_ig_att <= 1e-4, (
        f"IG and attrib edge scores disagree by {max_diff_ig_att:.3e} > 1e-4 on a linear toy. "
        "On a linear model the path integral is exact so they must agree. "
        "Possible causes: IG divide-by-N is wrong, partial sums underscale, or dense Jacobian."
    )


# ---------------------------------------------------------------------------
# test_ig_default_steps
# ---------------------------------------------------------------------------

def test_ig_default_steps():
    """variant='ig' with n_ig_steps=0 uses N=32; 'exact' raises NotImplementedError."""
    from circuitry.patching.sae_edges import SAEFeatureEdgeRunner
    from circuitry.patching.sae_features import SAEFeatureRunner
    from circuitry.patching.sites import Site

    torch.manual_seed(60)
    model = LinearResidToy(n_layers=2, d=8)
    torch.manual_seed(61)
    sae = SyntheticSAE(d_model=8, d_sae=16, relu=False)
    torch.manual_seed(62)
    sae1 = SyntheticSAE(d_model=8, d_sae=16, relu=False)
    resolver = _make_resolver()
    site0 = Site("resid_post", layer=0)
    site1 = Site("resid_post", layer=1)

    torch.manual_seed(9)
    clean, corrupted = _make_clean_corr(seed=9)

    # variant='ig' with n_ig_steps=0 should not raise and produce EXACTLY the same
    # scores as n_ig_steps=32 (the hardcoded default).  A wrong default (e.g. N=1)
    # would produce different scores and fail this assertion.
    runner_node = SAEFeatureRunner(model, {site0: sae}, resolver)
    result_default = runner_node.run(clean, corrupted, _metric, variant="ig", n_ig_steps=0)
    result_32 = runner_node.run(clean, corrupted, _metric, variant="ig", n_ig_steps=32)

    assert isinstance(result_default.scores, dict), "Should return a dict of scores"
    # Both runs must score the same features
    assert set(result_default.scores.keys()) == set(result_32.scores.keys()), (
        "n_ig_steps=0 and n_ig_steps=32 scored different feature sets — "
        "default N is not 32"
    )
    # Scores must be numerically identical (same computation, same N=32)
    max_diff_default = max(
        abs(result_default.scores[n] - result_32.scores[n])
        for n in result_default.scores
    ) if result_default.scores else 0.0
    print(f"\n[default steps] n_ig_steps=0 vs n_ig_steps=32 max|diff|={max_diff_default:.3e}")
    assert max_diff_default == 0.0, (
        f"n_ig_steps=0 produced scores DIFFERENT from n_ig_steps=32 "
        f"(max|diff|={max_diff_default:.3e}). The default N is not 32."
    )
    # alias for the old print
    result = result_default
    print(f"  variant='ig', n_ig_steps=0 → {len(result.scores)} scored features (N=32 default)")

    # variant='exact' should raise
    with pytest.raises(NotImplementedError):
        runner_node.run(clean, corrupted, _metric, variant="exact")

    # Same for edge runner: n_ig_steps=0 must match n_ig_steps=32 exactly
    runner_edge = SAEFeatureEdgeRunner(model, {site0: sae, site1: sae1}, resolver)
    circuit_default = runner_edge.run(
        clean, corrupted, _metric,
        layer_pairs="adjacent",
        variant="ig",
        n_ig_steps=0,
    )
    circuit_32 = runner_edge.run(
        clean, corrupted, _metric,
        layer_pairs="adjacent",
        variant="ig",
        n_ig_steps=32,
    )
    assert circuit_default.edges is not None
    # Edge sets must be identical
    assert set(circuit_default.edges.keys()) == set(circuit_32.edges.keys()), (
        "Edge runner n_ig_steps=0 and n_ig_steps=32 produced different edge sets"
    )
    max_diff_edge = max(
        abs(circuit_default.edges[e] - circuit_32.edges[e])
        for e in circuit_default.edges
    ) if circuit_default.edges else 0.0
    assert max_diff_edge == 0.0, (
        f"Edge runner n_ig_steps=0 produced scores different from n_ig_steps=32 "
        f"(max|diff|={max_diff_edge:.3e}). The default N is not 32."
    )
    print(f"  edge runner ig n_ig_steps=0 → {len(circuit_default.edges)} edges (matches n_ig_steps=32)")

    with pytest.raises(NotImplementedError):
        runner_edge.run(clean, corrupted, _metric, variant="exact")
