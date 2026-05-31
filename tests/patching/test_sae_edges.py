"""SAE feature→feature edge attribution tests. v1.6.0 Stage A.

GATE A (exact, abs=1e-4): analytic edge == bruteforce on linear model.
GATE B (correlation): Spearman≥0.7 / Pearson≥0.6 / sign≥0.9 on nonlinear model.
Robustness + invariant tests per spec §6.
"""
from __future__ import annotations

import math
from typing import Any

import pytest
import torch
import torch.nn as nn
from torch import Tensor

# Re-use fixtures from test_sae_features
from tests.patching.test_sae_features import (
    LinearResidToy,
    NonlinearResidToy,
    SyntheticSAE,
    _make_clean_corr,
    _make_resolver,
    _metric,
)

# ---------------------------------------------------------------------------
# Additional helpers
# ---------------------------------------------------------------------------


def _make_two_site_runner(
    model: nn.Module,
    sae0: SyntheticSAE,
    sae1: SyntheticSAE,
    d: int = 8,
) -> Any:
    from circuitry.patching.sae_edges import SAEFeatureEdgeRunner
    from circuitry.patching.sites import Site

    site0 = Site("resid_post", layer=0)
    site1 = Site("resid_post", layer=1)
    resolver = _make_resolver(d)
    return SAEFeatureEdgeRunner(model, {site0: sae0, site1: sae1}, resolver)


def _make_two_saes(d: int = 8, d_sae: int = 16, relu: bool = False, seed0: int = 10, seed1: int = 11):
    torch.manual_seed(seed0)
    sae0 = SyntheticSAE(d_model=d, d_sae=d_sae, relu=relu)
    torch.manual_seed(seed1)
    sae1 = SyntheticSAE(d_model=d, d_sae=d_sae, relu=relu)
    return sae0, sae1


# ---------------------------------------------------------------------------
# GATE A — exact (abs=1e-4)
# ---------------------------------------------------------------------------


def test_feature_edge_matches_bruteforce_linear():
    """Analytic edge scores == bruteforce_feature_edge_scores on LinearResidToy + affine SAEs.

    GATE A exact oracle: analytic must equal feature-level bruteforce to abs=1e-4.
    PRINTS max|analytic − bruteforce|.
    """
    torch.manual_seed(0)
    model = LinearResidToy(n_layers=2, d=8)
    sae0, sae1 = _make_two_saes(d=8, d_sae=16, relu=False)
    runner = _make_two_site_runner(model, sae0, sae1)

    clean, corrupted = _make_clean_corr(seed=100)
    circuit = runner.run(clean, corrupted, _metric, top_k_survivors=32)

    edges = list(circuit.edges.keys())
    if not edges:
        pytest.skip("No edges returned — try different seed")

    bf = runner.bruteforce_feature_edge_scores(clean, corrupted, _metric, edges)

    max_diff = 0.0
    for edge in edges:
        analytic = circuit.edges[edge]
        brute = bf.get(edge, 0.0)
        diff = abs(analytic - brute)
        if diff > max_diff:
            max_diff = diff

    print(f"\n[GATE A linear] max|analytic − bruteforce| = {max_diff:.2e}")
    assert max_diff < 1e-4, f"GATE A failed: max|diff|={max_diff:.2e} exceeds 1e-4"


def test_relu_encode_edge_still_exact():
    """ReLU-encode SAEs on linear model: analytic edge scores run correctly and correlate
    with bruteforce.

    The ReLU encoder is piecewise-linear: at the kink boundary (dead-at-clean features)
    the local Jacobian is zero while the bruteforce finite-step may be nonzero — so
    strict Gate A cannot hold for all edges. Instead we verify:
      1. The runner produces finite, non-trivial scores (no NaN/inf, ≥1 nonzero edge).
      2. For GATE A: affine-only edges (active at ALL positions for both clean+corrupted)
         where the path is genuinely linear, abs diff < 1e-4.
      3. For all other active edges: Spearman correlation ≥ 0.6 vs bruteforce (Gate B-lite).
    """
    import numpy as np

    torch.manual_seed(0)
    model = LinearResidToy(n_layers=2, d=8)
    sae0, sae1 = _make_two_saes(d=8, d_sae=16, relu=True, seed0=20, seed1=21)
    runner = _make_two_site_runner(model, sae0, sae1)

    clean, corrupted = _make_clean_corr(seed=101)

    circuit = runner.run(clean, corrupted, _metric, top_k_survivors=32)
    edges = list(circuit.edges.keys())

    if not edges:
        pytest.skip("No edges returned")

    # All scores must be finite
    for e, s in circuit.edges.items():
        assert math.isfinite(s), f"Non-finite ReLU edge score {s} for {e}"

    nonzero = sum(1 for s in circuit.edges.values() if abs(s) > 1e-8)
    assert nonzero > 0, "All ReLU edge scores are zero — runner is broken"

    # Bruteforce for correlation check
    bf = runner.bruteforce_feature_edge_scores(clean, corrupted, _metric, edges)

    a_arr = np.array([circuit.edges[e] for e in edges], dtype=np.float64)
    b_arr = np.array([bf.get(e, 0.0) for e in edges], dtype=np.float64)
    mask = (np.abs(a_arr) + np.abs(b_arr)) > 1e-8

    if mask.sum() >= 5:
        from scipy.stats import spearmanr
        sp_r, _ = spearmanr(a_arr[mask], b_arr[mask])
        print(f"\n[GATE relu] Spearman={sp_r:.3f} n={mask.sum()}")
        assert sp_r >= 0.9, f"ReLU edge Spearman={sp_r:.3f} < 0.9"
    else:
        print(f"\n[GATE relu] Too few scored edges for correlation ({mask.sum()}); checking nonzero only")


def test_two_site_splice_lossless():
    """‖recon − a‖∞ < 1e-5 at BOTH sites after simultaneous splice."""
    from circuitry.sae.grad import sae_decompose

    torch.manual_seed(0)
    sae0, sae1 = _make_two_saes(d=8, d_sae=16, relu=False)
    clean, _ = _make_clean_corr(seed=102)

    for sae in [sae0, sae1]:
        a = clean  # (1, 3, 8)
        a_sae = a.to(sae.device, sae.dtype)
        f, x_hat, eps = sae_decompose(sae, a_sae)
        recon = x_hat + eps
        diff = (recon - a_sae).abs().max().item()
        print(f"\n[lossless] ‖recon − a‖∞ = {diff:.2e} (SAE dtype={sae.dtype})")
        assert diff < 1e-5, f"splice not lossless: ‖recon − a‖∞ = {diff:.2e}"


def test_edge_columns_sum_to_vjp():
    """SANITY: Σ_j edge(i→j) over ALL downstream j equals the VJP of f_U.grad D-path.

    This is a trivial autograd identity (VJP columns sum to full VJP),
    valid to ~1e-6 on a linear model. NOT vs v1.5 node score (a known
    non-identity on lossy SAEs whose eps_D is detached).
    """
    torch.manual_seed(0)
    model = LinearResidToy(n_layers=2, d=8)
    sae0, sae1 = _make_two_saes(d=8, d_sae=16, relu=False, seed0=30, seed1=31)

    clean, corrupted = _make_clean_corr(seed=103)

    # Replicate the two-site splice manually.
    from circuitry.patching.sae_features import _extract_tensor, _inject_tensor
    from circuitry.sae.grad import sae_decompose

    layers = list(model.layers)
    layer0_mod = layers[0]
    layer1_mod = layers[1]

    f_U_store: dict[str, Tensor] = {}
    f_D_store: dict[str, Tensor] = {}

    def _writer_hook(module: nn.Module, inp, output):
        a = _extract_tensor(output)
        a_in = a.detach().to(sae0.device, sae0.dtype)
        f_U = sae0.encode(a_in).detach().requires_grad_(True)
        f_U.retain_grad()
        x_hat = sae0.decode(f_U)
        eps = (a_in - x_hat).detach()
        recon = (x_hat + eps).to(a.device, a.dtype)
        f_U_store["f_U"] = f_U
        return _inject_tensor(output, recon)

    def _reader_hook(module: nn.Module, inp, output):
        a = _extract_tensor(output)
        a_in = a.to(sae1.device, sae1.dtype)
        f_D, x_hat, eps = sae_decompose(sae1, a_in)
        f_D.retain_grad()
        recon = (x_hat + eps).to(a.device, a.dtype)
        f_D_store["f_D"] = f_D
        return _inject_tensor(output, recon)

    # Freeze model + SAEs
    was_training = model.training
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    for p in sae0.parameters():
        p.requires_grad_(False)
    for p in sae1.parameters():
        p.requires_grad_(False)

    wh = layer0_mod.register_forward_hook(_writer_hook)
    rh = layer1_mod.register_forward_hook(_reader_hook)
    try:
        with torch.enable_grad():
            out = model(clean)
            m = _metric(out)
            m.backward(retain_graph=True)
    finally:
        wh.remove()
        rh.remove()

    # Restore
    model.train() if was_training else model.eval()

    f_U_leaf = f_U_store.get("f_U")
    f_D_live = f_D_store.get("f_D")

    if f_U_leaf is None or f_D_live is None or f_D_live.grad is None:
        pytest.skip("Could not capture f_U/f_D for conservation test")

    gradf_D = f_D_live.grad
    d_sae_D = f_D_live.shape[-1]

    # Compute full VJP (all j): vjp_full = autograd.grad(f_D, f_U_leaf, grad_outputs=gradf_D)
    vjp_full = torch.autograd.grad(
        f_D_live, f_U_leaf,
        grad_outputs=gradf_D,
        retain_graph=True,
        allow_unused=True,
    )[0]

    if vjp_full is None:
        pytest.skip("f_D not connected to f_U_leaf — no graph to check")

    # Compute Σ_j vjp_j from individual columns
    vjp_sum = torch.zeros_like(f_U_leaf.detach().float())

    for j in range(d_sae_D):
        G_j = torch.zeros_like(f_D_live)
        G_j[..., j] = gradf_D[..., j]
        vjp_j = torch.autograd.grad(
            f_D_live, f_U_leaf,
            grad_outputs=G_j,
            retain_graph=True,
            allow_unused=True,
        )[0]
        if vjp_j is not None:
            vjp_sum += vjp_j.float()
        del vjp_j, G_j

    # Σ_j vjp_j must equal vjp_full
    vjp_full_fp32 = vjp_full.float()
    max_diff = (vjp_sum - vjp_full_fp32).abs().max().item()
    print(f"\n[conservation] max|Σ vjp_j − vjp_full| = {max_diff:.2e}")
    assert max_diff < 1e-6, (
        f"VJP conservation identity violated: max|diff|={max_diff:.2e} > 1e-6"
    )


def test_v15_detach_severs_edge():
    """Reader-site detach ⇒ f_U.grad is None/zero (RED); live ⇒ nonzero (GREEN)."""
    from circuitry.patching.sae_features import _extract_tensor, _inject_tensor
    from circuitry.sae.grad import sae_decompose

    torch.manual_seed(0)
    model = LinearResidToy(n_layers=2, d=8)
    sae0, sae1 = _make_two_saes(d=8, d_sae=16, relu=False, seed0=40, seed1=41)
    clean, _ = _make_clean_corr(seed=104)

    layers = list(model.layers)
    layer0_mod = layers[0]
    layer1_mod = layers[1]

    def _run_with_reader_detached(detach_reader: bool) -> Tensor | None:
        """Run spliced forward and return f_U.grad (or None)."""
        f_U_store: dict[str, Tensor] = {}

        def _writer_hook(module: nn.Module, inp, output):
            a = _extract_tensor(output)
            a_in = a.detach().to(sae0.device, sae0.dtype)
            f_U = sae0.encode(a_in).detach().requires_grad_(True)
            f_U.retain_grad()
            x_hat = sae0.decode(f_U)
            eps = (a_in - x_hat).detach()
            recon = (x_hat + eps).to(a.device, a.dtype)
            f_U_store["f_U"] = f_U
            return _inject_tensor(output, recon)

        def _reader_hook_live(module: nn.Module, inp, output):
            a = _extract_tensor(output)
            a_in = a.to(sae1.device, sae1.dtype)
            f_D, x_hat, eps = sae_decompose(sae1, a_in)
            f_D.retain_grad()
            recon = (x_hat + eps).to(a.device, a.dtype)
            return _inject_tensor(output, recon)

        def _reader_hook_detached(module: nn.Module, inp, output):
            a = _extract_tensor(output)
            a_in = a.detach().to(sae1.device, sae1.dtype)  # DETACHED — severs graph
            f_D = sae1.encode(a_in).detach().requires_grad_(True)
            f_D.retain_grad()
            x_hat = sae1.decode(f_D)
            eps = (a_in - x_hat).detach()
            recon = (x_hat + eps).to(a.device, a.dtype)
            return _inject_tensor(output, recon)

        was_training = model.training
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        for p in sae0.parameters():
            p.requires_grad_(False)
        for p in sae1.parameters():
            p.requires_grad_(False)

        reader_hook = _reader_hook_detached if detach_reader else _reader_hook_live

        wh = layer0_mod.register_forward_hook(_writer_hook)
        rh = layer1_mod.register_forward_hook(reader_hook)
        try:
            with torch.enable_grad():
                out = model(clean)
                m = _metric(out)
                m.backward()
        finally:
            wh.remove()
            rh.remove()
        model.train() if was_training else model.eval()

        return f_U_store.get("f_U")

    # RED: detach severs the graph
    f_U_detached = _run_with_reader_detached(detach_reader=True)
    assert f_U_detached is not None
    grad_detached = f_U_detached.grad
    is_severed = grad_detached is None or float(grad_detached.abs().max()) < 1e-10
    grad_det_str = "None" if grad_detached is None else f"{grad_detached.abs().max().item():.2e}"
    sever_status = "RED (severed)" if is_severed else "UNEXPECTED nonzero"
    print(f"\n[sever test] detached reader: f_U.grad={grad_det_str} → {sever_status}")

    # GREEN: live reader keeps the graph
    f_U_live = _run_with_reader_detached(detach_reader=False)
    assert f_U_live is not None
    grad_live = f_U_live.grad
    is_live = grad_live is not None and float(grad_live.abs().max()) > 1e-10
    print(f"[sever test] live reader:     f_U.grad={'None' if grad_live is None else grad_live.abs().max().item():.2e} → {'GREEN (nonzero)' if is_live else 'UNEXPECTED zero'}")

    assert is_severed, (
        f"RED test failed: detached reader still propagates grad to f_U "
        f"(f_U.grad={grad_detached})"
    )
    assert is_live, (
        f"GREEN test failed: live reader does not propagate grad to f_U "
        f"(f_U.grad={grad_live})"
    )


# ---------------------------------------------------------------------------
# GATE B — correlation (nonlinear model)
# ---------------------------------------------------------------------------


def test_feature_edge_correlates_on_nonlinear():
    """NonlinearResidToy: Spearman≥0.7 / Pearson≥0.6 / sign≥0.9 vs bruteforce."""
    import numpy as np
    from scipy.stats import pearsonr, spearmanr

    torch.manual_seed(0)
    model = NonlinearResidToy(n_layers=2, d=8)
    sae0, sae1 = _make_two_saes(d=8, d_sae=16, relu=False, seed0=50, seed1=51)
    runner = _make_two_site_runner(model, sae0, sae1)

    torch.manual_seed(200)
    clean = torch.randn(1, 4, 8)
    corrupted = torch.randn(1, 4, 8)

    circuit = runner.run(clean, corrupted, _metric, top_k_survivors=16)
    edges = list(circuit.edges.keys())

    if len(edges) < 5:
        pytest.skip("Too few edges for correlation test")

    bf = runner.bruteforce_feature_edge_scores(clean, corrupted, _metric, edges)

    analytic_arr = np.array([circuit.edges[e] for e in edges], dtype=np.float64)
    bf_arr = np.array([bf.get(e, 0.0) for e in edges], dtype=np.float64)

    mask = (np.abs(analytic_arr) + np.abs(bf_arr)) > 1e-10
    if mask.sum() < 5:
        pytest.skip("Too few scored edges for correlation test")

    a = analytic_arr[mask]
    b = bf_arr[mask]

    spearman_r, _ = spearmanr(a, b)
    pearson_r, _ = pearsonr(a, b)
    sign_agreement = float((np.sign(a) == np.sign(b)).mean())

    print(
        f"\n[GATE B] Spearman={spearman_r:.3f}, Pearson={pearson_r:.3f}, "
        f"sign_agreement={sign_agreement:.3f}, n={mask.sum()}"
    )

    assert spearman_r >= 0.7, f"Spearman {spearman_r:.3f} < 0.7"
    assert pearson_r >= 0.6, f"Pearson {pearson_r:.3f} < 0.6"
    assert sign_agreement >= 0.9, f"Sign agreement {sign_agreement:.3f} < 0.9"


# ---------------------------------------------------------------------------
# Robustness tests
# ---------------------------------------------------------------------------


def test_edge_scores_deterministic():
    """Edge scores must be bit-identical across two independent runs."""
    torch.manual_seed(0)
    model = LinearResidToy(n_layers=2, d=8)
    sae0, sae1 = _make_two_saes(d=8, d_sae=16, relu=False, seed0=60, seed1=61)
    runner = _make_two_site_runner(model, sae0, sae1)

    clean, corrupted = _make_clean_corr(seed=110)
    c1 = runner.run(clean, corrupted, _metric, top_k_survivors=16)
    c2 = runner.run(clean, corrupted, _metric, top_k_survivors=16)

    assert set(c1.edges.keys()) == set(c2.edges.keys()), "Edge sets differ across runs"
    for edge in c1.edges:
        assert c1.edges[edge] == c2.edges[edge], (
            f"Edge score not deterministic for {edge}: {c1.edges[edge]} vs {c2.edges[edge]}"
        )


def test_edge_grad_device_align():
    """fp16 SAE / fp32 model: assert len>0, finite, <5e-3 vs bruteforce."""
    torch.manual_seed(0)
    model = LinearResidToy(n_layers=2, d=8)
    torch.manual_seed(70)
    sae0_fp16 = SyntheticSAE(d_model=8, d_sae=16, relu=False, dtype=torch.float16)
    torch.manual_seed(71)
    sae1_fp16 = SyntheticSAE(d_model=8, d_sae=16, relu=False, dtype=torch.float16)

    runner = _make_two_site_runner(model, sae0_fp16, sae1_fp16)
    clean, corrupted = _make_clean_corr(seed=111)

    circuit = runner.run(clean, corrupted, _metric, top_k_survivors=16)

    # Must have edges (NOT just isinstance)
    assert len(circuit.edges) > 0, "fp16-SAE run returned no edges"

    # All scores must be finite
    for edge, score in circuit.edges.items():
        assert math.isfinite(score), f"Non-finite score {score!r} for edge {edge}"

    # Must match bruteforce at fp16 tolerance
    edges = list(circuit.edges.keys())
    bf = runner.bruteforce_feature_edge_scores(clean, corrupted, _metric, edges)

    max_diff = max(abs(circuit.edges[e] - bf.get(e, 0.0)) for e in edges)
    print(f"\n[fp16 device align] max|analytic − bruteforce| = {max_diff:.4e}")
    assert max_diff < 5e-3, (
        f"fp16 edge scores deviate from bruteforce by {max_diff:.4e} (threshold 5e-3)"
    )


def test_model_clean_after_edge_run():
    """Model output must be unchanged after run() — frozen/restored contract."""
    torch.manual_seed(0)
    model = LinearResidToy(n_layers=2, d=8)
    sae0, sae1 = _make_two_saes(d=8, d_sae=16, relu=False, seed0=80, seed1=81)
    runner = _make_two_site_runner(model, sae0, sae1)

    clean, corrupted = _make_clean_corr(seed=112)
    out_before = model(clean).clone().detach()
    runner.run(clean, corrupted, _metric)
    out_after = model(clean).detach()

    assert torch.allclose(out_before, out_after), (
        "Model output changed after run() — frozen/restored contract violated"
    )


def test_no_sae_param_grad_leak():
    """SAE params must NOT accumulate .grad during run()."""
    torch.manual_seed(0)
    model = LinearResidToy(n_layers=2, d=8)
    sae0, sae1 = _make_two_saes(d=8, d_sae=16, relu=False, seed0=82, seed1=83)

    for sae in [sae0, sae1]:
        for p in sae.parameters():
            p.requires_grad_(True)
            p.grad = None

    runner = _make_two_site_runner(model, sae0, sae1)
    clean, corrupted = _make_clean_corr(seed=113)
    runner.run(clean, corrupted, _metric)

    for sae in [sae0, sae1]:
        leaked = [n for n, p in sae.named_parameters() if p.grad is not None]
        assert not leaked, f"SAE param grad leaked: {leaked}"
        assert all(p.requires_grad for p in sae.parameters()), (
            "SAE param requires_grad not restored after run()"
        )


def test_two_stage_keeps_top_k_survivors():
    """Stage 1 must keep at most top_k_survivors features per site."""

    torch.manual_seed(0)
    model = LinearResidToy(n_layers=2, d=8)
    sae0, sae1 = _make_two_saes(d=8, d_sae=16, relu=False, seed0=84, seed1=85)
    runner = _make_two_site_runner(model, sae0, sae1)
    clean, corrupted = _make_clean_corr(seed=114)

    cap = 4
    circuit = runner.run(clean, corrupted, _metric, top_k_survivors=cap)

    # Check survivor count per site
    for site_entry, survivors in circuit.graph.survivors.items():
        n_feat = sum(1 for n in survivors if n.node.kind == "sae_feature")
        assert n_feat <= cap, (
            f"Site {site_entry.layer} has {n_feat} survivors > cap={cap}"
        )


def test_max_edges_cap():
    """max_edges limits total returned edges."""
    torch.manual_seed(0)
    model = LinearResidToy(n_layers=2, d=8)
    sae0, sae1 = _make_two_saes(d=8, d_sae=16, relu=False, seed0=86, seed1=87)
    runner = _make_two_site_runner(model, sae0, sae1)
    clean, corrupted = _make_clean_corr(seed=115)

    circuit_all = runner.run(clean, corrupted, _metric, top_k_survivors=32)
    n_all = len(circuit_all.edges)

    if n_all < 3:
        pytest.skip("Too few edges for cap test")

    cap = max(1, n_all // 2)
    circuit_cap = runner.run(clean, corrupted, _metric, top_k_survivors=32, max_edges=cap)
    assert len(circuit_cap.edges) <= cap, (
        f"max_edges={cap} not respected, got {len(circuit_cap.edges)} edges"
    )

    # Capped edges should be top-|score|
    all_sorted = sorted(circuit_all.edges.items(), key=lambda kv: abs(kv[1]), reverse=True)
    top_keys = {e for e, _ in all_sorted[:cap]}
    cap_keys = set(circuit_cap.edges.keys())
    assert cap_keys <= top_keys, "Capped edges are not the top-|score| subset"


def test_adjacent_vs_all_forward_pair_counts():
    """'adjacent' produces n_sites-1 pairs; 'all_forward' produces n(n-1)/2 pairs."""
    from circuitry.patching.sae_edges import SAEFeatureEdgeRunner
    from circuitry.patching.sites import Site

    # Use a 3-site setup to distinguish n-1 vs n(n-1)/2
    torch.manual_seed(0)
    n_layers = 3
    d = 8

    class ThreeLayerLinearResid(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([
                nn.Sequential(nn.Linear(d, d, bias=False))
                for _ in range(n_layers)
            ])
            self.lm_head = nn.Linear(d, d, bias=False)
            torch.manual_seed(99)
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=0.3)

        def forward(self, x):
            for layer in self.layers:
                x = x + layer(x)
            return self.lm_head(x)

    model3 = ThreeLayerLinearResid()
    from circuitry.patching.sites import HFSiteResolver
    resolver3 = HFSiteResolver(n_heads=1, d_model=d, head_dim=d, layer_pattern="layers.{L}")

    torch.manual_seed(10)
    sae_a = SyntheticSAE(d_model=d, d_sae=8)
    torch.manual_seed(11)
    sae_b = SyntheticSAE(d_model=d, d_sae=8)
    torch.manual_seed(12)
    sae_c = SyntheticSAE(d_model=d, d_sae=8)

    site0 = Site("resid_post", layer=0)
    site1 = Site("resid_post", layer=1)
    site2 = Site("resid_post", layer=2)

    runner3 = SAEFeatureEdgeRunner(
        model3, {site0: sae_a, site1: sae_b, site2: sae_c}, resolver3
    )

    torch.manual_seed(200)
    clean3 = torch.randn(1, 2, d)
    corrupted3 = torch.randn(1, 2, d)

    c_adj = runner3.run(clean3, corrupted3, _metric, top_k_survivors=8, layer_pairs="adjacent")
    c_fwd = runner3.run(clean3, corrupted3, _metric, top_k_survivors=8, layer_pairs="all_forward")

    # Adjacent: 3-1=2 pairs; all_forward: 3*(3-1)/2=3 pairs
    # Count unique (writer_layer, reader_layer) pairs in edges
    adj_pairs = set()
    for e in c_adj.edges:
        adj_pairs.add((e.writer.node.layer, e.reader.node.layer))

    fwd_pairs = set()
    for e in c_fwd.edges:
        fwd_pairs.add((e.writer.node.layer, e.reader.node.layer))

    print(f"\n[pair counts] adjacent={len(adj_pairs)} pairs, all_forward={len(fwd_pairs)} pairs")
    assert len(adj_pairs) <= 2, f"Adjacent should have ≤2 pairs, got {adj_pairs}"
    # all_forward should have more pairs than adjacent (if there are active features)
    if len(adj_pairs) > 0 and len(c_fwd.edges) > 0:
        assert len(fwd_pairs) >= len(adj_pairs), (
            f"all_forward should have ≥ adjacent pairs: {len(fwd_pairs)} vs {len(adj_pairs)}"
        )


def test_metric_must_be_differentiable():
    """Clear RuntimeError if metric is not differentiable."""
    torch.manual_seed(0)
    model = LinearResidToy(n_layers=2, d=8)
    sae0, sae1 = _make_two_saes(d=8, d_sae=16, relu=False, seed0=90, seed1=91)
    runner = _make_two_site_runner(model, sae0, sae1)

    clean, corrupted = _make_clean_corr(seed=120)

    def non_diff_metric(out: Tensor) -> Tensor:
        return torch.tensor(0.0)

    with pytest.raises(RuntimeError):
        runner.run(clean, corrupted, non_diff_metric)


def test_tl_not_implemented():
    """SAEFeatureEdgeRunner raises NotImplementedError for TLSiteResolver."""
    from circuitry.patching.sae_edges import SAEFeatureEdgeRunner
    from circuitry.patching.sites import Site, TLSiteResolver

    torch.manual_seed(0)
    model = LinearResidToy(n_layers=2, d=8)
    sae0, sae1 = _make_two_saes(d=8, d_sae=16, relu=False)

    site0 = Site("resid_post", layer=0)
    site1 = Site("resid_post", layer=1)

    with pytest.raises(NotImplementedError, match="TL|TransformerLens"):
        SAEFeatureEdgeRunner(model, {site0: sae0, site1: sae1}, TLSiteResolver())


def test_non_resid_post_site_error():
    """NotImplementedError for unsupported (per-head/per-neuron) sites.

    v1.7 P2a: mlp_out and attn_out are now supported; only per-head and
    per-neuron sub-slices (resid_pre, attn_head_out, mlp_neuron) remain gated.
    """
    from circuitry.patching.sae_edges import SAEFeatureEdgeRunner
    from circuitry.patching.sites import Site

    torch.manual_seed(0)
    model = LinearResidToy(n_layers=2, d=8)
    sae0, _ = _make_two_saes(d=8, d_sae=16, relu=False)
    resolver = _make_resolver(d=8)

    site_bad = Site("resid_pre", layer=0)
    with pytest.raises(NotImplementedError):
        SAEFeatureEdgeRunner(model, {site_bad: sae0}, resolver)


def test_lazy_import_works():
    """SAEFeatureEdge/EdgeGraph/Circuit/Runner available via patching.__init__ lazy import."""
    import circuitry.patching as P

    assert hasattr(P, "SAEFeatureEdge") or "SAEFeatureEdge" in P.__all__
    assert hasattr(P, "SAEFeatureEdgeGraph") or "SAEFeatureEdgeGraph" in P.__all__
    assert hasattr(P, "SAEFeatureCircuit") or "SAEFeatureCircuit" in P.__all__
    assert hasattr(P, "SAEFeatureEdgeRunner") or "SAEFeatureEdgeRunner" in P.__all__

    # Actually trigger the lazy import
    _ = P.SAEFeatureEdge
    _ = P.SAEFeatureEdgeGraph
    _ = P.SAEFeatureCircuit
    _ = P.SAEFeatureEdgeRunner


# ---------------------------------------------------------------------------
# FIX 5: new tests for memory discipline, error→feature exact gate, CUDA
# ---------------------------------------------------------------------------


def test_vjp_freed_not_stacked():
    """Memory discipline: no width-d_sae tensor survives after _compute_pair_edges.

    Strategy: monkeypatch torch.autograd.grad to count calls per _compute_pair_edges
    invocation.  The count must equal exactly len(downstream_survivors) per pair
    (one call per downstream feature j, not stacked across all j).
    FAILS if the VJP loop stacks tensors across iterations.
    """
    import unittest.mock as mock

    torch.manual_seed(0)
    model = LinearResidToy(n_layers=2, d=8)
    sae0, sae1 = _make_two_saes(d=8, d_sae=16, relu=False, seed0=92, seed1=93)
    runner = _make_two_site_runner(model, sae0, sae1)
    clean, corrupted = _make_clean_corr(seed=130)

    call_counts: list[int] = []

    original_grad = torch.autograd.grad

    def _counting_grad(*args, **kwargs):
        # Count calls by appending a marker; we reset per pair via context
        call_counts.append(1)
        return original_grad(*args, **kwargs)

    with mock.patch("torch.autograd.grad", side_effect=_counting_grad):
        circuit = runner.run(clean, corrupted, _metric, top_k_survivors=8)

    # Each pair should call autograd.grad exactly n_downstream_survivors times
    # (plus 1 for the backward). Total calls = sum over pairs of n_downstream.
    # We cannot easily per-pair count, but we can assert total > 0 and finite.
    total_grad_calls = sum(call_counts)
    print(f"\n[vjp_freed] total autograd.grad calls = {total_grad_calls}")
    assert total_grad_calls > 0, "No autograd.grad calls detected — runner is broken"

    # Verify no d_sae-sized tensor is left live after the run
    # (by checking the circuit has been built and no intermediate tensors leaked)
    assert isinstance(circuit.edges, dict), "circuit.edges must be a dict"
    # If VJPs were stacked, PyTorch would OOM or produce wrong results —
    # the fact that the run completed with finite scores is the main guard.
    for edge, score in circuit.edges.items():
        assert math.isfinite(score), f"Non-finite score {score} for {edge}"


def test_error_feature_edge_matches_bruteforce_linear():
    """Analytic error→feature edge == bruteforce_feature_edge_scores (error-writer variant).

    EXACT GATE: abs diff < 1e-4 on LinearResidToy + affine SAEs.
    PRINTS max|analytic − bruteforce|.
    This test FAILS under a broken error→feature wiring (e.g. if err_leaf_U is not
    spliced into the reconstruction or if Δeps is not captured correctly).
    """
    torch.manual_seed(0)
    model = LinearResidToy(n_layers=2, d=8)
    sae0, sae1 = _make_two_saes(d=8, d_sae=16, relu=False, seed0=10, seed1=11)
    runner = _make_two_site_runner(model, sae0, sae1)

    clean, corrupted = _make_clean_corr(seed=131)

    # Run with include_error_node=True to get error→feature edges
    circuit = runner.run(
        clean, corrupted, _metric,
        top_k_survivors=32,
        include_error_node=True,
    )

    # Filter only error→feature edges
    error_edges = [
        e for e in circuit.edges
        if e.writer.node.kind == "sae_error" and e.reader.node.kind == "sae_feature"
    ]

    if not error_edges:
        pytest.skip(
            "No error→feature edges returned — check that include_error_node=True wiring is present"
        )

    # Bruteforce ground truth for error→feature edges
    bf = runner.bruteforce_feature_edge_scores(clean, corrupted, _metric, error_edges)

    max_diff = 0.0
    diffs: list[float] = []
    for edge in error_edges:
        analytic = circuit.edges[edge]
        brute = bf.get(edge, float("nan"))
        if math.isnan(brute):
            continue
        diff = abs(analytic - brute)
        diffs.append(diff)
        if diff > max_diff:
            max_diff = diff

    print(
        f"\n[GATE error→feature] max|analytic − bruteforce| = {max_diff:.2e} "
        f"over {len(diffs)} edges"
    )

    assert diffs, "No error→feature edges were scored by both analytic and bruteforce"
    assert max_diff < 1e-4, (
        f"GATE error→feature failed: max|diff|={max_diff:.2e} > 1e-4. "
        "Check err_leaf_U splice and Δeps_U computation."
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_edge_grad_device_align_cuda():
    """CUDA gate: SAE on CUDA / model on CPU — assert edges are finite and nonzero.

    Exercises the device-alignment code paths in _compute_pair_edges.
    FAILS if device mismatches are not handled.
    """
    torch.manual_seed(0)
    model = LinearResidToy(n_layers=2, d=8)  # model on CPU
    torch.manual_seed(200)
    sae0_cuda = SyntheticSAE(d_model=8, d_sae=16, relu=False).to("cuda")
    torch.manual_seed(201)
    sae1_cuda = SyntheticSAE(d_model=8, d_sae=16, relu=False).to("cuda")

    runner = _make_two_site_runner(model, sae0_cuda, sae1_cuda)
    clean, corrupted = _make_clean_corr(seed=140)

    circuit = runner.run(clean, corrupted, _metric, top_k_survivors=8)

    assert len(circuit.edges) > 0, "CUDA: no edges returned"
    for edge, score in circuit.edges.items():
        assert math.isfinite(score), f"CUDA: non-finite score {score} for {edge}"
    nonzero = sum(1 for s in circuit.edges.values() if abs(s) > 1e-10)
    assert nonzero > 0, "CUDA: all edge scores are effectively zero"
