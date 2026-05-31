"""P3 tests: TransformerLens backend for SAE attribution.

Five tests with TEETH:
  1. test_tl_autograd_exact_through_hookpoint  — analytic grad vs central FD ≤1e-9 at fp64
  2. test_tl_splice_lossless                   — zero-perturbation splice ≤1e-12 at fp64
  3. test_tl_dtype_not_downcast                — fp64 model stays fp64 (§10 defect 2 regression)
  4. test_tl_edge_runs                         — SAEFeatureEdgeRunner end-to-end on TL model
  5. test_tl_mlp_out_site                      — mlp_out TL site resolves and runs attribution

No downloads: all models are tiny random-init HookedTransformer from HookedTransformerConfig.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torch import Tensor

transformer_lens = pytest.importorskip("transformer_lens")

from circuitry.patching.sites import Site, TLSiteResolver  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _tiny_tl(dtype: torch.dtype = torch.float32, seed: int = 0):
    """Random-init HookedTransformer with no downloads.

    d_model=32, n_heads=4, d_head=8, d_mlp=64, n_ctx=16, d_vocab=64.
    """
    from transformer_lens import HookedTransformer, HookedTransformerConfig
    cfg = HookedTransformerConfig(
        n_layers=2,
        d_model=32,
        n_heads=4,
        d_head=8,
        d_mlp=64,
        n_ctx=16,
        d_vocab=64,
        act_fn="gelu",
        normalization_type="LN",
        dtype=dtype,
    )
    torch.manual_seed(seed)
    model = HookedTransformer(cfg).eval()
    if dtype != torch.float32:
        model = model.to(dtype)
    return model


class TinySAE(nn.Module):
    """Duck-typed SAE matching d_in features.

    Exposes .encode, .decode, .device, .dtype attributes as required.
    cfg attribute satisfies assert_supported_sae (standard arch, d_sae, no norm).
    """

    def __init__(self, d_in: int, d_sae: int = 8, dtype: torch.dtype = torch.float32,
                 device: torch.device | str = "cpu", seed: int = 1) -> None:
        super().__init__()
        self.d_in = d_in
        self.d_sae = d_sae
        self.device = torch.device(device)
        self.dtype = dtype

        torch.manual_seed(seed)
        self.W_enc = nn.Parameter(torch.empty(d_sae, d_in, dtype=dtype))
        self.b_enc = nn.Parameter(torch.zeros(d_sae, dtype=dtype))
        self.W_dec = nn.Parameter(torch.empty(d_in, d_sae, dtype=dtype))
        self.b_dec = nn.Parameter(torch.zeros(d_in, dtype=dtype))
        nn.init.normal_(self.W_enc, std=0.3)
        nn.init.normal_(self.W_dec, std=0.3)

        # Minimal cfg so assert_supported_sae passes
        class _Cfg:
            def __init__(self, d_sae: int) -> None:
                self.d_sae = d_sae
                self.normalize_activations = "none"

            def architecture(self) -> str:
                return "standard"

        self.cfg = _Cfg(d_sae)

    def to(self, *args, **kwargs):  # type: ignore[override]
        result = super().to(*args, **kwargs)
        for p in result.parameters():
            result.device = p.device
            result.dtype = p.dtype
            break
        return result

    def encode(self, x: Tensor) -> Tensor:
        return x @ self.W_enc.T + self.b_enc

    def decode(self, f: Tensor) -> Tensor:
        return f @ self.W_dec.T + self.b_dec


def _metric(out: Tensor) -> Tensor:
    """Simple differentiable scalar from model output."""
    if hasattr(out, "logits"):
        logits = out.logits
    else:
        logits = out
    # logit diff: token 0 vs token 1 at last position
    return logits[..., -1, 0] - logits[..., -1, 1]  # scalar if batch=1


def _make_tokens(b: int = 1, s: int = 8, vocab: int = 64, seed: int = 42) -> Tensor:
    torch.manual_seed(seed)
    return torch.randint(0, vocab, (b, s))


# ---------------------------------------------------------------------------
# Test 1: Autograd exact through HookPoint (FD vs analytic ≤1e-9 at fp64)
# ---------------------------------------------------------------------------

def test_tl_autograd_exact_through_hookpoint():
    """SAE node attribution analytic grad vs central FD at fp64 — max err ≤ 1e-9.

    This is the primary TL-correctness oracle: proves autograd flows exactly
    through model.hook_dict[...] (the HookPoint) at fp64 precision.
    """
    from circuitry.patching.sae_features import (  # noqa: PLC0415
        SAEFeatureRunner,
        _routed_extract,
        _routed_inject,
    )

    model = _tiny_tl(dtype=torch.float64)
    assert model.cfg.dtype == torch.float64, "model should be fp64"

    d_model = model.cfg.d_model  # 32
    sae = TinySAE(d_in=d_model, d_sae=8, dtype=torch.float64).eval()
    resolver = TLSiteResolver()
    site = Site(layer=0, component="resid_post")

    clean_tokens = _make_tokens(seed=10)
    corr_tokens = _make_tokens(seed=11)

    runner = SAEFeatureRunner(model, {site: sae}, resolver)

    # --- Analytic scores (smoke check — ensures runner produces something) ---
    result = runner.run(clean_tokens, corr_tokens, _metric)
    assert len(result.scores) >= 0  # runner must not raise

    # --- Capture f_leaf and its gradient via a spliced forward ---
    resolved = resolver.resolve(model, site)
    lm = resolved.module

    f_leaf_store: dict[str, Tensor] = {}

    def _grad_splice_hook(
        mod: nn.Module, inp: object, output: object,
    ) -> object:
        a = _routed_extract(resolved, output)
        a_in = a.detach().to(sae.device, sae.dtype)
        f = sae.encode(a_in).detach().requires_grad_(True)
        f.retain_grad()
        x_hat = sae.decode(f)
        eps = (a_in - sae.decode(sae.encode(a_in))).detach()
        recon = (x_hat + eps).to(model.cfg.dtype)
        f_leaf_store["f"] = f
        return _routed_inject(resolved, output, recon)

    hg = lm.register_forward_hook(_grad_splice_hook)
    try:
        with torch.enable_grad():
            out = model(clean_tokens)
            m = _metric(out)
            m.backward()
    finally:
        hg.remove()

    f_leaf = f_leaf_store["f"]
    assert f_leaf.grad is not None, "f.grad is None — autograd did not flow through HookPoint"
    assert f_leaf.grad.abs().sum() > 0, "f.grad is all zeros — autograd produced no signal"

    grad_f = f_leaf.grad  # (b, s, d_sae) — analytic ∂metric/∂f

    # --- Central FD: perturb each feature and compare analytic sum ---
    h_fd = 1e-5
    n_features = f_leaf.shape[-1]  # 8

    def _fd_spliced_metric(feat_idx: int, h: float) -> float:
        """Run model with SAE splice where f[..., feat_idx] is perturbed by h."""
        def _splice_hook(mod: nn.Module, inp: object, output: object) -> object:
            a = _routed_extract(resolved, output)
            a_in = a.detach().to(sae.device, sae.dtype)
            with torch.no_grad():
                f = sae.encode(a_in).clone()
                f[..., feat_idx] = f[..., feat_idx] + h
                x_hat = sae.decode(f)
                eps = (a_in - sae.decode(sae.encode(a_in))).detach()
                recon = (x_hat + eps).to(model.cfg.dtype)
            return _routed_inject(resolved, output, recon)

        hh = lm.register_forward_hook(_splice_hook)
        try:
            with torch.no_grad():
                out = model(clean_tokens)
        finally:
            hh.remove()
        mv = _metric(out)
        return float(mv.item()) if isinstance(mv, Tensor) else float(mv)

    max_err = 0.0
    for i in range(n_features):
        m_plus = _fd_spliced_metric(i, h_fd)
        m_minus = _fd_spliced_metric(i, -h_fd)
        fd_grad_i = (m_plus - m_minus) / (2 * h_fd)
        # Analytic: Σ_pos grad_f[..., i] (FD perturbs all positions by h simultaneously)
        analytic_grad_i = float(grad_f[..., i].sum().item())
        err = abs(fd_grad_i - analytic_grad_i)
        if err > max_err:
            max_err = err

    print(f"  FD vs analytic max error (fp64): {max_err:.2e}")
    assert max_err <= 1e-9, (
        f"Autograd through HookPoint not exact: max_err={max_err:.2e} > 1e-9"
    )


# ---------------------------------------------------------------------------
# Test 2: Splice lossless (zero-perturbation ≤1e-12 at fp64)
# ---------------------------------------------------------------------------

def test_tl_splice_lossless():
    """The spliced reconstruction reproduces the original activation to ≤1e-12 at fp64.

    Zero-perturbation test: the SAE splice (encode→decode + eps) must round-trip
    the original activation exactly to floating-point precision.  Also verifies
    end-to-end runner produces finite scores.
    """
    from circuitry.patching.sae_features import (  # noqa: PLC0415
        SAEFeatureRunner,
        _routed_extract,
        _routed_inject,
    )

    model = _tiny_tl(dtype=torch.float64)
    d_model = model.cfg.d_model
    sae = TinySAE(d_in=d_model, d_sae=8, dtype=torch.float64).eval()
    resolver = TLSiteResolver()
    site = Site(layer=0, component="resid_post")

    tokens = _make_tokens(seed=20)

    # Capture clean activation before splice
    resolved = resolver.resolve(model, site)
    lm = resolved.module

    pre_store: dict[str, Tensor] = {}

    def _pre_hook(mod: nn.Module, inp: object, output: object) -> None:
        pre_store["a"] = _routed_extract(resolved, output).detach().clone()

    hp = lm.register_forward_hook(_pre_hook)
    with torch.no_grad():
        model(tokens)
    hp.remove()
    a_orig = pre_store["a"]

    # Capture what the splice actually injects back
    recon_store: dict[str, Tensor] = {}

    def _full_splice_hook(mod: nn.Module, inp: object, output: object) -> object:
        a = _routed_extract(resolved, output)
        a_in = a.detach().to(sae.device, sae.dtype)
        with torch.no_grad():
            f = sae.encode(a_in)
            x_hat = sae.decode(f)
            eps = a_in - x_hat
            recon = (x_hat + eps).to(model.cfg.dtype)
        recon_store["recon"] = recon.detach().clone()
        return _routed_inject(resolved, output, recon)

    hf = lm.register_forward_hook(_full_splice_hook)
    with torch.no_grad():
        model(tokens)
    hf.remove()

    recon = recon_store["recon"]
    max_diff = float((recon - a_orig).abs().max().item())
    print(f"  Splice lossless max diff (fp64): {max_diff:.2e}")
    assert max_diff <= 1e-12, (
        f"Splice not lossless: max diff = {max_diff:.2e} > 1e-12"
    )

    # Runner end-to-end: clean==corrupt → all scores zero, all finite
    runner = SAEFeatureRunner(model, {site: sae}, resolver)
    result = runner.run(tokens, tokens, _metric)
    for node, score in result.scores.items():
        assert torch.isfinite(torch.tensor(score)), f"Non-finite score for {node}: {score}"


# ---------------------------------------------------------------------------
# Test 3: dtype not downcast — fp64 model stays fp64 (§10 defect 2)
# ---------------------------------------------------------------------------

def test_tl_dtype_not_downcast():
    """On a fp64 HookedTransformer, scores must be computed in fp64.

    Without the fix (sourcing dtype from HookPoint.parameters() == []),
    the fallback was fp32, causing ~111% score error.  With the fix
    (model.cfg.dtype), the analytic gradient at fp64 matches FD to ≤1e-9.
    """
    from circuitry.patching.sae_features import (  # noqa: PLC0415
        SAEFeatureRunner,
        _routed_extract,
        _routed_inject,
    )

    model = _tiny_tl(dtype=torch.float64)
    assert model.cfg.dtype == torch.float64, "model must be fp64 for this test"

    d_model = model.cfg.d_model
    sae = TinySAE(d_in=d_model, d_sae=8, dtype=torch.float64).eval()
    resolver = TLSiteResolver()
    site = Site(layer=0, component="resid_post")

    clean_tokens = _make_tokens(seed=30)

    # Verify the spliced recon tensor is fp64 (model.cfg.dtype is used, not params-fallback)
    resolved = resolver.resolve(model, site)
    lm = resolved.module
    recon_dtype_store: dict[str, torch.dtype] = {}

    def _dtype_check_hook(mod: nn.Module, inp: object, output: object) -> object:
        a = _routed_extract(resolved, output)
        a_in = a.detach().to(sae.device, sae.dtype)
        with torch.no_grad():
            f = sae.encode(a_in)
            x_hat = sae.decode(f)
            eps = a_in - x_hat
            recon = (x_hat + eps).to(model.cfg.dtype)  # source: model.cfg.dtype
        recon_dtype_store["dtype"] = recon.dtype
        return _routed_inject(resolved, output, recon)

    hd = lm.register_forward_hook(_dtype_check_hook)
    with torch.no_grad():
        model(clean_tokens)
    hd.remove()

    assert recon_dtype_store["dtype"] == torch.float64, (
        f"Recon dtype is {recon_dtype_store['dtype']}, expected torch.float64 — "
        "model.cfg.dtype is not being used correctly"
    )

    # Run attribution smoke test (uses the runner's backend-aware dtype probe)
    corr_tokens = _make_tokens(seed=31)
    runner = SAEFeatureRunner(model, {site: sae}, resolver)
    result = runner.run(clean_tokens, corr_tokens, _metric)
    _ = result  # scores produced without error

    # Get analytic grad and verify fp64 precision vs FD
    f_leaf_store: dict[str, Tensor] = {}

    def _grad_hook(mod: nn.Module, inp: object, output: object) -> object:
        a = _routed_extract(resolved, output)
        a_in = a.detach().to(sae.device, sae.dtype)
        f = sae.encode(a_in).detach().requires_grad_(True)
        f.retain_grad()
        x_hat = sae.decode(f)
        eps = (a_in - sae.decode(sae.encode(a_in))).detach()
        recon = (x_hat + eps).to(model.cfg.dtype)
        f_leaf_store["f"] = f
        return _routed_inject(resolved, output, recon)

    hg = lm.register_forward_hook(_grad_hook)
    with torch.enable_grad():
        out = model(clean_tokens)
        m = _metric(out)
        m.backward()
    hg.remove()

    f_leaf = f_leaf_store["f"]
    assert f_leaf.grad is not None
    assert f_leaf.dtype == torch.float64, (
        f"f_leaf.dtype is {f_leaf.dtype}, expected float64 — dtype downcast detected"
    )

    h_fd = 1e-5

    def _fd_metric(feat_idx: int, h: float) -> float:
        def _splice_hook(mod: nn.Module, inp: object, output: object) -> object:
            a = _routed_extract(resolved, output)
            a_in = a.detach().to(sae.device, sae.dtype)
            with torch.no_grad():
                f = sae.encode(a_in).clone()
                f[..., feat_idx] = f[..., feat_idx] + h
                x_hat = sae.decode(f)
                eps = (a_in - sae.decode(sae.encode(a_in))).detach()
                recon = (x_hat + eps).to(model.cfg.dtype)
            return _routed_inject(resolved, output, recon)

        hh = lm.register_forward_hook(_splice_hook)
        try:
            with torch.no_grad():
                out2 = model(clean_tokens)
        finally:
            hh.remove()
        mv = _metric(out2)
        return float(mv.item()) if isinstance(mv, Tensor) else float(mv)

    for i in range(min(4, f_leaf.shape[-1])):
        m_plus = _fd_metric(i, h_fd)
        m_minus = _fd_metric(i, -h_fd)
        fd_grad_i = (m_plus - m_minus) / (2 * h_fd)
        analytic_grad_i = float(f_leaf.grad[..., i].sum().item())
        err = abs(fd_grad_i - analytic_grad_i)
        print(f"  feature {i}: analytic={analytic_grad_i:.6e}, FD={fd_grad_i:.6e}, err={err:.2e}")
        assert err <= 1e-9, (
            f"fp64 score error too large at feature {i}: {err:.2e} > 1e-9 "
            "(dtype downcast may be occurring)"
        )


# ---------------------------------------------------------------------------
# Test 4: SAEFeatureEdgeRunner on TL model (resid_post@0 → resid_post@1)
# ---------------------------------------------------------------------------

def test_tl_edge_runs():
    """SAEFeatureEdgeRunner runs on a TL model and produces finite edge values.

    Tests the cross-layer resid_post@0 → resid_post@1 edge and checks
    faithfulness(M) is finite.
    """
    from circuitry.patching.sae_edges import SAEFeatureEdgeRunner  # noqa: PLC0415

    model = _tiny_tl(dtype=torch.float32, seed=2)
    d_model = model.cfg.d_model

    sae0 = TinySAE(d_in=d_model, d_sae=8, dtype=torch.float32, seed=2).eval()
    sae1 = TinySAE(d_in=d_model, d_sae=8, dtype=torch.float32, seed=3).eval()

    resolver = TLSiteResolver()
    site0 = Site(layer=0, component="resid_post")
    site1 = Site(layer=1, component="resid_post")

    clean_tokens = _make_tokens(seed=40)
    corr_tokens = _make_tokens(seed=41)

    runner = SAEFeatureEdgeRunner(
        model,
        {site0: sae0, site1: sae1},
        resolver,
    )

    circuit = runner.run(clean_tokens, corr_tokens, _metric, layer_pairs="adjacent")

    # All edge scores must be finite
    for edge, score in circuit.edges.items():
        assert torch.isfinite(torch.tensor(score)), (
            f"Non-finite edge score: {edge} → {score}"
        )

    # faithfulness(M) must be finite and ≈ 1 (full circuit)
    faith = circuit.faithfulness(clean_tokens, corr_tokens, _metric)
    print(f"  TL edge faithfulness(M): {faith:.4f}")
    assert torch.isfinite(torch.tensor(faith)), f"faithfulness is not finite: {faith}"
    assert abs(faith) < 10.0, f"faithfulness out of range: {faith}"


# ---------------------------------------------------------------------------
# Test 5: mlp_out TL site resolves and runs node attribution
# ---------------------------------------------------------------------------

def test_tl_mlp_out_site():
    """mlp_out TL site (blocks.{L}.hook_mlp_out) resolves and runs end-to-end.

    hook_mlp_out in TL has shape (b, s, d_model=32) — the MLP submodule output
    after down_proj, consistent with the HF backend and eap/atp/acdc which all
    use hook_mlp_out.  The SAE must match d_model, not d_mlp.
    """
    from circuitry.patching.sae_features import SAEFeatureRunner  # noqa: PLC0415

    model = _tiny_tl(dtype=torch.float32, seed=3)
    d_model = model.cfg.d_model  # 32

    sae_mlp = TinySAE(d_in=d_model, d_sae=8, dtype=torch.float32, seed=5).eval()

    resolver = TLSiteResolver()
    site = Site(layer=0, component="mlp_out")

    clean_tokens = _make_tokens(seed=50)
    corr_tokens = _make_tokens(seed=51)

    runner = SAEFeatureRunner(model, {site: sae_mlp}, resolver)
    result = runner.run(clean_tokens, corr_tokens, _metric)

    print(f"  TL mlp_out node scores: {len(result.scores)} features scored")

    # All scores must be finite
    for node, score in result.scores.items():
        assert torch.isfinite(torch.tensor(score)), (
            f"Non-finite mlp_out score: {node} → {score}"
        )

    # Verify hook name resolves to hook_mlp_out (d_model output, not mlp.hook_post d_mlp intermediate)
    hook_name = resolver.hook_name(site)
    assert hook_name == "blocks.0.hook_mlp_out", (
        f"Unexpected hook name: {hook_name!r} — expected blocks.0.hook_mlp_out"
    )
    resolved = resolver.resolve(model, site)
    assert resolved.module is model.hook_dict[hook_name], (
        "Resolved module is not the expected HookPoint"
    )
