"""SAE feature attribution tests. v1.5.0.

GATE A (exact, abs=1e-4): analytic AtP == bruteforce on linear downstream model.
GATE B (correlation): Spearman/Pearson/sign agreement on nonlinear model.
Robustness + invariant tests per spec §4.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torch import Tensor

# ---------------------------------------------------------------------------
# Fixtures: LinearResidToy and SyntheticSAE
# ---------------------------------------------------------------------------

class LinearResidLayer(nn.Module):
    """A single layer of the LinearResidToy — a plain linear transform that
    returns the residual-stream value.  A resid_post forward hook on this
    module fires on the OUTPUT of forward().
    """
    def __init__(self, d: int) -> None:
        super().__init__()
        self.linear = nn.Linear(d, d, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.linear(x)  # residual write


class LinearResidToy(nn.Module):
    """Linear residual stack.  Each layer is an nn.Module RETURNING the residual
    stream, so a resid_post hook on layers[L] fires correctly.

    layer_pattern for HFSiteResolver: "layers.{L}"
    """
    def __init__(self, n_layers: int = 2, d: int = 8) -> None:
        super().__init__()
        self.layers = nn.ModuleList([LinearResidLayer(d) for _ in range(n_layers)])
        self.lm_head = nn.Linear(d, d, bias=False)
        torch.manual_seed(42)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.3)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(x)


class NonlinearResidToy(nn.Module):
    """Residual stack with ReLU nonlinearity downstream of the SAE site."""
    def __init__(self, n_layers: int = 2, d: int = 8) -> None:
        super().__init__()
        self.layers = nn.ModuleList([LinearResidLayer(d) for _ in range(n_layers)])
        self.lm_head = nn.Linear(d, d, bias=False)
        torch.manual_seed(7)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.3)

    def forward(self, x: Tensor) -> Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i == 0:
                # Insert ReLU after first layer (downstream of SAE site at layer 0)
                x = torch.relu(x)
        return self.lm_head(x)


class _SyntheticSAECfg:
    """Mimics sae_lens SAEConfig enough for assert_supported_sae."""
    def __init__(self, arch: str = "standard", d_sae: int = 16,
                 normalize_activations: str = "none") -> None:
        self._arch = arch
        self.d_sae = d_sae
        self.normalize_activations = normalize_activations

    def architecture(self) -> str:
        return self._arch


class SyntheticSAE(nn.Module):
    """Synthetic SAE with nn.Parameter W_enc/b_enc/W_dec/b_dec so the
    freeze-during-attribution is genuinely tested (default requires_grad=True).

    Variants:
      relu=False (default): affine encode — f = x @ W_enc.T + b_enc
      relu=True:            ReLU encode — f = relu(x @ W_enc.T + b_enc)
      topk=True:            hard top-k mask on f (jumprelu-style)
    """

    def __init__(
        self,
        d_model: int = 8,
        d_sae: int = 16,
        *,
        relu: bool = False,
        topk: int | None = None,
        arch: str = "standard",
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_sae = d_sae
        self.device = torch.device(device)
        self.dtype = dtype
        self._relu = relu
        self._topk = topk
        self.cfg = _SyntheticSAECfg(arch=arch, d_sae=d_sae)

        # Parameters: all default requires_grad=True — freeze contract tested here
        self.W_enc = nn.Parameter(torch.empty(d_sae, d_model, dtype=dtype))
        self.b_enc = nn.Parameter(torch.zeros(d_sae, dtype=dtype))
        self.W_dec = nn.Parameter(torch.empty(d_model, d_sae, dtype=dtype))
        self.b_dec = nn.Parameter(torch.zeros(d_model, dtype=dtype))

        nn.init.normal_(self.W_enc, std=0.3)
        nn.init.normal_(self.W_dec, std=0.3)

    def to(self, *args, **kwargs):  # type: ignore[override]
        result = super().to(*args, **kwargs)
        # Keep device/dtype attributes in sync
        for p in result.parameters():
            result.device = p.device
            result.dtype = p.dtype
            break
        return result

    def encode(self, x: Tensor) -> Tensor:
        f = x @ self.W_enc.T + self.b_enc
        if self._relu:
            f = torch.relu(f)
        if self._topk is not None:
            # Hard top-k mask: zero all but top-k activations per token
            k = min(self._topk, f.shape[-1])
            topk_vals, topk_idx = torch.topk(f.abs(), k, dim=-1)
            mask = torch.zeros_like(f)
            mask.scatter_(-1, topk_idx, 1.0)
            f = f * mask
        return f

    def decode(self, f: Tensor) -> Tensor:
        return f @ self.W_dec.T + self.b_dec


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def linear_toy():
    torch.manual_seed(0)
    return LinearResidToy(n_layers=2, d=8)


@pytest.fixture
def nonlinear_toy():
    torch.manual_seed(1)
    return NonlinearResidToy(n_layers=2, d=8)


@pytest.fixture
def affine_sae():
    torch.manual_seed(2)
    return SyntheticSAE(d_model=8, d_sae=16, relu=False)


@pytest.fixture
def relu_sae():
    torch.manual_seed(3)
    return SyntheticSAE(d_model=8, d_sae=16, relu=True)


@pytest.fixture
def topk_sae():
    torch.manual_seed(4)
    # topk=4 leaves some features active, some inactive — exactly what the
    # inactive-feature-has-nonzero-score test needs
    return SyntheticSAE(d_model=8, d_sae=16, relu=False, topk=4)


def _make_resolver(d: int = 8):
    from circuitry.patching.sites import HFSiteResolver
    # n_heads=1, head_dim=d avoids division-by-zero in HFSiteResolver while
    # keeping the toy model head-free in practice
    return HFSiteResolver(n_heads=1, d_model=d, head_dim=d, layer_pattern="layers.{L}")


def _metric(out: Tensor) -> Tensor:
    from circuitry.core.patching import logit_diff_t
    return logit_diff_t(out, correct=0, incorrect=1)


def _make_clean_corr(d: int = 8, b: int = 1, s: int = 3, seed: int = 0):
    torch.manual_seed(seed)
    clean = torch.randn(b, s, d)
    corrupted = torch.randn(b, s, d)
    return clean, corrupted


# ---------------------------------------------------------------------------
# GATE A — exact (abs=1e-4)
# ---------------------------------------------------------------------------

def test_feature_atp_matches_bruteforce_linear(linear_toy, affine_sae):
    """AtP feature scores == bruteforce_feature_scores on a linear model (affine SAE)."""
    from circuitry.patching.sae_features import SAEFeatureRunner
    from circuitry.patching.sites import Site

    clean, corrupted = _make_clean_corr()
    site = Site("resid_post", layer=0)
    runner = SAEFeatureRunner(linear_toy, {site: affine_sae}, _make_resolver())

    result = runner.run(clean, corrupted, _metric)
    nodes = list(result.scores.keys())

    if not nodes:
        pytest.skip("No active features — try a different seed")

    bf = runner.bruteforce_feature_scores(clean, corrupted, _metric, nodes)

    max_diff = 0.0
    for node in nodes:
        atp_score = result.scores[node]
        bf_score = bf.get(node, 0.0)
        diff = abs(atp_score - bf_score)
        if diff > max_diff:
            max_diff = diff

    print(f"\n[GATE A affine] max|AtP - bruteforce| = {max_diff:.2e}")
    assert max_diff < 1e-4, f"max|diff|={max_diff:.2e} exceeds 1e-4 gate"


def test_relu_encode_still_exact_on_linear_model(linear_toy, relu_sae):
    """ReLU-encode SAE on linear model: decode is still affine in f, so exact gate holds."""
    from circuitry.patching.sae_features import SAEFeatureRunner
    from circuitry.patching.sites import Site

    clean, corrupted = _make_clean_corr(seed=5)
    site = Site("resid_post", layer=0)
    runner = SAEFeatureRunner(linear_toy, {site: relu_sae}, _make_resolver())

    result = runner.run(clean, corrupted, _metric)
    nodes = list(result.scores.keys())

    if not nodes:
        pytest.skip("No active features")

    bf = runner.bruteforce_feature_scores(clean, corrupted, _metric, nodes)

    max_diff = 0.0
    for node in nodes:
        diff = abs(result.scores[node] - bf.get(node, 0.0))
        max_diff = max(max_diff, diff)

    print(f"\n[GATE A relu] max|AtP - bruteforce| = {max_diff:.2e}")
    assert max_diff < 1e-4, f"max|diff|={max_diff:.2e} exceeds 1e-4 gate"


def test_error_term_makes_splice_lossless(linear_toy, affine_sae):
    """‖recon − a‖∞ < 1e-5 — splice is lossless."""
    from circuitry.sae.grad import sae_decompose

    clean, _ = _make_clean_corr()
    a = clean  # (1, 3, 8)
    a_sae = a.to(affine_sae.device, affine_sae.dtype)

    f, x_hat, eps = sae_decompose(affine_sae, a_sae)
    recon = x_hat + eps

    diff = (recon - a_sae).abs().max().item()
    print(f"\n[lossless] ‖recon − a‖∞ = {diff:.2e}")
    assert diff < 1e-5, f"splice not lossless: ‖recon − a‖∞ = {diff:.2e}"


def test_error_node_exact_on_linear(linear_toy, affine_sae):
    """sae_error AtP == bruteforce error patching on a linear model."""
    from circuitry.patching.sae_features import SAEFeatureRunner
    from circuitry.patching.sites import Site

    clean, corrupted = _make_clean_corr(seed=10)
    site = Site("resid_post", layer=0)
    runner = SAEFeatureRunner(linear_toy, {site: affine_sae}, _make_resolver())

    result = runner.run(clean, corrupted, _metric, include_error_node=True)

    # Find error nodes in result
    err_nodes = [n for n in result.scores if n.node.kind == "sae_error"]
    if not err_nodes:
        pytest.skip("No error node returned (possibly Δeps == 0)")

    bf = runner.bruteforce_feature_scores(clean, corrupted, _metric, err_nodes)

    for err_node in err_nodes:
        atp_score = result.scores[err_node]
        bf_score = bf.get(err_node, 0.0)
        diff = abs(atp_score - bf_score)
        print(f"\n[GATE A error] AtP={atp_score:.6f}, bf={bf_score:.6f}, diff={diff:.2e}")
        assert diff < 1e-4, f"error node: max|diff|={diff:.2e} exceeds 1e-4"


def test_feature_scores_identical_with_error_node_on_off(linear_toy, affine_sae):
    """Feature scores must be bit-identical whether include_error_node is True or False."""
    from circuitry.patching.sae_features import SAEFeatureRunner
    from circuitry.patching.sites import Site

    clean, corrupted = _make_clean_corr(seed=11)
    site = Site("resid_post", layer=0)
    runner = SAEFeatureRunner(linear_toy, {site: affine_sae}, _make_resolver())

    result_off = runner.run(clean, corrupted, _metric, include_error_node=False)
    result_on = runner.run(clean, corrupted, _metric, include_error_node=True)

    # Filter to only sae_feature nodes
    feat_nodes_off = {n: s for n, s in result_off.scores.items() if n.node.kind == "sae_feature"}
    feat_nodes_on = {n: s for n, s in result_on.scores.items() if n.node.kind == "sae_feature"}

    assert set(feat_nodes_off.keys()) == set(feat_nodes_on.keys()), \
        "Feature node sets differ with error_node on vs off"

    for node in feat_nodes_off:
        s_off = feat_nodes_off[node]
        s_on = feat_nodes_on[node]
        assert s_off == s_on, (
            f"Feature score differs: error_node=off={s_off}, on={s_on} for {node}"
        )


def test_model_clean_after_feature_atp(linear_toy, affine_sae):
    """Model output must be unchanged after run() — frozen/restored contract."""
    from circuitry.patching.sae_features import SAEFeatureRunner
    from circuitry.patching.sites import Site

    clean, corrupted = _make_clean_corr(seed=12)
    site = Site("resid_post", layer=0)
    runner = SAEFeatureRunner(linear_toy, {site: affine_sae}, _make_resolver())

    out_before = linear_toy(clean).clone().detach()
    runner.run(clean, corrupted, _metric)
    out_after = linear_toy(clean).detach()

    assert torch.allclose(out_before, out_after), \
        "Model output changed after run() — frozen/restored contract violated"


def test_no_sae_param_grad_leak(linear_toy, affine_sae):
    """SAE params must NOT accumulate .grad during run()."""
    for p in affine_sae.parameters():
        p.requires_grad_(True)
        p.grad = None

    from circuitry.patching.sae_features import SAEFeatureRunner
    from circuitry.patching.sites import Site

    clean, corrupted = _make_clean_corr(seed=13)
    site = Site("resid_post", layer=0)
    runner = SAEFeatureRunner(linear_toy, {site: affine_sae}, _make_resolver())
    runner.run(clean, corrupted, _metric)

    leaked = [n for n, p in affine_sae.named_parameters() if p.grad is not None]
    assert not leaked, f"SAE param grad leaked: {leaked}"
    # requires_grad restored
    assert all(p.requires_grad for p in affine_sae.parameters()), \
        "SAE param requires_grad not restored"


def test_no_model_param_grad_leak(linear_toy, affine_sae):
    """Model params must NOT accumulate .grad during run()."""
    for p in linear_toy.parameters():
        p.requires_grad_(True)
        p.grad = None

    from circuitry.patching.sae_features import SAEFeatureRunner
    from circuitry.patching.sites import Site

    clean, corrupted = _make_clean_corr(seed=14)
    site = Site("resid_post", layer=0)
    runner = SAEFeatureRunner(linear_toy, {site: affine_sae}, _make_resolver())
    runner.run(clean, corrupted, _metric)

    leaked = [n for n, p in linear_toy.named_parameters() if p.grad is not None]
    assert not leaked, f"Model param grad leaked: {leaked}"


def test_inactive_feature_has_nonzero_score(linear_toy):
    """Clean-inactive / corrupted-active feature scores nonzero AND equals bruteforce.

    This guards the corrected enumeration rule (union Δf ≠ 0, NOT clean-live-only).
    """
    from circuitry.patching.atp import AtPNode
    from circuitry.patching.graph import Node
    from circuitry.patching.sae_features import SAEFeatureRunner
    from circuitry.patching.sites import Site

    # topk=4 means only 4 features active per token — many are inactive on clean
    # but may be active on corrupted (different inputs → different top-k)
    torch.manual_seed(99)
    sae = SyntheticSAE(d_model=8, d_sae=16, relu=False, topk=4)

    torch.manual_seed(55)
    clean = torch.randn(1, 3, 8)
    corrupted = torch.randn(1, 3, 8)

    site = Site("resid_post", layer=0)
    runner = SAEFeatureRunner(linear_toy, {site: sae}, _make_resolver())

    # Identify which features are clean-inactive but corrupted-active
    with torch.no_grad():
        # Run the layer to get clean and corrupted activations
        h_clean = clean
        for i, layer in enumerate(linear_toy.layers):
            h_clean = layer(h_clean)
            if i == 0:
                break

        h_corr = corrupted
        for i, layer in enumerate(linear_toy.layers):
            h_corr = layer(h_corr)
            if i == 0:
                break

        f_clean_np = sae.encode(h_clean.to(sae.device, sae.dtype)).detach()
        f_corr_np = sae.encode(h_corr.to(sae.device, sae.dtype)).detach()

    # Features that are clean-INACTIVE (all positions zero) but corrupted-ACTIVE (some nonzero)
    feature_dim = f_clean_np.shape[-1]
    clean_inactive = (f_clean_np.reshape(-1, feature_dim).abs() == 0).all(0)
    corr_active = (f_corr_np.reshape(-1, feature_dim).abs() > 0).any(0)
    interesting = (clean_inactive & corr_active).nonzero(as_tuple=True)[0].tolist()

    if not interesting:
        pytest.skip("No clean-inactive/corrupted-active features in this random seed — skip")

    result = runner.run(clean, corrupted, _metric)

    found_nonzero = False
    for i in interesting:
        node = AtPNode(Node("sae_feature", layer=0, neuron=i))
        score = result.scores.get(node)
        if score is not None and abs(score) > 0:
            found_nonzero = True
            # Also verify vs bruteforce
            bf = runner.bruteforce_feature_scores(clean, corrupted, _metric, [node])
            bf_score = bf.get(node, 0.0)
            diff = abs(score - bf_score)
            print(f"\n[inactive-feature] feat={i}, AtP={score:.6f}, bf={bf_score:.6f}, diff={diff:.2e}")
            assert diff < 1e-4, f"Inactive feature {i}: diff={diff:.2e}"

    assert found_nonzero, (
        f"All clean-inactive/corrupted-active features have score=0 or are missing: "
        f"features {interesting}. This means they were NOT enumerated — "
        f"the corrected Δf≠0 rule is not being applied."
    )


# ---------------------------------------------------------------------------
# GATE B — correlation (nonlinear model)
# ---------------------------------------------------------------------------

def test_feature_atp_correlates_on_nonlinear(nonlinear_toy, affine_sae):
    """On a nonlinear model: Spearman ≥ 0.7, Pearson ≥ 0.6, sign-agreement ≥ 0.9."""
    import numpy as np
    from scipy.stats import pearsonr, spearmanr

    from circuitry.patching.sae_features import SAEFeatureRunner
    from circuitry.patching.sites import Site

    torch.manual_seed(20)
    clean = torch.randn(1, 4, 8)
    corrupted = torch.randn(1, 4, 8)

    site = Site("resid_post", layer=0)
    runner = SAEFeatureRunner(nonlinear_toy, {site: affine_sae}, _make_resolver())

    result = runner.run(clean, corrupted, _metric)
    nodes = list(result.scores.keys())

    if len(nodes) < 5:
        pytest.skip("Too few active features for correlation test")

    bf = runner.bruteforce_feature_scores(clean, corrupted, _metric, nodes)

    atp_arr = np.array([result.scores[n] for n in nodes], dtype=np.float64)
    bf_arr = np.array([bf.get(n, 0.0) for n in nodes], dtype=np.float64)

    # Filter to nodes with any signal
    mask = (np.abs(atp_arr) + np.abs(bf_arr)) > 1e-10
    if mask.sum() < 5:
        pytest.skip("Too few scored features for correlation test")

    atp_arr = atp_arr[mask]
    bf_arr = bf_arr[mask]

    spearman_r, _ = spearmanr(atp_arr, bf_arr)
    pearson_r, _ = pearsonr(atp_arr, bf_arr)

    signs_agree = np.sign(atp_arr) == np.sign(bf_arr)
    sign_agreement = signs_agree.mean()

    print(f"\n[GATE B] Spearman={spearman_r:.3f}, Pearson={pearson_r:.3f}, "
          f"sign_agreement={sign_agreement:.3f}, n={mask.sum()}")

    assert spearman_r >= 0.7, f"Spearman {spearman_r:.3f} < 0.7"
    assert pearson_r >= 0.6, f"Pearson {pearson_r:.3f} < 0.6"
    assert sign_agreement >= 0.9, f"Sign agreement {sign_agreement:.3f} < 0.9"


# ---------------------------------------------------------------------------
# Robustness tests
# ---------------------------------------------------------------------------

def test_unsupported_architecture_error(linear_toy):
    """NotImplementedError for transcoder architecture."""
    from circuitry.patching.sae_features import SAEFeatureRunner
    from circuitry.patching.sites import Site

    transcoder_sae = SyntheticSAE(d_model=8, d_sae=16, arch="transcoder")
    site = Site("resid_post", layer=0)

    with pytest.raises(NotImplementedError, match="transcoder"):
        SAEFeatureRunner(linear_toy, {site: transcoder_sae}, _make_resolver())


def test_tl_not_implemented(linear_toy, affine_sae):
    """SAEFeatureRunner raises NotImplementedError for TLSiteResolver."""
    from circuitry.patching.sae_features import SAEFeatureRunner
    from circuitry.patching.sites import Site, TLSiteResolver

    site = Site("resid_post", layer=0)
    with pytest.raises(NotImplementedError, match="TL|TransformerLens"):
        SAEFeatureRunner(linear_toy, {site: affine_sae}, TLSiteResolver())


def test_non_resid_post_site_error(linear_toy, affine_sae):
    """NotImplementedError for non-resid_post sites."""
    from circuitry.patching.sae_features import SAEFeatureRunner
    from circuitry.patching.sites import Site

    site = Site("mlp_out", layer=0)
    with pytest.raises(NotImplementedError, match="resid_post"):
        SAEFeatureRunner(linear_toy, {site: affine_sae}, _make_resolver())


def test_metric_must_be_differentiable(linear_toy, affine_sae):
    """Clear error if metric returns a non-differentiable or f-independent value.

    Two cases:
      (1) Metric returns a constant no-grad tensor → backward() raises autograd error
          BEFORE reaching the bespoke f.grad guard.
      (2) Metric returns a differentiable tensor disconnected from f → backward() succeeds
          but f.grad stays None → the bespoke RuntimeError("f.grad is None") fires.
    Both must be caught.
    """
    from circuitry.patching.sae_features import SAEFeatureRunner
    from circuitry.patching.sites import Site

    clean, corrupted = _make_clean_corr()
    site = Site("resid_post", layer=0)
    runner = SAEFeatureRunner(linear_toy, {site: affine_sae}, _make_resolver())

    # Case 1: constant tensor — autograd error (no grad_fn) fires at m.backward()
    def bad_metric_no_grad(out: Tensor) -> Tensor:
        return torch.tensor(0.0)

    with pytest.raises(RuntimeError, match="[Gg]rad|differentiable"):
        runner.run(clean, corrupted, bad_metric_no_grad)

    # Case 2: differentiable tensor that is INDEPENDENT of the model output and of f.
    # backward() succeeds (the leaf has a grad_fn via .sum()), but since f_clean_leaf
    # is not in this computation graph, f_clean_leaf.grad stays None.
    # This directly exercises the bespoke guard at sae_features.py:323-328.
    def bad_metric_f_independent(out: Tensor) -> Tensor:
        # A fresh leaf through a trivial op — differentiable, but has nothing to do
        # with the model output or f.  backward() will complete without error yet
        # f.grad will remain None.
        leaf = torch.zeros(1, requires_grad=True)
        return (leaf * 0.0).sum()

    with pytest.raises(RuntimeError, match="f\\.grad is None"):
        runner.run(clean, corrupted, bad_metric_f_independent)


def test_feature_grad_device_align(linear_toy):
    """SAE dtype ≠ model dtype: results should still work (fp32 model, fp16 SAE)."""
    import math
    # Use float16 SAE on a float32 model — device alignment test
    torch.manual_seed(30)
    sae_fp16 = SyntheticSAE(d_model=8, d_sae=16, relu=False, dtype=torch.float16)

    from circuitry.patching.sae_features import SAEFeatureRunner
    from circuitry.patching.sites import Site

    clean, corrupted = _make_clean_corr()
    site = Site("resid_post", layer=0)
    runner = SAEFeatureRunner(linear_toy, {site: sae_fp16}, _make_resolver())

    # Should not raise
    result = runner.run(clean, corrupted, _metric)
    assert isinstance(result.scores, dict)

    # Must have at least some active features — empty result hides bugs
    assert len(result.scores) > 0, (
        "fp16-SAE runner returned no features: empty/zeroed result hiding a bug"
    )

    # All scores must be finite (no NaN/inf from dtype alignment errors)
    for node, score in result.scores.items():
        assert math.isfinite(score), (
            f"Non-finite score {score!r} for node {node} — dtype alignment error"
        )

    # Scores must match bruteforce on the linear toy at fp16 tolerance
    nodes = list(result.scores.keys())
    bf = runner.bruteforce_feature_scores(clean, corrupted, _metric, nodes)

    max_diff = 0.0
    for node in nodes:
        diff = abs(result.scores[node] - bf.get(node, 0.0))
        max_diff = max(max_diff, diff)

    print(f"\n[fp16-SAE device align] max|AtP - bruteforce| = {max_diff:.4e}")
    assert max_diff < 5e-3, (
        f"fp16-SAE scores deviate from bruteforce by {max_diff:.4e} (threshold 5e-3)"
    )


def test_max_features_cap(linear_toy, affine_sae):
    """max_features limits the number of returned feature nodes."""
    from circuitry.patching.sae_features import SAEFeatureRunner
    from circuitry.patching.sites import Site

    clean, corrupted = _make_clean_corr(seed=40)
    site = Site("resid_post", layer=0)
    runner = SAEFeatureRunner(linear_toy, {site: affine_sae}, _make_resolver())

    result_all = runner.run(clean, corrupted, _metric)
    n_all = len(result_all.scores)

    if n_all < 3:
        pytest.skip("Too few features for cap test")

    cap = max(1, n_all // 2)
    result_capped = runner.run(clean, corrupted, _metric, max_features=cap)
    feat_nodes = [n for n in result_capped.scores if n.node.kind == "sae_feature"]
    assert len(feat_nodes) <= cap, (
        f"max_features={cap} not respected, got {len(feat_nodes)} feature nodes"
    )

    # Capped features should be the top-|score| ones
    all_sorted = sorted(result_all.scores.items(), key=lambda kv: abs(kv[1]), reverse=True)
    top_cap_nodes = {n for n, _ in all_sorted[:cap]}
    capped_nodes = set(feat_nodes)
    assert capped_nodes <= top_cap_nodes, (
        "Capped features are not the top-|score| subset"
    )


def test_graddrop_feature_variant(linear_toy, affine_sae):
    """graddrop=True produces non-negative scores and differs from graddrop=False.

    Key invariant: graddrop score = Σ_pos |Δf_i · g_i|  (sum of abs of per-position
    products), NOT |Σ_pos Δf_i · g_i| (abs of the signed sum).  For a feature whose
    per-position contributions are SIGN-MIXED the two formulas give different results.

    This test uses a sum-over-all-positions metric so gradients are nonzero at EVERY
    sequence position (unlike logit_diff_t which is last-position-only), creating the
    sign-mixed contributions needed to distinguish Σ|.| from |Σ|.

    The expected sign-mixed behaviour is verified with seed=50, seq_len=4 where feature 1
    has per-position products ≈ [+1.10, -0.01, -0.85, -1.99] giving:
      Σ|Δf·g| ≈ 3.95   (correct graddrop formula)
      |Σ Δf·g| ≈ 1.75   (wrong |sum| formula)
    """
    from circuitry.patching.atp import AtPNode
    from circuitry.patching.graph import Node
    from circuitry.patching.sae_features import SAEFeatureRunner
    from circuitry.patching.sites import Site

    # Sum metric produces gradients at ALL positions — required for sign-mixed contributions.
    # (logit_diff_t only activates the last position, giving at most 1 nonzero per_pos
    #  per feature, which makes Σ|.| == |Σ| trivially.)
    def sum_metric(out: Tensor) -> Tensor:
        return out.sum()

    # seq_len=4 gives 4 positions for cancellations to occur
    clean, corrupted = _make_clean_corr(seed=50, s=4)
    site = Site("resid_post", layer=0)
    runner = SAEFeatureRunner(linear_toy, {site: affine_sae}, _make_resolver())

    result_std = runner.run(clean, corrupted, sum_metric, graddrop=False)
    result_gd = runner.run(clean, corrupted, sum_metric, graddrop=True)

    # GradDrop scores are non-negative
    for node, score in result_gd.scores.items():
        if node.node.kind == "sae_feature":
            assert score >= 0.0, f"graddrop score < 0: {node} -> {score}"

    # (a) At least one feature must actually differ between graddrop and signed sum
    feat_nodes = [n for n in result_std.scores if n.node.kind == "sae_feature"]
    assert feat_nodes, "No sae_feature nodes returned — cannot test graddrop"
    diffs = [abs(result_std.scores[n] - result_gd.scores.get(n, 0.0))
             for n in feat_nodes]
    assert any(d > 1e-9 for d in diffs), (
        "graddrop=True and graddrop=False gave identical scores for ALL features — "
        "the |per-position| formula is equivalent to the signed-sum formula for this "
        "input, which means sign-mixed contributions never occurred.  "
        "Either the implementation is wrong (returning |Σ|) or the input needs mixed signs."
    )

    # (b) For at least one feature with sign-mixed per-position contributions,
    #     hand-compute Σ_pos |Δf_i · g_i| and verify the runner matches to <1e-5.
    #
    # We replicate the runner's internal computation path to derive delta_f and grad_f.

    with torch.no_grad():
        h_clean = linear_toy.layers[0](clean)   # (1, 4, 8)
        h_corr = linear_toy.layers[0](corrupted)  # (1, 4, 8)

    # f_clean as a leaf requiring grad, decoded back, then forward through remainder
    f_clean_leaf = affine_sae.encode(h_clean.detach()).detach().requires_grad_(True)
    f_clean_leaf.retain_grad()
    x_hat_clean = affine_sae.decode(f_clean_leaf)
    eps_clean = h_clean.detach() - x_hat_clean.detach()
    recon_clean = x_hat_clean + eps_clean

    h1 = linear_toy.layers[1](recon_clean)
    out_clean = linear_toy.lm_head(h1)
    m_clean = sum_metric(out_clean)
    m_clean.backward()

    grad_f = f_clean_leaf.grad.float()  # (1, 4, 16)
    f_clean_f = f_clean_leaf.detach().float()

    with torch.no_grad():
        f_corr_f = affine_sae.encode(h_corr.to(affine_sae.device, affine_sae.dtype)).float()

    delta_f = f_corr_f - f_clean_f  # (1, 4, 16)

    # Find a feature where per-position contributions are sign-mixed (Σ|.| ≠ |Σ|)
    sign_mixed_feature: int | None = None
    hand_computed_score: float | None = None
    abs_sum: float | None = None
    signed_sum_abs: float | None = None

    feature_dim = delta_f.shape[-1]
    for feat_idx in range(feature_dim):
        df_i = delta_f[..., feat_idx].reshape(-1)   # (4,)
        g_i = grad_f[..., feat_idx].reshape(-1)
        per_pos = df_i * g_i
        s_abs = float(per_pos.abs().sum().item())
        s_signed_abs = abs(float(per_pos.sum().item()))
        if s_abs > 1e-4 and abs(s_abs - s_signed_abs) > 1e-4:
            sign_mixed_feature = feat_idx
            hand_computed_score = s_abs
            abs_sum = s_abs
            signed_sum_abs = s_signed_abs
            break

    if sign_mixed_feature is None:
        pytest.skip(
            "No sign-mixed feature found for hand-computed graddrop verification "
            "(all per-position contributions have the same sign for all features)"
        )

    print(
        f"\n[graddrop hand-check] feat={sign_mixed_feature}, "
        f"Σ|Δf·g|={abs_sum:.6f}, |Σ Δf·g|={signed_sum_abs:.6f}  "
        f"(differ by {abs(abs_sum - signed_sum_abs):.2e})"
    )

    # The runner's graddrop score for this feature must match Σ|.| (not |Σ|)
    node = AtPNode(Node("sae_feature", layer=0, neuron=sign_mixed_feature))
    runner_gd_score = result_gd.scores.get(node)
    assert runner_gd_score is not None, (
        f"Feature {sign_mixed_feature} (sign-mixed) not in graddrop result.scores"
    )
    diff = abs(runner_gd_score - hand_computed_score)
    assert diff < 1e-5, (
        f"graddrop score {runner_gd_score:.8f} ≠ hand-computed Σ|Δf·g|={hand_computed_score:.8f} "
        f"(diff={diff:.2e}). This would pass if the impl used |Σ Δf·g|={signed_sum_abs:.8f} "
        f"instead of the correct Σ|Δf·g|."
    )


def test_clean_corrupted_shape_mismatch_raises(linear_toy, affine_sae):
    """ValueError raised when clean and corrupted inputs have different seq lengths."""
    from circuitry.patching.sae_features import SAEFeatureRunner
    from circuitry.patching.sites import Site

    # clean: seq_len=3, corrupted: seq_len=1 — feature tensors will have different shapes
    torch.manual_seed(60)
    clean = torch.randn(1, 3, 8)
    corrupted = torch.randn(1, 1, 8)

    site = Site("resid_post", layer=0)
    runner = SAEFeatureRunner(linear_toy, {site: affine_sae}, _make_resolver())

    with pytest.raises(ValueError, match="[Ss]hape mismatch|shape"):
        runner.run(clean, corrupted, _metric)


# ---------------------------------------------------------------------------
# Additional: verify sae_decompose interface
# ---------------------------------------------------------------------------

def test_sae_decompose_returns_correct_shapes(affine_sae):
    from circuitry.sae.grad import sae_decompose

    x = torch.randn(2, 5, 8)
    f, x_hat, eps = sae_decompose(affine_sae, x)

    assert f.shape == (2, 5, 16), f"f shape wrong: {f.shape}"
    assert x_hat.shape == (2, 5, 8), f"x_hat shape wrong: {x_hat.shape}"
    assert eps.shape == (2, 5, 8), f"eps shape wrong: {eps.shape}"

    # eps must be detached
    assert not eps.requires_grad, "eps should be detached (frozen)"
    # f and x_hat should be in the graph
    assert f.requires_grad or x_hat.requires_grad, \
        "f and x_hat should be in the autograd graph"


def test_encode_features_differentiable(affine_sae):
    from circuitry.sae.grad import encode_features

    x = torch.randn(3, 8, requires_grad=True)
    f = encode_features(affine_sae, x)
    loss = f.sum()
    loss.backward()
    assert x.grad is not None, "encode_features must be differentiable"


def test_assert_supported_sae_passes_standard(affine_sae):
    from circuitry.sae.grad import assert_supported_sae
    # Should not raise
    assert_supported_sae(affine_sae)


def test_assert_supported_sae_raises_transcoder():
    from circuitry.sae.grad import assert_supported_sae
    sae = SyntheticSAE(d_model=8, d_sae=16, arch="transcoder")
    with pytest.raises(NotImplementedError):
        assert_supported_sae(sae)


def test_node_kinds_accepted():
    """sae_feature and sae_error Node kinds can be created without error."""
    from circuitry.patching.graph import Node

    feat_node = Node("sae_feature", layer=0, neuron=42)
    err_node = Node("sae_error", layer=0)

    assert feat_node.kind == "sae_feature"
    assert feat_node.neuron == 42
    assert err_node.kind == "sae_error"
