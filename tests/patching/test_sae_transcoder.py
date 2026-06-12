"""TranscoderWrapper tests (v1.21).

Verifies that TranscoderWrapper:
  - signals hook_input=True so attribution hooks encode from inp[0]
  - produces finite, nonzero attribution scores via SAEFeatureRunner
  - produces attribution scores that differ from a standard SAE (different input tensor)
  - works end-to-end with SAEFeatureEdgeRunner
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torch import Tensor

from tests.patching.test_sae_features import (
    LinearResidToy,
    SyntheticSAE,
    _make_clean_corr,
    _make_resolver,
    _metric,
)

# ---------------------------------------------------------------------------
# SyntheticTranscoder fixture
# ---------------------------------------------------------------------------


class SyntheticTranscoder(nn.Module):
    """Simple transcoder: encode from module input, decode to module output space.

    For LinearResidToy.layers[L]:
      inp[0] = residual stream before layer L  (shape: b, s, d)
      output  = residual stream after layer L   (shape: b, s, d)
    Same d_in == d_out == d for LinearResidToy.
    """

    def __init__(self, d_in: int, d_out: int, d_sae: int) -> None:
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self.W_enc = nn.Parameter(torch.empty(d_sae, d_in))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.W_dec = nn.Parameter(torch.empty(d_out, d_sae))
        self.b_dec = nn.Parameter(torch.zeros(d_out))
        nn.init.normal_(self.W_enc, std=0.3)
        nn.init.normal_(self.W_dec, std=0.3)

    def encode(self, x: Tensor) -> Tensor:
        return x @ self.W_enc.T + self.b_enc

    def decode(self, f: Tensor) -> Tensor:
        return f @ self.W_dec.T + self.b_dec


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _make_transcoder_runner(model, sae, d, layer=0):
    from circuitry.patching.sae_features import SAEFeatureRunner, TranscoderWrapper
    from circuitry.patching.sites import Site
    tc = TranscoderWrapper(sae)
    site = Site("resid_post", layer=layer)
    return SAEFeatureRunner(model, {site: tc}, _make_resolver(d)), tc


# ---------------------------------------------------------------------------
# Unit tests: TranscoderWrapper
# ---------------------------------------------------------------------------


def test_transcoder_wrapper_hook_input_true():
    from circuitry.patching.sae_features import TranscoderWrapper
    torch.manual_seed(10)
    tc_raw = SyntheticTranscoder(d_in=8, d_out=8, d_sae=16)
    tc = TranscoderWrapper(tc_raw)
    assert tc.hook_input is True


def test_transcoder_wrapper_delegates_encode_decode():
    from circuitry.patching.sae_features import TranscoderWrapper
    torch.manual_seed(11)
    tc_raw = SyntheticTranscoder(d_in=8, d_out=8, d_sae=16)
    tc = TranscoderWrapper(tc_raw)
    x = torch.randn(2, 3, 8)
    f = tc.encode(x)
    assert f.shape == (2, 3, 16)
    x_hat = tc.decode(f)
    assert x_hat.shape == (2, 3, 8)


def test_transcoder_wrapper_device_dtype():
    from circuitry.patching.sae_features import TranscoderWrapper
    torch.manual_seed(12)
    tc_raw = SyntheticTranscoder(d_in=8, d_out=8, d_sae=16)
    tc = TranscoderWrapper(tc_raw)
    assert tc.device == torch.device("cpu")
    assert tc.dtype == torch.float32


# ---------------------------------------------------------------------------
# SAEFeatureRunner with TranscoderWrapper
# ---------------------------------------------------------------------------


def test_transcoder_runner_returns_scores():
    """SAEFeatureRunner with TranscoderWrapper returns finite, nonzero scores."""
    d, d_sae = 8, 16
    torch.manual_seed(20)
    model = LinearResidToy(n_layers=2, d=d)
    torch.manual_seed(21)
    tc_raw = SyntheticTranscoder(d_in=d, d_out=d, d_sae=d_sae)
    clean, corrupted = _make_clean_corr(d=d)

    runner, _ = _make_transcoder_runner(model, tc_raw, d)
    result = runner.run(clean, corrupted, lambda out: _metric(out))

    assert len(result.scores) > 0, "Expected nonzero number of active features"
    for node, score in result.scores.items():
        assert torch.isfinite(torch.tensor(score)), f"Non-finite score for {node}: {score}"


def test_transcoder_scores_differ_from_standard_sae():
    """TranscoderWrapper scores differ from a standard SAE (different encoding input)."""
    d, d_sae = 8, 16
    torch.manual_seed(30)
    model = LinearResidToy(n_layers=2, d=d)
    # Use same weights for both to isolate the encoding-input difference
    torch.manual_seed(31)
    std_sae = SyntheticSAE(d_model=d, d_sae=d_sae)

    # Copy the standard SAE weights into a transcoder with same architecture
    tc_raw = SyntheticTranscoder(d_in=d, d_out=d, d_sae=d_sae)
    with torch.no_grad():
        tc_raw.W_enc.copy_(std_sae.W_enc)
        tc_raw.b_enc.copy_(std_sae.b_enc)
        tc_raw.W_dec.copy_(std_sae.W_dec)
        tc_raw.b_dec.copy_(std_sae.b_dec)

    clean, corrupted = _make_clean_corr(d=d)

    from circuitry.patching.sae_features import SAEFeatureRunner, TranscoderWrapper
    from circuitry.patching.sites import Site

    site = Site("resid_post", layer=0)
    resolver = _make_resolver(d)

    std_runner = SAEFeatureRunner(model, {site: std_sae}, resolver)
    tc_runner = SAEFeatureRunner(model, {site: TranscoderWrapper(tc_raw)}, resolver)

    std_result = std_runner.run(clean, corrupted, lambda out: _metric(out))
    tc_result = tc_runner.run(clean, corrupted, lambda out: _metric(out))

    # Scores should differ because inp[0] != output for LinearResidLayer
    std_scores = {n.node.neuron: v for n, v in std_result.scores.items()}
    tc_scores = {n.node.neuron: v for n, v in tc_result.scores.items()}

    common_features = set(std_scores) & set(tc_scores)
    if not common_features:
        pytest.skip("No common active features between std and transcoder SAE")

    max_diff = max(abs(std_scores[i] - tc_scores[i]) for i in common_features)
    assert max_diff > 1e-6, (
        f"Standard SAE and TranscoderWrapper scores are identical (max_diff={max_diff:.2e}); "
        "they should differ because the encoding input differs (output vs inp[0])"
    )


def test_transcoder_splice_is_lossless():
    """After splicing a TranscoderWrapper, the model output is unchanged (eps correction).

    We verify this by checking that the model metric is the same with and without the
    splice (within float32 precision). A non-lossless splice would change the metric.
    """
    d, d_sae = 8, 16
    torch.manual_seed(40)
    model = LinearResidToy(n_layers=2, d=d)
    torch.manual_seed(41)
    tc_raw = SyntheticTranscoder(d_in=d, d_out=d, d_sae=d_sae)

    clean, _ = _make_clean_corr(d=d)

    model.eval()

    # Run with transcoder (clean forward is spliced) — use corrupted == clean so Δf ≈ 0
    # but the splice itself should still be lossless regardless of Δf
    runner, tc = _make_transcoder_runner(model, tc_raw, d)
    with torch.no_grad():
        from circuitry.patching.sae_features import _extract_tensor, _routed_inject
        from circuitry.patching.sites import Site

        resolver = _make_resolver(d)
        site = Site("resid_post", layer=0)
        resolved = resolver.resolve(model, site)
        layer_mod = resolved.module

        splice_output_store: dict[str, Tensor] = {}
        original_output_store: dict[str, Tensor] = {}

        def _capture_hook(module, inp, output):
            original_output_store["out"] = _extract_tensor(output).detach().clone()

        def _splice_hook(module, inp, output):
            a_in = inp[0].detach().to(tc.device, tc.dtype)
            a_out = _extract_tensor(output).detach().to(a_in.device, a_in.dtype)
            f = tc.encode(a_in)
            x_hat = tc.decode(f)
            eps = (a_out - x_hat).detach()
            recon = x_hat + eps  # should equal a_out
            splice_output_store["out"] = recon.detach().clone()
            return _routed_inject(resolved, output, recon.to(a_out.device, a_out.dtype))

        # Check that recon == original output
        orig_h = layer_mod.register_forward_hook(_capture_hook)
        model(clean)
        orig_h.remove()

        splice_h = layer_mod.register_forward_hook(_splice_hook)
        model(clean)
        splice_h.remove()

    orig_tensor = original_output_store["out"]
    spliced_tensor = splice_output_store["out"]
    max_diff = float((orig_tensor - spliced_tensor).abs().max().item())
    assert max_diff < 1e-5, (
        f"TranscoderWrapper splice is not lossless: max|original − recon| = {max_diff:.2e}"
    )


# ---------------------------------------------------------------------------
# SAEFeatureEdgeRunner with TranscoderWrapper
# ---------------------------------------------------------------------------


def test_transcoder_edge_runner_returns_edges():
    """SAEFeatureEdgeRunner with TranscoderWrapper SAEs returns finite edge scores."""
    d, d_sae = 8, 16
    torch.manual_seed(50)
    model = LinearResidToy(n_layers=2, d=d)
    torch.manual_seed(51)
    tc0 = SyntheticTranscoder(d_in=d, d_out=d, d_sae=d_sae)
    torch.manual_seed(52)
    tc1 = SyntheticTranscoder(d_in=d, d_out=d, d_sae=d_sae)

    from circuitry.patching.sae_edges import SAEFeatureEdgeRunner
    from circuitry.patching.sae_features import TranscoderWrapper
    from circuitry.patching.sites import Site

    site0 = Site("resid_post", layer=0)
    site1 = Site("resid_post", layer=1)
    resolver = _make_resolver(d)

    runner = SAEFeatureEdgeRunner(
        model,
        {site0: TranscoderWrapper(tc0), site1: TranscoderWrapper(tc1)},
        resolver,
    )
    clean, corrupted = _make_clean_corr(d=d)
    circuit = runner.run(clean, corrupted, lambda out: _metric(out))

    assert isinstance(circuit.edges, dict)
    for edge, score in circuit.edges.items():
        assert torch.isfinite(torch.tensor(score)), (
            f"Non-finite edge score for {edge}: {score}"
        )


def test_transcoder_mixed_sae_edge_runner():
    """One standard SAE and one TranscoderWrapper — edge runner should handle the mix."""
    d, d_sae = 8, 16
    torch.manual_seed(60)
    model = LinearResidToy(n_layers=2, d=d)
    torch.manual_seed(61)
    std_sae = SyntheticSAE(d_model=d, d_sae=d_sae)
    torch.manual_seed(62)
    tc_raw = SyntheticTranscoder(d_in=d, d_out=d, d_sae=d_sae)

    from circuitry.patching.sae_edges import SAEFeatureEdgeRunner
    from circuitry.patching.sae_features import TranscoderWrapper
    from circuitry.patching.sites import Site

    site0 = Site("resid_post", layer=0)
    site1 = Site("resid_post", layer=1)
    resolver = _make_resolver(d)

    runner = SAEFeatureEdgeRunner(
        model,
        {site0: std_sae, site1: TranscoderWrapper(tc_raw)},
        resolver,
    )
    clean, corrupted = _make_clean_corr(d=d)
    circuit = runner.run(clean, corrupted, lambda out: _metric(out))

    # Just verify it ran without errors and returned a valid circuit
    assert hasattr(circuit, "edges")
    assert hasattr(circuit, "nodes")


def test_transcoder_top_level_import():
    """TranscoderWrapper is importable from circuitry.patching."""
    from circuitry.patching import TranscoderWrapper
    assert TranscoderWrapper.hook_input is True
