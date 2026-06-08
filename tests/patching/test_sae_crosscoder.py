"""CrosscoderWrapper tests.

Verifies that CrosscoderWrapper:
  - signals hook_input=False (residual-stream output hook, same as standard SAEs)
  - routes encode/decode through encode_at_layer/decode_at_layer when available
  - falls back to plain encode/decode when layer-specific methods are absent
  - exposes encode_all() for full cross-layer usage
  - produces finite, nonzero attribution scores via SAEFeatureRunner
  - is not blocked by assert_supported_sae (no .cfg attribute → early return)
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from tests.patching.test_sae_features import (
    LinearResidToy,
    _make_clean_corr,
    _make_resolver,
    _metric,
)


# ---------------------------------------------------------------------------
# SyntheticCrosscoder fixture
# ---------------------------------------------------------------------------


class SyntheticCrosscoder(nn.Module):
    """Minimal crosscoder: linear encode, two per-layer decoders.

    Implements both:
      encode(x)                  — single-tensor input (shared encoder)
      encode_at_layer(x, layer)  — explicit layer routing (same encoder for all)
      decode(f)                  — decodes via layer-0 decoder
      decode_at_layer(f, layer)  — layer-specific decode
    """

    def __init__(self, d_in: int, d_hidden: int, n_layers: int = 2) -> None:
        super().__init__()
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self.encoder = nn.Linear(d_in, d_hidden, bias=False)
        self.decoders = nn.ModuleList([
            nn.Linear(d_hidden, d_in, bias=False) for _ in range(n_layers)
        ])

    def encode(self, x: Tensor) -> Tensor:
        return self.encoder(x)

    def decode(self, f: Tensor) -> Tensor:
        return self.decoders[0](f)  # default to layer 0

    def encode_at_layer(self, x: Tensor, layer: int) -> Tensor:
        return self.encoder(x)  # same encoder for all layers

    def decode_at_layer(self, f: Tensor, layer: int) -> Tensor:
        return self.decoders[layer](f)


class SyntheticCrosscoderNoLayerMethods(nn.Module):
    """Minimal crosscoder without encode_at_layer / decode_at_layer.

    Tests the fallback path in CrosscoderWrapper.
    """

    def __init__(self, d_in: int, d_hidden: int) -> None:
        super().__init__()
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self.encoder = nn.Linear(d_in, d_hidden, bias=False)
        self.decoder = nn.Linear(d_hidden, d_in, bias=False)

    def encode(self, x: Tensor) -> Tensor:
        return self.encoder(x)

    def decode(self, f: Tensor) -> Tensor:
        return self.decoder(f)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _make_crosscoder_runner(model, cc, d, primary_layer=0, site_layer=0):
    from circuitry.patching.sae_features import CrosscoderWrapper, SAEFeatureRunner
    from circuitry.patching.sites import Site
    wrapper = CrosscoderWrapper(cc, primary_layer=primary_layer)
    site = Site("resid_post", layer=site_layer)
    return SAEFeatureRunner(model, {site: wrapper}, _make_resolver(d)), wrapper


# ---------------------------------------------------------------------------
# Unit tests: CrosscoderWrapper
# ---------------------------------------------------------------------------


def test_crosscoder_wrapper_hook_input_false():
    """hook_input is False — residual-stream output hook, unlike TranscoderWrapper."""
    from circuitry.patching.sae_features import CrosscoderWrapper
    torch.manual_seed(1)
    cc = SyntheticCrosscoder(d_in=8, d_hidden=16)
    wrapper = CrosscoderWrapper(cc)
    assert wrapper.hook_input is False


def test_crosscoder_wrapper_encode_decode_basic():
    """CrosscoderWrapper.encode returns (batch, d_hidden); decode returns (batch, d_in)."""
    from circuitry.patching.sae_features import CrosscoderWrapper
    torch.manual_seed(2)
    cc = SyntheticCrosscoder(d_in=8, d_hidden=16)
    wrapper = CrosscoderWrapper(cc)

    x = torch.randn(2, 3, 8)
    f = wrapper.encode(x)
    assert f.shape == (2, 3, 16), f"Expected (2, 3, 16), got {f.shape}"
    x_hat = wrapper.decode(f)
    assert x_hat.shape == (2, 3, 8), f"Expected (2, 3, 8), got {x_hat.shape}"


def test_crosscoder_wrapper_routes_to_encode_at_layer():
    """With primary_layer=1, encode/decode route through layer-1 decoder."""
    from circuitry.patching.sae_features import CrosscoderWrapper
    torch.manual_seed(3)
    cc = SyntheticCrosscoder(d_in=8, d_hidden=16, n_layers=2)

    # Initialize layer decoders with different random values to distinguish
    with torch.no_grad():
        nn.init.normal_(cc.decoders[0].weight, std=1.0)
        nn.init.normal_(cc.decoders[1].weight, std=2.0)

    wrapper_layer0 = CrosscoderWrapper(cc, primary_layer=0)
    wrapper_layer1 = CrosscoderWrapper(cc, primary_layer=1)

    x = torch.randn(4, 8)
    f0 = wrapper_layer0.encode(x)
    f1 = wrapper_layer1.encode(x)
    # encode_at_layer uses the same encoder — should be identical
    assert torch.allclose(f0, f1), "encode_at_layer should use the same shared encoder"

    out0 = wrapper_layer0.decode(f0)
    out1 = wrapper_layer1.decode(f1)
    # decode_at_layer uses different decoders — should differ
    assert not torch.allclose(out0, out1), (
        "Decode outputs should differ for primary_layer=0 vs primary_layer=1 "
        "(different decoder weights were used)"
    )


def test_crosscoder_wrapper_fallback_to_encode():
    """Without encode_at_layer, wrapper falls back to cc.encode()."""
    from circuitry.patching.sae_features import CrosscoderWrapper
    torch.manual_seed(4)
    cc = SyntheticCrosscoderNoLayerMethods(d_in=8, d_hidden=16)
    wrapper = CrosscoderWrapper(cc, primary_layer=0)

    x = torch.randn(2, 8)
    f = wrapper.encode(x)
    assert f.shape == (2, 16)
    x_hat = wrapper.decode(f)
    assert x_hat.shape == (2, 8)

    # Confirm no encode_at_layer attribute on the raw crosscoder
    assert not hasattr(cc, "encode_at_layer")
    assert not hasattr(cc, "decode_at_layer")


def test_crosscoder_wrapper_encode_all():
    """encode_all([x0, x1]) delegates to cc.encode with the list as input."""
    from circuitry.patching.sae_features import CrosscoderWrapper
    torch.manual_seed(5)

    # Stub that records calls to encode()
    class StubCC(nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = []

        def encode(self, x):
            self.calls.append(x)
            # Return a dummy tensor for shape testing
            if isinstance(x, list):
                return torch.zeros(x[0].shape[:-1] + (16,))
            return torch.zeros(x.shape[:-1] + (16,))

        def decode(self, f):
            return torch.zeros(f.shape[:-1] + (8,))

    stub = StubCC()
    wrapper = CrosscoderWrapper(stub)

    x0 = torch.randn(2, 8)
    x1 = torch.randn(2, 8)
    result = wrapper.encode_all([x0, x1])

    assert len(stub.calls) == 1, "encode_all should call cc.encode exactly once"
    called_with = stub.calls[0]
    assert isinstance(called_with, list), "encode_all should pass the list to cc.encode"
    assert len(called_with) == 2
    assert result.shape == (2, 16)


def test_crosscoder_wrapper_device_dtype():
    """device and dtype properties return valid torch types."""
    from circuitry.patching.sae_features import CrosscoderWrapper
    torch.manual_seed(6)
    cc = SyntheticCrosscoder(d_in=8, d_hidden=16)
    wrapper = CrosscoderWrapper(cc)
    assert wrapper.device == torch.device("cpu")
    assert wrapper.dtype == torch.float32


def test_crosscoder_wrapper_device_dtype_defaults():
    """device and dtype fall back to defaults when cc has no such attributes."""
    from circuitry.patching.sae_features import CrosscoderWrapper

    class BareCrosscoder:
        def encode(self, x):
            return x

        def decode(self, f):
            return f

    wrapper = CrosscoderWrapper(BareCrosscoder())
    assert wrapper.device == torch.device("cpu")
    assert wrapper.dtype == torch.float32


def test_crosscoder_wrapper_no_cfg_bypass():
    """CrosscoderWrapper has no .cfg — assert_supported_sae should not raise."""
    from circuitry.patching.sae_features import CrosscoderWrapper
    from circuitry.sae.grad import assert_supported_sae
    torch.manual_seed(7)
    cc = SyntheticCrosscoder(d_in=8, d_hidden=16)
    wrapper = CrosscoderWrapper(cc)

    assert not hasattr(wrapper, "cfg"), "CrosscoderWrapper must not have a .cfg attribute"
    # Should not raise
    assert_supported_sae(wrapper)


# ---------------------------------------------------------------------------
# SAEFeatureRunner integration
# ---------------------------------------------------------------------------


def test_crosscoder_wrapper_in_sae_feature_runner():
    """SAEFeatureRunner with CrosscoderWrapper completes and returns finite scores."""
    d, d_hidden = 8, 16
    torch.manual_seed(10)
    model = LinearResidToy(n_layers=2, d=d)
    torch.manual_seed(11)
    cc = SyntheticCrosscoder(d_in=d, d_hidden=d_hidden, n_layers=2)
    clean, corrupted = _make_clean_corr(d=d)

    runner, _ = _make_crosscoder_runner(model, cc, d)
    result = runner.run(clean, corrupted, lambda out: _metric(out))

    assert len(result.scores) > 0, "Expected at least one active feature with nonzero score"
    for node, score in result.scores.items():
        assert torch.isfinite(torch.tensor(score)), f"Non-finite score for {node}: {score}"


def test_crosscoder_wrapper_in_sae_feature_runner_primary_layer_1():
    """SAEFeatureRunner with primary_layer=1 CrosscoderWrapper also completes."""
    d, d_hidden = 8, 16
    torch.manual_seed(20)
    model = LinearResidToy(n_layers=2, d=d)
    torch.manual_seed(21)
    cc = SyntheticCrosscoder(d_in=d, d_hidden=d_hidden, n_layers=2)
    clean, corrupted = _make_clean_corr(d=d)

    runner, _ = _make_crosscoder_runner(model, cc, d, primary_layer=1, site_layer=1)
    result = runner.run(clean, corrupted, lambda out: _metric(out))

    for node, score in result.scores.items():
        assert torch.isfinite(torch.tensor(score)), f"Non-finite score for {node}: {score}"


def test_crosscoder_wrapper_fallback_cc_in_runner():
    """CrosscoderWrapper around a bare cc (no layer methods) works in SAEFeatureRunner."""
    d, d_hidden = 8, 16
    torch.manual_seed(30)
    model = LinearResidToy(n_layers=2, d=d)
    torch.manual_seed(31)
    cc = SyntheticCrosscoderNoLayerMethods(d_in=d, d_hidden=d_hidden)
    clean, corrupted = _make_clean_corr(d=d)

    runner, _ = _make_crosscoder_runner(model, cc, d)
    result = runner.run(clean, corrupted, lambda out: _metric(out))

    for node, score in result.scores.items():
        assert torch.isfinite(torch.tensor(score)), f"Non-finite score for {node}: {score}"


# ---------------------------------------------------------------------------
# Top-level import
# ---------------------------------------------------------------------------


def test_crosscoder_wrapper_top_level_import():
    """CrosscoderWrapper is importable from circuitry.patching."""
    from circuitry.patching import CrosscoderWrapper
    assert CrosscoderWrapper.hook_input is False


def test_crosscoder_wrapper_circuitry_top_level_import():
    """CrosscoderWrapper is importable from the top-level circuitry package."""
    from circuitry import CrosscoderWrapper
    assert CrosscoderWrapper.hook_input is False
