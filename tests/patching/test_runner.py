"""Tests for PatchRunner."""
from __future__ import annotations

import pytest
import torch

from circuitry.core.patching import logit_diff
from circuitry.patching.runner import PatchResult, PatchRunner
from circuitry.patching.sites import HFSiteResolver, Site


@pytest.fixture
def resolver():
    return HFSiteResolver(
        n_heads=1, d_model=4, d_mlp=8,
        layer_pattern="layers.{L}",
        attn_module="self_attn.o_proj",
        mlp_module="mlp",
        mlp_intermediate="mlp.down_proj",
    )


def test_run_patching_denoise(toy_model, resolver):
    """Denoising: patching clean activation into corrupted run recovers clean output."""
    torch.manual_seed(0)
    clean = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    corrupted = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    site = Site(component="resid_post", layer=0)

    runner = PatchRunner(toy_model, resolver)
    result = runner.run_patching(
        clean_inputs=clean,
        corrupted_inputs=corrupted,
        sites=[site],
        metric=lambda logits: logit_diff(logits, correct=0, incorrect=3),
        direction="denoise",
    )

    assert isinstance(result, PatchResult)
    assert site in result.metric_values
    clean_metric = logit_diff(toy_model(clean), correct=0, incorrect=3)
    assert result.metric_values[site] == pytest.approx(clean_metric, abs=1e-5)


def test_run_patching_noise(toy_model, resolver):
    """Noising: patching corrupted activation into clean run yields corrupted output."""
    torch.manual_seed(1)
    clean = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    corrupted = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    site = Site(component="resid_post", layer=0)

    runner = PatchRunner(toy_model, resolver)
    result = runner.run_patching(
        clean_inputs=clean,
        corrupted_inputs=corrupted,
        sites=[site],
        metric=lambda logits: logit_diff(logits, correct=0, incorrect=3),
        direction="noise",
    )

    corrupted_metric = logit_diff(toy_model(corrupted), correct=0, incorrect=3)
    assert result.metric_values[site] == pytest.approx(corrupted_metric, abs=1e-5)


def test_multiple_sites(toy_model, resolver):
    torch.manual_seed(2)
    clean = torch.randn(1, 4)
    corrupted = torch.randn(1, 4)
    sites = [
        Site(component="resid_post", layer=0),
        Site(component="resid_post", layer=1),
    ]

    runner = PatchRunner(toy_model, resolver)
    result = runner.run_patching(
        clean_inputs=clean,
        corrupted_inputs=corrupted,
        sites=sites,
        metric=lambda logits: float(logits.sum().item()),
        direction="denoise",
    )

    assert len(result.metric_values) == 2
    for s in sites:
        assert s in result.metric_values


def test_custom_metric(toy_model, resolver):
    clean = torch.ones(1, 4)
    corrupted = torch.zeros(1, 4)
    site = Site(component="resid_post", layer=0)

    def my_metric(logits: torch.Tensor) -> float:
        return float(logits.max().item())

    runner = PatchRunner(toy_model, resolver)
    result = runner.run_patching(
        clean_inputs=clean,
        corrupted_inputs=corrupted,
        sites=[site],
        metric=my_metric,
        direction="denoise",
    )

    assert isinstance(result.metric_values[site], float)


def test_model_clean_after_runner(toy_model, resolver):
    """Model state is clean after runner completes (no leftover hooks)."""
    torch.manual_seed(3)
    clean = torch.randn(1, 4)
    corrupted = torch.randn(1, 4)
    before = toy_model(clean).clone()

    runner = PatchRunner(toy_model, resolver)
    runner.run_patching(
        clean_inputs=clean,
        corrupted_inputs=corrupted,
        sites=[Site(component="resid_post", layer=0)],
        metric=lambda logits: float(logits.sum().item()),
        direction="denoise",
    )

    after = toy_model(clean)
    assert torch.equal(before, after)
