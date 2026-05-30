"""verify_top_k re-checks top nodes with real patch_site (ground truth)."""
from __future__ import annotations

import torch

from circuitry.core.patching import logit_diff_t
from circuitry.patching.atp import AtPRunner
from circuitry.patching.sites import HFSiteResolver


def test_verify_top_k_matches_direct_patch(linear_attn_toy):
    clean = torch.tensor([[0, 1, 2, 3]])
    corrupted = torch.tensor([[3, 2, 1, 0]])
    resolver = HFSiteResolver(n_heads=linear_attn_toy.n_heads, d_model=linear_attn_toy.d,
                              d_mlp=linear_attn_toy.d, layer_pattern="layers.{L}")
    runner = AtPRunner(linear_attn_toy, resolver)
    result = runner.run(clean_inputs=clean, corrupted_inputs=corrupted,
                        metric=lambda lg: logit_diff_t(lg, 0, 1), neurons=False)
    verified = result.verify_top_k(
        k=3, clean_inputs=clean, corrupted_inputs=corrupted,
        metric=lambda lg: logit_diff_t(lg, 0, 1), resolver=resolver, runner=runner)
    assert len(verified) == 3
    for node, (atp_score, true_effect) in verified.items():
        assert isinstance(atp_score, float) and isinstance(true_effect, float)
        # on the linear toy, vanilla AtP is exact, so for non-q/k nodes they match
        if node.slot not in ("q", "k"):
            assert abs(atp_score - true_effect) < 1e-3
