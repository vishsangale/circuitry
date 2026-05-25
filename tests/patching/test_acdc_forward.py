"""ACDC set-ablation forward: live-capture + empty/full anchors + live-vs-clean."""
from __future__ import annotations

import torch

from circuitry.patching.acdc import ACDCRunner
from circuitry.patching.graph import Node


def test_corr_act_cache_keys_match_writers(linear_attn_toy):
    runner = ACDCRunner(linear_attn_toy)
    clean = torch.tensor([[1, 2, 3, 4]])
    corrupted = torch.tensor([[4, 3, 2, 1]])
    corr = runner._cache_corrupted_acts(corrupted)
    assert set(corr.keys()) == set(runner.graph.writers)
    for v in corr.values():
        assert torch.isfinite(v).all()


def test_live_capture_on_corrupted_reproduces_corr_cache(linear_attn_toy):
    """Independent check of the live-capture hooks: running them on the corrupted
    input (no injection) must equal the corrupted cache from EAP's collector."""
    runner = ACDCRunner(linear_attn_toy)
    corrupted = torch.tensor([[4, 3, 2, 1]])
    corr = runner._cache_corrupted_acts(corrupted)
    live: dict[Node, torch.Tensor] = {}
    runner._run_capturing_live(corrupted, removed=set(), corr_act=corr, live_out=live)
    assert set(live.keys()) == set(runner.graph.writers)
    for n in runner.graph.writers:
        assert torch.allclose(live[n], corr[n], atol=1e-5), f"mismatch at {n}"
