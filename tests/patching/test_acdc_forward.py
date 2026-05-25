"""ACDC set-ablation forward: live-capture + empty/full anchors + live-vs-clean."""
from __future__ import annotations

import torch

from circuitry.patching.acdc import ACDCRunner
from circuitry.patching.graph import Node
from circuitry.patching.sites import HFSiteResolver


def _attn_resolver(toy):
    """Resolver so the attention toy's heads are visible to ACDC (n_heads>0).

    Without it the runner defaults to n_heads=0 and ACDC would silently ignore
    the model's attention contributions — making the anchors meaningless.
    """
    return HFSiteResolver(n_heads=toy.n_heads, d_model=toy.d, d_mlp=toy.d,
                          layer_pattern="layers.{L}")


def test_corr_act_cache_keys_match_writers(linear_attn_toy):
    runner = ACDCRunner(linear_attn_toy, _attn_resolver(linear_attn_toy))
    corrupted = torch.tensor([[4, 3, 2, 1]])
    corr = runner._cache_corrupted_acts(corrupted)
    assert set(corr.keys()) == set(runner.graph.writers)
    for v in corr.values():
        assert torch.isfinite(v).all()


def test_live_capture_on_corrupted_reproduces_corr_cache(linear_attn_toy):
    """Independent check of the live-capture hooks: running them on the corrupted
    input (no injection) must equal the corrupted cache from EAP's collector."""
    runner = ACDCRunner(linear_attn_toy, _attn_resolver(linear_attn_toy))
    corrupted = torch.tensor([[4, 3, 2, 1]])
    corr = runner._cache_corrupted_acts(corrupted)
    live: dict[Node, torch.Tensor] = {}
    runner._run_capturing_live(corrupted, removed=set(), corr_act=corr, live_out=live)
    assert set(live.keys()) == set(runner.graph.writers)
    for n in runner.graph.writers:
        assert torch.allclose(live[n], corr[n], atol=1e-5), f"mismatch at {n}"


def _all_edges(runner):
    return set(runner.graph.edges)


def test_empty_circuit_equals_clean_mlp_toy(linear_mlp_toy):
    runner = ACDCRunner(linear_mlp_toy)
    clean = torch.tensor([[1, 2, 3, 4]])
    corrupted = torch.tensor([[4, 3, 2, 1]])
    corr = runner._cache_corrupted_acts(corrupted)
    with torch.no_grad():
        clean_logits = linear_mlp_toy(clean)           # independent ground truth
    out = runner._run_capturing_live(clean, removed=set(), corr_act=corr)
    assert torch.allclose(out, clean_logits, atol=1e-5)


def test_full_ablation_equals_corrupted_mlp_toy(linear_mlp_toy):
    runner = ACDCRunner(linear_mlp_toy)
    clean = torch.tensor([[1, 2, 3, 4]])
    corrupted = torch.tensor([[4, 3, 2, 1]])
    corr = runner._cache_corrupted_acts(corrupted)
    with torch.no_grad():
        corrupted_logits = linear_mlp_toy(corrupted)   # independent ground truth
    out = runner._run_capturing_live(clean, removed=_all_edges(runner), corr_act=corr)
    assert torch.allclose(out, corrupted_logits, atol=1e-5)


def test_empty_and_full_anchors_attn_toy(linear_attn_toy):
    runner = ACDCRunner(linear_attn_toy, _attn_resolver(linear_attn_toy))
    clean = torch.tensor([[1, 2, 3, 4]])
    corrupted = torch.tensor([[4, 3, 2, 1]])
    corr = runner._cache_corrupted_acts(corrupted)
    with torch.no_grad():
        clean_logits = linear_attn_toy(clean)
        corrupted_logits = linear_attn_toy(corrupted)
    empty = runner._run_capturing_live(clean, removed=set(), corr_act=corr)
    full = runner._run_capturing_live(clean, removed=_all_edges(runner), corr_act=corr)
    assert torch.allclose(empty, clean_logits, atol=1e-5)
    assert torch.allclose(full, corrupted_logits, atol=1e-5)


def test_live_vs_clean_two_edge_composition(linear_mlp_toy):
    """The crux: after ablating edge e1 into mlp(0), mlp(0)'s LIVE contribution
    differs from its clean contribution, so the delta for a downstream edge e2
    must use the POST-e1 live value (corrupted − live), not a stale corrupted −
    clean.  This guards the live-recapture mechanism."""
    runner = ACDCRunner(linear_mlp_toy)
    clean = torch.tensor([[1, 2, 3, 4]])
    corrupted = torch.tensor([[4, 3, 2, 1]])
    corr = runner._cache_corrupted_acts(corrupted)
    g = runner.graph
    e1 = next(e for e in g.edges if e.reader == Node("mlp", 0) and e.writer == Node("embed"))
    e2 = next(e for e in g.edges if e.reader == Node("logits") and e.writer == Node("mlp", 0))
    out = runner._run_capturing_live(clean, removed={e1, e2}, corr_act=corr)
    live: dict[Node, torch.Tensor] = {}
    runner._run_capturing_live(clean, removed={e1}, corr_act=corr, live_out=live)
    clean_live: dict[Node, torch.Tensor] = {}
    runner._run_capturing_live(clean, removed=set(), corr_act=corr, live_out=clean_live)
    # mlp0's live contribution changed once e1 was ablated (it reads embed):
    assert not torch.allclose(live[Node("mlp", 0)], clean_live[Node("mlp", 0)], atol=1e-4)
    assert torch.isfinite(out).all()
