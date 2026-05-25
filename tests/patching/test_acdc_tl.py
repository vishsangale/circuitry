"""ACDC on a tiny TransformerLens model: full-ablation anchor holds."""
from __future__ import annotations

import pytest
import torch

transformer_lens = pytest.importorskip("transformer_lens")
from circuitry.patching.acdc import ACDCRunner  # noqa: E402
from circuitry.patching.sites import TLSiteResolver  # noqa: E402


def _tiny_tl():
    from transformer_lens import HookedTransformer, HookedTransformerConfig
    cfg = HookedTransformerConfig(
        n_layers=2, d_model=16, n_heads=2, d_head=8, d_mlp=32,
        n_ctx=16, d_vocab=32, act_fn="gelu",
    )
    torch.manual_seed(0)
    return HookedTransformer(cfg).eval()


def test_tl_full_ablation_anchor():
    model = _tiny_tl()
    runner = ACDCRunner(model, TLSiteResolver())
    clean = torch.tensor([[1, 2, 3, 4]])
    corrupted = torch.tensor([[4, 3, 2, 1]])
    corr = runner._cache_corrupted_acts(corrupted)
    with torch.no_grad():
        clean_logits = model(clean)
        corrupted_logits = model(corrupted)
    empty = runner._forward(clean, set(), corr)
    full = runner._forward(clean, set(runner.graph.edges), corr)
    assert torch.allclose(empty, clean_logits, atol=1e-4)
    assert torch.allclose(full, corrupted_logits, atol=1e-4)


def test_tl_run_executes():
    model = _tiny_tl()
    runner = ACDCRunner(model, TLSiteResolver())
    r = runner.run(clean_inputs=torch.tensor([[1, 2, 3, 4]]),
                   corrupted_inputs=torch.tensor([[4, 3, 2, 1]]), tau=0.05)
    assert isinstance(r.final_kl, float) and r.final_kl == r.final_kl  # finite/not-NaN
