"""EAP on a tiny real HF Llama: LN-scale two-hook, GQA, approximate cross-check."""
from __future__ import annotations

import pytest
import torch

transformers = pytest.importorskip("transformers")
from circuitry.core.patching import logit_diff_t  # noqa: E402
from circuitry.patching.eap import EAPRunner  # noqa: E402
from circuitry.patching.sites import HFSiteResolver  # noqa: E402


def _tiny_llama(n_kv_heads=4):
    cfg = transformers.LlamaConfig(
        vocab_size=32, hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=n_kv_heads,
        max_position_embeddings=16,
    )
    torch.manual_seed(0)
    m = transformers.LlamaForCausalLM(cfg)
    m.config._attn_implementation = "eager"
    return m.eval()


def _metric(out):
    logits = out.logits if hasattr(out, "logits") else out
    return logit_diff_t(logits, correct=0, incorrect=1)


def _resolver(model):
    return HFSiteResolver(
        n_heads=model.config.num_attention_heads, d_model=model.config.hidden_size,
        d_mlp=model.config.intermediate_size, layer_pattern="model.layers.{L}",
    )


def test_eap_runs_on_hf_llama_and_correlates_with_bruteforce():
    model = _tiny_llama()
    runner = EAPRunner(model, _resolver(model))
    clean = {"input_ids": torch.tensor([[1, 2, 3, 4]])}
    corrupted = {"input_ids": torch.tensor([[4, 3, 2, 1]])}
    result = runner.run(clean_inputs=clean, corrupted_inputs=corrupted, metric=_metric)
    bf = runner.bruteforce_edge_scores(clean_inputs=clean, corrupted_inputs=corrupted, metric=_metric)
    import numpy as np
    edges = [e for e in result.scores if e in bf]
    a = np.array([result.scores[e] for e in edges])
    b = np.array([bf[e] for e in edges])
    ra, rb = a.argsort().argsort(), b.argsort().argsort()
    corr = np.corrcoef(ra, rb)[0, 1]
    assert corr > 0.7, f"rank correlation {corr:.3f} too low"


def test_gqa_head_to_kv_group_mapping():
    assert EAPRunner._kv_head_for(0, 4, 2) == 0
    assert EAPRunner._kv_head_for(1, 4, 2) == 0
    assert EAPRunner._kv_head_for(2, 4, 2) == 1
    assert EAPRunner._kv_head_for(3, 4, 2) == 1


def test_eap_runs_with_gqa():
    model = _tiny_llama(n_kv_heads=2)
    runner = EAPRunner(model, _resolver(model))
    clean = {"input_ids": torch.tensor([[1, 2, 3, 4]])}
    corrupted = {"input_ids": torch.tensor([[4, 3, 2, 1]])}
    result = runner.run(clean_inputs=clean, corrupted_inputs=corrupted, metric=_metric)
    assert len(result.scores) == len(result.graph.edges)
    assert not any(s != s for s in result.scores.values())  # no NaN


def test_qkv_backmap_matches_reference():
    """Targeted gate for the back-map (the linear toy can't test q/k — fixed pattern)."""
    torch.manual_seed(0)
    n_heads, d_model = 2, 8
    head_dim = d_model // n_heads
    W_q = torch.randn(d_model, d_model)
    dL_dq = torch.randn(1, 3, d_model)
    ln_scale = torch.ones(1, 3, 1)
    for h in range(n_heads):
        dL_dq_h = dL_dq[..., h * head_dim:(h + 1) * head_dim]
        W_q_h = W_q[h * head_dim:(h + 1) * head_dim, :]
        got = EAPRunner._backmap_qkv_grad(dL_dq_h, W_q_h, ln_scale)
        expected = (dL_dq_h @ W_q_h) * ln_scale
        assert torch.allclose(got, expected, atol=1e-5), h
