"""QK fix gated on a real HF Llama: QK-fixed q/k attribution approximates
brute-force patch_site(q/k) BETTER than vanilla Δq·grad. The brute-force is
independent ground truth — never bend it to match."""
from __future__ import annotations

import numpy as np
import pytest
import torch

transformers = pytest.importorskip("transformers")
from circuitry.core.patching import logit_diff_t
from circuitry.patching.atp import AtPRunner
from circuitry.patching.sites import HFSiteResolver


def _tiny_llama():
    cfg = transformers.LlamaConfig(
        vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=16)
    torch.manual_seed(0)
    m = transformers.LlamaForCausalLM(cfg).eval()
    m.config._attn_implementation = "eager"
    return m


def _m(out):
    logits = out.logits if hasattr(out, "logits") else out
    return logit_diff_t(logits, correct=0, incorrect=1)


def _resolver():
    return HFSiteResolver(n_heads=4, d_model=16, d_mlp=32, layer_pattern="model.layers.{L}")


def test_qk_fix_beats_vanilla_against_bruteforce():
    model = _tiny_llama()
    runner = AtPRunner(model, _resolver())
    clean = {"input_ids": torch.tensor([[1, 2, 3, 4]])}
    corrupted = {"input_ids": torch.tensor([[4, 3, 2, 1]])}
    qk_fixed = runner.run(clean_inputs=clean, corrupted_inputs=corrupted, metric=_m)
    vanilla = runner.run(clean_inputs=clean, corrupted_inputs=corrupted, metric=_m, qk_fix=False)
    bf = runner.bruteforce_node_scores(clean_inputs=clean, corrupted_inputs=corrupted,
                                       metric=_m, nodes=list(qk_fixed.scores))
    qk_nodes = [n for n in qk_fixed.scores if n.slot in ("q", "k")]
    truth = np.array([bf[n] for n in qk_nodes])
    fixed = np.array([qk_fixed.scores[n] for n in qk_nodes])
    plain = np.array([vanilla.scores[n] for n in qk_nodes])
    assert np.abs(fixed - truth).sum() < np.abs(plain - truth).sum()  # closer to truth
    if np.std(fixed) > 1e-9 and np.std(truth) > 1e-9:
        assert np.corrcoef(fixed, truth)[0, 1] > 0.5


def test_qk_operand_space_match():
    model = _tiny_llama()
    runner = AtPRunner(model, _resolver())
    clean = {"input_ids": torch.tensor([[1, 2, 3, 4]])}
    corrupted = {"input_ids": torch.tensor([[4, 3, 2, 1]])}
    shapes = runner.qk_operand_shapes(clean_inputs=clean, corrupted_inputs=corrupted, metric=_m)
    for key, (d_head_out, d_grad) in shapes.items():
        assert d_head_out == d_grad == 16, key


def test_atp_v_slot_runs_on_gqa():
    """Regression: v-slot attribution must not crash on a GQA model
    (n_kv_heads < n_heads), and analytic v scores must correlate with brute-force."""
    cfg = transformers.LlamaConfig(
        vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=16)  # GQA: 4 q, 2 kv
    torch.manual_seed(0)
    model = transformers.LlamaForCausalLM(cfg).eval()
    model.config._attn_implementation = "eager"
    resolver = HFSiteResolver(n_heads=4, d_model=16, d_mlp=32, layer_pattern="model.layers.{L}")
    runner = AtPRunner(model, resolver)
    clean = {"input_ids": torch.tensor([[1, 2, 3, 4]])}
    corrupted = {"input_ids": torch.tensor([[4, 3, 2, 1]])}
    result = runner.run(clean_inputs=clean, corrupted_inputs=corrupted, metric=_m)
    v_nodes = [n for n in result.scores if n.slot == "v"]
    assert v_nodes  # produced v nodes, no crash
    assert all(s == s for s in result.scores.values())  # no NaN
    bf = runner.bruteforce_node_scores(clean_inputs=clean, corrupted_inputs=corrupted,
        metric=_m, nodes=v_nodes)
    a = np.array([result.scores[n] for n in v_nodes])
    b = np.array([bf[n] for n in v_nodes])
    if np.std(a) > 1e-9 and np.std(b) > 1e-9:
        assert np.corrcoef(a, b)[0, 1] > 0.5  # analytic v ≈ brute-force v
