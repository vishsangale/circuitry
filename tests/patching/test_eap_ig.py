"""EAP-IG (activation-path integrated gradients). Spec §3, §5."""
from __future__ import annotations

import pytest
import torch

from circuitry.core.patching import logit_diff_t
from circuitry.patching.eap import EAPRunner


def _metric(logits):
    return logit_diff_t(logits, correct=0, incorrect=1)


def test_ig_steps_1_equals_vanilla(linear_mlp_toy):
    clean = torch.tensor([[0, 1, 2]])
    corrupted = torch.tensor([[2, 0, 1]])
    runner = EAPRunner(linear_mlp_toy)
    vanilla = runner.run(clean_inputs=clean, corrupted_inputs=corrupted, metric=_metric)
    ig1 = runner.run(clean_inputs=clean, corrupted_inputs=corrupted, metric=_metric, ig_steps=1)
    for e in vanilla.scores:
        assert ig1.scores[e] == pytest.approx(vanilla.scores[e], abs=1e-5)


def test_ig_equals_vanilla_on_linear_model(linear_mlp_toy):
    # Linear model: gradient is path-constant, so IG (any N) == vanilla.
    clean = torch.tensor([[0, 1, 2]])
    corrupted = torch.tensor([[2, 0, 1]])
    runner = EAPRunner(linear_mlp_toy)
    vanilla = runner.run(clean_inputs=clean, corrupted_inputs=corrupted, metric=_metric)
    ig5 = runner.run(clean_inputs=clean, corrupted_inputs=corrupted, metric=_metric, ig_steps=5)
    assert len(ig5.scores) == len(vanilla.scores)
    for e in vanilla.scores:
        assert ig5.scores[e] == pytest.approx(vanilla.scores[e], abs=1e-4)


def test_ig_differs_from_vanilla_on_nonlinear_model():
    # Anti-stub: on a REAL nonlinear model, IG must actually differ from vanilla.
    transformers = pytest.importorskip("transformers")
    cfg = transformers.LlamaConfig(
        vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=16)
    torch.manual_seed(0)
    model = transformers.LlamaForCausalLM(cfg).eval()
    model.config._attn_implementation = "eager"
    from circuitry.patching.sites import HFSiteResolver
    resolver = HFSiteResolver(n_heads=4, d_model=16, d_mlp=32, layer_pattern="model.layers.{L}")
    runner = EAPRunner(model, resolver)
    clean = {"input_ids": torch.tensor([[1, 2, 3, 4]])}
    corrupted = {"input_ids": torch.tensor([[4, 3, 2, 1]])}

    def _m(out):
        logits = out.logits if hasattr(out, "logits") else out
        return logit_diff_t(logits, correct=0, incorrect=1)

    vanilla = runner.run(clean_inputs=clean, corrupted_inputs=corrupted, metric=_m)
    ig8 = runner.run(clean_inputs=clean, corrupted_inputs=corrupted, metric=_m, ig_steps=8)
    # at least one edge score should meaningfully differ
    diffs = [abs(ig8.scores[e] - vanilla.scores[e]) for e in vanilla.scores]
    assert max(diffs) > 1e-4, "IG produced identical scores to vanilla — ig_steps ignored?"
