"""Tests for patching/generation.py — generation-time analysis (v1.43)."""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from circuitry.patching.generation import (
    GenerationAttributionSetup,
    GenerationTrace,
    apply_steer_steps,
    generation_attribution,
    patch_site_steps,
    prepare_generation_attribution,
    trace_generation,
)

VOCAB, D = 13, 8


class TinyLM(nn.Module):
    """Causal-LM-shaped toy: model(ids) -> (batch, seq, vocab) logits."""

    def __init__(self, n_layers: int = 2):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, D)
        self.blocks = nn.ModuleList(nn.Linear(D, D) for _ in range(n_layers))
        self.unembed = nn.Linear(D, VOCAB)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(ids)
        for block in self.blocks:
            x = torch.tanh(block(x))
        return self.unembed(x)


class LogitsWrapper(nn.Module):
    """Wraps TinyLM to return an object with .logits (HF-style)."""

    class _Out:
        def __init__(self, logits):
            self.logits = logits

    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner

    def forward(self, ids):
        return self._Out(self.inner(ids))


@pytest.fixture
def lm() -> TinyLM:
    torch.manual_seed(0)
    return TinyLM()


@pytest.fixture
def prompt() -> torch.Tensor:
    return torch.tensor([[1, 2, 3]])


# ---------------------------------------------------------------------------
# trace_generation
# ---------------------------------------------------------------------------


def test_trace_basic(lm, prompt):
    trace = trace_generation(lm, prompt, n_steps=4)
    assert isinstance(trace, GenerationTrace)
    assert len(trace.records) == 4
    assert trace.prompt_len == 3
    assert all(0 <= t < VOCAB for t in trace.token_ids)


def test_trace_greedy_token_is_top1(lm, prompt):
    trace = trace_generation(lm, prompt, n_steps=3)
    for r in trace.records:
        assert r.token_id == r.top_token_ids[0]


def test_trace_entropy_positive_and_topk(lm, prompt):
    trace = trace_generation(lm, prompt, n_steps=2, top_k=3)
    for r in trace.records:
        assert r.entropy > 0
        assert len(r.top_token_ids) == 3
        # top logits sorted descending
        assert list(r.top_logits) == sorted(r.top_logits, reverse=True)


def test_trace_site_stats(lm, prompt):
    trace = trace_generation(
        lm, prompt, n_steps=2,
        modules={"block0": lm.blocks[0], "block1": lm.blocks[1]},
    )
    for r in trace.records:
        assert set(r.site_stats) == {"block0", "block1"}
        assert set(r.site_stats["block0"]) == {"norm", "mean", "std"}
    series = trace.site_series("block0", "norm")
    assert len(series) == 2 and all(v > 0 for v in series)


def test_trace_batch_gt1_raises(lm):
    with pytest.raises(ValueError, match=r"shape \(1, prompt_len\)"):
        trace_generation(lm, torch.ones(2, 3, dtype=torch.long), n_steps=1)


def test_trace_stop_token(lm, prompt):
    first = trace_generation(lm, prompt, n_steps=1).token_ids[0]
    trace = trace_generation(lm, prompt, n_steps=10, stop_token_id=first)
    assert len(trace.records) == 1
    assert trace.token_ids == [first]


def test_trace_custom_next_token_fn(lm, prompt):
    trace = trace_generation(lm, prompt, n_steps=3, next_token_fn=lambda last: 7)
    assert trace.token_ids == [7, 7, 7]


def test_trace_hf_style_output(lm, prompt):
    wrapped = LogitsWrapper(lm)
    a = trace_generation(wrapped, prompt, n_steps=3)
    b = trace_generation(lm, prompt, n_steps=3)
    assert a.token_ids == b.token_ids


def test_trace_bad_output_raises(prompt):
    class Weird(nn.Module):
        def forward(self, ids):
            return {"not": "logits"}

    with pytest.raises(TypeError, match="logits_fn"):
        trace_generation(Weird(), prompt, n_steps=1)


def test_trace_cleans_up_hooks_and_mode(lm, prompt):
    lm.train()
    trace_generation(lm, prompt, n_steps=1, modules={"b0": lm.blocks[0]})
    assert lm.training
    assert len(lm.blocks[0]._forward_hooks) == 0


def test_trace_to_markdown(lm, prompt):
    md = trace_generation(lm, prompt, n_steps=2).to_markdown()
    assert "## Generation Trace" in md
    assert "| step |" in md


# ---------------------------------------------------------------------------
# apply_steer_steps / patch_site_steps
# ---------------------------------------------------------------------------


def test_steer_steps_affects_only_selected_steps(lm, prompt):
    base = trace_generation(lm, prompt, n_steps=3, top_k=VOCAB)
    vector = torch.full((D,), 5.0)
    with apply_steer_steps(lm, lm.blocks[1], vector, steps={1}):
        steered = trace_generation(lm, prompt, n_steps=3, top_k=VOCAB)
    # step 0 untouched: identical logits
    assert steered.records[0].top_logits == base.records[0].top_logits
    # step 1 steered: logits differ
    assert steered.records[1].top_logits != base.records[1].top_logits


def test_steer_steps_range_and_coeff_zero(lm, prompt):
    base = trace_generation(lm, prompt, n_steps=2, top_k=VOCAB)
    with apply_steer_steps(lm, lm.blocks[0], torch.ones(D), steps=range(100), coeff=0.0):
        out = trace_generation(lm, prompt, n_steps=2, top_k=VOCAB)
    assert out.records[0].top_logits == base.records[0].top_logits
    assert out.records[1].top_logits == base.records[1].top_logits


def test_steer_steps_removes_hooks(lm, prompt):
    with apply_steer_steps(lm, lm.blocks[0], torch.ones(D), steps={0}):
        pass
    assert len(lm._forward_pre_hooks) == 0
    assert len(lm.blocks[0]._forward_hooks) == 0


def test_steer_steps_tuple_output(prompt):
    torch.manual_seed(0)

    class TupleBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(D, D)

        def forward(self, x):
            return self.lin(x), "aux"

    class TupleLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(VOCAB, D)
            self.block = TupleBlock()
            self.unembed = nn.Linear(D, VOCAB)

        def forward(self, ids):
            h, _aux = self.block(self.embed(ids))
            return self.unembed(h)

    lm = TupleLM()
    base = trace_generation(lm, prompt, n_steps=1, top_k=VOCAB)
    with apply_steer_steps(lm, lm.block, torch.full((D,), 3.0), steps={0}):
        steered = trace_generation(lm, prompt, n_steps=1, top_k=VOCAB)
    assert steered.records[0].top_logits != base.records[0].top_logits


def test_patch_site_steps_selected_steps_only(lm, prompt):
    base = trace_generation(lm, prompt, n_steps=2, top_k=VOCAB)
    with patch_site_steps(lm, lm.blocks[1], torch.zeros(D), steps={1}):
        patched = trace_generation(lm, prompt, n_steps=2, top_k=VOCAB)
    assert patched.records[0].top_logits == base.records[0].top_logits
    assert patched.records[1].top_logits != base.records[1].top_logits


def test_patch_site_steps_value_applied(lm, prompt):
    # Patching the final block's last position with zeros makes the final
    # logits equal unembed(0) regardless of the prompt.
    with patch_site_steps(lm, lm.blocks[1], torch.zeros(D), steps={0}):
        out = trace_generation(lm, prompt, n_steps=1, top_k=VOCAB)
    with torch.no_grad():
        # tanh is applied INSIDE the block loop before unembed in TinyLM's
        # forward, so the hook sees post-tanh values; expected logits are
        # unembed(zeros).
        expected = lm.unembed(torch.zeros(D))
    top = expected.topk(VOCAB)
    assert out.records[0].top_token_ids == tuple(top.indices.tolist())


def test_patch_site_steps_removes_hooks(lm):
    with patch_site_steps(lm, lm.blocks[0], torch.zeros(D), steps={0}):
        pass
    assert len(lm._forward_pre_hooks) == 0
    assert len(lm.blocks[0]._forward_hooks) == 0


# ---------------------------------------------------------------------------
# generation attribution
# ---------------------------------------------------------------------------


def test_prepare_setup_shapes_and_target(lm, prompt):
    corrupted = torch.tensor([[4, 5, 6]])
    setup = prepare_generation_attribution(lm, prompt, corrupted, target_step=2)
    assert isinstance(setup, GenerationAttributionSetup)
    assert setup.clean_inputs.shape == (1, 3 + 2)
    assert setup.corrupted_inputs.shape == (1, 3 + 2)
    # realized prefix matches a plain greedy trace
    trace = trace_generation(lm, prompt, n_steps=3)
    assert setup.clean_inputs[0, 3:].tolist() == trace.token_ids[:2]
    assert setup.target_token_id == trace.token_ids[2]
    assert setup.target_step == 2


def test_prepare_setup_metric_is_target_logit(lm, prompt):
    corrupted = torch.tensor([[4, 5, 6]])
    setup = prepare_generation_attribution(lm, prompt, corrupted, target_step=1)
    with torch.no_grad():
        out = lm(setup.clean_inputs)
    assert setup.metric(out) == pytest.approx(
        float(out[:, -1, setup.target_token_id].mean())
    )


def test_prepare_setup_shape_mismatch_raises(lm, prompt):
    with pytest.raises(ValueError, match="same shape"):
        prepare_generation_attribution(
            lm, prompt, torch.tensor([[1, 2]]), target_step=1,
        )


def test_generation_attribution_causal_trace(lm, prompt):
    result = generation_attribution(
        lm, prompt, torch.tensor([[4, 5, 6]]),
        target_step=1, module_pattern=r"blocks\.\d+$",
    )
    assert hasattr(result, "top_layers")
    assert result.recovery.shape == (2,)  # two blocks


def test_generation_attribution_patch_grid(lm, prompt):
    result = generation_attribution(
        lm, prompt, torch.tensor([[4, 5, 6]]),
        target_step=1, runner="patch_grid", module_pattern=r"blocks\.\d+$",
    )
    assert hasattr(result, "top_sites")
    # grid covers (layer, position) over the teacher-forced sequence
    assert result.recovery.shape[0] == 2


def test_generation_attribution_unknown_runner(lm, prompt):
    with pytest.raises(ValueError, match="unknown runner"):
        generation_attribution(
            lm, prompt, torch.tensor([[4, 5, 6]]),
            target_step=0, runner="nope", module_pattern=r"blocks\.\d+$",
        )


def test_exports():
    import circuitry
    from circuitry import patching

    assert circuitry.trace_generation is trace_generation
    assert patching.apply_steer_steps is apply_steer_steps
    assert "GenerationTrace" in circuitry.__all__
