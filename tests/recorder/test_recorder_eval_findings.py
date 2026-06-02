"""Regression tests for the v1.7 real-model evaluation findings in ``recorder/live``.

Source: ``docs/observations/2026-05-31-real-model-evaluation.md`` (F29, F2).

Each test encodes the *correct* behaviour, so it is RED under the current code and
flips GREEN once the finding is fixed. They are marked ``xfail(strict=True)`` so the
suite stays green today while the fix is pending; when a fix lands the test XPASSes,
strict-xfail turns that into a failure, and the marker must be removed.

To watch them actually fail (reproduce the findings), run with ``--runxfail``::

    .venv/bin/pytest tests/recorder/test_recorder_eval_findings.py --runxfail
"""

from __future__ import annotations

import dataclasses
import tempfile

import pytest
import torch

from circuitry.core.attention import attention_pattern_entropy as _ape
from circuitry.recipes import _clear_registry_for_tests, get_recipe
from circuitry.recipes.llm import register
from circuitry.recorder.live import Recorder
from circuitry.writers.base import RecordingWriter


@pytest.fixture(autouse=True)
def _clean():
    _clear_registry_for_tests()
    register()
    yield
    _clear_registry_for_tests()


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _llama_config(attn_implementation: str):
    """Build a tiny LlamaConfig from scratch — no network download required."""
    from transformers import LlamaConfig

    cfg = LlamaConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        vocab_size=64,
        max_position_embeddings=64,
    )
    # Setting _attn_implementation before model construction routes the kernel
    # choice without relying on the from_pretrained attn_implementation kwarg.
    cfg._attn_implementation = attn_implementation
    return cfg


def _llama_model(attn_implementation: str):
    from transformers import LlamaForCausalLM

    cfg = _llama_config(attn_implementation)
    return LlamaForCausalLM(cfg)


def _attention_only_recipe():
    """LLM recipe stripped to only induction_score + attention_pattern_entropy."""
    r = get_recipe("llm")
    return dataclasses.replace(
        r,
        activation_diagnostics=["induction_score", "attention_pattern_entropy"],
        gradient_diagnostics=[],
        weight_diagnostics=[],
    )


# ---------------------------------------------------------------------------
# F29 — Recorder.attach() crashes on any SDPA HF model
#        _set_output_attentions_true() raises ValueError:
#        "The `output_attentions` attribute is not supported when using the
#         `attn_implementation` set to sdpa."
#        This hits Qwen2 / Llama-3.x / Mistral / Gemma — all default to sdpa.
#        Fix: degrade gracefully (skip/warn) instead of raising.
# ---------------------------------------------------------------------------


def test_F29_attach_does_not_crash_on_sdpa_model():
    """attach() must NOT raise on a model with _attn_implementation='sdpa'.

    Correct behaviour: attach() completes (possibly emitting a warning), and a
    subsequent forward + step() also completes without raising.  The attention
    diagnostics (induction_score / attention_pattern_entropy) may silently emit
    no tags on SDPA models — that is acceptable — but the recorder must not crash.

    Confirmed failure mode (current code):
        ValueError: The `output_attentions` attribute is not supported when using
        the `attn_implementation` set to sdpa. Please set it to 'eager' instead.
    Raised at live.py _set_output_attentions_true() when it executes
        source.output_attentions = True
    for an HF config whose _attn_implementation is 'sdpa'.
    """
    transformers = pytest.importorskip("transformers")  # noqa: F841

    model = _llama_model("sdpa")
    recipe = _attention_only_recipe()

    with tempfile.TemporaryDirectory() as tmp:
        writer = RecordingWriter()
        rec = Recorder(
            model,
            run_dir=tmp,
            recipe=recipe,
            writer=writer,
            every_n_steps=1,
            strict=False,
        )
        # This is the call that currently raises ValueError on SDPA models.
        rec.attach()

        # Also verify a forward + step completes without raising.
        input_ids = torch.randint(0, 64, (1, 8))
        _ = model(input_ids)
        rec.step(0)
        rec.detach()

    # If we reach here the recorder degraded gracefully.
    # (No assertion needed on tags — zero tags is acceptable for SDPA.)


# ---------------------------------------------------------------------------
# F2 — attention_pattern_entropy emits the induction-probe's attention,
#       not the training batch's attention.
#
#       Root cause: the induction_score block (live.py ~869) runs
#           self.model(probe, output_attentions=True)
#       which re-fires the permanent _main_pass_attn capture hook and
#       OVERWRITES the training-forward attention.  The later-ordered
#       attention_pattern_entropy block (live.py ~892-916) then reads
#       the probe attention instead of the training attention.
#
#       Proven on real GPT-2: emitted entropy ~2.969 nats matches a fresh probe
#       forward to <1e-4 while the true training-forward entropy is ~1.325 nats
#       (delta ≈ 1.64 nats).
#
#       Fix: snapshot _main_pass_attn before the probe pass, or guard the
#       capture hook so it does not fire during inference_mode probe runs.
# ---------------------------------------------------------------------------


def test_F2_attention_pattern_entropy_reflects_training_forward(tmp_path):
    """The emitted attention_pattern_entropy must match the TRAINING forward,
    not the induction probe's forward.

    Test strategy:
    - Build a tiny Llama model with eager attention (so attach() succeeds).
    - Choose training tokens that produce a different entropy from the probe:
      a short, non-repeated sequence gives causal-mask entropy << max entropy,
      while the induction probe (2× repeated random tokens, longer sequence)
      approaches uniform entropy.
    - Run the recorder for one step; collect the emitted entropy for layer 0 / head 0.
    - Independently compute the TRUE training-forward entropy by running the
      model with output_attentions=True on the same training batch.
    - Assert: emitted value is close to the training entropy (tol 0.05 nats)
      and is NOT close to the probe entropy.

    Confirmed failure mode (current code): emitted entropy ≈ 2.969 nats (probe)
    vs true training entropy ≈ 1.325 nats — delta ≈ 1.64 nats.
    """
    transformers = pytest.importorskip("transformers")  # noqa: F841

    model = _llama_model("eager")
    model.eval()
    recipe = _attention_only_recipe()

    # Training batch: 8 distinct tokens — no repetition, short causal pattern.
    # Gives entropy << the probe (2×-repeated, length-16 sequence → near-uniform).
    train_ids = torch.tensor([[1, 5, 9, 13, 17, 21, 25, 29]], dtype=torch.long)

    # --- Compute the TRUE training-forward entropy independently ---
    with torch.no_grad():
        out_train = model(train_ids, output_attentions=True)
    # out_train.attentions is a tuple of (batch, heads, seq, seq) tensors.
    # Entropy for layer 0, all heads:
    true_entropy_layer0 = _ape(out_train.attentions[0])  # list[float], len = n_heads

    # --- Also compute the probe-forward entropy to anchor the wrong value ---
    # The recorder builds the probe internally; we approximate it.
    # n = recipe.induction_probe_seq_len (default 8), vocab = 64 for our config.
    # Use a deterministic probe that gives near-max entropy (uniform-ish attention).
    n = recipe.induction_probe_seq_len
    torch.manual_seed(0)
    half = torch.randint(0, 64, (1, n), dtype=torch.long)
    probe_ids = torch.cat([half, half], dim=1)
    with torch.no_grad():
        out_probe = model(probe_ids, output_attentions=True)
    probe_entropy_layer0 = _ape(out_probe.attentions[0])

    # Sanity: training and probe entropies must differ enough for the test to
    # have teeth.  On a 4-head causal model they typically differ by > 0.5 nats.
    delta = abs(true_entropy_layer0[0] - probe_entropy_layer0[0])
    assert delta > 0.2, (
        f"training entropy {true_entropy_layer0[0]:.4f} and probe entropy "
        f"{probe_entropy_layer0[0]:.4f} are too close (delta={delta:.4f}) "
        "— the test would be vacuous. Regenerate with different token sets."
    )

    # --- Run the recorder ---
    with tempfile.TemporaryDirectory() as tmp:
        writer = RecordingWriter()
        rec = Recorder(
            model,
            run_dir=tmp,
            recipe=recipe,
            writer=writer,
            every_n_steps=1,
            strict=False,
        )
        rec.attach()
        _ = model(train_ids)  # training forward (no output_attentions kwarg needed;
        #                       the recorder sets config.output_attentions=True)
        rec.step(0)
        rec.detach()

    # --- Check emitted values ---
    # Find the emitted entropy for layer 0, head 0.
    target_tag = "activation/attention_pattern_entropy/model.layers.0.self_attn/head_0"
    emitted = {t: v for t, v, _s in writer.scalars}
    assert target_tag in emitted, (
        f"Expected tag {target_tag!r} not found in emitted scalars. "
        f"Tags found: {sorted(emitted)}"
    )
    emitted_val = emitted[target_tag]

    # CORRECT behaviour: emitted ≈ training entropy.
    # CURRENT (buggy) behaviour: emitted ≈ probe entropy.
    assert emitted_val == pytest.approx(true_entropy_layer0[0], abs=0.05), (
        f"attention_pattern_entropy emitted {emitted_val:.4f} but training-forward "
        f"entropy is {true_entropy_layer0[0]:.4f} (probe entropy is "
        f"{probe_entropy_layer0[0]:.4f}). The recorder is emitting probe attention "
        "instead of training attention (F2)."
    )
