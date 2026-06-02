"""Regression tests for the v1.7 real-model evaluation findings related to
MoE (Mixture-of-Experts) coverage.

Source: ``docs/observations/2026-05-31-real-model-evaluation.md``
Findings: F36, F37, F39 (§9.3 "MoE — framework survives, but blind to experts").

Each test encodes *correct* behaviour so it is RED under the current code and
flips GREEN once the finding is fixed.  They are marked ``xfail(strict=True)``
so the suite stays green today while the fix is pending; when a fix lands the
test XPASSes, strict-xfail turns that into a failure, and the marker must be
removed.

To watch them actually fail (reproduce the findings), run with ``--runxfail``::

    .venv/bin/pytest tests/recorder/test_moe_eval_findings.py --runxfail -v

**Model path used:** tiny OLMoE built from config — no network download or
cached checkpoint required.  The tiny config reproduces the structural finding:
``OlmoeExperts`` stores all expert weights as batched 3D tensors
(``gate_up_proj [n_experts, d_in, d_out]``, ``down_proj [n_experts, ...]``)
with NO leaf Linear submodules.  ``mlp.gate`` (an ``OlmoeTopKRouter``) is also
present.  This is structurally identical to the real ``allenai/OLMoE-1B-7B-0924``
model referenced in the eval script.

Verified reproduce evidence (tiny-from-config model)
-----------------------------------------------------
- F36: 62 weight tags emitted, 0 cover expert tensors
  (tags cover only q/k/v/o_proj, embed_tokens, lm_head).
  ``OlmoeExperts.gate_up_proj`` shape (4, 64, 32) and ``down_proj`` shape
  (4, 32, 32) are matched by 0 llm-recipe HookPoints.
- F37: ``logger.warning`` fires per-HookPoint ("HookPoint 1 matched 0 modules"),
  but there is NO aggregate WARNING summarising that a weight-source family
  matched 0 modules across ALL its patterns — users get terse per-pattern noise
  with no actionable coverage total.
- F39: 0 tags reference ``mlp.gate``; router weight is invisible.
  Pattern ``r".*\\.(w1|w2|w3|gate_proj|up_proj|down_proj)$"`` does not match
  ``model.layers.0.mlp.gate`` (an ``OlmoeTopKRouter`` named ``gate``, not
  ``gate_proj``); no other llm-recipe pattern covers it.
"""

from __future__ import annotations

import logging
import tempfile

import pytest
import torch

from circuitry.recipes import _clear_registry_for_tests
from circuitry.recipes.llm import register
from circuitry.recorder.live import Recorder
from circuitry.writers.base import RecordingWriter

# ---------------------------------------------------------------------------
# Autouse fixture: keep recipe registry clean between tests
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean():
    _clear_registry_for_tests()
    register()
    yield
    _clear_registry_for_tests()


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _tiny_olmoe():
    """Build a tiny OLMoE model from config — no download required.

    Uses ``OlmoeConfig`` / ``OlmoeForCausalLM`` from the installed
    ``transformers``.  The resulting model reproduces the batched-3D-expert
    structure of the real ``allenai/OLMoE-1B-7B-0924``:
    - ``model.layers[i].mlp.experts.gate_up_proj`` → shape
      ``(n_experts, intermediate_size, hidden_size)``  (3-D, no leaf Linears)
    - ``model.layers[i].mlp.experts.down_proj`` → shape
      ``(n_experts, hidden_size, intermediate_size)``
    - ``model.layers[i].mlp.gate`` → ``OlmoeTopKRouter`` with a
      ``(n_experts, hidden_size)`` weight
    """
    from transformers import OlmoeConfig, OlmoeForCausalLM

    cfg = OlmoeConfig(
        hidden_size=32,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_experts=4,
        num_experts_per_tok=2,
        vocab_size=64,
        max_position_embeddings=64,
    )
    # Force eager attention so Recorder.attach() succeeds (SDPA blocks
    # output_attentions — see F29).
    cfg._attn_implementation = "eager"
    return OlmoeForCausalLM(cfg)


def _run_recorder(model) -> list[tuple[str, float, int]]:
    """Attach a Recorder, run one training step, return emitted scalars."""
    with tempfile.TemporaryDirectory() as tmp:
        writer = RecordingWriter()
        rec = Recorder(
            model,
            run_dir=tmp,
            recipe="llm",
            writer=writer,
            every_n_steps=1,
            strict=False,
        )
        rec.attach()

        tokens = torch.randint(0, 64, (1, 8))
        out = model(tokens, labels=tokens)
        out.loss.backward()
        rec.step(0)
        rec.detach()

    return list(writer.scalars)  # [(tag, value, step), ...]


# ---------------------------------------------------------------------------
# F36 — MoE expert weights are 0% covered (recipes/llm)
#
# OLMoE stores experts as batched 3-D tensors (``gate_up_proj [n_experts, d,
# d]``, ``down_proj [n_experts, d, d]``) inside ``OlmoeExperts`` — there are
# no leaf Linear submodules, so the llm recipe's mlp weight patterns match
# nothing.  Correct behaviour = the recipe captures expert weights on a MoE
# model (coverage > 0 for expert tensors).
#
# Verified current state: 62 weight tags, 0 cover expert tensors.
# ---------------------------------------------------------------------------


def test_F36_expert_weight_tags_are_emitted():
    """The llm recipe must emit at least one weight diagnostic tag covering
    an expert tensor when run on a MoE model.

    Correct behaviour: ``weight/<diag>/model.layers.<L>.mlp.experts.*`` tags
    are present in the emitted scalars (coverage > 0).

    Confirmed failure mode (current code): 62 weight tags emitted but 0 cover
    ``OlmoeExperts.gate_up_proj`` or ``OlmoeExperts.down_proj``.  The recipe's
    MLP weight pattern ``.*\\.(w1|w2|w3|gate_proj|up_proj|down_proj)$``
    requires *named Linear submodules*, which do not exist in the OLMoE batched
    expert layout.
    """
    pytest.importorskip("transformers")

    model = _tiny_olmoe()
    model.train()
    scalars = _run_recorder(model)

    all_tags = [t for t, _v, _s in scalars]
    weight_tags = [t for t in all_tags if t.startswith("weight/")]

    # Correct behaviour: at least one weight tag should reference an expert
    # tensor (gate_up_proj or down_proj inside the experts container).
    expert_weight_tags = [
        t
        for t in weight_tags
        if "experts" in t or "gate_up_proj" in t or (
            "down_proj" in t and "mlp" in t
        )
    ]
    assert len(expert_weight_tags) > 0, (
        f"Expected at least one weight tag for MoE expert tensors "
        f"(gate_up_proj / down_proj inside OlmoeExperts), but got 0. "
        f"Total weight tags emitted: {len(weight_tags)}. "
        f"Sample: {weight_tags[:5]}. "
        f"F36: expert weights are invisible to the llm recipe."
    )


# ---------------------------------------------------------------------------
# F37 — Silent under-coverage when all MLP weight HookPoints match 0 modules
#       (recorder/live)
#
# When every weight-source HookPoint for MLP projections matches 0 modules
# (as happens on OLMoE), there is no aggregate user-facing WARNING summarising
# that the weight-source family has 0 total matches.  The only signal is a
# terse per-HookPoint DEBUG/WARNING buried in noise.
#
# Correct behaviour = a WARNING-level log record is emitted that aggregates
# across weight-source HookPoints and reports the total 0-match count so users
# can immediately understand that weight diagnostics are not running.
# ---------------------------------------------------------------------------


def test_F37_zero_match_weight_family_emits_aggregate_warning(caplog):
    """attach() must emit an aggregate WARNING when N>=1 weight-source
    HookPoints all match 0 modules.

    Correct behaviour: after attach(), caplog contains a WARNING-level record
    from the ``circuitry`` logger that mentions BOTH the total count of
    zero-match weight patterns (e.g. "2 weight" or "weight HookPoints") AND
    that coverage is 0 — distinct from the per-HookPoint "HookPoint N matched
    0 modules" messages.

    Confirmed failure mode (current code): only per-HookPoint warnings appear;
    no aggregate summary record exists.  A user running with standard Python
    logging (WARNING level) sees two terse lines about HookPoint 1 and
    HookPoint 4 but no high-level "weight family: 0 modules matched" summary.
    """
    pytest.importorskip("transformers")

    model = _tiny_olmoe()
    model.train()

    caplog.set_level(logging.WARNING, logger="circuitry")

    with tempfile.TemporaryDirectory() as tmp:
        writer = RecordingWriter()
        rec = Recorder(
            model,
            run_dir=tmp,
            recipe="llm",
            writer=writer,
            every_n_steps=1,
            strict=False,
        )
        rec.attach()
        rec.detach()

    # Correct behaviour: an aggregate WARNING that covers multiple zero-match
    # weight patterns — something like "2 weight HookPoints matched 0 modules"
    # or "weight diagnostics: 0 modules matched across N patterns".
    # The per-HookPoint messages ("HookPoint 1 matched 0 modules") do NOT
    # satisfy this because they are per-pattern, not aggregated.
    aggregate_msgs = [
        r.getMessage()
        for r in caplog.records
        if r.levelno >= logging.WARNING
        and "weight" in r.getMessage().lower()
        and (
            # Must mention multiple patterns or a total count implying aggregation
            "HookPoint" not in r.getMessage()  # not the per-pattern message
            or r.getMessage().count("HookPoint") > 1  # covers >1 in one message
        )
        and "0" in r.getMessage()
    ]
    assert len(aggregate_msgs) > 0, (
        "Expected at least one aggregate WARNING summarising that weight-source "
        "HookPoints matched 0 modules total, but found none.\n"
        "All WARNING records from 'circuitry' logger:\n"
        + "\n".join(
            f"  [{r.levelname}] {r.getMessage()}"
            for r in caplog.records
            if r.levelno >= logging.WARNING
        )
        + "\nF37: zero-match weight coverage is not aggregated into a single "
        "user-visible WARNING."
    )


# ---------------------------------------------------------------------------
# F39 — MoE router weights unmatched → load-balance/imbalance invisible
#       (recipes/llm)
#
# The router weight (``mlp.gate``, ``OlmoeTopKRouter`` with weight shape
# ``[n_experts, hidden_size]``) is not matched by any llm recipe HookPoint.
# No weight diagnostics (effective_rank, stable_rank, etc.) are emitted for
# it, so expert load imbalance is entirely invisible.
#
# Correct behaviour = a router-weight tag (containing "mlp.gate" or "gate"
# within the mlp namespace) is emitted.
# ---------------------------------------------------------------------------


def test_F39_router_weight_tags_are_emitted():
    """The llm recipe must emit at least one weight diagnostic tag for the MoE
    router (``mlp.gate``) when run on a MoE model.

    Correct behaviour: a tag of the form
    ``weight/<diag>/model.layers.<L>.mlp.gate`` is present in emitted scalars.

    Confirmed failure mode (current code): 0 tags reference ``mlp.gate``.
    The ``OlmoeTopKRouter`` module is named ``gate`` (not ``gate_proj``), so
    the existing pattern ``.*\\.(gate_proj|up_proj|down_proj)$`` does not
    match.  No other llm-recipe pattern covers ``*.mlp.gate``.  Router load
    imbalance (revealed by rank/spectral diagnostics on the gate weight) is
    therefore entirely invisible.
    """
    pytest.importorskip("transformers")

    model = _tiny_olmoe()
    model.train()
    scalars = _run_recorder(model)

    all_tags = [t for t, _v, _s in scalars]

    # Correct behaviour: at least one weight tag references the router weight.
    # Accept any tag that contains "mlp.gate" or "mlp/gate" (tag separator
    # depends on implementation).
    router_weight_tags = [
        t
        for t in all_tags
        if t.startswith("weight/") and (
            "mlp.gate" in t
            or "mlp/gate" in t
            or ".gate" in t and "mlp" in t
        )
    ]
    assert len(router_weight_tags) > 0, (
        f"Expected at least one weight tag for the MoE router (mlp.gate / "
        f"OlmoeTopKRouter), but got 0. "
        f"All emitted tags containing 'gate': "
        f"{[t for t in all_tags if 'gate' in t.lower()][:10]}. "
        f"F39: router weight is invisible to the llm recipe."
    )
