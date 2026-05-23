"""Shared pytest fixtures for the circuitry test suite."""

from __future__ import annotations

import dataclasses

import pytest

from circuitry.recipes import get_recipe

# Activation diagnostics in the stock `llm` recipe that need a real HF model
# (per-head attention weights via output_attentions, or a true unembed matrix)
# and therefore cannot run on the tiny stand-in models used by the e2e and
# perf tests. Tests that train a hand-rolled tiny model filter these out.
HF_ONLY_ACTIVATION_DIAGNOSTICS = (
    "induction_score",
    "logit_lens_kl",
    "attention_pattern_entropy",
)


@pytest.fixture
def llm_recipe_no_hf_diagnostics():
    """The stock `llm` recipe with HF-only activation diagnostics stripped,
    so it runs against the tiny models used in e2e + perf tests."""
    r = get_recipe("llm")
    return dataclasses.replace(
        r,
        activation_diagnostics=[
            d
            for d in r.activation_diagnostics
            if d not in HF_ONLY_ACTIVATION_DIAGNOSTICS
        ],
    )
