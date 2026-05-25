"""AtP* runs end-to-end on a TransformerLens HookedTransformer (skipif)."""
from __future__ import annotations

import pytest
import torch

tl = pytest.importorskip("transformer_lens")
from circuitry.core.patching import logit_diff_t
from circuitry.patching.atp import AtPRunner
from circuitry.patching.sites import TLSiteResolver


def test_atp_runs_on_hooked_transformer():
    model = tl.HookedTransformer.from_pretrained("gelu-1l")
    clean = model.to_tokens("The cat sat")
    corrupted = model.to_tokens("The dog ran")
    runner = AtPRunner(model, TLSiteResolver())
    result = runner.run(clean_inputs=clean, corrupted_inputs=corrupted,
                        metric=lambda l: logit_diff_t(l, 0, 1))
    assert len(result.scores) > 0
    assert not any(s != s for s in result.scores.values())  # no NaN
