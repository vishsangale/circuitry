"""EAP on a small TransformerLens HookedTransformer (skipif not installed)."""
from __future__ import annotations

import pytest

tl = pytest.importorskip("transformer_lens")
from circuitry.core.patching import logit_diff_t  # noqa: E402
from circuitry.patching.eap import EAPRunner  # noqa: E402
from circuitry.patching.sites import TLSiteResolver  # noqa: E402


def test_eap_runs_end_to_end_on_hooked_transformer():
    model = tl.HookedTransformer.from_pretrained("gelu-1l")
    clean = model.to_tokens("The cat sat")
    corrupted = model.to_tokens("The dog ran")
    runner = EAPRunner(model, TLSiteResolver())
    result = runner.run(
        clean_inputs=clean, corrupted_inputs=corrupted,
        metric=lambda logits: logit_diff_t(logits, correct=0, incorrect=1),
    )
    assert len(result.scores) == len(result.graph.edges)
    assert all(abs(s) < float("inf") for s in result.scores.values())
    assert not any(s != s for s in result.scores.values())  # no NaN
