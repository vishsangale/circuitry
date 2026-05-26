"""Tests for the to_hooked_transformer TL bridge. Design spec §4.6 / v1.1."""
import importlib.util

import pytest

tl_missing = importlib.util.find_spec("transformer_lens") is None


def test_export_present():
    import circuitry.patching as p
    assert hasattr(p, "to_hooked_transformer")


@pytest.mark.skipif(tl_missing, reason="transformer_lens not installed")
def test_bridge_gpt2_eap_recovers_name_mover():
    transformers = pytest.importorskip("transformers")
    try:
        hf = transformers.GPT2LMHeadModel.from_pretrained("gpt2")
    except Exception as e:  # pragma: no cover - network/cache gated
        pytest.skip(f"gpt2 unavailable: {e}")

    from circuitry.core.patching import logit_diff_t
    from circuitry.patching import EAPRunner, to_hooked_transformer
    from circuitry.patching.sites import TLSiteResolver

    tl = to_hooked_transformer(hf, "gpt2", device="cpu")
    io = tl.to_single_token(" Mary")
    s = tl.to_single_token(" John")
    clean = tl.to_tokens("When John and Mary went to the store, John gave a drink to")
    corrupt = tl.to_tokens("When John and Mary went to the store, Mary gave a drink to")

    def metric(out):
        return logit_diff_t(out.logits if hasattr(out, "logits") else out, io, s)

    res = EAPRunner(tl, resolver=TLSiteResolver()).run(clean, corrupt, metric)
    top_heads = [(e.writer.layer, e.writer.head) for e, _ in res.top_k(40)
                 if e.writer.kind == "attn_head"]
    assert (9, 9) in top_heads[:6]   # the canonical IOI name-mover
