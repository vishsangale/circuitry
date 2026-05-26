"""v1.0 patching-pillar backend probe.

Goal: empirically pin which backend can run the IOI circuit-recovery validation,
BEFORE committing the campaign to scripts. Checks three things:

  1. TL backend on GPT-2 small  — expected OK (canonical IOI vehicle).
  2. HF-eager backend on GPT-2   — expected FAIL (transformer.h / Conv1D layout
     is not the Llama/Gemma nesting the HF path assumes).
  3. HF-eager backend on Gemma-2-2b — expected OK (correct nesting + separate
     q/k/v/o_proj), so HF-eager cross-validation runs here on a prompt-pair task.

Run:
  venv/bin/python scripts/v1_probe_backends.py
"""

from __future__ import annotations

import torch

from circuitry.core.patching import logit_diff_t
from circuitry.patching import EAPRunner
from circuitry.patching.sites import HFSiteResolver, TLSiteResolver

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# A single IOI prompt pair (ABBA clean -> answer IO="Mary"; corrupted swaps the
# subject so the IO-vs-S signal is destroyed). Names are single GPT-2 tokens.
CLEAN = "When John and Mary went to the store, John gave a drink to"
CORRUPT = "When John and Mary went to the store, Mary gave a drink to"
IO_WORD = " Mary"
S_WORD = " John"


def _attn_head_edges(result, top=20):
    rows = []
    for edge, score in result.ranked():
        w = edge.writer
        if w.kind == "attn_head":
            rows.append((w.layer, w.head, edge.reader.kind, edge.slot, score))
        if len(rows) >= top:
            break
    return rows


def probe_tl_gpt2():
    print("\n=== [1] TransformerLens — GPT-2 small (IOI) ===")
    try:
        from transformer_lens import HookedTransformer
        model = HookedTransformer.from_pretrained("gpt2", device=DEVICE)
        io_id = model.to_single_token(IO_WORD)
        s_id = model.to_single_token(S_WORD)
        clean = model.to_tokens(CLEAN)
        corrupt = model.to_tokens(CORRUPT)

        def metric(out):
            logits = out.logits if hasattr(out, "logits") else out
            return logit_diff_t(logits, io_id, s_id)

        runner = EAPRunner(model, resolver=TLSiteResolver())
        res = runner.run(clean, corrupt, metric)
        print(f"OK — {len(res.scores)} edges scored. Top attn-head edges:")
        for layer, head, rk, slot, sc in _attn_head_edges(res):
            print(f"   head {layer}.{head:<2} -> {rk}/{slot:<10} {sc:+.4f}")
    except Exception as e:  # noqa: BLE001
        print(f"FAIL: {type(e).__name__}: {e}")


def probe_hf_gpt2():
    print("\n=== [2] HF-eager — GPT-2 small (expected FAIL) ===")
    try:
        from transformers import GPT2LMHeadModel
        model = GPT2LMHeadModel.from_pretrained("gpt2", attn_implementation="eager").to(DEVICE)
        cfg = model.config
        resolver = HFSiteResolver(
            n_heads=cfg.num_attention_heads,
            d_model=cfg.hidden_size,
            layer_pattern="transformer.h.{L}",
            attn_module="attn.c_proj",
            mlp_module="mlp",
        )
        runner = EAPRunner(model, resolver=resolver)
        print(f"   (built runner; n_heads={runner.n_heads}, graph edges={len(runner.graph.edges)})")
        print("   -> would still fail at run() on transformer.h / c_attn layout")
    except Exception as e:  # noqa: BLE001
        print(f"FAIL (as expected): {type(e).__name__}: {e}")


def probe_hf_gemma():
    print("\n=== [3] HF-eager — Gemma-2-2b (prompt-pair smoke) ===")
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained("google/gemma-2-2b")
        model = AutoModelForCausalLM.from_pretrained(
            "google/gemma-2-2b", attn_implementation="eager", dtype=torch.float32
        ).to(DEVICE)
        io_id = tok(IO_WORD, add_special_tokens=False)["input_ids"][-1]
        s_id = tok(S_WORD, add_special_tokens=False)["input_ids"][-1]
        clean = tok(CLEAN, return_tensors="pt").to(DEVICE)
        corrupt = tok(CORRUPT, return_tensors="pt").to(DEVICE)

        def metric(out):
            logits = out.logits if hasattr(out, "logits") else out
            return logit_diff_t(logits, io_id, s_id)

        resolver = HFSiteResolver.from_config(model.config)
        runner = EAPRunner(model, resolver=resolver)
        res = runner.run(dict(clean), dict(corrupt), metric)
        print(f"OK — {len(res.scores)} edges scored. Top attn-head edges:")
        for layer, head, rk, slot, sc in _attn_head_edges(res, top=10):
            print(f"   head {layer}.{head:<2} -> {rk}/{slot:<10} {sc:+.4f}")
    except Exception as e:  # noqa: BLE001
        import traceback
        print(f"FAIL: {type(e).__name__}: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    print(f"device={DEVICE} torch={torch.__version__}")
    probe_tl_gpt2()
    probe_hf_gpt2()
    probe_hf_gemma()
