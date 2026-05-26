"""Probe AtP* and ACDC on TL-GPT-2 (IOI single pair) before writing campaign scripts.

EAP already probed clean. This confirms the other two runners execute on the TL
backend and surface IOI heads, so Track 1 produces three runners not one.

Run:  venv/bin/python scripts/v1_probe_atp_acdc.py
"""

from __future__ import annotations

import time

import torch

from circuitry.core.patching import logit_diff_t
from circuitry.patching import ACDCRunner, AtPRunner
from circuitry.patching.sites import TLSiteResolver

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLEAN = "When John and Mary went to the store, John gave a drink to"
CORRUPT = "When John and Mary went to the store, Mary gave a drink to"


def _node_str(n):
    kind = getattr(n, "kind", "?")
    layer = getattr(n, "layer", None)
    head = getattr(n, "head", None)
    if kind == "attn_head":
        return f"head {layer}.{head}"
    return f"{kind}{'' if layer is None else ' L' + str(layer)}"


def main():
    from transformer_lens import HookedTransformer
    model = HookedTransformer.from_pretrained("gpt2", device=DEVICE)
    io_id = model.to_single_token(" Mary")
    s_id = model.to_single_token(" John")
    clean = model.to_tokens(CLEAN)
    corrupt = model.to_tokens(CORRUPT)

    def metric(out):
        logits = out.logits if hasattr(out, "logits") else out
        return logit_diff_t(logits, io_id, s_id)

    print(f"device={DEVICE} seq_len={clean.shape[1]}")

    # --- AtP* ---
    print("\n=== AtP* (TL, qk_fix=True -> vanilla q/k on TL) ===")
    t0 = time.time()
    atp = AtPRunner(model, resolver=TLSiteResolver())
    ares = atp.run(clean, corrupt, metric, qk_fix=True)
    print(f"OK in {time.time()-t0:.1f}s — {len(ares.scores)} nodes scored. Top attn-head nodes:")
    shown = 0
    for node, sc in ares.ranked():
        if getattr(node, "kind", None) == "attn_head":
            print(f"   {_node_str(node):<12} {sc:+.4f}")
            shown += 1
        if shown >= 12:
            break

    # --- ACDC (single run + tiny sweep) ---
    print("\n=== ACDC (TL, last-token KL recovery) ===")
    t0 = time.time()
    acdc = ACDCRunner(model, resolver=TLSiteResolver())
    print(f"   graph: {len(acdc._eap.graph.edges)} edges — O(edges) forwards, timing one run...")
    res = acdc.run(clean, corrupt, tau=0.05)
    print(f"OK in {time.time()-t0:.1f}s — kept {res.n_kept()} / {len(acdc._eap.graph.edges)} edges, final_kl={res.final_kl:.4f}")
    # Which attn heads survived (as edge writers)?
    kept_heads = sorted({(e.writer.layer, e.writer.head) for e in res.kept_edges
                         if e.writer.kind == "attn_head"})
    print(f"   surviving attn heads ({len(kept_heads)}): {kept_heads[:25]}")


if __name__ == "__main__":
    main()
