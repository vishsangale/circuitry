"""Track 1a/1b — EAP + AtP* circuit recovery on TL GPT-2 small vs the published
IOI circuit (Wang et al. 2022). Multi-prompt IOIDataset (batched), per-head
aggregation, overlap@k and per-class recall against the 26-head ground truth.

Run:  venv/bin/python scripts/v1_validation/track1_eap_atp_tl.py
Saves: scripts/v1_validation/track1_eap_atp_tl.results.json
"""

from __future__ import annotations

import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(__file__))
from ioi_common import CIRCUIT, all_circuit_heads, batched_logit_diff_metric, build_ioi_batch

from circuitry.patching import AtPRunner, EAPRunner
from circuitry.patching.sites import TLSiteResolver

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_PROMPTS = 50
SEED = 0


def eap_head_importance(result) -> dict[tuple[int, int], float]:
    """Aggregate EAP edge scores to per-head importance = sum |score| over the
    head's outgoing edges."""
    imp: dict[tuple[int, int], float] = {}
    for edge, sc in result.scores.items():
        w = edge.writer
        if w.kind == "attn_head":
            imp[(w.layer, w.head)] = imp.get((w.layer, w.head), 0.0) + abs(sc)
    return imp


def atp_head_importance(result) -> dict[tuple[int, int], float]:
    """Aggregate AtP* node scores to per-head importance = sum |score| over q/k/v."""
    imp: dict[tuple[int, int], float] = {}
    for an, sc in result.scores.items():
        if an.node.kind == "attn_head":
            key = (an.node.layer, an.node.head)
            imp[key] = imp.get(key, 0.0) + abs(sc)
    return imp


def overlap_at_k(ranked_heads: list[tuple[int, int]], k: int) -> tuple[int, float]:
    truth = all_circuit_heads()
    topk = ranked_heads[:k]
    hits = sum(1 for h in topk if h in truth)
    return hits, hits / k


def class_recall(ranked_heads: list[tuple[int, int]], k: int) -> dict[str, str]:
    topk = set(ranked_heads[:k])
    out = {}
    for cls, hs in CIRCUIT.items():
        found = sum(1 for h in hs if h in topk)
        out[cls] = f"{found}/{len(hs)}"
    return out


def report(name: str, imp: dict[tuple[int, int], float]) -> dict:
    ranked = sorted(imp, key=lambda h: imp[h], reverse=True)
    print(f"\n=== {name}: top-15 heads by aggregated importance ===")
    from ioi_common import head_class
    for h in ranked[:15]:
        print(f"  {h[0]}.{h[1]:<2}  {imp[h]:.3f}   [{head_class(*h) or '-'}]")
    res = {"top20": [list(h) for h in ranked[:20]], "overlap": {}, "class_recall_at26": class_recall(ranked, 26)}
    print(f"  overlap@k vs 26-head published circuit:")
    for k in (10, 15, 20, 26):
        hits, frac = overlap_at_k(ranked, k)
        res["overlap"][str(k)] = {"hits": hits, "frac": round(frac, 3)}
        print(f"    @{k:<3} {hits}/{k}  ({frac:.0%})")
    print(f"  per-class recall @26: {res['class_recall_at26']}")
    return res


def main():
    torch.manual_seed(SEED)
    from transformer_lens import HookedTransformer
    model = HookedTransformer.from_pretrained("gpt2", device=DEVICE)
    batch = build_ioi_batch(model.tokenizer, n=N_PROMPTS, seed=SEED, device=DEVICE)
    metric = batched_logit_diff_metric(batch.io_ids, batch.s_ids)
    print(f"device={DEVICE} prompts={N_PROMPTS} seq_len={batch.clean.shape[1]}")

    results = {"n_prompts": N_PROMPTS, "seed": SEED}

    t0 = time.time()
    eap = EAPRunner(model, resolver=TLSiteResolver())
    eap_res = eap.run(batch.clean, batch.corrupt, metric)
    print(f"\nEAP done in {time.time()-t0:.1f}s ({len(eap_res.scores)} edges)")
    results["eap"] = report("EAP", eap_head_importance(eap_res))

    t0 = time.time()
    atp = AtPRunner(model, resolver=TLSiteResolver())
    atp_res = atp.run(batch.clean, batch.corrupt, metric, qk_fix=True)
    print(f"\nAtP* done in {time.time()-t0:.1f}s ({len(atp_res.scores)} nodes)")
    results["atp"] = report("AtP*", atp_head_importance(atp_res))

    out = os.path.join(os.path.dirname(__file__), "track1_eap_atp_tl.results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
