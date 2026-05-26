"""Track 1c — ACDC circuit recovery on TL GPT-2 small (IOI).

ACDC is O(edges) forwards (~32k on GPT-2), so this is a long-running batch job.
We seed traversal with EAP scores (ordering="eap") and run a small tau sweep,
reporting the Pareto frontier and, for one tau, which heads survive vs the
published circuit. Smaller batch keeps each forward cheap.

Run (background):  venv/bin/python -u scripts/v1_validation/track1_acdc_tl.py
Saves: scripts/v1_validation/track1_acdc_tl.results.json
"""

from __future__ import annotations

import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(__file__))
from ioi_common import all_circuit_heads, batched_logit_diff_metric, build_ioi_batch, head_class

from circuitry.patching import ACDCRunner, EAPRunner
from circuitry.patching.sites import TLSiteResolver

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_PROMPTS = 12
SEED = 0
TAUS = [0.05]  # single tau: ACDC is O(edges) forwards; one tau demonstrates recovery


def main():
    torch.manual_seed(SEED)
    from transformer_lens import HookedTransformer
    model = HookedTransformer.from_pretrained("gpt2", device=DEVICE)
    batch = build_ioi_batch(model.tokenizer, n=N_PROMPTS, seed=SEED, device=DEVICE)
    metric = batched_logit_diff_metric(batch.io_ids, batch.s_ids)
    print(f"device={DEVICE} prompts={N_PROMPTS} seq_len={batch.clean.shape[1]}", flush=True)

    # EAP scores to seed ACDC traversal order.
    eap = EAPRunner(model, resolver=TLSiteResolver())
    eap_scores = eap.run(batch.clean, batch.corrupt, metric).scores
    n_edges = len(eap.graph.edges)
    print(f"graph: {n_edges} edges. Running ACDC sweep {TAUS} (KL recovery)...", flush=True)

    acdc = ACDCRunner(model, resolver=TLSiteResolver())

    results = {"n_prompts": N_PROMPTS, "seed": SEED, "n_edges": n_edges, "sweep": []}
    detailed = None
    for tau in TAUS:
        t0 = time.time()
        res = acdc.run(batch.clean, batch.corrupt, tau=tau,
                       ordering="eap", eap_scores=eap_scores)
        dt = time.time() - t0
        kept_heads = sorted({(e.writer.layer, e.writer.head) for e in res.kept_edges
                             if e.writer.kind == "attn_head"})
        in_circuit = sum(1 for h in kept_heads if h in all_circuit_heads())
        row = {"tau": tau, "n_kept_edges": res.n_kept(), "final_kl": round(res.final_kl, 5),
               "n_kept_heads": len(kept_heads), "heads_in_circuit": in_circuit}
        results["sweep"].append(row)
        print(f"  tau={tau:<5} kept {res.n_kept():>5}/{n_edges} edges, "
              f"{len(kept_heads)} heads ({in_circuit} in published circuit), "
              f"KL={res.final_kl:.4f}  [{dt:.0f}s]", flush=True)
        if tau == 0.05:
            detailed = [[h[0], h[1], head_class(*h) or "-"] for h in kept_heads]

    results["kept_heads_tau0.05"] = detailed
    out = os.path.join(os.path.dirname(__file__), "track1_acdc_tl.results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsurviving heads @tau=0.05: {detailed}", flush=True)
    print(f"saved -> {out}", flush=True)


if __name__ == "__main__":
    main()
