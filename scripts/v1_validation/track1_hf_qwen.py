"""Track 1d — HF-eager backend on a REAL pretrained model (Qwen2.5-0.5B, GQA).

The HF backend failed on every cached model (head_dim!=d/h on Gemma; Conv1D layout
on GPT-2). Qwen2.5-0.5B satisfies head_dim = d_model/n_heads (896/14=64) and has
GQA (kv_heads=2), so it's the vehicle for:
  - confirming EAP + AtP* run end-to-end on the HF path on a real model (GQA),
  - patch_site FAITHFULNESS: verify_top_k runs real patch_site ablation (ground
    truth) and we correlate AtP* scores against it. This is HF-only
    (bruteforce_node_scores uses the HF layers list; unsupported on TL).

Run:  venv/bin/python scripts/v1_validation/track1_hf_qwen.py
Saves: scripts/v1_validation/track1_hf_qwen.results.json
"""

from __future__ import annotations

import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(__file__))
from ioi_common import batched_logit_diff_metric, build_ioi_batch

from circuitry.patching import AtPRunner, EAPRunner
from circuitry.patching.sites import HFSiteResolver

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = "Qwen/Qwen2.5-0.5B"
N_PROMPTS = 16
SEED = 0


def _spearman(xs, ys):
    """Spearman rank correlation without scipy."""
    import numpy as np
    def ranks(a):
        order = np.argsort(a)
        r = np.empty_like(order, dtype=float)
        r[order] = np.arange(len(a))
        return r
    rx, ry = ranks(np.asarray(xs)), ranks(np.asarray(ys))
    rx -= rx.mean(); ry -= ry.mean()
    denom = (np.sqrt((rx**2).sum()) * np.sqrt((ry**2).sum()))
    return float((rx * ry).sum() / denom) if denom else 0.0


def main():
    torch.manual_seed(SEED)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, attn_implementation="eager", dtype=torch.float32
    ).to(DEVICE).eval()
    cfg = model.config
    print(f"device={DEVICE} model={MODEL} layers={cfg.num_hidden_layers} "
          f"heads={cfg.num_attention_heads} kv_heads={cfg.num_key_value_heads} "
          f"d_model={cfg.hidden_size}", flush=True)

    batch = build_ioi_batch(tok, n=N_PROMPTS, seed=SEED, device=DEVICE)
    clean = {"input_ids": batch.clean}
    corrupt = {"input_ids": batch.corrupt}
    metric = batched_logit_diff_metric(batch.io_ids, batch.s_ids)

    # Sanity: does the model do the IOI task (clean logit-diff > corrupt)?
    with torch.no_grad():
        ld_clean = metric(model(**clean)).item()
        ld_corrupt = metric(model(**corrupt)).item()
    print(f"IOI logit-diff: clean={ld_clean:+.3f}  corrupt={ld_corrupt:+.3f}  "
          f"(task signal = {ld_clean - ld_corrupt:+.3f})", flush=True)

    resolver = HFSiteResolver.from_config(cfg)
    results = {"model": MODEL, "n_prompts": N_PROMPTS, "seed": SEED,
               "logit_diff_clean": ld_clean, "logit_diff_corrupt": ld_corrupt}

    # EAP — confirm the HF path runs end-to-end on a real GQA model.
    t0 = time.time()
    eap = EAPRunner(model, resolver=resolver)
    eap_res = eap.run(clean, corrupt, metric)
    print(f"\nEAP (HF) OK in {time.time()-t0:.1f}s — {len(eap_res.scores)} edges scored", flush=True)
    top_eap = [(f"{e.writer.kind}{e.writer.layer}.{e.writer.head}", round(s, 3))
               for e, s in eap_res.top_k(8) if e.writer.kind == "attn_head"]
    print(f"  top attn-head edges: {top_eap}", flush=True)
    results["eap_ran"] = True
    results["eap_top_attn_edges"] = top_eap

    # AtP* + faithfulness: verify_top_k runs real patch_site ablation (ground truth).
    t0 = time.time()
    atp = AtPRunner(model, resolver=resolver)
    atp_res = atp.run(clean, corrupt, metric, qk_fix=True)
    print(f"AtP* (HF) OK in {time.time()-t0:.1f}s — {len(atp_res.scores)} nodes scored", flush=True)

    t0 = time.time()
    verified = atp_res.verify_top_k(24, clean, corrupt, metric, resolver, atp)
    pairs = [(float(a), float(t)) for (a, t) in verified.values()]
    # Dedupe GQA-redundant nodes: query heads sharing a kv-group give identical
    # v-slot effects (correct GQA semantics, but degenerate for a correlation).
    distinct = sorted(set((round(a, 4), round(t, 4)) for a, t in pairs))
    atp_scores = [p[0] for p in pairs]
    true_effects = [p[1] for p in pairs]
    rho_all = _spearman(atp_scores, true_effects)
    rho_distinct = _spearman([p[0] for p in distinct], [p[1] for p in distinct])
    sign_agree = sum(1 for a, t in pairs if (a >= 0) == (t >= 0)) / len(pairs)
    print(f"\nverify_top_k(24) (real patch_site ground truth) in {time.time()-t0:.1f}s:", flush=True)
    for node, (a, t) in sorted(verified.items(), key=lambda kv: abs(kv[1][0]), reverse=True):
        nd = node.node
        print(f"  {nd.kind} {nd.layer}.{nd.head} slot={node.slot}: "
              f"atp={a:+.4f}  true_patch={t:+.4f}", flush=True)
    print(f"  {len(pairs)} nodes -> {len(distinct)} distinct (atp,true) points (GQA-deduped)", flush=True)
    print(f"  Spearman(atp,true) all={rho_all:+.3f}  distinct={rho_distinct:+.3f} ; "
          f"sign-agreement={sign_agree:.0%}", flush=True)
    results["faithfulness"] = {"spearman_all": round(rho_all, 3),
                               "spearman_distinct": round(rho_distinct, 3),
                               "n_distinct": len(distinct),
                               "sign_agreement": round(sign_agree, 3),
                               "pairs": [[round(a, 4), round(t, 4)] for a, t in pairs]}

    out = os.path.join(os.path.dirname(__file__), "track1_hf_qwen.results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved -> {out}", flush=True)


if __name__ == "__main__":
    main()
