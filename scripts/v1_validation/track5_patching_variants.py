"""Track 5 (gap-closing, Gemini review) — patching algorithm variants not covered
by Track 1: EAP-IG (ig_steps>1), AtP* GradDrop, AtP* neuron-level nodes, and
PatchRunner denoise vs noise. Run on Qwen2.5-0.5B (HF, the working backend).

Run:  venv/bin/python scripts/v1_validation/track5_patching_variants.py
Saves: scripts/v1_validation/track5_patching_variants.results.json
"""

from __future__ import annotations

import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from ioi_common import batched_logit_diff_metric, build_ioi_batch

from circuitry.patching import AtPRunner, EAPRunner, PatchRunner, Site
from circuitry.patching.sites import HFSiteResolver

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = "Qwen/Qwen2.5-0.5B"


def top_heads_eap(res, k=10):
    imp = {}
    for e, s in res.scores.items():
        if e.writer.kind == "attn_head":
            imp[(e.writer.layer, e.writer.head)] = imp.get((e.writer.layer, e.writer.head), 0.0) + abs(s)
    return sorted(imp, key=lambda h: imp[h], reverse=True)[:k]


def main():
    torch.manual_seed(0)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, attn_implementation="eager", dtype=torch.float32).to(DEVICE).eval()
    resolver = HFSiteResolver.from_config(model.config)
    batch = build_ioi_batch(tok, n=12, seed=0, device=DEVICE)
    clean = {"input_ids": batch.clean}
    corrupt = {"input_ids": batch.corrupt}
    metric_t = batched_logit_diff_metric(batch.io_ids, batch.s_ids)        # tensor (EAP/AtP)
    metric_f = lambda out: float(metric_t(out).item())                     # float (PatchRunner)
    results = {"model": MODEL}
    print(f"device={DEVICE} model={MODEL}", flush=True)

    # --- EAP vanilla vs EAP-IG (ig_steps=4) ---
    eap = EAPRunner(model, resolver=resolver)
    van = top_heads_eap(eap.run(clean, corrupt, metric_t, ig_steps=1))
    ig = top_heads_eap(eap.run(clean, corrupt, metric_t, ig_steps=4))
    overlap = len(set(van) & set(ig))
    print(f"\n[EAP-IG] vanilla top-10 vs IG(4) top-10 heads: overlap={overlap}/10", flush=True)
    print(f"  vanilla: {van[:6]}\n  ig(4):   {ig[:6]}", flush=True)
    results["eap_ig"] = {"vanilla_top6": [list(h) for h in van[:6]],
                         "ig4_top6": [list(h) for h in ig[:6]], "overlap_top10": overlap}

    # --- AtP* GradDrop ---
    atp = AtPRunner(model, resolver=resolver)
    gd = atp.run(clean, corrupt, metric_t, graddrop=True, qk_fix=True)
    top_gd = [(n.node.layer, n.node.head, n.slot) for n, _ in gd.top_k(20)
              if n.node.kind == "attn_head"][:6]
    print(f"\n[AtP* GradDrop] {len(gd.scores)} nodes; top attn-head: {top_gd}", flush=True)
    results["atp_graddrop"] = {"n_nodes": len(gd.scores), "top_attn": [list(x) for x in top_gd]}

    # --- AtP* neuron-level (mlp_neuron nodes) ---
    try:
        neu = atp.run(clean, corrupt, metric_t, neurons=True, qk_fix=False)
        n_neuron = sum(1 for n in neu.scores if n.node.kind == "mlp_neuron")
        top_neu = [(n.node.layer, n.node.neuron) for n, _ in neu.top_k(50)
                   if n.node.kind == "mlp_neuron"][:5]
        print(f"\n[AtP* neuron] {n_neuron} mlp_neuron nodes scored; top: {top_neu}", flush=True)
        results["atp_neuron"] = {"n_mlp_neuron_nodes": n_neuron, "top5": [list(x) for x in top_neu]}
    except Exception as e:  # noqa: BLE001
        print(f"\n[AtP* neuron] FAILED: {type(e).__name__}: {e}", flush=True)
        results["atp_neuron"] = {"error": f"{type(e).__name__}: {e}"}

    # --- PatchRunner denoise vs noise on a high-attribution head ---
    L, H = top_gd[0][0], top_gd[0][1]
    site = Site(component="attn_head_out", layer=L, head=H)
    pr = PatchRunner(model, resolver)
    with torch.no_grad():
        ld_clean = metric_f(model(**clean))
        ld_corrupt = metric_f(model(**corrupt))
    den = pr.run_patching(clean, corrupt, [site], metric_f, direction="denoise")
    noi = pr.run_patching(clean, corrupt, [site], metric_f, direction="noise")
    print(f"\n[PatchRunner] head {L}.{H}  baseline: clean={ld_clean:+.3f} corrupt={ld_corrupt:+.3f}", flush=True)
    print(f"  denoise (clean->corrupt) logit-diff = {den.metric_values[site]:+.3f} "
          f"(moves corrupt toward clean)", flush=True)
    print(f"  noise   (corrupt->clean) logit-diff = {noi.metric_values[site]:+.3f} "
          f"(moves clean toward corrupt)", flush=True)
    results["patchrunner"] = {"site": f"attn_head_out L{L}.{H}", "clean": round(ld_clean, 3),
                              "corrupt": round(ld_corrupt, 3),
                              "denoise": round(den.metric_values[site], 3),
                              "noise": round(noi.metric_values[site], 3)}

    out = os.path.join(os.path.dirname(__file__), "track5_patching_variants.results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved -> {out}", flush=True)


if __name__ == "__main__":
    main()
