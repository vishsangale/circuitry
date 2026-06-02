#!/usr/bin/env python3
"""v1.7 eval — SAE machinery on a PARALLEL-ATTENTION model (pythia-70m, GPT-NeoX).

The v1.7 caveat says attn_out/mlp_out hook the submodule output BEFORE residual
add, and equivalence is "scoped to Llama/GPT-2 SEQUENTIAL blocks; parallel-attn
differs." pythia uses parallel attention+MLP (both read the same normed input,
summed into the residual), so mlp_out does NOT depend on attn_out.

Does circuitry behave SAFELY on this real model?
  - resid_post->resid_post (cross-layer): genuinely connected -> edges produced.
  - intra-layer attn_out@L -> mlp_out@L: genuinely SEVERED (parallel). The v1.7
    no-grad guard should WARN + return {} for this sub-block pair (NOT raise,
    NOT silently emit garbage, NOT crash).
  - node attribution + splice losslessness at each site: should still work.

Run from repo root:
    .venv/bin/python scripts/v17_validation/sae_parallel_attn.py
"""

from __future__ import annotations

import traceback
import warnings

import torch

from sae_lens import SAE
from transformer_lens import HookedTransformer

from circuitry.patching.sae_edges import SAEFeatureEdgeRunner
from circuitry.patching.sae_features import SAEFeatureRunner
from circuitry.patching.sites import Site, TLSiteResolver

DEV = "cpu"
MODEL = "pythia-70m-deduped"


def load_sae(release, sae_id):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = SAE.from_pretrained(release, sae_id, device=DEV)
    return res[0] if isinstance(res, tuple) else res


def hr(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78, flush=True)


def main():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        print(f"loading TL {MODEL} (cpu)...", flush=True)
        model = HookedTransformer.from_pretrained(MODEL, device=DEV)
        model.eval()

    print(f"  n_layers={model.cfg.n_layers} d_model={model.cfg.d_model}")
    print(f"  parallel_attn_mlp={getattr(model.cfg,'parallel_attn_mlp','?')}  "
          f"original_architecture={getattr(model.cfg,'original_architecture','?')}")

    # generic differentiable metric: logit diff between two tokens at last pos
    clean = model.to_tokens("The capital of France is")
    corrupt = model.to_tokens("The capital of Japan is")
    t_a = model.to_single_token(" Paris")
    t_b = model.to_single_token(" Tokyo")

    def metric(logits):
        return logits[0, -1, t_a] - logits[0, -1, t_b]

    print(f"  baseline metric = {metric(model(clean)).item():+.4f}")

    resolver = TLSiteResolver()
    print("loading pythia SAEs...", flush=True)
    sae_r2 = load_sae("pythia-70m-deduped-res-sm", "blocks.2.hook_resid_post")  # resid_post@2
    sae_r3 = load_sae("pythia-70m-deduped-res-sm", "blocks.3.hook_resid_post")  # resid_post@3
    sae_mlp3 = load_sae("pythia-70m-deduped-mlp-sm", "blocks.3.hook_mlp_out")
    sae_attn3 = load_sae("pythia-70m-deduped-att-sm", "blocks.3.hook_attn_out")

    # ---------- splice losslessness + node attribution at each site ----------
    try:
        hr("A. splice losslessness + node attribution (parallel-attn model)")
        base = model(clean)
        site_specs = [
            ("resid_post@2", Site("resid_post", 2), sae_r2, "blocks.2.hook_resid_post"),
            ("mlp_out@3", Site("mlp_out", 3), sae_mlp3, "blocks.3.hook_mlp_out"),
            ("attn_out@3", Site("attn_out", 3), sae_attn3, "blocks.3.hook_attn_out"),
        ]
        for label, site, sae, tl_hook in site_specs:
            def splice(t, hook, _sae=sae):
                a = t.detach().reshape(-1, t.shape[-1]).float()
                xh = _sae.decode(_sae.encode(a))
                eps = (a - xh).detach()
                return (xh + eps).reshape(t.shape).to(t.dtype)
            spliced = model.run_with_hooks(clean, fwd_hooks=[(tl_hook, splice)])
            dmax = (spliced - base).abs().max().item()
            runner = SAEFeatureRunner(model=model, sae_sites={site: sae}, resolver=resolver)
            res = runner.run(clean, corrupt, metric, max_features=5)
            top = sorted(res.scores.items(), key=lambda kv: abs(kv[1]), reverse=True)[:3]
            comps = {a.node.component for a in res.scores}
            print(f"  {label:14} splice max|Δlogit|={dmax:.2e}  #feat={len(res.scores)}  comps={comps}")
            for a, s in top:
                print(f"        feat={a.node.neuron:>5} score={s:+.4f}")
    except Exception:
        print("  SECTION A FAILED:"); traceback.print_exc()

    # ---------- connected edge: resid_post@2 -> resid_post@3 ----------
    try:
        hr("B. CONNECTED edge resid_post@2 -> resid_post@3 (expect edges)")
        r = SAEFeatureEdgeRunner(model=model,
                                 sae_sites={Site("resid_post", 2): sae_r2,
                                            Site("resid_post", 3): sae_r3},
                                 resolver=resolver)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            circ = r.run(clean, corrupt, metric, layer_pairs="adjacent", top_k_survivors=10)
        print(f"  #edges={len(circ.edges)}  warnings={len(w)}")
        for e, s in circ.top_k(4):
            wn, rn = e.writer.node, e.reader.node
            print(f"    L{wn.layer}:{wn.neuron} -> L{rn.layer}:{rn.neuron}  {s:+.4f}")
        print("  EXPECT: non-empty edges, no sever warning.")
    except Exception:
        print("  SECTION B FAILED:"); traceback.print_exc()

    # ---------- SEVERED edge: attn_out@3 -> mlp_out@3 (parallel => severed) ----------
    try:
        hr("C. SEVERED intra-layer edge attn_out@3 -> mlp_out@3 (parallel-attn)")
        r = SAEFeatureEdgeRunner(model=model,
                                 sae_sites={Site("attn_out", 3): sae_attn3,
                                            Site("mlp_out", 3): sae_mlp3},
                                 resolver=resolver)
        raised = None
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                circ = r.run(clean, corrupt, metric, layer_pairs="adjacent", top_k_survivors=10)
                n_edges = len(circ.edges)
            except Exception as e:
                raised = e
                n_edges = None
        sever_warnings = [str(x.message) for x in w
                          if "sever" in str(x.message).lower()
                          or "disconnect" in str(x.message).lower()
                          or "no grad" in str(x.message).lower()
                          or "grad is none" in str(x.message).lower()]
        print(f"  raised={type(raised).__name__ if raised else None}")
        print(f"  #edges={n_edges}")
        print(f"  total warnings={len(w)}; sever-related={len(sever_warnings)}")
        for sw in sever_warnings[:3]:
            print(f"    WARN: {sw[:140]}")
        print("\n  EXPECTED (v1.7 guard): WARN + empty edges for this parallel sub-block pair")
        print("  (NOT a raise, NOT silent garbage edges, NOT a crash).")
        # verdict
        if raised is not None:
            print(f"  >>> got a RAISE ({type(raised).__name__}) — check whether guard mis-classifies "
                  "parallel pair as always-connected.")
        elif n_edges == 0 and sever_warnings:
            print("  >>> CORRECT: warned + returned empty edges. Guard handles parallel-attn safely.")
        elif n_edges == 0 and not sever_warnings:
            print("  >>> empty edges but NO sever warning — silent {}; user can't tell sever from "
                  "'no significant edges'. (honesty gap)")
        elif n_edges and n_edges > 0:
            print(f"  >>> PRODUCED {n_edges} EDGES on a severed parallel pair — possible SILENT-WRONG. "
                  "Inspect whether these are spurious.")
    except Exception:
        print("  SECTION C FAILED:"); traceback.print_exc()


if __name__ == "__main__":
    main()
