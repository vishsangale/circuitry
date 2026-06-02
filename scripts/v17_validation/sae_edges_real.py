#!/usr/bin/env python3
"""v1.7 eval — SAE feature edges on REAL GPT-2 + REAL SAEs (TL backend, CPU).

Sections (each isolated so one failure doesn't sink the rest):
  A. resid_post@6 -> resid_post@7 edges (both res-jb, FVU~0.001).
     Compare analytic attrib edges AND ig edges vs the INDEPENDENT bruteforce
     oracle on a real nonlinear model. Faithfulness/completeness of the circuit.
  B. Intra-layer attn_out@8 -> mlp_out@8 edges (v1.7 headline; forward rank
     attn_out < mlp_out).
  C. BUG #8 probe: is sae_lens layer_norm SAE.encode STATEFUL? (compute_f_per_site
     calls bare encode w/o paired decode — grad.py warns this breaks layer_norm SAEs.)
  D. IG error->feature edge gap: include_error_node=True under attrib vs ig.
     Static review + Gemini both flagged ig silently drops error->feature edges.

Run from repo root:
    .venv/bin/python scripts/v17_validation/sae_edges_real.py
"""

from __future__ import annotations

import traceback
import warnings

import torch

warnings.filterwarnings("ignore")

from sae_lens import SAE  # noqa: E402
from transformer_lens import HookedTransformer  # noqa: E402

from circuitry.patching.sae_edges import SAEFeatureEdgeRunner  # noqa: E402
from circuitry.patching.sites import Site, TLSiteResolver  # noqa: E402

DEV = "cpu"


def load_sae(release, sae_id):
    res = SAE.from_pretrained(release, sae_id, device=DEV)
    return res[0] if isinstance(res, tuple) else res


def hr(title):
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78, flush=True)


def main():
    print("loading TL gpt2 (cpu)...", flush=True)
    model = HookedTransformer.from_pretrained("gpt2", device=DEV)
    model.eval()

    clean = "When John and Mary went to the store, John gave a drink to"
    corrupt = "When John and Mary went to the store, Mary gave a drink to"
    clean_tok = model.to_tokens(clean)
    corrupt_tok = model.to_tokens(corrupt)
    mary, john = model.to_single_token(" Mary"), model.to_single_token(" John")

    def metric(logits):
        return logits[0, -1, mary] - logits[0, -1, john]

    resolver = TLSiteResolver()

    # ---- load SAEs ----
    print("loading SAEs...", flush=True)
    sae_r6 = load_sae("gpt2-small-res-jb", "blocks.7.hook_resid_pre")   # == resid_post@6
    sae_r7 = load_sae("gpt2-small-res-jb", "blocks.8.hook_resid_pre")   # == resid_post@7
    sae_mlp8 = load_sae("gpt2-small-mlp-out-v5-32k", "blocks.8.hook_mlp_out")
    sae_attn8 = load_sae("gpt2-small-attn-out-v5-32k", "blocks.8.hook_attn_out")

    # ================================================================== A
    try:
        hr("A. resid_post@6 -> resid_post@7 edges + bruteforce oracle")
        runner = SAEFeatureEdgeRunner(
            model=model,
            sae_sites={Site("resid_post", 6): sae_r6, Site("resid_post", 7): sae_r7},
            resolver=resolver,
        )
        circ = runner.run(clean_tok, corrupt_tok, metric,
                          layer_pairs="adjacent", top_k_survivors=10, max_edges=200)
        print(f"  #nodes scored={len(circ.nodes.scores)}  #edges={len(circ.edges)}")
        top = circ.top_k(6)
        edges = [e for e, _ in top]

        # IG edges over the SAME survivor set
        circ_ig = runner.run(clean_tok, corrupt_tok, metric,
                             layer_pairs="adjacent", top_k_survivors=10, max_edges=200,
                             variant="ig", n_ig_steps=16)
        ig_map = dict(circ_ig.edges)

        # bruteforce oracle (independent path) on the top edges
        bf = runner.bruteforce_feature_edge_scores(clean_tok, corrupt_tok, metric, edges)

        print(f"\n  {'edge (U:feat -> D:feat)':42} {'attrib':>10} {'ig':>10} {'bruteforce':>11}")
        a_list, b_list = [], []
        for e, a_score in top:
            w, r = e.writer.node, e.reader.node
            tag = f"L{w.layer}:{w.neuron}->L{r.layer}:{r.neuron}"
            ig_s = ig_map.get(e, float("nan"))
            bf_s = bf.get(e, float("nan"))
            a_list.append(a_score); b_list.append(bf_s)
            print(f"  {tag:42} {a_score:+10.4f} {ig_s:+10.4f} {bf_s:+11.4f}")

        # agreement attrib vs bruteforce
        at = torch.tensor(a_list); bt = torch.tensor(b_list)
        sign_agree = (torch.sign(at) == torch.sign(bt)).float().mean().item()
        if at.std() > 0 and bt.std() > 0:
            corr = torch.corrcoef(torch.stack([at, bt]))[0, 1].item()
        else:
            corr = float("nan")
        print(f"\n  attrib vs bruteforce: sign-agreement={sign_agree:.2f}  pearson={corr:.3f}")

        # faithfulness / completeness
        faith = circ.faithfulness(clean_tok, corrupt_tok, metric, ablation_mode="corrupted")
        comp = circ.completeness(clean_tok, corrupt_tok, metric, ablation_mode="corrupted")
        print(f"  full-circuit faithfulness={faith:.4f}  completeness={comp:.4f}")
    except Exception:
        print("  SECTION A FAILED:"); traceback.print_exc()

    # ================================================================== B
    try:
        hr("B. intra-layer attn_out@8 -> mlp_out@8 edges (v1.7 headline)")
        runner_il = SAEFeatureEdgeRunner(
            model=model,
            sae_sites={Site("attn_out", 8): sae_attn8, Site("mlp_out", 8): sae_mlp8},
            resolver=resolver,
        )
        circ_il = runner_il.run(clean_tok, corrupt_tok, metric,
                               layer_pairs="adjacent", top_k_survivors=10, max_edges=50)
        print(f"  #edges={len(circ_il.edges)}")
        for e, s in circ_il.top_k(5):
            w, r = e.writer.node, e.reader.node
            print(f"    {w.component}@L{w.layer}:{w.neuron} -> {r.component}@L{r.layer}:{r.neuron}  {s:+.4f}")
        # verify forward order: every edge writer rank < reader rank
        def rank(nd):
            off = {"attn_out": 0, "mlp_out": 1, None: 2, "resid_post": 2}[nd.component]
            return 3 * nd.layer + off
        bad = [e for e in circ_il.edges if rank(e.writer.node) >= rank(e.reader.node)]
        print(f"  forward-order violations: {len(bad)} (expect 0)")
    except Exception:
        print("  SECTION B FAILED:"); traceback.print_exc()

    # ================================================================== C
    try:
        hr("C. BUG #8 probe: is layer_norm SAE.encode stateful?")
        sae = sae_mlp8  # normalize_activations='layer_norm'
        print(f"  SAE normalize_activations={getattr(sae.cfg,'normalize_activations','?')!r}")
        torch.manual_seed(0)
        a1 = torch.randn(4, 768)
        a2 = torch.randn(4, 768) * 5.0 + 2.0  # very different scale/mean
        with torch.no_grad():
            f1 = sae.encode(a1)
            xhat1_immediate = sae.decode(f1)        # decode right after encode(a1)
            _ = sae.encode(a2)                       # encode a DIFFERENT activation
            xhat1_later = sae.decode(f1)             # decode the OLD f1 again
        drift = (xhat1_immediate - xhat1_later).abs().max().item()
        verdict = "STATEFUL -> BUG #8 BITES" if drift > 1e-6 else "stateless -> BUG #8 benign here"
        print(f"  max|decode(f1)_immediate - decode(f1)_after_encode(a2)| = {drift:.2e}  [{verdict}]")
    except Exception:
        print("  SECTION C FAILED:"); traceback.print_exc()

    # ================================================================== D
    try:
        hr("D. IG error->feature edge gap (include_error_node attrib vs ig)")
        runner = SAEFeatureEdgeRunner(
            model=model,
            sae_sites={Site("resid_post", 6): sae_r6, Site("resid_post", 7): sae_r7},
            resolver=resolver,
        )

        def count_error_writers(circuit):
            n = 0
            for e in circuit.edges:
                w = e.writer.node
                if getattr(w, "kind", "") == "sae_error" or "error" in repr(w).lower():
                    n += 1
            return n

        c_at = runner.run(clean_tok, corrupt_tok, metric, layer_pairs="adjacent",
                         top_k_survivors=10, include_error_node=True, variant="attrib")
        c_ig = runner.run(clean_tok, corrupt_tok, metric, layer_pairs="adjacent",
                         top_k_survivors=10, include_error_node=True, variant="ig", n_ig_steps=16)
        n_at = count_error_writers(c_at)
        n_ig = count_error_writers(c_ig)
        print(f"  error->feature edges: attrib={n_at}  ig={n_ig}")
        if n_at > 0 and n_ig == 0:
            print("  >>> CONFIRMED: variant='ig' SILENTLY drops all error->feature edges "
                  "while attrib produces them (honesty gap).")
        elif n_at == n_ig == 0:
            print("  (neither produced error->feature edges on this prompt — inconclusive; "
                  "try a prompt where the error term moves the metric)")
        else:
            print(f"  (attrib={n_at}, ig={n_ig} — inspect)")
    except Exception:
        print("  SECTION D FAILED:"); traceback.print_exc()


if __name__ == "__main__":
    main()
