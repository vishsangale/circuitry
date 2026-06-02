"""v1.7 patching backends re-validation.

Runs EAP / AtP* (TL path), ACDC (TL path), EAP+AtP*+faithfulness (HF Qwen path)
and compares fresh outputs against pre-v1.7 baseline JSONs stored in
scripts/v1_validation/*.results.json.

Run (CPU-only, no side-effects to baseline files):
    .venv/bin/python scripts/v17_validation/track1_patching_revalidation.py

Saves fresh results to scripts/v17_validation/*.results.json and prints a
severity-tagged diff summary.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import torch

# Make ioi_common importable from both the old and new directories
V1_DIR = Path(__file__).parent.parent / "v1_validation"
sys.path.insert(0, str(V1_DIR))
from ioi_common import (  # noqa: E402
    CIRCUIT,
    all_circuit_heads,
    batched_logit_diff_metric,
    build_ioi_batch,
    head_class,
)

from circuitry.patching import ACDCRunner, AtPRunner, EAPRunner  # noqa: E402
from circuitry.patching.sites import HFSiteResolver, TLSiteResolver  # noqa: E402

DEVICE = "cpu"  # macbook: MPS flagged unreliable; use CPU
V17_DIR = Path(__file__).parent
BASELINE_DIR = V1_DIR

# ─────────────────────────────────────────────────────────────────
# Helpers shared with track1_eap_atp_tl
# ─────────────────────────────────────────────────────────────────

def eap_head_importance(result) -> dict[tuple[int, int], float]:
    imp: dict[tuple[int, int], float] = {}
    for edge, sc in result.scores.items():
        w = edge.writer
        if w.kind == "attn_head":
            imp[(w.layer, w.head)] = imp.get((w.layer, w.head), 0.0) + abs(sc)
    return imp


def atp_head_importance(result) -> dict[tuple[int, int], float]:
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


def report_eap_atp(name: str, imp: dict[tuple[int, int], float]) -> dict:
    ranked = sorted(imp, key=lambda h: imp[h], reverse=True)
    res = {
        "top20": [list(h) for h in ranked[:20]],
        "overlap": {},
        "class_recall_at26": class_recall(ranked, 26),
    }
    for k in (10, 15, 20, 26):
        hits, frac = overlap_at_k(ranked, k)
        res["overlap"][str(k)] = {"hits": hits, "frac": round(frac, 3)}
    return res


def _spearman(xs, ys):
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


# ─────────────────────────────────────────────────────────────────
# Track A: EAP + AtP* on TL GPT-2
# ─────────────────────────────────────────────────────────────────

def run_eap_atp_tl(n_prompts: int = 50, seed: int = 0) -> dict:
    print("\n" + "=" * 60, flush=True)
    print("TRACK A: EAP + AtP* on TransformerLens GPT-2", flush=True)
    print("=" * 60, flush=True)

    torch.manual_seed(seed)
    from transformer_lens import HookedTransformer
    model = HookedTransformer.from_pretrained("gpt2", device=DEVICE)
    batch = build_ioi_batch(model.tokenizer, n=n_prompts, seed=seed, device=DEVICE)
    metric = batched_logit_diff_metric(batch.io_ids, batch.s_ids)
    print(f"device={DEVICE} prompts={n_prompts} seq_len={batch.clean.shape[1]}", flush=True)

    results = {"n_prompts": n_prompts, "seed": seed}

    t0 = time.time()
    eap = EAPRunner(model, resolver=TLSiteResolver())
    eap_res = eap.run(batch.clean, batch.corrupt, metric)
    print(f"EAP done in {time.time()-t0:.1f}s ({len(eap_res.scores)} edges)", flush=True)
    results["eap"] = report_eap_atp("EAP", eap_head_importance(eap_res))

    t0 = time.time()
    atp = AtPRunner(model, resolver=TLSiteResolver())
    atp_res = atp.run(batch.clean, batch.corrupt, metric, qk_fix=True)
    print(f"AtP* done in {time.time()-t0:.1f}s ({len(atp_res.scores)} nodes)", flush=True)
    results["atp"] = report_eap_atp("AtP*", atp_head_importance(atp_res))

    return results


# ─────────────────────────────────────────────────────────────────
# Track B: ACDC on TL GPT-2
# ─────────────────────────────────────────────────────────────────

def run_acdc_tl(n_prompts: int = 12, seed: int = 0) -> dict:
    print("\n" + "=" * 60, flush=True)
    print("TRACK B: ACDC on TransformerLens GPT-2", flush=True)
    print("=" * 60, flush=True)

    torch.manual_seed(seed)
    from transformer_lens import HookedTransformer
    model = HookedTransformer.from_pretrained("gpt2", device=DEVICE)
    batch = build_ioi_batch(model.tokenizer, n=n_prompts, seed=seed, device=DEVICE)
    metric = batched_logit_diff_metric(batch.io_ids, batch.s_ids)
    print(f"device={DEVICE} prompts={n_prompts}", flush=True)

    eap = EAPRunner(model, resolver=TLSiteResolver())
    eap_scores = eap.run(batch.clean, batch.corrupt, metric).scores
    n_edges = len(eap.graph.edges)
    print(f"graph: {n_edges} edges", flush=True)

    acdc = ACDCRunner(model, resolver=TLSiteResolver())
    TAUS = [0.05]
    results = {"n_prompts": n_prompts, "seed": seed, "n_edges": n_edges, "sweep": []}
    detailed = None

    for tau in TAUS:
        t0 = time.time()
        res = acdc.run(batch.clean, batch.corrupt, tau=tau,
                       ordering="eap", eap_scores=eap_scores)
        dt = time.time() - t0
        kept_heads = sorted({(e.writer.layer, e.writer.head) for e in res.kept_edges
                             if e.writer.kind == "attn_head"})
        in_circuit = sum(1 for h in kept_heads if h in all_circuit_heads())
        row = {
            "tau": tau,
            "n_kept_edges": res.n_kept(),
            "final_kl": round(res.final_kl, 5),
            "n_kept_heads": len(kept_heads),
            "heads_in_circuit": in_circuit,
        }
        results["sweep"].append(row)
        print(f"  tau={tau} kept {res.n_kept()}/{n_edges} edges, "
              f"{len(kept_heads)} heads ({in_circuit} in published circuit), "
              f"KL={res.final_kl:.4f}  [{dt:.0f}s]", flush=True)
        if tau == 0.05:
            detailed = [[h[0], h[1], head_class(*h) or "-"] for h in kept_heads]

    results["kept_heads_tau0.05"] = detailed
    return results


# ─────────────────────────────────────────────────────────────────
# Track C: EAP + AtP* + faithfulness on HF Qwen2.5-0.5B
# ─────────────────────────────────────────────────────────────────

def run_hf_qwen(n_prompts: int = 16, seed: int = 0) -> dict:
    print("\n" + "=" * 60, flush=True)
    print("TRACK C: EAP + AtP* + faithfulness on HF Qwen2.5-0.5B", flush=True)
    print("=" * 60, flush=True)

    MODEL = "Qwen/Qwen2.5-0.5B"
    torch.manual_seed(seed)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, attn_implementation="eager", torch_dtype=torch.float32
    ).to(DEVICE).eval()
    cfg = model.config
    print(f"device={DEVICE} model={MODEL} layers={cfg.num_hidden_layers} "
          f"heads={cfg.num_attention_heads} kv_heads={cfg.num_key_value_heads} "
          f"d_model={cfg.hidden_size}", flush=True)

    batch = build_ioi_batch(tok, n=n_prompts, seed=seed, device=DEVICE)
    clean = {"input_ids": batch.clean}
    corrupt = {"input_ids": batch.corrupt}
    metric = batched_logit_diff_metric(batch.io_ids, batch.s_ids)

    with torch.no_grad():
        ld_clean = metric(model(**clean)).item()
        ld_corrupt = metric(model(**corrupt)).item()
    print(f"IOI logit-diff: clean={ld_clean:+.3f}  corrupt={ld_corrupt:+.3f}  "
          f"(signal = {ld_clean - ld_corrupt:+.3f})", flush=True)

    resolver = HFSiteResolver.from_config(cfg)
    results = {
        "model": MODEL, "n_prompts": n_prompts, "seed": seed,
        "logit_diff_clean": ld_clean, "logit_diff_corrupt": ld_corrupt,
    }

    t0 = time.time()
    eap = EAPRunner(model, resolver=resolver)
    eap_res = eap.run(clean, corrupt, metric)
    print(f"EAP (HF) OK in {time.time()-t0:.1f}s — {len(eap_res.scores)} edges scored", flush=True)
    top_eap = [(f"{e.writer.kind}{e.writer.layer}.{e.writer.head}", round(s, 3))
               for e, s in eap_res.top_k(8) if e.writer.kind == "attn_head"]
    results["eap_ran"] = True
    results["eap_top_attn_edges"] = top_eap

    t0 = time.time()
    atp = AtPRunner(model, resolver=resolver)
    atp_res = atp.run(clean, corrupt, metric, qk_fix=True)
    print(f"AtP* (HF) OK in {time.time()-t0:.1f}s — {len(atp_res.scores)} nodes scored", flush=True)

    t0 = time.time()
    verified = atp_res.verify_top_k(24, clean, corrupt, metric, resolver, atp)
    pairs = [(float(a), float(t)) for (a, t) in verified.values()]
    distinct = sorted(set((round(a, 4), round(t, 4)) for a, t in pairs))
    atp_scores = [p[0] for p in pairs]
    true_effects = [p[1] for p in pairs]
    rho_all = _spearman(atp_scores, true_effects)
    rho_distinct = _spearman([p[0] for p in distinct], [p[1] for p in distinct])
    sign_agree = sum(1 for a, t in pairs if (a >= 0) == (t >= 0)) / len(pairs)
    print(f"verify_top_k(24) in {time.time()-t0:.1f}s", flush=True)
    print(f"  Spearman(atp,true): all={rho_all:+.3f}  distinct={rho_distinct:+.3f}  "
          f"sign-agreement={sign_agree:.0%}", flush=True)

    results["faithfulness"] = {
        "spearman_all": round(rho_all, 3),
        "spearman_distinct": round(rho_distinct, 3),
        "n_distinct": len(distinct),
        "sign_agreement": round(sign_agree, 3),
        "pairs": [[round(a, 4), round(t, 4)] for a, t in pairs],
    }
    return results


# ─────────────────────────────────────────────────────────────────
# Comparison helpers
# ─────────────────────────────────────────────────────────────────

def load_baseline(name: str) -> dict:
    p = BASELINE_DIR / f"{name}.results.json"
    with open(p) as f:
        return json.load(f)


def compare_top20(tag: str, fresh: list, baseline: list) -> list[str]:
    findings = []
    fresh_set = {(r[0], r[1]) for r in fresh}
    base_set = {(r[0], r[1]) for r in baseline}
    overlap = len(fresh_set & base_set)
    overlap_pct = overlap / len(base_set) * 100
    if fresh == baseline:
        findings.append(f"GOOD | {tag} top20 | Exact match (deterministic) | top20 identical")
    elif overlap >= 18:
        findings.append(f"GOOD | {tag} top20 | High overlap {overlap}/20 ({overlap_pct:.0f}%) with baseline | ranking preserved")
    elif overlap >= 15:
        findings.append(f"ERGO | {tag} top20 | Moderate overlap {overlap}/20 ({overlap_pct:.0f}%) with baseline | minor reordering expected")
    else:
        findings.append(f"REGRESSION | {tag} top20 | Low overlap {overlap}/20 ({overlap_pct:.0f}%) with baseline | circuit ranking diverged")
    return findings


def compare_overlap(tag: str, fresh: dict, baseline: dict) -> list[str]:
    findings = []
    for k_str in ("10", "15", "20", "26"):
        fh = fresh.get(k_str, {}).get("hits", -1)
        bh = baseline.get(k_str, {}).get("hits", -1)
        if fh == bh:
            findings.append(f"GOOD | {tag} overlap@{k_str} | {fh}/{k_str} matches baseline exactly")
        elif abs(fh - bh) <= 1:
            findings.append(f"ERGO | {tag} overlap@{k_str} | fresh={fh} vs baseline={bh} (±1 within tolerance)")
        else:
            findings.append(f"REGRESSION | {tag} overlap@{k_str} | fresh={fh} vs baseline={bh} (delta={fh-bh})")
    return findings


def compare_acdc(fresh: dict, baseline: dict) -> list[str]:
    findings = []
    f_row = fresh["sweep"][0] if fresh.get("sweep") else {}
    b_row = baseline["sweep"][0] if baseline.get("sweep") else {}

    # n_edges
    if fresh.get("n_edges") == baseline.get("n_edges"):
        findings.append(f"GOOD | ACDC graph | n_edges={fresh['n_edges']} matches baseline exactly")
    else:
        findings.append(f"REGRESSION | ACDC graph | n_edges: fresh={fresh.get('n_edges')} vs baseline={baseline.get('n_edges')}")

    # kept edges at tau=0.05
    fk, bk = f_row.get("n_kept_edges"), b_row.get("n_kept_edges")
    if fk == bk:
        findings.append(f"GOOD | ACDC tau=0.05 | n_kept_edges={fk} matches baseline exactly")
    else:
        findings.append(f"REGRESSION | ACDC tau=0.05 | n_kept_edges: fresh={fk} vs baseline={bk}")

    # kept heads in circuit
    fhi = f_row.get("heads_in_circuit"); bhi = b_row.get("heads_in_circuit")
    if fhi == bhi:
        findings.append(f"GOOD | ACDC tau=0.05 | heads_in_circuit={fhi} matches baseline exactly")
    else:
        findings.append(f"REGRESSION | ACDC tau=0.05 | heads_in_circuit: fresh={fhi} vs baseline={bhi}")

    # final KL
    fkl = f_row.get("final_kl"); bkl = b_row.get("final_kl")
    if fkl is not None and bkl is not None:
        delta_kl = abs(fkl - bkl)
        if delta_kl < 0.01:
            findings.append(f"GOOD | ACDC KL | final_kl={fkl:.5f} vs baseline={bkl:.5f} (delta={delta_kl:.5f})")
        elif delta_kl < 0.05:
            findings.append(f"ERGO | ACDC KL | final_kl={fkl:.5f} vs baseline={bkl:.5f} (delta={delta_kl:.5f})")
        else:
            findings.append(f"REGRESSION | ACDC KL | final_kl={fkl:.5f} vs baseline={bkl:.5f} (large delta={delta_kl:.5f})")

    # kept heads set
    fresh_heads = set(tuple(h[:2]) for h in (fresh.get("kept_heads_tau0.05") or []))
    base_heads = set(tuple(h[:2]) for h in (baseline.get("kept_heads_tau0.05") or []))
    if fresh_heads == base_heads:
        findings.append(f"GOOD | ACDC kept_heads@tau=0.05 | Exact head set match: {sorted(fresh_heads)}")
    else:
        extra = fresh_heads - base_heads
        missing = base_heads - fresh_heads
        findings.append(f"REGRESSION | ACDC kept_heads@tau=0.05 | "
                        f"extra={sorted(extra)} missing={sorted(missing)}")
    return findings


def compare_qwen(fresh: dict, baseline: dict) -> list[str]:
    findings = []

    # Logit-diff sanity (within 0.1 of baseline)
    for key in ("logit_diff_clean", "logit_diff_corrupt"):
        fv = fresh.get(key); bv = baseline.get(key)
        if fv is not None and bv is not None:
            d = abs(fv - bv)
            if d < 0.01:
                findings.append(f"GOOD | HF-Qwen {key} | {fv:.4f} vs baseline {bv:.4f} (delta={d:.4f})")
            elif d < 0.1:
                findings.append(f"ERGO | HF-Qwen {key} | {fv:.4f} vs baseline {bv:.4f} (delta={d:.4f})")
            else:
                findings.append(f"REGRESSION | HF-Qwen {key} | {fv:.4f} vs baseline {bv:.4f} (large delta={d:.4f})")

    # EAP ran
    if fresh.get("eap_ran"):
        findings.append("GOOD | HF-Qwen EAP | eap_ran=True (HF+GQA pipeline succeeded)")
    else:
        findings.append("REGRESSION | HF-Qwen EAP | eap_ran=False (EAP did not complete)")

    # Faithfulness: Spearman
    ff = fresh.get("faithfulness", {})
    bf = baseline.get("faithfulness", {})
    for k in ("spearman_all", "spearman_distinct", "sign_agreement"):
        fv = ff.get(k); bv = bf.get(k)
        if fv is None or bv is None:
            findings.append(f"GAP | HF-Qwen {k} | not present in results")
            continue
        d = abs(fv - bv)
        label = f"HF-Qwen faithfulness.{k}"
        if d < 0.05:
            findings.append(f"GOOD | {label} | fresh={fv:.3f} vs baseline={bv:.3f} (delta={d:.3f})")
        elif d < 0.15:
            findings.append(f"ERGO | {label} | fresh={fv:.3f} vs baseline={bv:.3f} (delta={d:.3f})")
        else:
            findings.append(f"REGRESSION | {label} | fresh={fv:.3f} vs baseline={bv:.3f} (large delta={d:.3f})")

    # n_distinct
    fn = ff.get("n_distinct"); bn = bf.get("n_distinct")
    if fn == bn:
        findings.append(f"GOOD | HF-Qwen n_distinct | {fn} matches baseline")
    else:
        findings.append(f"ERGO | HF-Qwen n_distinct | fresh={fn} vs baseline={bn}")

    return findings


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

def main():
    all_findings: list[str] = []

    # ── Track A: EAP + AtP* TL ──────────────────────────────────
    try:
        fresh_a = run_eap_atp_tl()
        out_a = V17_DIR / "track1_eap_atp_tl.results.json"
        with open(out_a, "w") as f:
            json.dump(fresh_a, f, indent=2)
        print(f"\nSaved: {out_a}", flush=True)

        base_a = load_baseline("track1_eap_atp_tl")
        for method in ("eap", "atp"):
            all_findings += compare_top20(
                f"TL-{method.upper()}",
                fresh_a[method]["top20"],
                base_a[method]["top20"],
            )
            all_findings += compare_overlap(
                f"TL-{method.upper()}",
                fresh_a[method]["overlap"],
                base_a[method]["overlap"],
            )
    except Exception as e:
        import traceback
        all_findings.append(f"BUG | TL EAP+AtP* | Exception: {type(e).__name__}: {e} | {traceback.format_exc()[:300]}")
        print(f"TRACK A FAILED: {e}", flush=True)
        traceback.print_exc()

    # ── Track B: ACDC TL ─────────────────────────────────────────
    try:
        fresh_b = run_acdc_tl()
        out_b = V17_DIR / "track1_acdc_tl.results.json"
        with open(out_b, "w") as f:
            json.dump(fresh_b, f, indent=2)
        print(f"\nSaved: {out_b}", flush=True)

        base_b = load_baseline("track1_acdc_tl")
        all_findings += compare_acdc(fresh_b, base_b)
    except Exception as e:
        import traceback
        all_findings.append(f"BUG | TL ACDC | Exception: {type(e).__name__}: {e} | {traceback.format_exc()[:300]}")
        print(f"TRACK B FAILED: {e}", flush=True)
        traceback.print_exc()

    # ── Track C: HF Qwen ─────────────────────────────────────────
    try:
        fresh_c = run_hf_qwen()
        out_c = V17_DIR / "track1_hf_qwen.results.json"
        with open(out_c, "w") as f:
            json.dump(fresh_c, f, indent=2)
        print(f"\nSaved: {out_c}", flush=True)

        base_c = load_baseline("track1_hf_qwen")
        all_findings += compare_qwen(fresh_c, base_c)
    except Exception as e:
        import traceback
        all_findings.append(f"BUG | HF Qwen | Exception: {type(e).__name__}: {e} | {traceback.format_exc()[:300]}")
        print(f"TRACK C FAILED: {e}", flush=True)
        traceback.print_exc()

    # ── Print summary ──────────────────────────────────────────────
    print("\n" + "=" * 70, flush=True)
    print("SEVERITY-TAGGED FINDINGS", flush=True)
    print("=" * 70, flush=True)
    for f in all_findings:
        sev = f.split("|")[0].strip()
        if sev == "REGRESSION":
            marker = "[!!]"
        elif sev == "BUG":
            marker = "[BUG]"
        elif sev == "GAP":
            marker = "[GAP]"
        elif sev == "ERGO":
            marker = "[~]"
        else:
            marker = "[OK]"
        print(f"{marker} {f}", flush=True)

    regressions = [f for f in all_findings if f.startswith("REGRESSION")]
    bugs = [f for f in all_findings if f.startswith("BUG")]
    print(f"\nSUMMARY: {len(all_findings)} findings — "
          f"{len([f for f in all_findings if f.startswith('GOOD')])} GOOD, "
          f"{len([f for f in all_findings if f.startswith('ERGO')])} ERGO, "
          f"{len([f for f in all_findings if f.startswith('GAP')])} GAP, "
          f"{len(regressions)} REGRESSION, "
          f"{len(bugs)} BUG", flush=True)

    if not regressions and not bugs:
        print("\nVERDICT: PASS — v1.7 refactor preserved EAP/ATP/ACDC behavior YES", flush=True)
    else:
        print("\nVERDICT: FAIL — regressions detected", flush=True)
        for r in regressions + bugs:
            print(f"  {r}", flush=True)


if __name__ == "__main__":
    main()
