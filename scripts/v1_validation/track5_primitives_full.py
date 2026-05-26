"""Track 5 (gap-closing, Gemini review) — the remaining Tier-1 primitives not
covered by track3a, exercised on real GPT-2 tensors.

Covers: weight.condition_number / singular_values / attention_head_rank;
activation.dead_fraction / norm_stats / kurtosis / participation_ratio / gate_stats;
gradient.signal_propagation_depth; spectral.esd / rank_trajectory.

Run:  venv/bin/python scripts/v1_validation/track5_primitives_full.py
Saves: scripts/v1_validation/track5_primitives_full.results.json
"""

from __future__ import annotations

import json
import os

import torch

from circuitry.core.activation import (
    dead_fraction,
    gate_stats,
    kurtosis,
    norm_stats,
    participation_ratio,
)
from circuitry.core.gradient import signal_propagation_depth
from circuitry.core.spectral import esd, rank_trajectory
from circuitry.core.weight import (
    attention_head_rank,
    condition_number,
    singular_values,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    torch.manual_seed(0)
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2", attn_implementation="eager").to(DEVICE).eval()
    results = {}
    W = model.transformer.h[0].mlp.c_fc.weight.detach().float()  # (768, 3072)

    # --- weight: condition_number, singular_values ---
    cond = condition_number(W)
    sv = singular_values(W)
    print(f"[weight] condition_number={cond:.1f}  singular_values: "
          f"{sv.shape[0]} vals, max={sv[0]:.2f} min={sv[-1]:.4f} descending={bool((sv[:-1]>=sv[1:]).all())}", flush=True)
    assert cond > 1 and bool((sv[:-1] >= sv[1:]).all())
    results["condition_number"] = round(float(cond), 3)
    results["singular_values"] = {"n": int(sv.shape[0]), "max": round(float(sv[0]), 3)}

    # --- weight: attention_head_rank on GPT-2's Q block of fused c_attn ---
    # c_attn.weight is (d_model, 3*d_model) [Conv1D in×out]; Q = [:, :768], head dim in OUT axis.
    Wq = model.transformer.h[0].attn.c_attn.weight.detach().float()[:, :768]  # (768, 768)
    ahr = attention_head_rank(Wq, n_heads=12, head_dim=64, axis=1)
    print(f"[weight] attention_head_rank (12 heads): "
          f"min={min(ahr):.1f} max={max(ahr):.1f}  n={len(ahr)}", flush=True)
    assert len(ahr) == 12 and all(0 < r <= 64 for r in ahr)
    results["attention_head_rank"] = {"n_heads": len(ahr), "min": round(min(ahr), 2), "max": round(max(ahr), 2)}

    # --- capture a post-activation MLP tensor for activation primitives ---
    captured = {}
    h = model.transformer.h[0].mlp.act.register_forward_hook(
        lambda m, i, o: captured.__setitem__("act", o.detach()))
    ids = tok("Paris is the capital of France and a major European city.",
              return_tensors="pt").input_ids.to(DEVICE)
    with torch.no_grad():
        model(ids)
    h.remove()
    act = captured["act"]  # (1, seq, 3072) post-GELU

    df = dead_fraction(act, threshold=0.0)
    ns = norm_stats(act)
    kt = kurtosis(act)
    pr = participation_ratio(act)
    gs = gate_stats(act)
    print(f"\n[activation] on post-GELU MLP act {tuple(act.shape)}:", flush=True)
    print(f"  dead_fraction={df:.3f}  participation_ratio={pr:.2f}", flush=True)
    print(f"  norm_stats: mean={ns.mean:.3f} std={ns.std:.3f} max={ns.max:.2f} "
          f"frac_above_3med={ns.frac_above_k_median:.3f}", flush=True)
    print(f"  kurtosis (mean over tokens)={float(kt.mean()):.2f}", flush=True)
    print(f"  gate_stats: {{frac_active={gs.get('frac_active'):.3f}}}", flush=True)
    assert 0.0 <= df <= 1.0 and pr > 0
    results["activation"] = {"dead_fraction": round(df, 4), "participation_ratio": round(pr, 3),
                             "norm_stats_mean": round(ns.mean, 4), "kurtosis_mean": round(float(kt.mean()), 3),
                             "gate_frac_active": round(gs.get("frac_active"), 4)}

    # --- gradient: signal_propagation_depth ---
    model.zero_grad()
    loss = model(ids, labels=ids).loss
    loss.backward()
    grads_by_depth = [model.transformer.h[i].mlp.c_proj.weight.grad.detach()
                      for i in range(model.config.n_layer)]
    spd = signal_propagation_depth(grads_by_depth)
    print(f"\n[gradient] signal_propagation_depth = {spd} / {len(grads_by_depth)} layers "
          f"(grad reaches layer {spd})", flush=True)
    results["signal_propagation_depth"] = {"depth": spd, "n_layers": len(grads_by_depth)}

    # --- spectral: esd, rank_trajectory ---
    centers, counts = esd(W, bins=50)
    print(f"\n[spectral] esd: {centers.shape[0]} bins, total mass={float(counts.sum()):.0f}", flush=True)
    # rank_trajectory over 2 SGD snapshots
    opt = torch.optim.SGD(model.parameters(), lr=1e-2)
    snaps = [{k: v.detach().clone() for k, v in model.state_dict().items()}]
    model.zero_grad(); model(ids, labels=ids).loss.backward(); opt.step()
    snaps.append({k: v.detach().clone() for k, v in model.state_dict().items()})
    traj = rank_trajectory(snaps)
    sample = next(k for k in traj if "c_fc" in k)
    print(f"[spectral] rank_trajectory: {len(traj)} 2D params; "
          f"e.g. {sample}: {[round(x,1) for x in traj[sample]]}", flush=True)
    results["spectral"] = {"esd_bins": int(centers.shape[0]), "rank_trajectory_params": len(traj),
                           "rank_trajectory_sample": [round(x, 2) for x in traj[sample]]}

    out = os.path.join(os.path.dirname(__file__), "track5_primitives_full.results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nALL remaining Tier-1 primitives exercised. saved -> {out}", flush=True)


if __name__ == "__main__":
    main()
