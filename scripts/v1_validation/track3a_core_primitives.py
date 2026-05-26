"""Track 3a — Tier-1 core primitives on a REAL pretrained model (GPT-2 small, eager).

Validates the primitives produce CORRECT signals on a known model (not just that
they emit): logit-lens KL should fall toward the final layer; GPT-2's known
induction heads (5.5, 5.8, 5.9, 6.9) should top induction_score; etc. Eager attn
makes attention diagnostics available (they were silently zero under SDPA on Gemma).

Run:  venv/bin/python scripts/v1_validation/track3a_core_primitives.py
Saves: scripts/v1_validation/track3a_core_primitives.results.json
"""

from __future__ import annotations

import json
import os

import torch

from circuitry.core.activation import token_similarity
from circuitry.core.attention import attention_pattern_entropy, induction_score
from circuitry.core.gradient import grad_norm_per_module, total_grad_norm
from circuitry.core.inventory import ModelInventory
from circuitry.core.lens import logit_lens_kl
from circuitry.core.weight import (
    direction_cosine,
    effective_rank,
    heavy_tail_alpha,
    stable_rank,
    update_delta,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
KNOWN_INDUCTION = {(5, 5), (5, 8), (5, 9), (6, 9)}


def main():
    torch.manual_seed(0)
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2", attn_implementation="eager").to(DEVICE).eval()
    results = {"model": "gpt2", "device": DEVICE}
    print(f"device={DEVICE} model=gpt2 (eager)", flush=True)

    # --- inventory ---
    inv = ModelInventory(model)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n[inventory] {n_params/1e6:.1f}M params; "
          f"{len(list(model.named_parameters()))} named params", flush=True)
    results["n_params"] = n_params

    # --- weight-space primitives on a real matrix (layer-0 mlp c_fc) ---
    W = model.transformer.h[0].mlp.c_fc.weight.detach().float()
    er, sr, ht = effective_rank(W), stable_rank(W), heavy_tail_alpha(W)
    print(f"\n[weight] layer0 mlp.c_fc {tuple(W.shape)}: "
          f"effective_rank={er:.1f} stable_rank={sr:.1f} heavy_tail_alpha={ht:.2f}", flush=True)
    assert 0 < er <= min(W.shape), "effective_rank out of range"
    results["weight_layer0_mlp"] = {"shape": list(W.shape), "effective_rank": round(er, 2),
                                    "stable_rank": round(sr, 2), "heavy_tail_alpha": round(ht, 3)}

    # --- logit lens: KL should fall toward the final layer ---
    text = "The Eiffel Tower is located in the city of Paris, the capital of France."
    ids = tok(text, return_tensors="pt").input_ids.to(DEVICE)
    with torch.no_grad():
        out = model(ids, output_hidden_states=True)
    final_logits = out.logits
    unembed = model.lm_head.weight  # (vocab, d_model)
    ln_f = model.transformer.ln_f
    kper = []
    for layer_idx, h in enumerate(out.hidden_states):
        kl = logit_lens_kl(h, unembed, final_logits, layer_norm=ln_f)
        kper.append(round(kl, 4))
    print(f"\n[lens] logit_lens_kl per layer (0=embed .. {len(kper)-1}=final):", flush=True)
    print("  " + " ".join(f"{k:.2f}" for k in kper), flush=True)
    monotone_tail = kper[-1] < kper[len(kper) // 2] < kper[1]
    print(f"  KL falls toward final layer (embed→mid→final {kper[1]:.2f}→"
          f"{kper[len(kper)//2]:.2f}→{kper[-1]:.3f}): {monotone_tail}", flush=True)
    results["logit_lens_kl_per_layer"] = kper
    results["lens_falls_toward_final"] = bool(monotone_tail)

    # --- induction_score on a repeated-random-token probe ---
    L = 25
    rand = torch.randint(0, tok.vocab_size, (1, L), device=DEVICE)
    probe = torch.cat([rand, rand], dim=1)  # seq = 2L
    with torch.no_grad():
        att = model(probe, output_attentions=True).attentions  # tuple[(1,H,2L,2L)]
    head_scores = {}
    for layer_idx, a in enumerate(att):
        for head, sc in enumerate(induction_score(a, seq_len_repeat=L)):
            head_scores[(layer_idx, head)] = sc
    top = sorted(head_scores, key=lambda h: head_scores[h], reverse=True)[:6]
    hits = sum(1 for h in top if h in KNOWN_INDUCTION)
    print(f"\n[attention] top-6 induction heads (probe L={L}):", flush=True)
    for h in top:
        mark = " <- known" if h in KNOWN_INDUCTION else ""
        print(f"  {h[0]}.{h[1]}  score={head_scores[h]:.3f}{mark}", flush=True)
    print(f"  known induction heads {sorted(KNOWN_INDUCTION)} in top-6: {hits}/4", flush=True)
    results["induction_top6"] = [[h[0], h[1], round(head_scores[h], 3)] for h in top]
    results["known_induction_in_top6"] = hits

    # --- attention_pattern_entropy on natural text ---
    with torch.no_grad():
        att2 = model(ids, output_attentions=True).attentions
    ent_l0 = attention_pattern_entropy(att2[0])
    print(f"\n[attention] layer-0 per-head entropy (nats): "
          f"{[round(e,2) for e in ent_l0]}", flush=True)
    results["attn_entropy_layer0"] = [round(e, 3) for e in ent_l0]

    # --- token_similarity on real hidden states (mean off-diagonal cosine) ---
    sim = float(token_similarity(out.hidden_states[6]))  # scalar in [-1, 1]
    print(f"\n[activation] token_similarity (mean off-diag cosine, layer 6) = {sim:.4f}", flush=True)
    assert -1.0 <= sim <= 1.0, "token_similarity out of [-1,1]"
    results["token_similarity_layer6"] = round(sim, 4)

    # --- gradient primitives after a real backward ---
    model.zero_grad()
    loss = model(ids, labels=ids).loss
    loss.backward()
    grads = {n: p.grad for n, p in model.named_parameters() if p.grad is not None}
    per = grad_norm_per_module(grads)
    tot = total_grad_norm(per)
    print(f"\n[gradient] {len(per)} modules with grads; total_grad_norm={tot:.3f}", flush=True)
    results["total_grad_norm"] = round(tot, 4)

    # --- weight deltas across 2 SGD steps (update_delta + direction_cosine) ---
    snaps = []
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    for _ in range(3):
        snaps.append({n: p.detach().clone() for n, p in model.named_parameters()})
        opt.zero_grad()
        model(ids, labels=ids).loss.backward()
        opt.step()
    delta = update_delta(snaps[2], snaps[1])
    cos = direction_cosine(snaps[2], snaps[1], snaps[0])
    sample_key = next(iter(delta))
    print(f"\n[weight] update_delta+direction_cosine over 2 SGD steps:", flush=True)
    print(f"  e.g. {sample_key}: ||Δ||={delta[sample_key]:.4f}  cos(update1,update2)={cos[sample_key]:+.3f}", flush=True)
    results["update_delta_sample"] = {sample_key: round(delta[sample_key], 5)}
    results["direction_cosine_sample"] = {sample_key: round(cos[sample_key], 4)}

    out_path = os.path.join(os.path.dirname(__file__), "track3a_core_primitives.results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
