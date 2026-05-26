"""Track 3b — SAE workflow on a real model (Gemma-2-2b + Gemma Scope), realistic prompt.

v0.9 validated the SAE path on a 4-token prompt and got l0=1293 vs the SAE's design
point ~71 (out-of-distribution short input). This re-runs on a realistic
paragraph-length prompt (BOS-prepended, the SAE's training regime) to confirm the
sparsity collapses toward the design point — closing the v0.9 short-sequence caveat.

Run:  venv/bin/python scripts/v1_validation/track3b_sae.py
Saves: scripts/v1_validation/track3b_sae.results.json
"""

from __future__ import annotations

import json
import os

import torch

from circuitry.sae.loader import load_sae
from circuitry.sae.metrics import sae_reconstruction_error

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RELEASE = "gemma-scope-2b-pt-res"
SAE_ID = "layer_8/width_16k/average_l0_71"
LAYER = 8

PARAGRAPH = (
    "The history of artificial intelligence began in antiquity with myths and "
    "stories of artificial beings endowed with intelligence by master craftsmen. "
    "The modern field of AI research was founded at a workshop held on the campus "
    "of Dartmouth College in the summer of 1956. Those who attended would become "
    "the leaders of AI research for decades. Many of them predicted that a machine "
    "as intelligent as a human being would exist within a generation, and they were "
    "given millions of dollars to make this vision come true. Eventually it became "
    "obvious that they had grossly underestimated the difficulty of the project."
)


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("google/gemma-2-2b")
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-2-2b", dtype=torch.bfloat16
    ).to(DEVICE).eval()
    sae = load_sae(RELEASE, SAE_ID, device=DEVICE)

    ids = tok(PARAGRAPH, return_tensors="pt").input_ids.to(DEVICE)  # BOS auto-prepended
    n_tok = ids.shape[1]
    print(f"device={DEVICE} prompt tokens={n_tok} (vs v0.9: 4) layer={LAYER}", flush=True)
    print(f"SAE: {RELEASE} / {SAE_ID}", flush=True)

    captured = {}
    def hook(_m, _inp, out):
        captured["resid"] = out[0] if isinstance(out, tuple) else out
    h = model.model.layers[LAYER].register_forward_hook(hook)
    with torch.no_grad():
        model(ids)
    h.remove()

    resid = captured["resid"]  # (1, seq, d_model)
    print(f"captured resid_post[{LAYER}] shape={tuple(resid.shape)}", flush=True)
    # Gemma-2 has a huge-norm BOS/attention-sink token (pos 0) that dominates MSE/l0.
    bos_norm = float(resid[0, 0].norm())
    rest_norm = float(resid[0, 1:].norm(dim=-1).mean())
    print(f"BOS-token norm={bos_norm:.0f} vs mean non-BOS norm={rest_norm:.0f}", flush=True)
    m = sae_reconstruction_error(resid, sae)
    m_nobos = sae_reconstruction_error(resid[:, 1:, :], sae)

    print("\nSAE reconstruction metrics — all tokens / excluding BOS:", flush=True)
    for k in ("recon_mse", "l0", "l1", "frac_alive", "ce_recovered_proxy"):
        print(f"  {k:20s} {m[k]:10.3f}   {m_nobos[k]:10.3f}", flush=True)
    print(f"\n  l0: all={m['l0']:.0f}  ex-BOS={m_nobos['l0']:.0f}  "
          f"(design ~71; v0.9 4-token=1293)", flush=True)

    results = {"model": "google/gemma-2-2b", "release": RELEASE, "sae_id": SAE_ID,
               "layer": LAYER, "n_tokens": n_tok, "bos_norm": round(bos_norm, 1),
               "non_bos_mean_norm": round(rest_norm, 1),
               "metrics_all": {k: round(m[k], 4) for k in m},
               "metrics_ex_bos": {k: round(m_nobos[k], 4) for k in m_nobos},
               "baseline_v0.9_4tok": {"l0": 1293, "recon_mse": 3130, "ce_recovered_proxy": -19.1,
                                      "frac_alive": 0.384}}
    out = os.path.join(os.path.dirname(__file__), "track3b_sae.results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved -> {out}", flush=True)


if __name__ == "__main__":
    main()
