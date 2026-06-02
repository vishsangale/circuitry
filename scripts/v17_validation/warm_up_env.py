#!/usr/bin/env python3
"""Warm the real-model cache for the v1.7 evaluation.

- Confirms GPT-2-small loads via HuggingFace AND TransformerLens.
- Enumerates available sae_ids for the three GPT-2 SAE releases we will use
  (resid_post / mlp_out / attn_out), then downloads a minimal subset:
  two resid layers (a writer/reader pair for edges) + one mlp + one attn.
- Prints d_in / d_sae / hook_name / load time for each, so we know the exact
  shapes and ID format before writing the validation scripts.

Pure environment prep — touches no circuitry src. Run from repo root:
    .venv/bin/python scripts/v17_validation/warm_up_env.py
"""

from __future__ import annotations

import time

import torch


def _t(label: str, fn):
    t0 = time.time()
    out = fn()
    print(f"[{time.time() - t0:6.1f}s] {label}", flush=True)
    return out


def main() -> None:
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device={dev} torch={torch.__version__}\n", flush=True)

    # --- GPT-2 via HF ---
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast

    hf = _t("HF GPT2LMHeadModel.from_pretrained('gpt2')",
            lambda: GPT2LMHeadModel.from_pretrained("gpt2"))
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    print(f"  HF n_layer={hf.config.n_layer} d_model={hf.config.n_embd}\n", flush=True)

    # --- GPT-2 via TransformerLens ---
    from transformer_lens import HookedTransformer

    tl = _t("TL HookedTransformer.from_pretrained('gpt2')",
            lambda: HookedTransformer.from_pretrained("gpt2", device=dev))
    print(f"  TL n_layers={tl.cfg.n_layers} d_model={tl.cfg.d_model} dtype={tl.cfg.dtype}\n",
          flush=True)

    # --- SAE catalog (metadata only, no download) ---
    from sae_lens import SAE
    from sae_lens.loading.pretrained_saes_directory import get_pretrained_saes_directory

    d = get_pretrained_saes_directory()
    for rel in ("gpt2-small-res-jb", "gpt2-small-mlp-out-v5-32k", "gpt2-small-attn-out-v5-32k"):
        info = d.get(rel)
        if info is None:
            print(f"  RELEASE MISSING: {rel}", flush=True)
            continue
        ids = list(info.saes_map.keys())
        print(f"  {rel}: {len(ids)} sae_ids; sample={ids[:3]}", flush=True)
    print(flush=True)

    # --- Download a minimal subset ---
    targets = [
        ("gpt2-small-res-jb", "blocks.7.hook_resid_pre"),
        ("gpt2-small-res-jb", "blocks.8.hook_resid_pre"),
        ("gpt2-small-mlp-out-v5-32k", "blocks.8.hook_mlp_out"),
        ("gpt2-small-attn-out-v5-32k", "blocks.8.hook_attn_out"),
    ]
    for rel, sae_id in targets:
        def _load(rel=rel, sae_id=sae_id):
            res = SAE.from_pretrained(rel, sae_id, device=dev)
            return res[0] if isinstance(res, tuple) else res

        try:
            sae = _t(f"SAE.from_pretrained({rel}, {sae_id})", _load)
            cfg = sae.cfg
            d_in = getattr(cfg, "d_in", getattr(cfg, "d_sae", "?"))
            d_sae = getattr(cfg, "d_sae", "?")
            hook = getattr(cfg, "hook_name", getattr(cfg, "metadata", "?"))
            print(f"  -> d_in={d_in} d_sae={d_sae} hook={hook}\n", flush=True)
        except Exception as e:  # noqa: BLE001
            import traceback
            print(f"  !! FAILED {rel}/{sae_id}: {e.__class__.__name__}: {e}", flush=True)
            traceback.print_exc()

    print("WARM-UP DONE", flush=True)


if __name__ == "__main__":
    main()
