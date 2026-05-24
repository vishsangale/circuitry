"""SAE reconstruction validation. Not part of the shipped surface.

Loads google/gemma-2-2b, attaches Recorder with the llm recipe
.with_sae(...) + 'sae_reconstruction' opted in, runs one step on a
representative-length sequence, and prints wall-clock + the layer-8 SAE
reconstruction metrics (recon_mse / l0 / l1 / frac_alive / ce_recovered_proxy).

v0.9.0 validated on a 4-token input ("The quick brown fox") which gave a
non-representative l0 (1293 vs the SAE's design point ~71). This run uses a
≥128-token prose passage. See docs/observations/2026-05-23-v091-validation.md.

Usage:
    venv/bin/python scripts/v09_sae_validation.py                 # SAE on, CUDA, ~256 tok
    venv/bin/python scripts/v09_sae_validation.py --no-sae        # SAE off
    venv/bin/python scripts/v09_sae_validation.py --seqlen 64     # shorter
    venv/bin/python scripts/v09_sae_validation.py --device cpu    # CPU
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from circuitry import Recorder
from circuitry.recipes import get_recipe

# Representative English prose (public-domain style). Truncated to --seqlen
# tokens. The point is a natural in-distribution sequence, not random tokens
# or a 4-token fragment.
DEFAULT_TEXT = (
    "The history of science is the study of the development of science, "
    "including both the natural and social sciences. Science is a body of "
    "empirical, theoretical, and practical knowledge about the natural world, "
    "produced by researchers making use of observation, explanation, and "
    "prediction. The earliest roots of science can be traced to ancient Egypt "
    "and Mesopotamia. Their contributions to mathematics, astronomy, and "
    "medicine entered and shaped Greek natural philosophy of classical "
    "antiquity, whereby formal attempts were made to provide explanations of "
    "events in the physical world based on natural causes. After the fall of "
    "the Western Roman Empire, knowledge of Greek conceptions of the world "
    "deteriorated in Western Europe, but was preserved in the Muslim world "
    "during the Islamic Golden Age. The recovery and assimilation of Greek "
    "works and Islamic inquiries into Western Europe from the tenth to the "
    "thirteenth century revived natural philosophy, which was later "
    "transformed by the Scientific Revolution that began in the sixteenth "
    "century as new ideas and discoveries departed from previous Greek "
    "conceptions and traditions."
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--no-sae", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                   choices=["cpu", "cuda"])
    p.add_argument("--seqlen", type=int, default=256,
                   help="Truncate the input to this many tokens.")
    p.add_argument("--text", default=DEFAULT_TEXT)
    args = p.parse_args()

    sae_on = not args.no_sae
    model_id = "google/gemma-2-2b"
    print(f"Loading {model_id} (sae_on={sae_on}, device={args.device})...")
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16)
    model = model.to(args.device)

    recipe = get_recipe("llm").with_prefix("model")
    if sae_on:
        recipe = recipe.with_sae({
            r".*\.layers\.8$": (
                "gemma-scope-2b-pt-res",
                "layer_8/width_16k/average_l0_71",
            ),
        })
        recipe = dataclasses.replace(
            recipe,
            activation_diagnostics=recipe.activation_diagnostics + ["sae_reconstruction"],
        )

    suffix = "sae" if sae_on else "no_sae"
    run_dir = pathlib.Path(f"runs/gemma2_2b_v091_{suffix}")
    run_dir.mkdir(parents=True, exist_ok=True)

    rec = Recorder(model, run_dir, recipe, writer="jsonl",
                   every_n_steps=1, strict=False)
    rec.attach()

    inputs = tok(args.text, return_tensors="pt", truncation=True,
                 max_length=args.seqlen)
    n_tok = inputs["input_ids"].shape[-1]
    inputs = {k: v.to(args.device) for k, v in inputs.items()}
    print(f"input tokens: {n_tok}")

    if args.device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        model(**inputs)
    rec.step(0)
    if args.device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    rec.detach()

    print(f"wall-clock 1 step ({suffix}, {n_tok} tok): {elapsed:.2f}s")

    # Read back the layer-8 SAE metrics so the validation is self-contained.
    if sae_on:
        jsonl = run_dir / "metrics.jsonl"
        sae_vals: dict[str, float] = {}
        for line in jsonl.read_text().splitlines():
            rec_obj = json.loads(line)
            tag = rec_obj.get("tag", "")
            if "/sae/" in tag:
                sub = tag.rsplit("/", 1)[-1]
                sae_vals[sub] = rec_obj.get("value")
        print(f"\nlayer-8 SAE metrics ({n_tok} tok):")
        for k in ("recon_mse", "l0", "l1", "frac_alive", "ce_recovered_proxy"):
            if k in sae_vals:
                print(f"  {k:18s} = {sae_vals[k]}")
        print("\nbaselines: 4-tok (v0.9.0) recon_mse=3130 l0=1293 frac_alive=0.384"
              " | SAE design point l0 ~= 71")
    print(f"\nmetrics written to {run_dir / 'metrics.jsonl'}")


if __name__ == "__main__":
    main()
