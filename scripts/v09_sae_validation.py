"""One-off v0.9.0 SAE validation. Not part of the shipped surface.

Loads google/gemma-2-2b, attaches Recorder with the llm recipe
.with_sae(...) + 'sae_reconstruction' opted in, runs one step, prints
wall-clock + recon metrics for layer 8.

Usage:
    venv/bin/python scripts/v09_sae_validation.py            # SAE on
    venv/bin/python scripts/v09_sae_validation.py --no-sae   # SAE off
"""
from __future__ import annotations

import dataclasses
import pathlib
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from circuitry import Recorder
from circuitry.recipes import get_recipe


def main() -> None:
    sae_on = "--no-sae" not in sys.argv
    model_id = "google/gemma-2-2b"
    print(f"Loading {model_id} (sae_on={sae_on})...")
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16)

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
    run_dir = pathlib.Path(f"runs/gemma2_2b_v09_{suffix}")
    run_dir.mkdir(parents=True, exist_ok=True)

    rec = Recorder(model, run_dir, recipe, writer="jsonl",
                   every_n_steps=1, strict=False)
    rec.attach()

    inputs = tok("The quick brown fox", return_tensors="pt")
    t0 = time.perf_counter()
    with torch.inference_mode():
        model(**inputs)
    rec.step(0)
    elapsed = time.perf_counter() - t0
    rec.detach()

    print(f"wall-clock 1 step ({suffix}): {elapsed:.2f}s")
    print(f"metrics written to {run_dir / 'metrics.jsonl'}")


if __name__ == "__main__":
    main()
