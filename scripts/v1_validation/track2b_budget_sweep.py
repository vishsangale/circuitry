"""Track 2b — characterize the Recorder wall-clock overhead vs emit cadence and
isolate its source. The headline budget run (every_n=100) showed +23.8% on a tiny
28M model with ~15ms steps; this shows the overhead is fixed per-emit cost
(dominated by the induction-probe second forward) that amortizes with cadence.

Run:  venv/bin/python scripts/v1_validation/track2b_budget_sweep.py
Saves: scripts/v1_validation/track2b_budget_sweep.results.json
"""

from __future__ import annotations

import dataclasses
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from track2_train_recorder import RUN_ROOT, SEED, STEPS, build_model, load_tokens, train

from circuitry import Recorder
from circuitry.recipes import get_recipe


def run_recorded(model_fn, tokens, every_n, recipe):
    torch.manual_seed(SEED)
    model = model_fn().to("cuda" if torch.cuda.is_available() else "cpu")
    rd = os.path.join(RUN_ROOT, f"budget_n{every_n}_{getattr(recipe,'name','x')}")
    rec = Recorder(model, run_dir=rd, recipe=recipe, writer="null",
                   every_n_steps=every_n, strict=False)
    rec.attach()
    _, wall = train(model, tokens, STEPS, recorder=rec)
    rec.detach()
    return wall


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("gpt2")
    tokens = load_tokens(tok)
    vocab = tok.vocab_size
    model_fn = lambda: build_model(vocab)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # Baseline (no recorder), median of 3 to stabilize timing.
    torch.manual_seed(SEED)
    base_model = model_fn().to(dev)
    walls = []
    for _ in range(3):
        torch.manual_seed(SEED)
        m = model_fn().to(dev)
        _, w = train(m, tokens, STEPS, recorder=None)
        walls.append(w)
    base = sorted(walls)[1]
    print(f"baseline (median of 3): {base:.2f}s over {STEPS} steps", flush=True)

    results = {"steps": STEPS, "baseline_s": round(base, 2), "device": dev, "sweep": []}

    # Full "llm" recipe at several cadences.
    full = get_recipe("llm")
    for every_n in (50, 100, 200):
        w = run_recorded(model_fn, tokens, every_n, full)
        ov = (w - base) / base
        results["sweep"].append({"every_n": every_n, "recipe": "llm-full",
                                 "wall_s": round(w, 2), "overhead": round(ov, 4)})
        print(f"  llm-full   every_n={every_n:<4} {w:.2f}s  overhead {ov:+.1%}", flush=True)

    # Ablation: same cadence (200=default), induction_score removed (the 2nd-forward probe).
    full_no_induction = dataclasses.replace(
        full, activation_diagnostics=tuple(d for d in full.activation_diagnostics
                                           if d != "induction_score"))
    w = run_recorded(model_fn, tokens, 200, full_no_induction)
    ov = (w - base) / base
    results["sweep"].append({"every_n": 200, "recipe": "llm-no-induction",
                             "wall_s": round(w, 2), "overhead": round(ov, 4)})
    print(f"  no-induction every_n=200  {w:.2f}s  overhead {ov:+.1%}  "
          f"(isolates the induction-probe 2nd-forward cost)", flush=True)

    out = os.path.join(os.path.dirname(__file__), "track2b_budget_sweep.results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved -> {out}", flush=True)


if __name__ == "__main__":
    main()
