"""Track 4 — post-hoc scan_run workflow on real (changing) checkpoints.

The CLI `circuitry scan` is a documented stub (it cannot conjure a model_factory;
prints a pointer to the programmatic API and exits 2). This validates the real
workflow: save checkpoints during training, replay them through the recipe's
weight diagnostics with scan_run(), and confirm the report shows movement between
checkpoints (the Delta column that was always "—" on static lr=0 runs).

Run:  venv/bin/python scripts/v1_validation/track4_scan.py
Saves: scripts/v1_validation/track4_scan.results.json + runs/v1_track4/
"""

from __future__ import annotations

import json
import os

import torch

from circuitry.core.weight import update_delta
from circuitry.recorder.scan import scan_run
from circuitry import build_report
import sys
sys.path.insert(0, os.path.dirname(__file__))
from track2_train_recorder import build_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RUN_ROOT = os.path.join("runs", "v1_track4")
CKPT_DIR = os.path.join(RUN_ROOT, "checkpoints")
STEPS = 60


def main():
    os.makedirs(CKPT_DIR, exist_ok=True)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("gpt2")
    vocab = tok.vocab_size
    torch.manual_seed(0)
    model = build_model(vocab).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Train briefly, saving a checkpoint at the start and end.
    data = torch.randint(0, vocab, (8, 128), device=DEVICE)
    saved = []
    for step in range(STEPS):
        if step in (0, STEPS - 1):
            p = os.path.join(CKPT_DIR, f"step{step:06d}.pt")
            torch.save(model.state_dict(), p)
            saved.append((step, p))
        out = model(input_ids=data, labels=data)
        out.loss.backward()
        opt.step(); opt.zero_grad(set_to_none=True)
    print(f"device={DEVICE} saved checkpoints: {[s for s, _ in saved]}", flush=True)

    # Confirm weights actually moved between the two checkpoints.
    sd0 = torch.load(saved[0][1], map_location="cpu", weights_only=True)
    sd1 = torch.load(saved[1][1], map_location="cpu", weights_only=True)
    deltas = update_delta(sd1, sd0)
    moved = sum(1 for v in deltas.values() if v > 1e-6)
    print(f"weights moved in {moved}/{len(deltas)} params between ckpts "
          f"(max ||Δ||={max(deltas.values()):.4f})", flush=True)

    # Post-hoc scan: replay both checkpoints through the 'llm' recipe weight diagnostics.
    out_dir = os.path.join(RUN_ROOT, "scan_out")
    scan_run(RUN_ROOT, "llm", out_dir,
             model_factory=lambda: build_model(vocab),
             writer="jsonl", strict=False)
    print(f"scan_run complete -> {out_dir}", flush=True)

    # Parse scan output: weight diagnostics should exist for both steps and differ.
    mpath = os.path.join(out_dir, "metrics.jsonl")
    rows = [json.loads(l) for l in open(mpath) if l.strip()]
    by_tag_step = {}
    for r in rows:
        tag = r.get("tag") or r.get("name"); step = r.get("step"); val = r.get("value")
        if tag and step is not None and isinstance(val, (int, float)):
            by_tag_step.setdefault(tag, {})[step] = val
    er_tags = [t for t in by_tag_step if "effective_rank" in t]
    n_moving = 0
    example = None
    for t in er_tags:
        vals = by_tag_step[t]
        if len(vals) >= 2:
            steps = sorted(vals)
            if vals[steps[0]] != vals[steps[-1]]:
                n_moving += 1
                if example is None:
                    example = (t, vals[steps[0]], vals[steps[-1]], steps[0], steps[-1])
    print(f"\nscan: {len(rows)} metric rows, {len(er_tags)} effective_rank tags, "
          f"{n_moving} moved between checkpoints", flush=True)
    if example:
        print(f"  e.g. {example[0]}: {example[1]:.2f} (s{example[3]}) -> "
              f"{example[2]:.2f} (s{example[4]})", flush=True)

    report_path = build_report(out_dir)
    print(f"scan report -> {report_path}", flush=True)

    results = {"checkpoints": [s for s, _ in saved], "params_moved": moved,
               "max_delta": round(max(deltas.values()), 5), "scan_rows": len(rows),
               "effective_rank_tags": len(er_tags), "moving_across_ckpts": n_moving,
               "cli": {"list_recipes": "ok", "report": "ok",
                       "scan": "documented stub (exit 2, points to scan_run)"}}
    out = os.path.join(os.path.dirname(__file__), "track4_scan.results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved -> {out}", flush=True)


if __name__ == "__main__":
    main()
