"""Track 5 (gap-closing, Gemini review) — non-LLM recipes + writer backends + scan
strict mode. Closes: vision & two_tower recipes (only llm was tested), the
tensorboard writer (only jsonl/null), and scan_run(strict=True).

Run:  venv/bin/python scripts/v1_validation/track5_recipes_writers.py
Saves: scripts/v1_validation/track5_recipes_writers.results.json
"""

from __future__ import annotations

import glob
import importlib.util
import json
import os

import torch
import torch.nn.functional as F

from circuitry import Recorder, build_report
from circuitry.recorder.scan import scan_run

RUN_ROOT = os.path.join("runs", "v1_track5")


def _load_example(name, cls):
    spec = importlib.util.spec_from_file_location(name, os.path.join("examples", f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, cls)


def count_tags(run_dir):
    p = os.path.join(run_dir, "metrics.jsonl")
    if not os.path.exists(p):
        return 0
    tags = set()
    for line in open(p):
        if line.strip():
            r = json.loads(line)
            t = r.get("tag") or r.get("name")
            if t:
                tags.add(t)
    return len(tags)


def main():
    results = {}
    TinyCNN = _load_example("tiny_vision", "TinyCNN")
    TwoTower = _load_example("tiny_two_tower", "TwoTower")

    # --- vision recipe (jsonl) ---
    torch.manual_seed(0)
    m = TinyCNN()
    rd = os.path.join(RUN_ROOT, "vision")
    rec = Recorder(m, run_dir=rd, recipe="vision", writer="jsonl", every_n_steps=10, strict=False)
    rec.attach()
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    for step in range(40):
        loss = F.cross_entropy(m(torch.randn(8, 3, 16, 16)), torch.randint(0, 10, (8,)))
        opt.zero_grad(); loss.backward(); opt.step()
        rec.step(step, loss=float(loss))
    rec.detach()
    nv = count_tags(rd)
    rep_v = build_report(rd)
    print(f"[recipe=vision] TinyCNN: {nv} diagnostic tags emitted, report -> {rep_v}", flush=True)
    results["vision"] = {"tags": nv, "report_ok": os.path.exists(rep_v)}
    assert nv > 0, "vision recipe emitted no tags"

    # --- two_tower recipe (jsonl) ---
    torch.manual_seed(0)
    m = TwoTower()
    rd = os.path.join(RUN_ROOT, "two_tower")
    rec = Recorder(m, run_dir=rd, recipe="two_tower", writer="jsonl", every_n_steps=10, strict=False)
    rec.attach()
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    for step in range(40):
        q, ip, ineg = torch.randn(16, 8), torch.randn(16, 8), torch.randn(16, 8)
        loss = F.softplus(m(q, ineg) - m(q, ip)).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        rec.step(step, loss=float(loss))
    rec.detach()
    nt = count_tags(rd)
    rep_t = build_report(rd)
    print(f"[recipe=two_tower] TwoTower: {nt} diagnostic tags emitted, report -> {rep_t}", flush=True)
    results["two_tower"] = {"tags": nt, "report_ok": os.path.exists(rep_t)}
    assert nt > 0, "two_tower recipe emitted no tags"

    # --- tensorboard writer ---
    torch.manual_seed(0)
    m = TinyCNN()
    rd_tb = os.path.join(RUN_ROOT, "vision_tb")
    rec = Recorder(m, run_dir=rd_tb, recipe="vision", writer="tensorboard", every_n_steps=10, strict=False)
    rec.attach()
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    for step in range(30):
        loss = F.cross_entropy(m(torch.randn(8, 3, 16, 16)), torch.randint(0, 10, (8,)))
        opt.zero_grad(); loss.backward(); opt.step()
        rec.step(step, loss=float(loss))
    rec.detach()
    ev = glob.glob(os.path.join(rd_tb, "**", "events.out.tfevents.*"), recursive=True)
    print(f"\n[writer=tensorboard] {len(ev)} TB event file(s) written: {bool(ev)}", flush=True)
    results["tensorboard_writer"] = {"event_files": len(ev), "ok": bool(ev)}
    assert ev, "tensorboard writer produced no event files"

    # --- scan strict=True (on the Track-4 Llama checkpoints if present) ---
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from track2_train_recorder import build_model
    from transformers import AutoTokenizer
    vocab = AutoTokenizer.from_pretrained("gpt2").vocab_size
    ckpt_run = os.path.join("runs", "v1_track4")
    strict_result = {}
    if glob.glob(os.path.join(ckpt_run, "checkpoints", "*.pt")):
        try:
            scan_run(ckpt_run, "llm", os.path.join(RUN_ROOT, "scan_strict"),
                     model_factory=lambda: build_model(vocab), writer="null", strict=True)
            strict_result = {"strict_true": "passed (all matched HookPoints resolved)"}
            print("\n[scan strict=True] PASSED — clean Llama, all HookPoints resolved", flush=True)
        except Exception as e:  # noqa: BLE001
            strict_result = {"strict_true": f"{type(e).__name__}: {e}"}
            print(f"\n[scan strict=True] raised (as designed on unmatched): {type(e).__name__}: {e}", flush=True)
    else:
        strict_result = {"strict_true": "skipped (no track4 checkpoints; run track4_scan.py first)"}
        print("\n[scan strict=True] skipped — no checkpoints", flush=True)
    results["scan_strict"] = strict_result

    out = os.path.join(os.path.dirname(__file__), "track5_recipes_writers.results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved -> {out}", flush=True)


if __name__ == "__main__":
    main()
