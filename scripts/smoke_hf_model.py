#!/usr/bin/env python3
"""End-to-end smoke test on a real HuggingFace decoder-LM.

Loads a small pretrained model, exercises circuitry's recipe matcher against
it, runs a handful of forward+backward passes with synthetic data through the
``Recorder`` lifecycle, and produces a markdown report.

This is a one-off integration test for surfacing gaps between circuitry's
stock recipes and real-world HF model naming. It deliberately does NOT
pre-patch the LLM recipe — the goal is to capture what the stock recipe sees
on an unmodified HF model.

``transformers`` is a script-only dependency; not in circuitry's pyproject.

Run from the repo root:
    venv/bin/python scripts/smoke_hf_model.py --model Qwen/Qwen2.5-0.5B
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
import traceback
from contextlib import contextmanager

import torch

from circuitry import Recipe, Recorder, build_report
from circuitry.recipes import get_recipe
from circuitry.recorder.hooks import HookPoint, match_modules


@contextmanager
def _section(title: str):
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}", flush=True)
    t0 = time.time()
    yield
    print(f"  (took {time.time() - t0:.2f}s)", flush=True)


def _load_model(name: str, dtype: torch.dtype):
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        sys.exit("transformers not installed. Run: venv/bin/pip install transformers")
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype)
    return model, tok


def _diagnose_recipe(model, recipe: Recipe) -> dict:
    """For every HookPoint in `recipe`, count matched modules against `model`.
    Returns a dict the script logs and pickles into the observations doc."""
    diag: dict[str, list[dict]] = {"hook_points": []}
    for idx, hp in enumerate(recipe.hook_points):
        label = (
            hp.pattern if hp.pattern is not None
            else "<modules>" if hp.modules is not None
            else "<selector>"
        )
        names = match_modules(model, hp)
        entry = {
            "idx": idx,
            "source": hp.source.value,
            "target": label,
            "matched_count": len(names),
            "matched_examples": names[:3],
        }
        diag["hook_points"].append(entry)
        marker = "OK" if names else "MISS"
        print(f"  [{marker}] hook_point[{idx}] source={hp.source.value:7s} "
              f"target={label!r:55s} matched={len(names)}")
        if names[:3]:
            for n in names[:3]:
                print(f"        - {n}")
            if len(names) > 3:
                print(f"        - ... ({len(names) - 3} more)")
    return diag


def _build_safe_recipe(stock: Recipe, model) -> Recipe:
    """Filter the stock recipe's HookPoints to only those that match at least
    one module on `model`. Returns a new Recipe with the survivors and the
    same diagnostic lists. Empty-result fallback: caller checks."""
    survivors: list[HookPoint] = []
    for hp in stock.hook_points:
        if match_modules(model, hp):
            survivors.append(hp)
    return Recipe(
        name=f"{stock.name}_filtered",
        hook_points=survivors,
        weight_diagnostics=stock.weight_diagnostics,
        activation_diagnostics=stock.activation_diagnostics,
        gradient_diagnostics=stock.gradient_diagnostics,
        custom=list(getattr(stock, "custom", [])),
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--run-dir", default="runs/hf_smoke")
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--seqlen", type=int, default=32)
    p.add_argument("--every-n-steps", type=int, default=2)
    p.add_argument("--dtype", default="float32",
                   choices=["float32", "bfloat16", "float16"])
    args = p.parse_args()

    run_dir = pathlib.Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    dtype = {"float32": torch.float32,
             "bfloat16": torch.bfloat16,
             "float16": torch.float16}[args.dtype]

    print(f"run_dir: {run_dir}")
    print(f"model:   {args.model}  dtype={args.dtype}")

    # ------------------------------------------------------------------ load
    with _section(f"Phase 1: load {args.model}"):
        model, tok = _load_model(args.model, dtype)
        n_params = sum(p.numel() for p in model.parameters())
        n_modules = sum(1 for _ in model.named_modules())
        print(f"  params:  {n_params / 1e6:.1f}M  ({n_params * dtype.itemsize / 1e9:.2f}GB)")
        print(f"  modules: {n_modules}")
        print(f"  vocab:   {model.config.vocab_size}")
        # Sample module names — helpful for understanding what regexes need to match
        all_names = [n for n, _ in model.named_modules()][:30]
        print("  first 30 named_modules:")
        for n in all_names:
            print(f"    {n}")

    # ----------------------------------------------------- stock recipe diag
    stock = get_recipe("llm")
    with _section("Phase 2: stock 'llm' recipe coverage on this model"):
        stock_diag = _diagnose_recipe(model, stock)
        miss_count = sum(1 for hp in stock_diag["hook_points"] if hp["matched_count"] == 0)
        print(f"\n  summary: {miss_count}/{len(stock_diag['hook_points'])} HookPoints matched 0 modules")

    # ------------------------------------------------------ try stock attach
    with _section("Phase 3: try Recorder(recipe='llm') — expected to fail if stock has misses"):
        stock_run = run_dir / "stock"
        stock_run.mkdir(parents=True, exist_ok=True)
        rec_stock = Recorder(model, run_dir=stock_run, recipe="llm",
                             writer="jsonl", every_n_steps=args.every_n_steps,
                             strict=False)
        stock_attach_result = {"ok": False, "error": None}
        try:
            rec_stock.attach()
            stock_attach_result["ok"] = True
            rec_stock.detach()
            print("  OK: stock recipe attached cleanly")
        except Exception as e:
            stock_attach_result["error"] = f"{type(e).__name__}: {e}"
            print(f"  FAIL: {stock_attach_result['error']}")

    # ----------------------------------------------------- filtered recipe run
    with _section("Phase 4: build filtered recipe + run training loop"):
        filtered = _build_safe_recipe(stock, model)
        print(f"  filtered recipe: {len(filtered.hook_points)}/{len(stock.hook_points)} "
              f"HookPoints survive")
        filt_run = run_dir / "filtered"
        filt_run.mkdir(parents=True, exist_ok=True)
        rec = Recorder(model, run_dir=filt_run, recipe=filtered,
                       writer="jsonl", every_n_steps=args.every_n_steps,
                       strict=False)
        rec.attach()
        print(f"  matched_modules.txt -> {filt_run / 'circuitry' / 'matched_modules.txt'}")

        torch.manual_seed(0)
        vocab = model.config.vocab_size
        input_ids = torch.randint(0, vocab, (1, args.seqlen))

        model.train()
        opt = torch.optim.SGD(model.parameters(), lr=0.0)
        losses = []
        for step in range(args.steps):
            out = model(input_ids=input_ids, labels=input_ids)
            loss = out.loss
            loss.backward()
            rec.step(step=step, loss=float(loss))
            opt.zero_grad()
            losses.append(float(loss))
            print(f"  step {step}: loss={loss.item():.4f}")
        rec.detach()

    # -------------------------------------------------------- generate report
    with _section("Phase 5: build_report"):
        report_path = build_report(filt_run)
        print(f"  report -> {report_path}")
        size = report_path.stat().st_size
        print(f"  size: {size} bytes")
        # First few lines for stdout context
        with report_path.open() as f:
            head = "".join(f.readlines()[:20])
        print(f"  first 20 lines:\n{head}")

    # ----------------------------------------------- dump structured findings
    findings = {
        "model": args.model,
        "params_M": round(n_params / 1e6, 2),
        "stock_recipe": {
            "hook_points": stock_diag["hook_points"],
            "attach_result": stock_attach_result,
        },
        "filtered_run": {
            "run_dir": str(filt_run),
            "hook_points_kept": len(filtered.hook_points),
            "hook_points_total": len(stock.hook_points),
            "report": str(report_path),
            "losses": losses,
        },
    }
    findings_path = run_dir / "findings.json"
    findings_path.write_text(json.dumps(findings, indent=2))
    print(f"\nstructured findings -> {findings_path}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
