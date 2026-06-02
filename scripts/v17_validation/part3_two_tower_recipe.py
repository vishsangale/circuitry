"""Part 3 — Two-tower recipe validation.

Run with:  .venv/bin/python scripts/v17_validation/part3_two_tower_recipe.py
"""
from __future__ import annotations

import sys
import pathlib
import tempfile
import json
import traceback
import re

REPO = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "src"))

import torch
import torch.nn as nn
import torch.nn.functional as F

import circuitry
from circuitry import Recorder, build_report

print(f"circuitry version: {circuitry.__version__}")

# ── Build two-tower model matching the recipe pattern ─────────────────────
print("\n=== Building TwoTower model ===")

class TwoTower(nn.Module):
    def __init__(self, d_in=16, d_out=32):
        super().__init__()
        self.query_tower = nn.Sequential(
            nn.Linear(d_in, 64), nn.ReLU(), nn.Linear(64, d_out),
        )
        self.item_tower = nn.Sequential(
            nn.Linear(d_in, 64), nn.ReLU(), nn.Linear(64, d_out),
        )

    def forward(self, q, i):
        return (self.query_tower(q) * self.item_tower(i)).sum(-1)

model = TwoTower()

# Check recipe module matching
TWO_TOWER_WEIGHT_PAT = r"(query_tower|item_tower|interaction).*"
TWO_TOWER_OUTPUT_PAT = r"(query_tower|item_tower)$"

matched_w = [n for n, _ in model.named_modules() if re.search(TWO_TOWER_WEIGHT_PAT, n)]
matched_o = [n for n, _ in model.named_modules() if re.search(TWO_TOWER_OUTPUT_PAT, n)]

print(f"  All modules: {[n for n, _ in model.named_modules() if n]}")
print(f"  Matched by weight pattern: {matched_w}")
print(f"  Matched by output pattern: {matched_o}")

# ── Run Recorder with two_tower recipe ────────────────────────────────────
print("\n=== Running Recorder (two_tower recipe, 30 steps) ===")
run_dir = pathlib.Path(tempfile.mkdtemp(prefix="circuitry_part3_"))
print(f"  run_dir: {run_dir}")

model.train()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
rec = Recorder(model, run_dir=run_dir, recipe="two_tower", writer="jsonl", every_n_steps=5)
rec.attach()

for step in range(30):
    q = torch.randn(16, 16)
    i_pos = torch.randn(16, 16)
    i_neg = torch.randn(16, 16)
    score_pos = model(q, i_pos)
    score_neg = model(q, i_neg)
    loss = F.softplus(score_neg - score_pos).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()
    rec.step(step, loss=float(loss.item()))

rec.detach()
print("  Training complete.")

# ── Check written metrics ──────────────────────────────────────────────────
print("\n=== Checking written metrics ===")
with open(run_dir / "metrics.jsonl") as f:
    entries = [json.loads(line) for line in f]

print(f"  Total JSONL entries: {len(entries)}")

tags_by_category = {}
for e in entries:
    tag = e.get("tag", "")
    if "/" in str(tag):
        cat = tag.split("/")[0]
        tags_by_category.setdefault(cat, set()).add(tag)
    elif tag:
        tags_by_category.setdefault("other", set()).add(tag)

for cat, tags in sorted(tags_by_category.items()):
    print(f"  [{cat}] {sorted(tags)}")

# Check for custom embedding_alignment metric
align_entries = [e for e in entries if "embedding_alignment" in str(e.get("tag", ""))]
print(f"\n  embedding_alignment entries: {len(align_entries)}")
if align_entries:
    vals = [e["value"] for e in align_entries]
    import numpy as np
    print(f"    values: mean={np.mean(vals):.4f}, min={min(vals):.4f}, max={max(vals):.4f}")
else:
    print("  WARN: No embedding_alignment entries found (custom diagnostic not emitting)")

# Check for NaN/Inf
import math
nans = [e for e in entries if isinstance(e.get("value"), float) and
        (math.isnan(e["value"]) or math.isinf(e["value"]))]
print(f"\n  NaN/Inf entries: {len(nans)}")
if nans:
    for n in nans[:5]:
        print(f"    {n}")

# Check per-step coverage
steps_with_data = sorted(set(e.get("step") for e in entries if isinstance(e.get("step"), int)))
print(f"  Steps with data: {steps_with_data}")

# Check that output-hook metrics appear for both towers
tower_output_tags = [e["tag"] for e in entries if "dead_fraction" in str(e.get("tag", "")) or
                     "participation_ratio" in str(e.get("tag", ""))]
query_tower_tags = [t for t in tower_output_tags if "query_tower" in str(t)]
item_tower_tags  = [t for t in tower_output_tags if "item_tower" in str(t)]
print(f"\n  query_tower activation tags: {sorted(set(query_tower_tags))}")
print(f"  item_tower activation tags:  {sorted(set(item_tower_tags))}")

# ── Verify recipe pattern captures weight metrics for sub-modules ─────────
print("\n=== Weight metric coverage ===")
weight_tags = [e["tag"] for e in entries if "effective_rank" in str(e.get("tag", "")) or
               "stable_rank" in str(e.get("tag", ""))]
print(f"  Weight metric tags (unique): {sorted(set(weight_tags))}")

# The pattern r"(query_tower|item_tower|interaction).*" for weight hooks
# should capture all sub-modules: query_tower.0 (Linear), query_tower.2 (Linear), etc.
# Let's check if sub-linear layers within the tower are captured
sub_linear_w_tags = [t for t in weight_tags if re.search(r"\.\d", str(t))]
print(f"  Sub-module weight tags: {sorted(set(sub_linear_w_tags))[:10]}")

# ── build_report ──────────────────────────────────────────────────────────
print("\n=== build_report ===")
try:
    report = build_report(run_dir)
    print(f"  build_report returned: {type(report).__name__} — {report}")
    print("  OK: build_report succeeded")
except Exception as e:
    print(f"  FAIL: {type(e).__name__}: {e}")
    traceback.print_exc()

print("\n=== PART 3 COMPLETE ===")
print(f"  Script: /Users/vishsangale/workspace/circuitry/scripts/v17_validation/part3_two_tower_recipe.py")
