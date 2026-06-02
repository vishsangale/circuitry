"""Part 2 — Vision recipe validation with ResNet-18 on random image batches.

Run with:  .venv/bin/python scripts/v17_validation/part2_vision_recipe.py
"""
from __future__ import annotations

import sys
import pathlib
import tempfile
import json
import traceback

REPO = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "src"))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm

import circuitry
from circuitry import Recorder, build_report

print(f"circuitry version: {circuitry.__version__}")

# ── Build a real ResNet-18 (untrained is fine) ─────────────────────────────
print("\n=== Building ResNet-18 (untrained, CPU) ===")
model = tvm.resnet18(weights=None)
model.eval()

# inspect named modules matching vision recipe pattern
import re
VISION_WEIGHT_PAT = r"(conv\d+|fc\d+|patch_embed|blocks\.\d+\.(attn|mlp))(\.weight)?$"
VISION_OUT_PAT    = r"(conv\d+|fc\d+|blocks\.\d+\.(attn|mlp))$"

matched_weight = [n for n, _ in model.named_modules() if re.search(VISION_WEIGHT_PAT, n)]
matched_output = [n for n, _ in model.named_modules() if re.search(VISION_OUT_PAT, n)]
print(f"  Module names matching weight pattern: {matched_weight}")
print(f"  Module names matching output pattern: {matched_output}")

if not matched_weight:
    print("\n  NOTE: ResNet-18 uses 'layer1.0.conv1', 'layer2.0.conv1', 'fc' naming.")
    print("  The vision recipe pattern 'conv\\d+|fc\\d+' only matches top-level 'conv1', 'fc'.")
    print("  Let's check what actually matches vs resnet module names:")
    all_mods = [n for n, _ in model.named_modules()]
    top_conv = [n for n in all_mods if re.match(r"^conv\d+$", n)]
    top_fc   = [n for n in all_mods if re.match(r"^fc\d+$", n) or n == "fc"]
    print(f"  top-level conv modules: {top_conv}")
    print(f"  top-level fc modules:   {top_fc}")

# ── Use a small custom CNN like tiny_vision.py example ────────────────────
print("\n=== Using custom TinyCNN (matches vision recipe pattern) ===")

class TinyCNN(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 32 * 32, 64)
        self.fc2 = nn.Linear(64, n_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.relu(self.fc1(x.flatten(1)))
        return self.fc2(x)

model_tiny = TinyCNN()
model_tiny.train()

# Check what the vision recipe matches on TinyCNN
matched_w_tiny = [n for n, _ in model_tiny.named_modules() if re.search(VISION_WEIGHT_PAT, n)]
matched_o_tiny = [n for n, _ in model_tiny.named_modules() if re.search(VISION_OUT_PAT, n)]
print(f"  TinyCNN modules matching weight pattern: {matched_w_tiny}")
print(f"  TinyCNN modules matching output pattern:  {matched_o_tiny}")

# ── Run Recorder with vision recipe ───────────────────────────────────────
print("\n=== Running Recorder (vision recipe, jsonl writer, 30 steps) ===")
run_dir = pathlib.Path(tempfile.mkdtemp(prefix="circuitry_part2_"))
print(f"  run_dir: {run_dir}")

opt = torch.optim.SGD(model_tiny.parameters(), lr=1e-3)
rec = Recorder(model_tiny, run_dir=run_dir, recipe="vision", writer="jsonl", every_n_steps=5)
rec.attach()

steps_recorded = []
for step in range(30):
    x = torch.randn(8, 3, 64, 64)
    y = torch.randint(0, 10, (8,))
    loss = F.cross_entropy(model_tiny(x), y)
    opt.zero_grad()
    loss.backward()
    opt.step()
    rec.step(step, loss=float(loss.item()))
    if step % 5 == 0:
        steps_recorded.append(step)

rec.detach()
print(f"  Training complete. Steps with recordings: {steps_recorded}")

# ── Check what was written ─────────────────────────────────────────────────
print("\n=== Checking written metrics ===")
jsonl_files = sorted(run_dir.glob("**/*.jsonl"))
print(f"  JSONL files: {[str(f) for f in jsonl_files]}")

metric_keys_seen = set()
entries_per_step = {}
for jf in jsonl_files:
    with open(jf) as f:
        for line in f:
            entry = json.loads(line)
            step_key = entry.get("step", "?")
            metric_keys_seen.update(entry.keys())
            if step_key not in entries_per_step:
                entries_per_step[step_key] = 0
            entries_per_step[step_key] += 1

print(f"  Metric keys seen: {sorted(metric_keys_seen)}")
print(f"  Steps with data: {sorted(entries_per_step.keys())}")
print(f"  Entries per step sample: {dict(list(entries_per_step.items())[:5])}")

# Check for NaN values
nan_count = 0
total_count = 0
for jf in jsonl_files:
    with open(jf) as f:
        for line in f:
            entry = json.loads(line)
            for k, v in entry.items():
                if isinstance(v, float):
                    total_count += 1
                    import math
                    if math.isnan(v) or math.isinf(v):
                        nan_count += 1
                        print(f"  NaN/Inf: step={entry.get('step')}, key={k}, val={v}")

print(f"  Float values: {total_count} total, {nan_count} NaN/Inf")

# ── build_report ───────────────────────────────────────────────────────────
print("\n=== build_report ===")
try:
    report = build_report(run_dir)
    print(f"  Report type: {type(report).__name__}")
    if hasattr(report, "__dict__"):
        print(f"  Report attrs: {list(vars(report).keys())[:10]}")
    elif isinstance(report, dict):
        print(f"  Report keys: {list(report.keys())[:10]}")
    print(f"  OK: build_report returned successfully")
except Exception as e:
    print(f"  FAIL build_report: {type(e).__name__}: {e}")
    traceback.print_exc()

# ── Also run on ResNet-18 to check recipe matching on it ─────────────────
print("\n=== ResNet-18 module matching vs. vision recipe pattern ===")
model_rn18 = tvm.resnet18(weights=None).train()
run_dir_rn = pathlib.Path(tempfile.mkdtemp(prefix="circuitry_rn18_"))

# Count modules matching the pattern
matched_rn = [n for n, _ in model_rn18.named_modules() if re.search(VISION_WEIGHT_PAT, n)]
print(f"  ResNet-18 modules matched by vision recipe: {len(matched_rn)} — {matched_rn}")
total_modules = len(list(model_rn18.named_modules()))
print(f"  Total ResNet-18 modules: {total_modules}")
if len(matched_rn) == 0:
    print("  GAP: Vision recipe pattern misses all ResNet-18 conv/linear layers")
    print("  ResNet-18 layer names: conv1, layer1.0.conv1, layer1.0.conv2, ... layer4.1.conv2, fc")
    # Check fc
    has_fc = any(n == "fc" for n, _ in model_rn18.named_modules())
    print(f"  Has 'fc' module: {has_fc}")
    # Check if pattern matches 'fc' (no digit after) -- it requires fc\d+
    print(f"  Does pattern match 'fc'? {bool(re.search(VISION_WEIGHT_PAT, 'fc'))}")
    print(f"  Does pattern match 'conv1'? {bool(re.search(VISION_WEIGHT_PAT, 'conv1'))}")

print("\n=== PART 2 COMPLETE ===")
print(f"  Script written: /Users/vishsangale/workspace/circuitry/scripts/v17_validation/part2_vision_recipe.py")
