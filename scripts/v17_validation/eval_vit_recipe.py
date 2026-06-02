"""Evaluate the vision recipe against ViT-B/16 (torchvision).

Run with: .venv/bin/python scripts/v17_validation/eval_vit_recipe.py

Investigates:
  1. Which ViT modules the current vision recipe pattern matches (capture %).
  2. Whether weight/spectral primitives handle large 2-D Linear weights (768×3072).
  3. Whether the condition_number max_dim=512 subsampling bites (768>512 for some dims).
  4. Whether anything breaks on the patch-embedding conv or class token.
  5. Whether the Recorder emits or silently misses ViT modules.
"""
from __future__ import annotations

import json
import math
import pathlib
import re
import sys
import tempfile
import traceback

REPO = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "src"))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm

import circuitry
from circuitry import Recorder, build_report
from circuitry.core.weight import singular_values, effective_rank, stable_rank, condition_number

print(f"circuitry {circuitry.__version__}  |  torch {torch.__version__}")

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Module-name audit: what does the vision recipe match?
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("PART 1 — Module-name audit")
print("=" * 70)

model = tvm.vit_b_16(weights=None)
model.train()

# Current vision recipe patterns (from src/circuitry/recipes/vision.py)
WEIGHT_PAT = r"(conv\d+|fc\d+|patch_embed|blocks\.\d+\.(attn|mlp))(\.weight)?$"
OUTPUT_PAT = r"(conv\d+|fc\d+|blocks\.\d+\.(attn|mlp))$"

all_module_names = [n for n, _ in model.named_modules() if n]
total_modules = len(all_module_names)

matched_weight = [n for n in all_module_names if re.search(WEIGHT_PAT, n)]
matched_output = [n for n in all_module_names if re.search(OUTPUT_PAT, n)]
capture_pct = 100.0 * len(matched_weight) / total_modules

print(f"Total named modules (non-root): {total_modules}")
print(f"Matched by WEIGHT pattern: {len(matched_weight)}  → {capture_pct:.1f}% capture")
print(f"  hits: {matched_weight}")
print(f"Matched by OUTPUT pattern: {len(matched_output)}")
print(f"  hits: {matched_output}")

# Enumerate the important ViT layers that ARE NOT captured
important_categories = {
    "patch_embed (conv_proj)": [n for n in all_module_names if n == "conv_proj"],
    "self_attention (MultiheadAttention)": [n for n in all_module_names if "self_attention" in n and "out_proj" not in n],
    "self_attention.out_proj (Linear)": [n for n in all_module_names if n.endswith("self_attention.out_proj")],
    "mlp blocks (Sequential)": [n for n in all_module_names if re.fullmatch(r"encoder\.layers\.encoder_layer_\d+\.mlp", n)],
    "mlp Linear (mlp.0, mlp.3)": [n for n in all_module_names if re.search(r"mlp\.[03]$", n)],
    "layer norm (ln_1, ln_2, encoder.ln)": [n for n in all_module_names if "ln" in n],
    "heads.head (classifier Linear)": [n for n in all_module_names if n == "heads.head"],
}
print("\nCategories of ViT modules NOT captured by current pattern:")
for cat, mods in important_categories.items():
    captured = sum(1 for m in mods if re.search(WEIGHT_PAT, m))
    print(f"  {cat}: {len(mods)} modules, {captured} captured by recipe")

# ─────────────────────────────────────────────────────────────────────────────
# 2.  Spectral / weight primitives on large 2-D Linear weights
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("PART 2 — Spectral primitives on ViT weight shapes")
print("=" * 70)

# Key weight shapes in ViT-B/16:
#   mlp.0: (3072, 768)  — 3072 > 512, triggers subsampling on the column/row dim
#   mlp.3: (768, 3072)  — same, other orientation
#   out_proj: (768, 768) — square, fine
#   in_proj_weight: (2304, 768) — fused QKV, 2304 > 512
#   conv_proj weight after reshape: (768, 3*16*16=768) — square after _as_2d

test_shapes = {
    "mlp.0 (3072×768)": (3072, 768),
    "mlp.3 (768×3072)": (768, 3072),
    "out_proj (768×768)": (768, 768),
    "in_proj_weight (2304×768)": (2304, 768),
    "heads.head (1000×768)": (1000, 768),
    "conv_proj reshaped (768×768)": (768, 768),   # 768 out_ch × (3*16*16=768)
}

torch.manual_seed(42)
for label, shape in test_shapes.items():
    W = torch.randn(*shape)
    # Check subsampling: max_dim=512, singular_values clips the LARGER dim
    min_dim, max_dim_shape = min(shape), max(shape)
    will_subsample = max_dim_shape > 512  # default max_dim=512
    try:
        sv = singular_values(W)  # default max_dim=512
        er = effective_rank(W)
        sr = stable_rank(W)
        # condition_number uses use_gram=False; max_dim=512 → subsampling bites if min_dim>512
        cn_subsampled = condition_number(W)  # uses singular_values(W, use_gram=False) with max_dim=512
        print(f"  {label}")
        print(f"    subsample_triggered={will_subsample}, sv.shape={tuple(sv.shape)}, "
              f"eff_rank={er:.2f}, stable_rank={sr:.2f}, cond_num={cn_subsampled:.2f}")
        # Also run without subsampling to compare condition_number
        sv_full = singular_values(W, max_dim=None)
        cn_full_val = float(sv_full[0] / sv_full[-1]) if sv_full[-1] > 1e-12 else float("inf")
        if will_subsample:
            print(f"    condition_number(full, no subsample)={cn_full_val:.2f}  ← REAL value vs subsampled {cn_subsampled:.2f}")
    except Exception as e:
        print(f"  {label}: EXCEPTION {type(e).__name__}: {e}")
        traceback.print_exc()

# Specifically check: is min(3072, 768)=768 > 512? YES → subsampling on min_dim axis
print(f"\n  NOTE: singular_values subsamples the LARGER axis when max_dim=512.")
print(f"  For mlp.0 (3072×768): larger axis=3072 → subsampled to 512 rows.")
print(f"  condition_number uses subsampled SVD → σ_min is from 512-sample, NOT full matrix.")
print(f"  This means condition_number is ESTIMATED, not exact, for any ViT Linear weight.")
print(f"  Concretely: min(3072,768)=768 > 512? {768 > 512} — but axis subsampled is ROWS not cols.")
print(f"  Key: for (3072,768), max_dim clips the larger dim (3072), so we get (512,768) matrix.")
print(f"  σ_min of (512,768) != σ_min of (3072,768). condition_number is biased low.")

# ─────────────────────────────────────────────────────────────────────────────
# 3.  Recorder + vision recipe live run on ViT-B/16
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("PART 3 — Recorder + vision recipe live run on ViT-B/16 (15 steps)")
print("=" * 70)

run_dir = pathlib.Path(tempfile.mkdtemp(prefix="circuitry_vit_eval_"))
print(f"run_dir: {run_dir}")

model = tvm.vit_b_16(weights=None)
model.train()

opt = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

rec = Recorder(
    model,
    run_dir=run_dir,
    recipe="vision",
    writer="jsonl",
    every_n_steps=5,
    strict=False,   # vision recipe matches 0 ViT modules → attach raises without this
)
rec.attach()

losses = []
for step in range(15):
    x = torch.randn(2, 3, 224, 224)   # small batch to keep CPU time sane
    y = torch.randint(0, 1000, (2,))
    logits = model(x)
    loss = F.cross_entropy(logits, y)
    opt.zero_grad()
    loss.backward()
    opt.step()
    rec.step(step, loss=float(loss.item()))
    losses.append(float(loss.item()))
    if step % 5 == 0:
        print(f"  step {step:3d}  loss={loss.item():.4f}")

rec.detach()
print(f"  Final loss: {losses[-1]:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 4.  Inspect what was emitted
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("PART 4 — Emitted metrics analysis")
print("=" * 70)

metrics_path = run_dir / "metrics.jsonl"
if not metrics_path.exists():
    print("  WARN: metrics.jsonl not found — recipe emitted NOTHING")
    entries = []
else:
    entries = [json.loads(l) for l in metrics_path.read_text().splitlines() if l.strip()]

print(f"Total JSONL entries: {len(entries)}")

if entries:
    # Unique tags
    tags = sorted(set(e.get("tag", "") for e in entries))
    print(f"Unique tags ({len(tags)}): {tags[:30]}")
    if len(tags) > 30:
        print(f"  ... {len(tags)-30} more")

    # Steps covered
    steps = sorted(set(e.get("step") for e in entries if isinstance(e.get("step"), int)))
    print(f"Steps with data: {steps}")

    # Check for NaN/Inf
    bad = [(e.get("tag"), e.get("step"), e.get("value"))
           for e in entries
           if isinstance(e.get("value"), float) and (math.isnan(e["value"]) or math.isinf(e["value"]))]
    print(f"NaN/Inf entries: {len(bad)}")
    for tag, step, val in bad[:10]:
        print(f"  step={step} tag={tag} val={val}")

    # Check what module names appear in tags
    module_tags = [t for t in tags if "/" in t]
    unique_modules = sorted(set(t.split("/")[0] for t in module_tags))
    print(f"\nModules mentioned in tags ({len(unique_modules)}): {unique_modules}")
else:
    print("  No entries emitted — recipe produced 0 metrics for ViT.")

# ─────────────────────────────────────────────────────────────────────────────
# 5.  build_report
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("PART 5 — build_report")
print("=" * 70)

try:
    report = build_report(run_dir)
    print(f"  build_report OK: {type(report).__name__}")
except Exception as e:
    print(f"  build_report FAILED: {type(e).__name__}: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# 6.  Summary
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"  Total ViT-B/16 modules: {total_modules}")
print(f"  Matched by vision recipe WEIGHT pattern: {len(matched_weight)} ({capture_pct:.1f}%)")
print(f"  Matched by vision recipe OUTPUT pattern: {len(matched_output)}")
print(f"  Emitted metric entries: {len(entries)}")
print(f"  Coverage verdict: {'ZERO coverage (0/151 modules)' if len(matched_weight) == 0 else f'{len(matched_weight)} modules'}")

print(f"\nScript: {pathlib.Path(__file__).resolve()}")
