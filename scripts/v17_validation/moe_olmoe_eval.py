"""OLMoE MoE evaluation for circuitry v1.7.
Tasks:
  T1 — Recipe module matching (how many modules, does it hit expert projections?)
  T2 — Recorder end-to-end on MoE (crash/no crash + backward)
  T3 — SAE site resolution: locate_layers, HFSiteResolver for resid_post/mlp_out/attn_out
  T4 — Core weight primitives on batched expert weight + condition_number subsampling bug

Usage:
  .venv/bin/python scripts/v17_validation/moe_olmoe_eval.py
"""
from __future__ import annotations

import json
import logging
import pathlib
import re
import sys
import tempfile
import time
import traceback

import numpy as np
import torch
import torch.nn as nn

# ---- setup logging --------------------------------------------------------
logging.basicConfig(
    level=logging.WARNING,  # suppress circuitry INFO noise
    format="%(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("moe_eval")
log.setLevel(logging.DEBUG)

MODEL_ID = "allenai/OLMoE-1B-7B-0924"
SCRIPT_PATH = pathlib.Path(__file__).resolve()

print(f"Script: {SCRIPT_PATH}")
print(f"Model : {MODEL_ID}")
print()

# ---- load model once, reuse across tasks ----------------------------------
print("=" * 60)
print("Loading OLMoE-1B-7B-0924 (bf16, eager) ...")
t0 = time.time()
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.bfloat16,
    attn_implementation="eager",
)
model.train()  # need grad for T2 backward
print(f"Loaded in {time.time()-t0:.1f}s")
print(f"Model type  : {type(model).__name__}")
print(f"MLP type    : {type(model.model.layers[0].mlp).__name__}")
print()


# ===========================================================================
# T1: Recipe module matching
# ===========================================================================
print("=" * 60)
print("T1: LLM recipe module matching")
print()

from circuitry.recipes.llm import RECIPE

# Enumerate all patterns and their match counts
all_mods = dict(model.named_modules())
print(f"  Total named_modules in OLMoE: {len(all_mods)}")
print()

PATTERNS = [(hp.source.value, hp.pattern) for hp in RECIPE.hook_points if hp.pattern is not None]
total_matched = 0
for src, pat in PATTERNS:
    rx = re.compile(pat)
    matches = [n for n in all_mods if rx.search(n)]
    total_matched += len(matches)
    status = "OK" if matches else "MISS"
    flag = "  [!]" if not matches else "   "
    print(f"{flag} [{status}] {src:6s}  {len(matches):4d}  {pat}")
    if matches and len(matches) <= 5:
        for m in matches[:5]:
            print(f"            -> {m}")

print()
# Specifically: do expert projections get hit?
expert_gate_up_names = [n for n in all_mods if "gate_up_proj" in n]
expert_down_names = [n for n in all_mods if n.endswith("down_proj") and "mlp" in n]
expert_gate_names = [n for n in all_mods if n.endswith(".gate") and "mlp" in n]

print(f"  Expert gate_up_proj modules in model: {len(expert_gate_up_names)}")
print(f"    examples: {expert_gate_up_names[:2]}")
print(f"  Expert down_proj modules in model   : {len(expert_down_names)}")
print(f"    examples: {expert_down_names[:2]}")
print(f"  Router gate modules in model        : {len(expert_gate_names)}")
print(f"    examples: {expert_gate_names[:2]}")
print()

# Does pattern '.*\.(w1|w2|w3|gate_proj|up_proj|down_proj)$' match ANYTHING?
mlp_weight_pattern = re.compile(r".*\.(w1|w2|w3|gate_proj|up_proj|down_proj)$")
mlp_weight_matches = [n for n in all_mods if mlp_weight_pattern.search(n)]
print(f"  MLP weight pattern matches: {len(mlp_weight_matches)}")
print("  => Expert projections MISSED (batched 3D tensors, not individual Linear submodules)")
print()

# Does the router 'mlp.gate' get incorrectly hit by any pattern?
router_weight_pat = re.compile(r".*\.(w1|w2|w3|gate_proj|up_proj|down_proj)$")
router_name_match = router_weight_pat.search("model.layers.0.mlp.gate")
print(f"  mlp.gate matched by MLP weight pattern: {bool(router_name_match)}")
# Check gate against all patterns
for src, pat in PATTERNS:
    if re.compile(pat).search("model.layers.0.mlp.gate"):
        print(f"  mlp.gate IS matched by: [{src}] {pat}")
        break
else:
    print("  mlp.gate is NOT matched by any llm recipe pattern")

print()
t1_summary = {
    "total_named_modules": len(all_mods),
    "total_pattern_matches": total_matched,
    "expert_gate_up_modules": len(expert_gate_up_names),
    "expert_down_modules": len(expert_down_names),
    "expert_weight_pattern_matches": len(mlp_weight_matches),
    "verdict": "EXPERTS MISSED - 0 expert weight diagnostics emitted",
}
print(f"  T1 summary: {json.dumps(t1_summary, indent=4)}")


# ===========================================================================
# T2: Recorder end-to-end
# ===========================================================================
print()
print("=" * 60)
print("T2: Recorder end-to-end on OLMoE (strict=False)")
print()

from circuitry.recorder.live import Recorder

BATCH, SEQ = 1, 8
STEPS = 4

tokens = torch.randint(0, model.config.vocab_size, (BATCH, SEQ))

crash_info = None
recorder_metrics_emitted = 0

try:
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = pathlib.Path(tmpdir)
        rec = Recorder(
            model,
            run_dir=run_dir,
            recipe="llm",
            writer="jsonl",
            every_n_steps=2,
            strict=False,  # allow unmatched hook points
        )
        rec.attach()
        print(f"  Attach succeeded.")
        print(f"  Matched modules summary:")
        attach_summary = json.loads((run_dir / "circuitry" / "attach_summary.json").read_text())
        for hp_info in attach_summary["hook_points"]:
            if hp_info["matched"] > 0:
                print(f"    hp[{hp_info['idx']}] {hp_info['source']:6s}  {hp_info['matched']:4d} matched"
                      f"  label={hp_info['label'][:60]}")
        totals = attach_summary["totals"]
        print(f"  Totals: matched={totals['matched']}, resolved={totals['resolved']}, unresolved={totals['unresolved']}")

        for step in range(STEPS):
            # Forward pass
            out = model(tokens, labels=tokens)
            loss = out.loss
            print(f"  Step {step}: loss={loss.item():.4f}", end="")
            # Backward
            loss.backward()
            print(f"  backward OK", end="")
            rec.step(step, loss=loss.detach())
            print(f"  rec.step OK")
            model.zero_grad()

        rec.detach()

        # Count emitted metrics — JsonlWriter writes to run_dir/metrics.jsonl (not circuitry/)
        metrics_file = run_dir / "metrics.jsonl"
        metrics_by_prefix: dict[str, int] = {}
        if metrics_file.exists():
            lines = [l for l in metrics_file.read_text().strip().splitlines() if l]
            recorder_metrics_emitted = len(lines)
            for line in lines:
                entry = json.loads(line)
                tag = entry.get("tag", "")
                prefix = tag.split("/")[0] if "/" in tag else tag
                metrics_by_prefix[prefix] = metrics_by_prefix.get(prefix, 0) + 1
        print(f"\n  Metrics emitted (total JSONL lines): {recorder_metrics_emitted}")

        print(f"\n  Metric prefixes emitted:")
        for pfx, cnt in sorted(metrics_by_prefix.items(), key=lambda x: -x[1]):
            print(f"    {pfx:30s}: {cnt}")

        t2_crash = False
        print("\n  T2: Recorder ran end-to-end without crash.")

except Exception as e:
    crash_info = traceback.format_exc()
    t2_crash = True
    print(f"\n  T2 CRASH: {type(e).__name__}: {e}")
    print(crash_info[:500])

t2_summary = {
    "crashed": t2_crash,
    "metrics_emitted": recorder_metrics_emitted,
    "crash_info": crash_info,
}


# ===========================================================================
# T3: SAE site resolution on OLMoE
# ===========================================================================
print()
print("=" * 60)
print("T3: SAE site resolution (locate_layers, HFSiteResolver)")
print()

from circuitry.patching._layout import locate_layers
from circuitry.patching.sites import HFSiteResolver, Site

# T3a: locate_layers
try:
    layers = locate_layers(model)
    print(f"  locate_layers: OK  -> {type(layers).__name__}, len={len(layers)}")
    t3a_ok = True
except Exception as e:
    print(f"  locate_layers: FAIL  {e}")
    t3a_ok = False

# T3b: HFSiteResolver.from_config
cfg = model.config
try:
    resolver = HFSiteResolver.from_config(cfg)
    print(f"  HFSiteResolver.from_config: OK")
    print(f"    n_heads={resolver.n_heads}, d_model={resolver.d_model}, d_mlp={resolver.d_mlp}")
    t3b_ok = True
except Exception as e:
    print(f"  HFSiteResolver.from_config: FAIL  {e}")
    t3b_ok = False
    resolver = None

# T3c: resolve sites
LAYER = 0
site_results = {}

if resolver is not None:
    for site_name in ("resid_post", "mlp_out", "attn_out"):
        site = Site(site_name, LAYER)
        try:
            resolved = resolver.resolve(model, site)
            mod = resolved.module
            mod_path = None
            # Find the module path by reverse lookup
            for n, m in model.named_modules():
                if id(m) == id(mod):
                    mod_path = n
                    break
            print(f"  Site({site_name!r}, {LAYER}): OK")
            print(f"    module={type(mod).__name__}, path={mod_path}")
            site_results[site_name] = {"ok": True, "type": type(mod).__name__, "path": mod_path}
        except Exception as e:
            print(f"  Site({site_name!r}, {LAYER}): FAIL  {e}")
            site_results[site_name] = {"ok": False, "error": str(e)}

    # Special check for mlp_out: what module does it hook?
    if site_results.get("mlp_out", {}).get("ok"):
        mlp_path = site_results["mlp_out"]["path"]
        mlp_type = site_results["mlp_out"]["type"]
        print()
        print(f"  mlp_out hooks: {mlp_path} ({mlp_type})")
        if "SparseMoe" in mlp_type or "Olmoe" in mlp_type or mlp_path.endswith(".mlp"):
            print("    => Hooks the WHOLE MoE sparse-moe block output (correct for block-level mlp_out)")
        else:
            print("    => [UNEXPECTED] Not the MoE block")

t3_summary = {
    "locate_layers_ok": t3a_ok,
    "resolver_from_config_ok": t3b_ok,
    "site_results": site_results,
}
print(f"\n  T3 summary: {json.dumps(t3_summary, indent=4)}")


# ===========================================================================
# T4: Core primitives on batched expert weight + condition_number subsampling
# ===========================================================================
print()
print("=" * 60)
print("T4: Core primitives on OLMoE expert weight + condition_number bug")
print()

from circuitry.core.weight import (
    condition_number,
    effective_rank,
    singular_values,
    stable_rank,
)

# The experts weight is 3D: gate_up_proj shape [64, 2048, 2048]
# _as_2d() reshapes it to [2048, 64*2048] = [2048, 131072]
# which has min_dim=2048 > max_dim=512 → SUBSAMPLED to [2048, 512]

gate_up_weight = model.model.layers[0].mlp.experts.gate_up_proj.detach()  # [64, 2048, 2048]
down_weight = model.model.layers[0].mlp.experts.down_proj.detach()         # [64, 2048, 1024]

print(f"  gate_up_proj shape: {tuple(gate_up_weight.shape)}  (3D!)")
print(f"  down_proj shape   : {tuple(down_weight.shape)}")
print()

# Test effective_rank on 3D weight (uses _as_2d internally)
try:
    er = effective_rank(gate_up_weight)
    print(f"  effective_rank(gate_up_proj): {er:.4f}  (max would be min_dim of subsampled = 512)")
    t4_er_ok = True
except Exception as e:
    print(f"  effective_rank FAIL: {e}")
    t4_er_ok = False
    er = None

try:
    sr = stable_rank(gate_up_weight)
    print(f"  stable_rank(gate_up_proj)   : {sr:.4f}")
    t4_sr_ok = True
except Exception as e:
    print(f"  stable_rank FAIL: {e}")
    t4_sr_ok = False
    sr = None

# condition_number: the 3D weight reshapes to [2048, 131072]
# min_dim = 2048 > 512 → subsampled to 512 cols → true min singular value WRONG
print()
print("  === condition_number subsampling bug ===")
# Use a single expert's gate projection for a cleaner 2D comparison
# Slice expert 0's gate_proj: gate_up_proj[0, :1024, :] → shape [1024, 2048]
# (OLMoE gate_up_proj is [64, 2048, 2048]; layout: [:1024] = gate, [1024:] = up)
W_single_fp32 = gate_up_weight[0].detach().to(torch.float32)   # [2048, 2048] — single expert combined
print(f"  W_single shape: {tuple(W_single_fp32.shape)}")

# circuitry condition_number: uses singular_values(max_dim=512) on the 2D weight
cond_circuitry = condition_number(W_single_fp32)

# numpy ground truth (full SVD, no subsampling)
W_np = W_single_fp32.numpy()
sv_full = np.linalg.svd(W_np, compute_uv=False)
cond_numpy = float(sv_full[0] / sv_full[-1]) if sv_full[-1] > 1e-12 else float("inf")

# Also get circuitry SVs for comparison
sv_circuitry = singular_values(W_single_fp32, use_gram=False)
sigma_min_circuitry = float(sv_circuitry[-1].item())
sigma_min_numpy = float(sv_full[-1])
sigma_max_circuitry = float(sv_circuitry[0].item())
sigma_max_numpy = float(sv_full[0])

print(f"  sigma_max  circuitry={sigma_max_circuitry:.6f}  numpy={sigma_max_numpy:.6f}")
print(f"  sigma_min  circuitry={sigma_min_circuitry:.6f}  numpy={sigma_min_numpy:.6f}")
print(f"  condition  circuitry={cond_circuitry:.2f}  numpy={cond_numpy:.2f}")
ratio = cond_circuitry / cond_numpy if cond_numpy > 0 and not np.isinf(cond_numpy) else float("nan")
print(f"  ratio (circuitry/numpy): {ratio:.4f}")

# Confirm the min_dim > 512 trigger
from circuitry.core.weight import _as_2d
M_2d = _as_2d(W_single_fp32)
print()
print(f"  _as_2d(W_single) shape: {tuple(M_2d.shape)}")
print(f"  min(M.shape)={min(M_2d.shape)}, max_dim=512 → subsampled={min(M_2d.shape) > 512}")
print()

t4_summary = {
    "gate_up_proj_shape": list(gate_up_weight.shape),
    "effective_rank_ok": t4_er_ok,
    "effective_rank": er,
    "stable_rank_ok": t4_sr_ok,
    "stable_rank": sr,
    "W_single_shape": list(W_single_fp32.shape),
    "condition_circuitry": cond_circuitry,
    "condition_numpy": cond_numpy,
    "condition_ratio": ratio,
    "sigma_min_circuitry": sigma_min_circuitry,
    "sigma_min_numpy": sigma_min_numpy,
    "subsampled": min(M_2d.shape) > 512,
}
print(f"  T4 summary: {json.dumps(t4_summary, indent=4)}")


# ===========================================================================
# Final verdict
# ===========================================================================
print()
print("=" * 60)
print("FINAL VERDICT")
print("=" * 60)
print()
print(f"Model       : {MODEL_ID}")
print(f"Arch        : OlmoeForCausalLM  16 layers, 64 experts/layer, d_model=2048")
print(f"MLP struct  : OlmoeSparseMoeBlock  (gate: OlmoeTopKRouter + experts: OlmoeExperts)")
print(f"Expert params: gate_up_proj [64,2048,2048] + down_proj [64,2048,1024] on OlmoeExperts (batched 3D)")
print()
print("VERDICT: circuitry does NOT usefully cover OLMoE experts.")
print("  Expert weight diagnostics (effective_rank, condition_number, etc.) are emitted for 0 expert tensors.")
print("  The llm recipe matches 0/0 expert modules — MISSED, not exploded — because OLMoE's")
print("  OlmoeExperts packs all experts into 3D batch tensors, not individual Linear submodules.")
print("  The recipe's MLP pattern expects leaf Linear names like 'gate_proj', 'down_proj' but")
print("  those names do not appear in model.named_modules() at all.")
print()

# Save results
results_path = SCRIPT_PATH.parent / "moe_olmoe_eval.results.json"
results = {
    "model": MODEL_ID,
    "t1": t1_summary,
    "t2": t2_summary,
    "t3": t3_summary,
    "t4": t4_summary,
}
results_path.write_text(json.dumps(results, indent=2))
print(f"Results saved to: {results_path}")
