"""Part 1 — Core primitives validation on real GPT-2 activations/weights.

Run with:  .venv/bin/python scripts/v17_validation/part1_core_primitives.py
"""
from __future__ import annotations

import sys
import pathlib
import traceback

REPO = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "src"))

import numpy as np
import torch
import torch.nn.functional as F

# ── load GPT-2 on CPU ──────────────────────────────────────────────────────
print("=== Loading GPT-2 (CPU) ===")
from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
lm_model = GPT2LMHeadModel.from_pretrained("gpt2").eval()
# Force eager attention so we can capture attention weights
lm_model.config._attn_implementation = "eager"

block0 = lm_model.transformer.h[0]
print(f"  GPT-2 block0: {type(block0).__name__}")

# pull real weights
W_mlp      = block0.mlp.c_fc.weight.detach()          # (768, 3072) min_dim=768 > 512 → subsampled
W_attn     = block0.attn.c_attn.weight.detach()        # (768, 2304) min_dim=768 > 512
W_attn_proj = block0.attn.c_proj.weight.detach()       # (768, 768)  min_dim=768 > 512
W_small_ok  = W_mlp[:256, :256].clone()                # (256, 256)  min_dim=256 < 512 ✓

print(f"W_mlp shape: {tuple(W_mlp.shape)}, min_dim={min(W_mlp.shape)}")
print(f"W_attn shape: {tuple(W_attn.shape)}, min_dim={min(W_attn.shape)}")
print(f"W_attn_proj shape: {tuple(W_attn_proj.shape)}, min_dim={min(W_attn_proj.shape)}")
print(f"W_small_ok shape: {tuple(W_small_ok.shape)}, min_dim={min(W_small_ok.shape)}")

# ── get real activations via forward pass ──────────────────────────────────
text = "The quick brown fox jumps over the lazy dog"
tokens_in = tokenizer(text, return_tensors="pt")
seq_len = tokens_in["input_ids"].shape[1]
print(f"\nTokenized seq_len: {seq_len}")

hidden_states_by_layer: dict[int, torch.Tensor] = {}
attn_patterns_by_layer: dict[int, torch.Tensor] = {}

def make_block_hook(idx: int):
    def hook(module, input, output):
        t = output[0].detach() if isinstance(output, tuple) else output.detach()
        hidden_states_by_layer[idx] = t
    return hook

def make_attn_hook(idx: int):
    def hook(module, input, output):
        # output = (attn_out, attn_weights)
        if isinstance(output, tuple) and len(output) >= 2 and output[1] is not None:
            if hasattr(output[1], "shape"):
                attn_patterns_by_layer[idx] = output[1].detach()
    return hook

handles = []
for i, block in enumerate(lm_model.transformer.h):
    handles.append(block.register_forward_hook(make_block_hook(i)))
    handles.append(block.attn.register_forward_hook(make_attn_hook(i)))

with torch.no_grad():
    lm_out = lm_model(**tokens_in)

for h in handles:
    h.remove()

final_logits = lm_out.logits  # (1, seq, 50257)

print(f"Hidden states captured: {len(hidden_states_by_layer)} layers")
print(f"Attn patterns captured: {len(attn_patterns_by_layer)} layers")
print(f"hidden[0] shape: {tuple(hidden_states_by_layer[0].shape)}")
if attn_patterns_by_layer:
    print(f"attn[0] shape: {tuple(attn_patterns_by_layer[0].shape)}")

# ── helper ──────────────────────────────────────────────────────────────────
def run(label, fn, *args, **kwargs):
    try:
        result = fn(*args, **kwargs)
        if isinstance(result, (float, int)):
            ok = result == float("inf") or np.isfinite(result)
            flag = "OK" if ok else "WARN_NAN_INF"
            print(f"  {flag} {label} = {result:.6g}")
        elif isinstance(result, torch.Tensor):
            has_nan = result.isnan().any().item()
            has_inf = result.isinf().any().item()
            flag = "WARN_NAN_INF" if (has_nan or has_inf) else "OK"
            print(f"  {flag} {label} shape={tuple(result.shape)} "
                  f"mean={result.float().mean():.6g} min={result.float().min():.6g} max={result.float().max():.6g}")
        elif isinstance(result, list):
            arr = np.array(result, dtype=float)
            print(f"  OK {label} = list[{len(result)}] "
                  f"mean={arr.mean():.4g} min={arr.min():.4g} max={arr.max():.4g}")
        elif isinstance(result, dict):
            print(f"  OK {label} = dict[{len(result)}] sample={list(result.items())[:2]}")
        elif isinstance(result, tuple):
            shapes = [tuple(r.shape) if isinstance(r, torch.Tensor) else type(r).__name__ for r in result]
            print(f"  OK {label} = tuple[{len(result)}] shapes={shapes}")
        else:
            print(f"  OK {label} = {result}")
        return result
    except Exception as e:
        print(f"  FAIL {label}: {type(e).__name__}: {e}")
        traceback.print_exc()
        return None


# ─────────────────────────────────────────────────────────────────────────────
# WEIGHT PRIMITIVES
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== weight.py primitives ===")
from circuitry.core import weight as W

# singular_values
sv_default = run("singular_values(W_mlp, default/max_dim=512)", W.singular_values, W_mlp)
sv_full    = run("singular_values(W_mlp, max_dim=None)", W.singular_values, W_mlp, max_dim=None)
sv_small   = run("singular_values(W_small_ok)", W.singular_values, W_small_ok)

if sv_default is not None and sv_full is not None:
    print(f"    sigma_max: subsampled={sv_default[0]:.4f} full={sv_full[0]:.4f}")
    print(f"    sigma_min: subsampled={sv_default[-1]:.4f} full={sv_full[-1]:.4f}")
    print(f"    count: subsampled={len(sv_default)} full={len(sv_full)}")

run("effective_rank(W_mlp)", W.effective_rank, W_mlp)
run("effective_rank(W_small_ok)", W.effective_rank, W_small_ok)
run("stable_rank(W_mlp)", W.stable_rank, W_mlp)
run("stable_rank(W_small_ok)", W.stable_rank, W_small_ok)

# ── THE KEY TEST: condition_number subsampling bug ─────────────────────────
print("\n--- condition_number subsampling-bias test ---")
# condition_number(W_mlp): internally calls singular_values(W, use_gram=False)
# but max_dim=512 (default).  W_mlp has min_dim=768 > 512 → columns subsampled.
# Subsampled sigma_min is artificially large → condition_number too low.
cn_default = run("condition_number(W_mlp, default max_dim=512)", W.condition_number, W_mlp)

# numpy reference: full matrix, no subsampling
W_np = W_mlp.float().numpy()
cn_numpy = float(np.linalg.cond(W_np))
print(f"  REF numpy.linalg.cond(W_mlp, full matrix) = {cn_numpy:.6g}")

if cn_default is not None:
    ratio = cn_default / cn_numpy
    pct_diff = abs(ratio - 1.0) * 100
    print(f"  circuitry/numpy ratio = {ratio:.4f}  ({pct_diff:.1f}% off)")
    if pct_diff > 5:
        print(f"  ==> BUG CONFIRMED: condition_number is {pct_diff:.1f}% off from numpy reference")
        print(f"      (subsampled min_dim=512 < actual min_dim=768; sigma_min too large)")
    else:
        print(f"  ==> OK: within 5% tolerance")

# max_dim=None should match numpy closely (only float32 vs float64 precision differs)
cn_full = run("condition_number(W_mlp, max_dim=None)", W.condition_number, W_mlp, max_dim=None)
if cn_full is not None:
    ratio_full = cn_full / cn_numpy
    print(f"  max_dim=None ratio = {ratio_full:.4f} (should be ~1.0, float32 rounding only)")

# Small matrix — no subsampling (min_dim=256 < 512)
cn_small = run("condition_number(W_small_ok, min_dim<512 → no subsample)", W.condition_number, W_small_ok)
cn_small_np = float(np.linalg.cond(W_small_ok.float().numpy()))
print(f"  REF numpy(W_small_ok) = {cn_small_np:.6g}")
if cn_small is not None:
    ratio_s = cn_small / cn_small_np
    print(f"  small ratio = {ratio_s:.4f} (should be ~1.0)")

# Heavy-tail alpha
print()
run("heavy_tail_alpha(W_mlp)", W.heavy_tail_alpha, W_mlp)
run("heavy_tail_alpha(W_small_ok)", W.heavy_tail_alpha, W_small_ok)

# attention_head_rank — GPT-2: n_heads=12, head_dim=64
# c_attn weight (768, 2304): 2304 = Q+K+V, each (768, 768).
# For Q-slice: (768, 768) — axis=1 since head dimension is in output (dim 1)
W_q = W_attn[:, :768]  # (768, 768) Q-projection slice
run("attention_head_rank(W_q, n_heads=12, head_dim=64, axis=1)",
    W.attention_head_rank, W_q, 12, 64, 1)
run("attention_head_rank(W_attn_proj, axis=1)",
    W.attention_head_rank, W_attn_proj, 12, 64, 1)

# update_delta and direction_cosine
print()
sd1 = {k: v.detach().clone() for k, v in lm_model.state_dict().items() if v.ndim >= 2}
sd2 = {k: v + 1e-3 * torch.randn_like(v) for k, v in sd1.items()}
sd3 = {k: v + 1e-3 * torch.randn_like(v) for k, v in sd2.items()}

delta_d = run("update_delta(sd2, sd1)", W.update_delta, sd2, sd1)
dc_d    = run("direction_cosine(sd3, sd2, sd1)", W.direction_cosine, sd3, sd2, sd1)
if delta_d:
    vals = list(delta_d.values())
    print(f"    update_delta: n={len(vals)}, mean={np.mean(vals):.4g}, max={max(vals):.4g}")
if dc_d:
    vals = list(dc_d.values())
    print(f"    direction_cosine: n={len(vals)}, mean_cos={np.mean(vals):.4g}")


# ─────────────────────────────────────────────────────────────────────────────
# ACTIVATION PRIMITIVES
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== activation.py primitives ===")
from circuitry.core import activation as A

h0 = hidden_states_by_layer[0]   # (1, seq, 768)
h1 = hidden_states_by_layer[1]   # (1, seq, 768)
h_flat = h0.squeeze(0)            # (seq, 768)

run("norm_stats(h0)", A.norm_stats, h0)
run("dead_fraction(h0)", A.dead_fraction, h0)
relu_h0 = F.relu(h0)
df_relu = run("dead_fraction(relu(h0)) should be ~0.5", A.dead_fraction, relu_h0)
if df_relu is not None:
    print(f"    dead_fraction(relu(h0)) = {df_relu:.4f} (expect ~0.3–0.7 for typical hidden states)")

run("kurtosis(h0, dim=-1)", A.kurtosis, h0, dim=-1)
run("participation_ratio(h_flat[0])", A.participation_ratio, h_flat[0])
run("gate_stats(h0)", A.gate_stats, h0)
run("token_similarity(h0)", A.token_similarity, h0)

# repr_drift: h0 (layer 0) vs h1 (layer 1) — expect > 0
drift_lc  = run("repr_drift(h0,h1, linear_cka)", A.repr_drift, h0, h1, method="linear_cka")
drift_rbf = run("repr_drift(h0,h1, rbf_cka)",    A.repr_drift, h0, h1, method="rbf_cka")
drift_cos = run("repr_drift(h0,h1, cosine)",      A.repr_drift, h0, h1, method="cosine")

# self-drift must be 0
drift_self_lc  = run("repr_drift(h0, h0, linear_cka) =? 0", A.repr_drift, h0, h0, method="linear_cka")
drift_self_rbf = run("repr_drift(h0, h0, rbf_cka)    =? 0", A.repr_drift, h0, h0, method="rbf_cka")
if drift_self_lc is not None and drift_self_lc > 1e-9:
    print(f"  BUG: self-drift(linear_cka)={drift_self_lc:.2e} expected 0")
if drift_self_rbf is not None and drift_self_rbf > 1e-9:
    print(f"  BUG: self-drift(rbf_cka)={drift_self_rbf:.2e} expected 0")


# ─────────────────────────────────────────────────────────────────────────────
# GRADIENT PRIMITIVES
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== gradient.py primitives ===")
from circuitry.core import gradient as G

lm_train = GPT2LMHeadModel.from_pretrained("gpt2").train()
with torch.enable_grad():
    out_t = lm_train(**tokens_in)
    loss = out_t.logits.mean()
    loss.backward()

grads_dict: dict[str, torch.Tensor] = {}
grads_by_depth: list[torch.Tensor] = []
for name, param in lm_train.named_parameters():
    if param.grad is not None:
        grads_dict[name] = param.grad
        if "weight" in name and param.grad.ndim == 2:
            grads_by_depth.append(param.grad)

print(f"  {len(grads_dict)} gradient tensors, {len(grads_by_depth)} 2D weight grads")

per_module_norms = run("grad_norm_per_module", G.grad_norm_per_module, grads_dict)
if per_module_norms:
    vals = list(per_module_norms.values())
    print(f"    n={len(vals)}, mean_norm={np.mean(vals):.4g}, max_norm={max(vals):.4g}")

run("total_grad_norm", G.total_grad_norm, per_module_norms or {})
run("signal_propagation_depth (12 weight grads)", G.signal_propagation_depth, grads_by_depth[:12])


# ─────────────────────────────────────────────────────────────────────────────
# SPECTRAL PRIMITIVES
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== spectral.py primitives ===")
from circuitry.core import spectral as S

run("esd(W_mlp)", S.esd, W_mlp)
run("esd(W_small_ok)", S.esd, W_small_ok)

sd_list = [sd1, sd2, sd3]
rt = run("rank_trajectory(3 state_dicts)", S.rank_trajectory, sd_list)
if rt:
    first_key = next(iter(rt))
    traj = rt[first_key]
    print(f"    {len(rt)} params, first '{first_key}' trajectory={[f'{x:.2f}' for x in traj]}")


# ─────────────────────────────────────────────────────────────────────────────
# ATTENTION PRIMITIVES
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== attention.py primitives ===")
from circuitry.core import attention as ATT

if attn_patterns_by_layer:
    A0 = attn_patterns_by_layer[0]  # (1, 12, seq, seq)
    print(f"  Using real GPT-2 attn pattern shape: {tuple(A0.shape)}")
else:
    # Fallback: create realistic synthetic attention pattern
    seq = seq_len
    A0 = torch.softmax(torch.randn(1, 12, seq, seq) - 1e9 * (1 - torch.tril(torch.ones(seq, seq))), dim=-1)
    print(f"  Using synthetic causal attn pattern shape: {tuple(A0.shape)} [FALLBACK]")

run("attention_pattern_entropy", ATT.attention_pattern_entropy, A0)

# 3-D input (no batch dimension)
run("attention_pattern_entropy(3D no-batch)", ATT.attention_pattern_entropy, A0.squeeze(0))

seq = A0.shape[-1]
slr = max(1, seq // 4)
if seq >= 2 * slr:
    run(f"induction_score(seq_len_repeat={slr})", ATT.induction_score, A0, seq_len_repeat=slr)
else:
    print(f"  SKIP induction_score: seq={seq} too short (need ≥{2*slr})")


# ─────────────────────────────────────────────────────────────────────────────
# LENS PRIMITIVE
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== lens.py primitives ===")
from circuitry.core import lens as L

# lm_head.weight is (vocab=50257, d_model=768) — auto-detect transposes
unembed = lm_model.lm_head.weight  # (50257, 768)
residual_last  = hidden_states_by_layer[11]  # (1, seq, 768) — layer 11 output
residual_first = hidden_states_by_layer[0]   # (1, seq, 768) — layer 0 output

run("logit_lens_kl(last layer, unembed (vocab,d), auto-orient)",
    L.logit_lens_kl, residual_last, unembed, final_logits)
run("logit_lens_kl(last layer, unembed.T (d,vocab))",
    L.logit_lens_kl, residual_last, unembed.T, final_logits)

kl_last  = run("logit_lens_kl with ln_f (last layer)",
    L.logit_lens_kl, residual_last, unembed, final_logits,
    layer_norm=lm_model.transformer.ln_f)
kl_first = run("logit_lens_kl(first layer, should be higher)",
    L.logit_lens_kl, residual_first, unembed, final_logits)

if kl_last is not None and kl_first is not None:
    if kl_first > kl_last:
        print(f"    OK: KL decreases from layer 0 ({kl_first:.4f}) to layer 11 ({kl_last:.4f})")
    else:
        print(f"    WARN: KL didn't decrease: layer0={kl_first:.4f} layer11={kl_last:.4f}")


print("\n=== PART 1 COMPLETE ===")
