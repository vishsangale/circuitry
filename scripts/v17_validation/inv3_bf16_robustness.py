"""Investigation 3: Lower-precision (bf16) robustness.

Reloads model in bf16, casts SAEs to bf16.
Runs node attribution (SAEFeatureRunner) and edges (SAEFeatureEdgeRunner).

Checks:
  1. Does splice losslessness hold at bf16 tolerance (~1e-2)?
  2. Are scores finite and sign-consistent with fp32?
  3. Does anything raise / NaN / silently degrade?
  4. Reports fp32-vs-bf16 comparison for top features/edges.
"""

import warnings
warnings.filterwarnings("ignore")

import math

import torch
from sae_lens import SAE
from transformer_lens import HookedTransformer

from circuitry.patching.sae_edges import SAEFeatureEdgeRunner
from circuitry.patching.sae_features import SAEFeatureRunner
from circuitry.patching.sites import Site, TLSiteResolver
from circuitry.sae.grad import sae_decompose

print("=" * 70)
print("INVESTIGATION 3 — bf16 robustness")
print("=" * 70)


def load_sae(rel, sid, device="cpu"):
    r = SAE.from_pretrained(rel, sid, device=device)
    return r[0] if isinstance(r, tuple) else r


# ---------------------------------------------------------------------------
# 1. fp32 baseline
# ---------------------------------------------------------------------------
print("\n[1] Loading fp32 model and SAEs (baseline) ...")
model_fp32 = HookedTransformer.from_pretrained("gpt2", device="cpu")
model_fp32.eval()

sae_r6_fp32 = load_sae("gpt2-small-res-jb", "blocks.7.hook_resid_pre")
sae_r7_fp32 = load_sae("gpt2-small-res-jb", "blocks.8.hook_resid_pre")

resolver_fp32 = TLSiteResolver()
clean_fp32 = model_fp32.to_tokens("When John and Mary went to the store, John gave a drink to")
corrupt_fp32 = model_fp32.to_tokens("When John and Mary went to the store, Mary gave a drink to")
mary_fp32 = model_fp32.to_single_token(" Mary")
john_fp32 = model_fp32.to_single_token(" John")
metric_fp32 = lambda logits: logits[0, -1, mary_fp32] - logits[0, -1, john_fp32]

runner_fp32 = SAEFeatureEdgeRunner(
    model=model_fp32,
    sae_sites={Site("resid_post", 6): sae_r6_fp32, Site("resid_post", 7): sae_r7_fp32},
    resolver=resolver_fp32,
)
node_runner_fp32 = SAEFeatureRunner(
    model=model_fp32,
    sae_sites={Site("resid_post", 6): sae_r6_fp32, Site("resid_post", 7): sae_r7_fp32},
    resolver=resolver_fp32,
)

TOP_K = 16
print("  Running fp32 node attribution ...")
nodes_fp32 = node_runner_fp32.run(clean_fp32, corrupt_fp32, metric_fp32, max_features=TOP_K)
print("  Running fp32 edge attribution ...")
circ_fp32 = runner_fp32.run(clean_fp32, corrupt_fp32, metric_fp32, layer_pairs="adjacent", top_k_survivors=TOP_K)

print(f"  fp32 nodes: {len(nodes_fp32.scores)}, fp32 edges: {len(circ_fp32.edges)}")

# ---------------------------------------------------------------------------
# 2. bf16 run
# ---------------------------------------------------------------------------
print("\n[2] Loading bf16 model and SAEs ...")
model_bf16 = HookedTransformer.from_pretrained("gpt2", device="cpu", dtype=torch.bfloat16)
model_bf16.eval()

sae_r6_bf16 = load_sae("gpt2-small-res-jb", "blocks.7.hook_resid_pre")
sae_r6_bf16 = sae_r6_bf16.to(torch.bfloat16)

sae_r7_bf16 = load_sae("gpt2-small-res-jb", "blocks.8.hook_resid_pre")
sae_r7_bf16 = sae_r7_bf16.to(torch.bfloat16)

resolver_bf16 = TLSiteResolver()
clean_bf16 = model_bf16.to_tokens("When John and Mary went to the store, John gave a drink to")
corrupt_bf16 = model_bf16.to_tokens("When John and Mary went to the store, Mary gave a drink to")
mary_bf16 = model_bf16.to_single_token(" Mary")
john_bf16 = model_bf16.to_single_token(" John")
metric_bf16 = lambda logits: logits[0, -1, mary_bf16] - logits[0, -1, john_bf16]

bf16_raised = False
bf16_nan = False
error_msg = ""

try:
    runner_bf16 = SAEFeatureEdgeRunner(
        model=model_bf16,
        sae_sites={Site("resid_post", 6): sae_r6_bf16, Site("resid_post", 7): sae_r7_bf16},
        resolver=resolver_bf16,
    )
    node_runner_bf16 = SAEFeatureRunner(
        model=model_bf16,
        sae_sites={Site("resid_post", 6): sae_r6_bf16, Site("resid_post", 7): sae_r7_bf16},
        resolver=resolver_bf16,
    )

    print("  Running bf16 node attribution ...")
    nodes_bf16 = node_runner_bf16.run(clean_bf16, corrupt_bf16, metric_bf16, max_features=TOP_K)
    print("  Running bf16 edge attribution ...")
    circ_bf16 = runner_bf16.run(clean_bf16, corrupt_bf16, metric_bf16, layer_pairs="adjacent", top_k_survivors=TOP_K)

    print(f"  bf16 nodes: {len(nodes_bf16.scores)}, bf16 edges: {len(circ_bf16.edges)}")

except Exception as e:
    bf16_raised = True
    error_msg = str(e)
    print(f"  EXCEPTION raised in bf16 run: {e}")

# ---------------------------------------------------------------------------
# 3. Splice losslessness at bf16
# ---------------------------------------------------------------------------
print("\n[3] Checking splice losslessness (bf16) ...")


def check_splice_losslessness(model, sae, site, resolver, inputs, tol=5e-2):
    """
    Inject SAE-reconstructed activations and compare reconstructed logits
    to bypass logits, checking relative agreement under spliced forward.
    Returns max absolute error in logit space across vocab.
    """
    from circuitry.patching.sae_features import _routed_extract, _routed_inject

    # Get clean logits without any splice
    with torch.no_grad():
        if isinstance(inputs, dict):
            out_clean = model(**inputs)
        else:
            out_clean = model(inputs)
    logits_clean = (out_clean.logits if hasattr(out_clean, "logits") else out_clean)[0, -1]

    # Spliced forward
    store: dict = {}
    resolved = resolver.resolve(model, site)
    layer_mod = resolved.module

    def _hook(module, inp, output, _sae=sae, _res=resolved, _st=store):
        a = _routed_extract(_res, output)
        a_in = a.detach().to(getattr(_sae, "device", a.device), getattr(_sae, "dtype", a.dtype))
        with torch.no_grad():
            f, x_hat, eps = sae_decompose(_sae, a_in)
            recon = x_hat + eps
        # Cast back to model dtype/device
        params = list(layer_mod.parameters())
        if params:
            m_dtype, m_dev = params[0].dtype, params[0].device
        else:
            m_dtype, m_dev = torch.bfloat16, torch.device("cpu")
        recon_cast = recon.to(m_dev, m_dtype)
        _st["splice_error"] = (a_in - recon).abs().max().item()
        return _routed_inject(_res, output, recon_cast)

    h = layer_mod.register_forward_hook(_hook)
    try:
        with torch.no_grad():
            if isinstance(inputs, dict):
                out_spliced = model(**inputs)
            else:
                out_spliced = model(inputs)
    finally:
        h.remove()

    logits_spliced = (out_spliced.logits if hasattr(out_spliced, "logits") else out_spliced)[0, -1]

    # Compare in fp32 for fair comparison
    lc_fp32 = logits_clean.float()
    ls_fp32 = logits_spliced.float()
    max_logit_err = (lc_fp32 - ls_fp32).abs().max().item()
    sae_recon_err = store.get("splice_error", float("nan"))

    return max_logit_err, sae_recon_err


# fp32 losslessness
err_fp32_r6, rec_fp32_r6 = check_splice_losslessness(
    model_fp32, sae_r6_fp32, Site("resid_post", 6), resolver_fp32, clean_fp32
)
err_fp32_r7, rec_fp32_r7 = check_splice_losslessness(
    model_fp32, sae_r7_fp32, Site("resid_post", 7), resolver_fp32, clean_fp32
)
print(f"  fp32 splice: site6 logit_err={err_fp32_r6:.2e}, recon_err={rec_fp32_r6:.2e}")
print(f"  fp32 splice: site7 logit_err={err_fp32_r7:.2e}, recon_err={rec_fp32_r7:.2e}")

# bf16 losslessness
if not bf16_raised:
    err_bf16_r6, rec_bf16_r6 = check_splice_losslessness(
        model_bf16, sae_r6_bf16, Site("resid_post", 6), resolver_bf16, clean_bf16
    )
    err_bf16_r7, rec_bf16_r7 = check_splice_losslessness(
        model_bf16, sae_r7_bf16, Site("resid_post", 7), resolver_bf16, clean_bf16
    )
    print(f"  bf16 splice: site6 logit_err={err_bf16_r6:.2e}, recon_err={rec_bf16_r6:.2e}")
    print(f"  bf16 splice: site7 logit_err={err_bf16_r7:.2e}, recon_err={rec_bf16_r7:.2e}")

    splice_ok_bf16 = err_bf16_r6 < 5e-2 and err_bf16_r7 < 5e-2
else:
    splice_ok_bf16 = False
    err_bf16_r6 = err_bf16_r7 = float("nan")

# ---------------------------------------------------------------------------
# 4. NaN/finite check for bf16 scores
# ---------------------------------------------------------------------------
if not bf16_raised:
    print("\n[4] Checking for NaN/inf in bf16 scores ...")
    node_scores_bf16 = [float(s) for s in nodes_bf16.scores.values()]
    edge_scores_bf16 = list(circ_bf16.edges.values())

    node_nan = sum(1 for s in node_scores_bf16 if not math.isfinite(s))
    edge_nan = sum(1 for s in edge_scores_bf16 if not math.isfinite(s))

    print(f"  bf16 node scores: {len(node_scores_bf16)} total, {node_nan} NaN/inf")
    print(f"  bf16 edge scores: {len(edge_scores_bf16)} total, {edge_nan} NaN/inf")

    bf16_nan = node_nan > 0 or edge_nan > 0

# ---------------------------------------------------------------------------
# 5. Sign-consistency comparison: fp32 vs bf16 top nodes
# ---------------------------------------------------------------------------
print("\n[5] Sign-consistency: fp32 vs bf16 top nodes and edges ...")

if not bf16_raised:
    # Node scores: build map (layer, neuron) → score
    def node_score_map(result):
        return {
            (n.node.layer, n.node.neuron, n.node.component or "resid_post"): float(score)
            for n, score in result.scores.items()
            if n.node.neuron is not None
        }

    fp32_nodes = node_score_map(nodes_fp32)
    bf16_nodes = node_score_map(nodes_bf16)

    common_nodes = set(fp32_nodes) & set(bf16_nodes)
    print(f"\n  Top node scores (fp32 vs bf16) — {len(common_nodes)} common nodes:")
    print(f"  {'(layer,neuron,comp)':<32} {'fp32':>12} {'bf16':>12} {'sign_match':>12}")
    print("  " + "-" * 72)

    # Show top 10 by absolute fp32 score
    top_by_fp32 = sorted(common_nodes, key=lambda k: abs(fp32_nodes[k]), reverse=True)[:10]
    sign_matches = 0
    sign_total = 0
    for k in top_by_fp32:
        s32 = fp32_nodes[k]
        s16 = bf16_nodes[k]
        sign_match = (s32 >= 0) == (s16 >= 0)
        sign_matches += int(sign_match)
        sign_total += 1
        print(f"  {str(k):<32} {s32:>12.6f} {s16:>12.6f} {'YES' if sign_match else 'NO':>12}")

    node_sign_frac = sign_matches / max(sign_total, 1)

    # Edge scores: build map by (writer_layer, writer_neuron, reader_layer, reader_neuron) → score
    def edge_score_map(circ):
        m = {}
        for edge, score in circ.edges.items():
            w, r = edge.writer.node, edge.reader.node
            k = (w.layer, w.neuron, w.component or "resid_post",
                 r.layer, r.neuron, r.component or "resid_post")
            m[k] = float(score)
        return m

    fp32_edges = edge_score_map(circ_fp32)
    bf16_edges = edge_score_map(circ_bf16)

    common_edges = set(fp32_edges) & set(bf16_edges)
    print(f"\n  Top edge scores (fp32 vs bf16) — {len(common_edges)} common edges:")
    print(f"  {'edge_key':<52} {'fp32':>10} {'bf16':>10} {'sign':>6}")
    print("  " + "-" * 82)

    top_by_fp32_e = sorted(common_edges, key=lambda k: abs(fp32_edges[k]), reverse=True)[:10]
    edge_sign_matches = 0
    edge_sign_total = 0
    for k in top_by_fp32_e:
        s32 = fp32_edges[k]
        s16 = bf16_edges[k]
        sign_match = (s32 >= 0) == (s16 >= 0)
        edge_sign_matches += int(sign_match)
        edge_sign_total += 1
        print(f"  {str(k):<52} {s32:>10.6f} {s16:>10.6f} {'Y' if sign_match else 'N':>6}")

    edge_sign_frac = edge_sign_matches / max(edge_sign_total, 1)
    print(f"\n  Node sign match: {sign_matches}/{sign_total} ({node_sign_frac:.1%})")
    print(f"  Edge sign match: {edge_sign_matches}/{edge_sign_total} ({edge_sign_frac:.1%})")

# ---------------------------------------------------------------------------
# Final Verdict
# ---------------------------------------------------------------------------
print()
print("=" * 70)
print("FINAL VERDICT: bf16 ROBUSTNESS")
print("=" * 70)

if bf16_raised:
    print(f"  RESULT: FAIL — bf16 run raised an exception: {error_msg}")
    print("  bf16-robust = NO")
elif bf16_nan:
    print("  RESULT: PARTIAL — NaN/inf found in bf16 scores")
    print("  bf16-robust = NO (NaN scores)")
else:
    splice_label = "YES" if splice_ok_bf16 else "BORDERLINE"
    print(f"  Splice losslessness at bf16 tol=5e-2: {splice_label}")
    print(f"    site6: logit_err={err_bf16_r6:.2e} vs fp32 {err_fp32_r6:.2e}")
    print(f"    site7: logit_err={err_bf16_r7:.2e} vs fp32 {err_fp32_r7:.2e}")
    print(f"  Sign-consistency: nodes {node_sign_frac:.1%}, edges {edge_sign_frac:.1%}")

    robust = splice_ok_bf16 and not bf16_nan and node_sign_frac >= 0.7
    print(f"  bf16-robust = {'YES' if robust else 'PARTIAL/NO'}")
    if not robust:
        reasons = []
        if not splice_ok_bf16:
            reasons.append(f"splice losslessness failed (logit_err too high)")
        if bf16_nan:
            reasons.append("NaN in scores")
        if node_sign_frac < 0.7:
            reasons.append(f"sign-consistency only {node_sign_frac:.0%}")
        print(f"  Reasons: {'; '.join(reasons)}")

print()
print("Script complete.")
print(f"Absolute path: /Users/vishsangale/workspace/circuitry/scripts/v17_validation/inv3_bf16_robustness.py")
