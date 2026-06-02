"""Investigation 1: Bounded-memory claim.

Measures peak process RSS and tracemalloc delta across runner.run() while
sweeping top_k_survivors ∈ [4, 16, 64, 256].

The design claim: peak memory does NOT scale with top_k_survivors² or d_sae².
A dense d_sae×d_sae Jacobian would be ~24576² × 4 bytes ≈ 2.4 GB.
We verify that actual peaks are O(d_sae·top_k) not O(d_sae²).
"""

import gc
import resource
import tracemalloc
import warnings

warnings.filterwarnings("ignore")

import torch
from sae_lens import SAE
from transformer_lens import HookedTransformer

from circuitry.patching.sae_edges import SAEFeatureEdgeRunner
from circuitry.patching.sites import Site, TLSiteResolver

print("=" * 70)
print("INVESTIGATION 1 — Bounded-memory claim")
print("=" * 70)

# ---------------------------------------------------------------------------
# Setup (exact pattern from spec)
# ---------------------------------------------------------------------------
print("\nLoading model and SAEs ...")
model = HookedTransformer.from_pretrained("gpt2", device="cpu")
model.eval()


def load_sae(rel, sid):
    r = SAE.from_pretrained(rel, sid, device="cpu")
    return r[0] if isinstance(r, tuple) else r


sae_r6 = load_sae("gpt2-small-res-jb", "blocks.7.hook_resid_pre")  # == resid_post@6
sae_r7 = load_sae("gpt2-small-res-jb", "blocks.8.hook_resid_pre")  # == resid_post@7

resolver = TLSiteResolver()
clean = model.to_tokens("When John and Mary went to the store, John gave a drink to")
corrupt = model.to_tokens("When John and Mary went to the store, Mary gave a drink to")
mary = model.to_single_token(" Mary")
john = model.to_single_token(" John")
metric = lambda logits: logits[0, -1, mary] - logits[0, -1, john]

d_sae = sae_r7.W_dec.shape[0]  # feature dimension (should be 24576)
print(f"d_sae = {d_sae}")
print(f"Dense d_sae×d_sae Jacobian would be: {d_sae**2 * 4 / 1e9:.3f} GB (4 bytes/float32)\n")

TOP_K_VALUES = [4, 16, 64, 256]

# Pre-build the runner (SAE objects shared across sweeps — fixed allocation)
runner = SAEFeatureEdgeRunner(
    model=model,
    sae_sites={Site("resid_post", 6): sae_r6, Site("resid_post", 7): sae_r7},
    resolver=resolver,
)


def get_rss_bytes() -> int:
    """Return current process RSS in bytes (macOS: getrusage returns bytes)."""
    usage = resource.getrusage(resource.RUSAGE_SELF)
    # On macOS, ru_maxrss is in bytes; on Linux it's kilobytes
    import platform
    if platform.system() == "Darwin":
        return usage.ru_maxrss
    else:
        return usage.ru_maxrss * 1024


# Warm-up run to avoid cold-start noise
print("Warm-up run (top_k=4) ...")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    _ = runner.run(clean, corrupt, metric, layer_pairs="adjacent", top_k_survivors=4)
gc.collect()

print("\n{:<15} {:>20} {:>20} {:>20}".format(
    "top_k_survivors", "tracemalloc_peak_MB", "rss_max_MB", "n_edges"
))
print("-" * 78)

results = []
for top_k in TOP_K_VALUES:
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    tracemalloc.start()
    rss_before = get_rss_bytes()

    circ = runner.run(clean, corrupt, metric, layer_pairs="adjacent", top_k_survivors=top_k)

    rss_after = get_rss_bytes()
    current_tm, peak_tm = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    peak_mb = peak_tm / 1e6
    rss_delta_mb = (rss_after - rss_before) / 1e6  # can be ≤0 if RSS was already high
    n_edges = len(circ.edges)

    print("{:<15} {:>20.2f} {:>20.2f} {:>20}".format(
        top_k, peak_mb, rss_delta_mb, n_edges
    ))
    results.append((top_k, peak_mb, rss_delta_mb, n_edges))
    gc.collect()

# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
print()
print("ANALYSIS")
print("-" * 70)
print(f"d_sae = {d_sae}, so dense Jacobian = {d_sae**2 * 4 / 1e9:.2f} GB")
print()

# Check if memory scales sub-quadratically in top_k
if len(results) >= 2:
    for idx in range(1, len(results)):
        prev_topk, prev_mem, _, _ = results[idx - 1]
        curr_topk, curr_mem, _, _ = results[idx]
        ratio_topk = curr_topk / prev_topk
        ratio_mem = curr_mem / max(prev_mem, 1e-9)
        print(f"  top_k {prev_topk}→{curr_topk} ({ratio_topk:.0f}×): "
              f"tracemalloc peak {prev_mem:.1f}→{curr_mem:.1f} MB "
              f"(ratio={ratio_mem:.2f}×)")

print()
# Verdict
max_peak_mb = max(r[1] for r in results)
print(f"MAX tracemalloc peak across all top_k values: {max_peak_mb:.2f} MB")
if max_peak_mb < 500:  # well under 1 GB
    print("VERDICT: BOUNDED-MEMORY CLAIM = YES")
    print("  Peak stays well under 1 GB even at top_k=256.")
    print("  The dense d_sae×d_sae Jacobian is NEVER materialized (claim confirmed).")
else:
    print("VERDICT: BOUNDED-MEMORY CLAIM = SUSPECT")
    print(f"  Peak {max_peak_mb:.0f} MB is surprisingly high — review for regressions.")

# Check if memory grows quadratically with top_k
if len(results) >= 3:
    k4_mem = results[0][1]
    k64_mem = results[2][1]
    k256_mem = results[3][1] if len(results) > 3 else None
    if k64_mem > 0 and k4_mem > 0:
        observed_scaling = (k64_mem / k4_mem)
        topk_ratio = 64 / 4
        print(f"\n  top_k 4→64 ({topk_ratio}×): memory ratio {observed_scaling:.2f}×")
        if observed_scaling < topk_ratio ** 2 * 0.5:
            print("  Memory does NOT scale quadratically — sub-quadratic confirmed.")
        else:
            print("  WARNING: memory may be scaling quadratically with top_k.")

print()
print("Script complete.")
print(f"Absolute path: /Users/vishsangale/workspace/circuitry/scripts/v17_validation/inv1_bounded_memory.py")
