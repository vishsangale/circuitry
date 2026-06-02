"""Investigation 2: IG N×-cost claim.

Times variant='attrib' vs variant='ig' with n_ig_steps ∈ [8, 32, 64].
Design claim: IG costs ~N× attrib with peak memory == attrib.

Verifies:
  - time scales ~linearly in N (IG N× more expensive than attrib)
  - IG peak memory ≈ attrib peak memory (NOT N× more)
"""

import gc
import time
import tracemalloc
import warnings

warnings.filterwarnings("ignore")

import torch
from sae_lens import SAE
from transformer_lens import HookedTransformer

from circuitry.patching.sae_edges import SAEFeatureEdgeRunner
from circuitry.patching.sites import Site, TLSiteResolver

print("=" * 70)
print("INVESTIGATION 2 — IG N×-cost claim")
print("=" * 70)

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
print("\nLoading model and SAEs ...")
model = HookedTransformer.from_pretrained("gpt2", device="cpu")
model.eval()


def load_sae(rel, sid):
    r = SAE.from_pretrained(rel, sid, device="cpu")
    return r[0] if isinstance(r, tuple) else r


sae_r6 = load_sae("gpt2-small-res-jb", "blocks.7.hook_resid_pre")
sae_r7 = load_sae("gpt2-small-res-jb", "blocks.8.hook_resid_pre")

resolver = TLSiteResolver()
clean = model.to_tokens("When John and Mary went to the store, John gave a drink to")
corrupt = model.to_tokens("When John and Mary went to the store, Mary gave a drink to")
mary = model.to_single_token(" Mary")
john = model.to_single_token(" John")
metric = lambda logits: logits[0, -1, mary] - logits[0, -1, john]

d_sae = sae_r7.W_dec.shape[0]
print(f"d_sae = {d_sae}")

TOP_K = 16  # fixed; representative of real use
N_REPEATS = 3  # time each config N_REPEATS times and take the median

runner = SAEFeatureEdgeRunner(
    model=model,
    sae_sites={Site("resid_post", 6): sae_r6, Site("resid_post", 7): sae_r7},
    resolver=resolver,
)


def time_run(variant: str, n_ig_steps: int = 0, top_k: int = TOP_K) -> tuple[float, float, int]:
    """Run once, return (elapsed_seconds, tracemalloc_peak_bytes, n_edges)."""
    gc.collect()
    tracemalloc.start()
    t0 = time.perf_counter()
    circ = runner.run(
        clean, corrupt, metric,
        layer_pairs="adjacent",
        top_k_survivors=top_k,
        variant=variant,
        n_ig_steps=n_ig_steps,
    )
    t1 = time.perf_counter()
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return t1 - t0, peak_bytes, len(circ.edges)


# ---------------------------------------------------------------------------
# Warm up
# ---------------------------------------------------------------------------
print("\nWarm-up run (attrib) ...")
_ = time_run("attrib")
gc.collect()

# ---------------------------------------------------------------------------
# Measure attrib baseline
# ---------------------------------------------------------------------------
print("\nMeasuring attrib baseline ...")
attrib_times = []
attrib_mems = []
for _ in range(N_REPEATS):
    t, m, n = time_run("attrib")
    attrib_times.append(t)
    attrib_mems.append(m)
    gc.collect()

attrib_time_med = sorted(attrib_times)[N_REPEATS // 2]
attrib_mem_med = sorted(attrib_mems)[N_REPEATS // 2]
print(f"  attrib: median time={attrib_time_med:.3f}s, peak mem={attrib_mem_med/1e6:.2f} MB, n_edges={n}")

# ---------------------------------------------------------------------------
# Measure IG at different N values
# ---------------------------------------------------------------------------
IG_N_VALUES = [8, 32, 64]

print("\n{:<12} {:>16} {:>16} {:>16} {:>16} {:>12}".format(
    "variant", "n_ig_steps", "median_time_s", "expected_t_s", "peak_mem_MB", "n_edges"
))
print("-" * 88)

# Print attrib row
print("{:<12} {:>16} {:>16.3f} {:>16} {:>16.2f} {:>12}".format(
    "attrib", "N/A", attrib_time_med, "1×", attrib_mem_med / 1e6, n
))

ig_results = []
for n_steps in IG_N_VALUES:
    times = []
    mems = []
    for _ in range(N_REPEATS):
        t, m, n_e = time_run("ig", n_ig_steps=n_steps)
        times.append(t)
        mems.append(m)
        gc.collect()
    t_med = sorted(times)[N_REPEATS // 2]
    m_med = sorted(mems)[N_REPEATS // 2]
    time_ratio = t_med / max(attrib_time_med, 1e-9)
    expected_t = attrib_time_med * n_steps
    print("{:<12} {:>16} {:>16.3f} {:>16.3f} {:>16.2f} {:>12}".format(
        "ig", n_steps, t_med, expected_t, m_med / 1e6, n_e
    ))
    ig_results.append((n_steps, t_med, m_med, n_e, time_ratio))

# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
print()
print("ANALYSIS")
print("-" * 70)

print("\n  TIME SCALING (IG vs attrib):")
for n_steps, t, m, n_e, ratio in ig_results:
    print(f"    n_ig_steps={n_steps:3d}: actual {ratio:.2f}× attrib "
          f"(expected ~{n_steps}×, actual/expected = {ratio/n_steps:.2f})")

print("\n  MEMORY SCALING (IG vs attrib):")
for n_steps, t, m, n_e, ratio in ig_results:
    mem_ratio = m / max(attrib_mem_med, 1)
    print(f"    n_ig_steps={n_steps:3d}: IG peak mem = {m/1e6:.2f} MB, "
          f"attrib peak = {attrib_mem_med/1e6:.2f} MB, ratio = {mem_ratio:.2f}×")

print()
# Verdicts
time_ok = True
mem_ok = True

for n_steps, t, m, n_e, ratio in ig_results:
    # Time should scale roughly N× (within 50% of expected)
    if ratio < n_steps * 0.3 or ratio > n_steps * 3.0:
        print(f"  WARNING: IG n={n_steps} time ratio {ratio:.2f}× is FAR from expected {n_steps}×")
        time_ok = False
    # Memory ratio should be ≤ 2× attrib (not N×)
    mem_ratio = m / max(attrib_mem_med, 1)
    if mem_ratio > 3.0:  # generous threshold; N× would be 8×, 32×, 64×
        print(f"  WARNING: IG n={n_steps} memory ratio {mem_ratio:.2f}× is too high")
        mem_ok = False

if time_ok:
    print("TIME VERDICT: IG N×-cost claim = YES (time scales ~linearly in N)")
else:
    print("TIME VERDICT: IG N×-cost claim = PARTIAL/NO (time scaling unexpected)")

if mem_ok:
    print("MEMORY VERDICT: IG peak memory ≈ attrib peak memory = YES")
else:
    print("MEMORY VERDICT: IG peak memory exceeds attrib — NOT equal")

print()
print("Script complete.")
print(f"Absolute path: /Users/vishsangale/workspace/circuitry/scripts/v17_validation/inv2_ig_cost.py")
