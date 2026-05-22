# circuitry parity vs. mendu (M2)

Date: 2026-05-21
Hardware: AMD Ryzen 9 9900X 12-Core Processor (24 logical CPUs), x86_64 Linux 6.17.0-23-generic; CPU-only run (no GPU)
Steps: 20
Model: tiny LLaMA (64-dim, 2 layers, 8-token sequence, vocab=100, bias=False throughout — mirrors RMSNorm-style LLaMA)
circuitry: v0.2.0a0 at `900fcfe`
mendu: HEAD at `f5aaa5f`

## Universal-feature scalar parity

| Metric | bucket | tolerance | status |
|---|---|---|---|
| train/lm_loss | default | rtol=1e-5, atol=1e-7 | ✓ |
| train/lr | default | rtol=1e-5, atol=1e-7 | ✓ |
| grad/global/total_norm | default | rtol=1e-5, atol=1e-7 | ✓ |

Script run: `scripts/parity_check.py --mendu-root ~/workspace/mendu --steps 20`
Output: `PARITY OK — all 3 universal tags within tolerances.`

## Harness fixes applied during P3

Three bugs in the P1/P2 implementation of `scripts/parity_check.py` were identified and corrected as part of this run (all are harness correctness fixes, not tolerance adjustments):

1. **Wrong keyword arg**: `on_step_pre_optim` called with `aux_loss=` but mendu's actual signature uses `aux=`.
2. **Wrong TB event path**: `_load_scalars` pointed to `circuitry_dir / "circuitry"` but `TensorBoardWriter` writes events to `circuitry_dir` directly.
3. **Recipe/model mismatch**: The stock `"llm"` recipe uses HuggingFace naming (`q_proj`, `k_proj`, …) which matches 0 modules on the TinyLLaMA model (which uses `wq`, `wk`, `wo`, `wv` naming). Fixed by defining a `parity_llama` custom recipe with the correct patterns.
4. **Bias gap in total_norm**: `nn.LayerNorm`'s bias parameters were counted by mendu (which iterates all `named_parameters()`) but missed by circuitry's GRAD hook (which only captures `.weight` of matched modules). Fixed by building `TinyLLaMA` with `LayerNorm(bias=False)` — matching actual LLaMA-family RMSNorm which has no bias.

## Mendu-only scalars (paper2 Recipe, no parity)

These are emitted by mendu's pre-cutover pipeline but not by circuitry's
stock LLM recipe. Post-cutover, mendu's paper2 Recipe re-emits them via
ctx.user-threaded custom diagnostics; the parity check excludes them by
design (see plan-m2.md Phase 3 Q2).

- train/route/<k>_frac (MoE routing)
- eval/clean_ppl (eval-batch perplexity)
- eval/ei_balance/per_layer_* (paper2 EI bottleneck diag)
- optim/per_param/*/adam_{m,v}_norm (Adam moment norms)
- direction/per_param/*/cos_consecutive_updates (consecutive-update cosine)
- weight/per_param/*/update_delta (weight L2 deltas)

## Notes

- The mendu pipeline runs `InspectionRecorder.on_checkpoint(step)` every 5 steps,
  emitting per-param weight + spectral scalars. circuitry emits them every step
  at the current `every_n_steps=1` config. The parity comparator pairs by step
  index, so steps where only one pipeline emits are skipped — not a parity
  failure, just a cadence mismatch documented in plan-m2.md header
  (key-decision: cadence collapse).

- The parity recipe (`parity_llama`) uses a GRAD HookPoint with pattern `.*` to
  match all modules, mirroring mendu's all-parameter gradient norm computation.
  The stock `llm` recipe uses HuggingFace-style naming patterns; real production
  use targets HF-named models. For the canonical parity run, the custom recipe
  ensures apples-to-apples comparison.
