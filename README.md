# circuitry

> **Scope:** Statistical diagnostics on neural-network weights, activations, and gradients — usable live during training or post-hoc on saved checkpoints. The statistical weight/activation/gradient primitives are the core foundation; an opt-in interventional **activation-patching / attribution** pillar (EAP, AtP\*, ACDC), an SAE-reconstruction workflow, and **SAE-feature circuits** (node-level attribution + feature→feature edges + greedy `FeatureACDC`) have since been added. v1.7 extends SAE circuits to `resid_post`, `mlp_out`, and `attn_out` sites, enables the TransformerLens backend, and adds an integrated-gradients variant (`variant='ig'`). v1.8 is a correctness release — it fixes 20 findings from a whole-library real-model evaluation (accurate-by-default weight SVD, graceful SDPA/Qwen/Llama-3 attach, MoE/recsys/ViT recorder coverage, layer_norm-SAE faithfulness); see [`docs/observations/2026-05-31-real-model-evaluation.md`](docs/observations/2026-05-31-real-model-evaluation.md). **v1.10 adds the tuned lens** (Belrose et al. 2023): post-hoc `fit_tuned_lens` + the opt-in `tuned_lens_kl` diagnostic, plus a polish bucket (signed report Δ, scale-invariant `update_delta_vanishing`). **v1.11 adds `copy_suppression_score`** — the per-head same-token attention metric that identifies copy-suppression heads on the repeated-token probe (McDougall et al. 2023). **v1.12 adds `attention_sink_score`** — per-head mean attention weight on the initial token, the signature of attention sink heads (Xiao et al. 2023). **v1.13 adds `head_specialization`** — classifies each attention head as induction / copy_suppression / sink / uniform from the three behavioral scores, rendered as a `## Head Specialization` table in the report. **v1.14 adds training-dynamics primitives** (`phase_transition_steps`, `head_formation_step` in `core/dynamics`) and a `## Training Dynamics` report section surfacing head-formation events and phase transitions in rank/health metrics. **v1.15 polish sprint**: `grokking_step` helper, per-hook-family tag counts in the report summary, Grokking Signals sub-table in Training Dynamics, and `docs/positioning.md` tool-by-tool landscape comparison. **v1.16 training-dynamics depth**: weight-norm direction labels (↑ norm growth / ↓ norm collapse) in Phase Transitions, `activation/repr_drift` trending as a `### Representation Drift` sub-table (start drift, end drift, Δ, trend label), and a `repr_drift_high` flag rule. **v1.17 circuit rendering**: `.to_markdown()` on `EAPResult`, `AtPResult`, `ACDCResult` (top-K edge/node tables); `.to_json()` / `.from_json()` on `EAPResult` and `ACDCResult`; `circuitry circuit-compare A.json B.json` CLI for edge-set diffs; `circuitry scan --model-factory pkg.module:factory` now fully functional.

Mechanistic-interpretability diagnostics for PyTorch — works across LLMs, vision (CNNs / ViTs), and recsys models with a single API, live during training or post-hoc on a checkpoint.

**Status:** v1.17.0 (beta). Research code; no support promise. Design contract: [`docs/design.md`](docs/design.md).

## Install

```bash
pip install -e .                 # core: torch + numpy only
pip install -e ".[tensorboard]"  # + TensorBoard writer
pip install -e ".[sae]"          # + SAE features (sae-lens)
pip install -e ".[all]"          # everything
```

`tensorboard` and `sae-lens` are optional extras. With a core-only install the
default `writer="auto"` falls back to the no-dep JSONL writer.

## Quickstart

```python
from circuitry import Recorder

recorder = Recorder(
    model,
    run_dir="runs/my_run",
    recipe="llm",            # or "vision", "two_tower", "recsys"
    writer="auto",           # tensorboard if installed, else jsonl; or "jsonl"/"null"
    every_n_steps=200,
)
recorder.attach()
for step, batch in enumerate(loader):
    loss = train_step(model, batch)
    recorder.step(step, loss=loss)
recorder.detach()
```

Retrospective scan + report from saved checkpoints:

```bash
circuitry scan    --run runs/my_run --recipe llm
circuitry report  --run runs/my_run
circuitry report  --run runs/my_run --compact
circuitry compare runs/run_a runs/run_b
```

`circuitry report` has two entry points:

- **Live run** — the `Recorder` writes `metrics.jsonl` during training; run
  `circuitry report --run <dir>` directly on it. No `scan` step needed.
- **Retrospective** — `circuitry scan` reads saved checkpoints and writes
  `metrics.jsonl` (default `writer="jsonl"`), which `circuitry report` then renders.

`--compact` renders only the `## Summary` and `## Flags` verdict blocks, suppressing per-tag tables.
`circuitry compare run_a run_b` compares two runs at family/diagnostic granularity.

## What you get

- **Primitives** (`circuitry.core.*`) — `effective_rank`, `stable_rank`, `heavy_tail_alpha`, `dead_fraction`, `kurtosis`, `participation_ratio`, `grad_norm_per_module`, ESD, rank trajectory, cross-step weight dynamics (`update_delta`, `direction_cosine`), **representational drift** (`repr_drift` — configurable linear-CKA / cosine / RBF-CKA), and more.
- **Recorder** — attach to a training loop, write TensorBoard events every N steps, dump a markdown report at the end. Live **training-dynamics** diagnostics (`weight/update_delta`, `weight/direction_cosine`, `weight/rank_trajectory`) track weight formation/collapse across emit steps with no extra forward pass. `Recipe.disable(names)` / `Recipe.only(names)` select diagnostics; `circuitry report --compact` renders Summary + Flags only; `circuitry compare` diffs two runs at family/diagnostic granularity. **Opt-in representational-drift probe** (`Recipe.probe_batch`): pass a fixed probe batch to track per-layer `activation/repr_drift/<module>` across training — requires a second forward pass per emit step, off by default.
- **Recipes** — `llm` / `vision` / `two_tower` plug the right hooks and diagnostics into your model; subclass `Recipe` or `register_recipe(...)` for custom architectures.
- **Activation patching / attribution** (`circuitry.patching`) — opt-in causal activation patching (`patch_site`, `PatchRunner`) plus EAP, AtP\*, and ACDC circuit-attribution over a frozen model. HF-eager (Llama-family + `head_dim`-aware) and TransformerLens backends; `to_hooked_transformer` bridge for non-Llama HF models. ACDC `run()`/`sweep()` gained `ablation_mode` (`"corrupted"` / `"zero"` / `"mean"`) and `eap_skip_threshold` for faster circuit discovery.
- **SAE-feature attribution** (`circuitry.patching.SAEFeatureRunner`, v1.5+) — node-level attribution to individual **SAE features** via error-term substitution (Marks "Sparse Feature Circuits"). Splice a SAELens SAE in losslessly on the clean pass, then score each feature `Σ_pos(Δf·gradf)`. **v1.7:** supports `resid_post`, `mlp_out`, and `attn_out` sites (Llama/GPT-2-style sequential blocks; Gemma2/parallel-attn caveated); HF-eager and **TransformerLens** backends; `variant='ig'` (integrated gradients, `n_ig_steps=32` default) refines scores past the `grad@clean` saturation blind spot — feature-IG completes to the eps-frozen spliced delta; adding `include_error_node=True` closes the error contribution and completes to the real forward delta. Opt-in `sae_error` node; differentiable `sae.encode_features` / `decode_features` / `sae_decompose` helpers.
- **SAE-feature circuits** (`circuitry.patching.SAEFeatureEdgeRunner` + `FeatureACDCRunner`, v1.6+) — feature→feature **edges** and a prunable **sparse-feature circuit**. Two-stage (node attribution → top-K active survivors → edges among them); all sites spliced simultaneously so an upstream feature's decode reaches the downstream encode; edges scored by a per-downstream-survivor VJP (no dense Jacobian). Opt-in error→feature edges (feature→error is structurally zero). `SAEFeatureCircuit.prune('threshold'|'acdc'|'both')` + `faithfulness()`/`completeness()`; `FeatureACDC` is greedy reverse-topo node pruning with a `sweep` Pareto helper. **v1.7:** multi-site-per-layer composite `(layer, component)` keying, forward-position ordering (`attn_out@L → mlp_out@L` edges), `variant='ig'` (full EAP-IG, per-j VJP — no dense Jacobian, peak memory == attrib), TL backend.
- **Accurate weight SVD by default (v1.8)** — `weight.singular_values` and the SVD-derived diagnostics compute the **full, deterministic SVD by default** (`max_dim=None`). The old `max_dim=512` random-column subsample silently biased every diagnostic on any matrix with min-dim > 512 (every real LLM layer) and varied run-to-run; it is now an **opt-in perf hatch** (`max_dim=...`, with `seed` for reproducibility). The `use_gram='auto'` Gram fast path remains for strongly-rectangular matrices; `condition_number` stays on full SVD. The scalar rank primitives require ≤2-D input — batched MoE-expert tensors are emitted per-expert.
- **MetricWriter protocol** — TB by default; `jsonl` (storage format for the `scan` / `report` workflow) and `null` (test plumbing) adapters ship in-tree. Bring-your-own writer is a ~50-LOC subclass of `MetricWriter`.

## Performance

Default settings target ≤10% wall-clock overhead at `every_n_steps=200` on a ~50M-param decoder transformer. **At a realistic training step the budget holds — +5.3% on GPU** (RTX 5080, 88M-param decoder, batch 16 × seq 512, full `llm` recipe).

Measured overhead (88M decoder, `every_n_steps=200`):

| device | training step | overhead |
| --- | --- | -------: |
| RTX 5080 | batch 16 × seq 512 (8192 tok) | **+5.3%** |
| RTX 5080 | batch 4 × seq 64 (256 tok) | +45.3% |
| CPU 16-core (v0.2.0a0) | batch 4 × seq 64 | +14.9% |

The overhead is dominated by the roughly *fixed* per-emit diagnostic cost (the SVD set + logit-lens + induction-score), so the **ratio** is very sensitive to how heavy the baseline step is: a tiny 256-token step on GPU (~12 ms) inflates it to +45%, while a realistic 8192-token step amortises it to +5.3%. On small/fast steps, raise `every_n_steps` or trim diagnostics with `Recipe.disable` / `Recipe.only`. The v1.4 Gram fast path (`use_gram='auto'`) helps narrowly-rectangular matrices; the drift probe is off by default (zero overhead at default settings).

Run the harness yourself (defaults to the tiny batch; pass a realistic one for the budget scenario):

```bash
.venv/bin/python scripts/bench_50m.py --device cuda --steps 1000 --every-n-steps 200 --batch-size 16 --seq-len 512
```

## Known limits

- Single-process training only. In a multi-rank DDP/FSDP run `circuitry` no-ops on non-zero ranks; FSDP-sharded parameters will produce **incorrect** diagnostics on rank 0. Multi-process support is planned for a future release; see `docs/design.md` §11 for the upgrade path.

## License

MIT.
