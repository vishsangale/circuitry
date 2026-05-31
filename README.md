# circuitry

> **Scope:** Statistical diagnostics on neural-network weights, activations, and gradients — usable live during training or post-hoc on saved checkpoints. The statistical weight/activation/gradient primitives are the core foundation; an opt-in interventional **activation-patching / attribution** pillar (EAP, AtP\*, ACDC), an SAE-reconstruction workflow, and **SAE-feature circuits** (node-level attribution + feature→feature edges + greedy `FeatureACDC`) have since been added. Tuned lens remains future work. The name is borrowed from electronics.

Mechanistic-interpretability diagnostics for PyTorch — works across LLMs, vision (CNNs / ViTs), and recsys models with a single API, live during training or post-hoc on a checkpoint.

**Status:** v1.4.0 (beta). Research code; no support promise. Design contract: [`docs/design.md`](docs/design.md).

## Install

```bash
pip install -e .          # editable, from a checkout
```

## Quickstart

```python
from circuitry import Recorder

recorder = Recorder(
    model,
    run_dir="runs/my_run",
    recipe="llm",            # or "vision", "two_tower"
    writer="tensorboard",    # or "jsonl", "null"
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
- **SAE-feature attribution** (`circuitry.patching.SAEFeatureRunner`, v1.5) — node-level attribution to individual **SAE features** at residual-stream sites, via error-term substitution (Marks "Sparse Feature Circuits"). Splice a SAELens SAE in losslessly on the clean pass, then score each feature `Σ_pos(Δf·gradf)` against a clean/corrupted prompt pair; opt-in `sae_error` reconstruction-error node. Differentiable `sae.encode_features` / `decode_features` / `sae_decompose` helpers back it. HF-eager + `resid_post`.
- **SAE-feature circuits** (`circuitry.patching.SAEFeatureEdgeRunner` + `FeatureACDCRunner`, v1.6) — feature→feature **edges** and a prunable **sparse-feature circuit**. Two-stage (node attribution → top-K active survivors → edges among them); all sites spliced simultaneously so an upstream feature's decode reaches the downstream encode; edges scored by a per-downstream-survivor VJP (no dense Jacobian). Opt-in error→feature edges (feature→error is structurally zero). `SAEFeatureCircuit.prune('threshold'|'acdc'|'both')` + `faithfulness()`/`completeness()`; `FeatureACDC` is greedy reverse-topo node pruning with a `sweep` Pareto helper.
- **Deterministic SVD subsample** — `weight.singular_values` gained a `seed` kwarg (CPU-deterministic column subsample for cross-step comparison) and a `use_gram='auto'` Gram fast path for strongly-rectangular matrices (eigvalsh(W^T W) path; `condition_number` stays on full SVD).
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
