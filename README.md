# circuitry

> **Scope:** Statistical diagnostics on neural-network weights, activations, and gradients — usable live during training or post-hoc on saved checkpoints. The statistical weight/activation/gradient primitives are the core foundation; an opt-in interventional **activation-patching / attribution** pillar (EAP, AtP\*, ACDC) and an SAE-reconstruction workflow have since been added. Tuned lens and SAE-feature circuits remain future work. The name is borrowed from electronics.

Mechanistic-interpretability diagnostics for PyTorch — works across LLMs, vision (CNNs / ViTs), and recsys models with a single API, live during training or post-hoc on a checkpoint.

**Status:** v1.3.0 (beta). Research code; no support promise. Design contract: [`docs/design.md`](docs/design.md).

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

- **Primitives** (`circuitry.core.*`) — `effective_rank`, `stable_rank`, `heavy_tail_alpha`, `dead_fraction`, `kurtosis`, `participation_ratio`, `grad_norm_per_module`, ESD, rank trajectory, cross-step weight dynamics (`update_delta`, `direction_cosine`), and more.
- **Recorder** — attach to a training loop, write TensorBoard events every N steps, dump a markdown report at the end. Live **training-dynamics** diagnostics (`weight/update_delta`, `weight/direction_cosine`, `weight/rank_trajectory`) track weight formation/collapse across emit steps with no extra forward pass. `Recipe.disable(names)` / `Recipe.only(names)` select diagnostics; `circuitry report --compact` renders Summary + Flags only; `circuitry compare` diffs two runs at family/diagnostic granularity.
- **Recipes** — `llm` / `vision` / `two_tower` plug the right hooks and diagnostics into your model; subclass `Recipe` or `register_recipe(...)` for custom architectures.
- **Activation patching / attribution** (`circuitry.patching`) — opt-in causal activation patching (`patch_site`, `PatchRunner`) plus EAP, AtP\*, and ACDC circuit-attribution over a frozen model. HF-eager (Llama-family + `head_dim`-aware) and TransformerLens backends; `to_hooked_transformer` bridge for non-Llama HF models.
- **MetricWriter protocol** — TB by default; `jsonl` (storage format for the `scan` / `report` workflow) and `null` (test plumbing) adapters ship in-tree. Bring-your-own writer is a ~50-LOC subclass of `MetricWriter`.

## Performance

Default settings target ≤10% wall-clock overhead at `every_n_steps=200` on a 50M-param decoder transformer (see `docs/design.md` §10).

**Measured at v0.2.0a0** (the v0.3.0 wandb removal does not touch perf-sensitive code paths, so these numbers remain representative). 88M-param decoder, 100 steps, `every_n_steps=200`, CPU on a 16-core consumer machine (`scripts/bench_50m.py --n-layers 8 --d-model 768`):

| run | baseline | instrumented | overhead |
| --- | -------: | -----------: | -------: |
| 1   |  23.90 s |      27.46 s |   +14.9% |
| 2   |  21.15 s |      24.26 s |   +14.7% |

Run-to-run noise on CPU is high (±5% typical, occasional 30% spikes when the bench shares cores), and CPU inflates the ratio versus the GPU production scenario the budget was sized against. GPU re-measurement is on the to-do list.

Run the harness yourself:

```bash
venv/bin/python scripts/bench_50m.py --n-layers 8 --d-model 768 --steps 100
```

## Known limits

- Single-process training only. In a multi-rank DDP/FSDP run `circuitry` no-ops on non-zero ranks; FSDP-sharded parameters will produce **incorrect** diagnostics on rank 0. Multi-process support is planned for a future release; see `docs/design.md` §11 for the upgrade path.

## License

MIT.
