# circuitry

> **Scope:** Statistical diagnostics on neural-network weights, activations, and gradients — usable live during training or post-hoc on saved checkpoints. Lens-style and attribution primitives (logit lens, activation patching, SAE probes) are on the future-work list; today's surface is statistical and modality-agnostic. The name is borrowed from electronics.

Mechanistic-interpretability diagnostics for PyTorch — works across LLMs, vision (CNNs / ViTs), and recsys models with a single API, live during training or post-hoc on a checkpoint.

**Status:** v0.3.0 (alpha). Research code; no support promise. Design contract: [`docs/design.md`](docs/design.md).

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
circuitry scan   --run runs/my_run --recipe llm
circuitry report --run runs/my_run
```

## What you get

- **Primitives** (`circuitry.core.*`) — `effective_rank`, `stable_rank`, `heavy_tail_alpha`, `dead_fraction`, `kurtosis`, `participation_ratio`, `layer_norm`, ESD, rank trajectory, and more.
- **Recorder** — attach to a training loop, write TensorBoard events every N steps, dump a markdown report at the end.
- **Recipes** — `llm` / `vision` / `two_tower` plug the right hooks and diagnostics into your model; subclass `Recipe` or `register_recipe(...)` for custom architectures.
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
