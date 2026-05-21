# circuitry

> **Disambiguation:** `circuitry` is statistical diagnostics on neural-network weights / activations / gradients during training. It is **not** causal-pathway discovery in the Anthropic "circuits" research sense. The name is borrowed from electronics.

Training-time mechanistic-interpretability diagnostics for PyTorch — works across LLMs, vision (CNNs / ViTs), and recommender (two-tower) models with a single API.

**Status:** v0.1.0 (alpha). Research code; no support promise. Design contract: [`docs/design.md`](docs/design.md). Implementation plan: [`docs/plan.md`](docs/plan.md).

## Install

```bash
pip install -e .          # editable, from a checkout
pip install -e ".[wandb]" # with wandb writer
```

## Quickstart

```python
from circuitry import Recorder

recorder = Recorder(
    model,
    run_dir="runs/my_run",
    recipe="llm",            # or "vision", "two_tower"
    writer="tensorboard",    # or "wandb", "jsonl", "null"
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
- **MetricWriter protocol** — TB by default; wandb / jsonl / null adapters ship in-tree.

## Performance

Default settings target ≤10% wall-clock overhead at `every_n_steps=200` on a 50M-param decoder transformer (see `docs/design.md` §10). Benchmark numbers will land alongside the M2 mendu cutover. Run the harness yourself:

```bash
python scripts/bench_50m.py --n-layers 8 --d-model 768 --steps 100
```

## v0.1.0 limits

- Single-process training only. In a multi-rank DDP/FSDP run `circuitry` no-ops on non-zero ranks; FSDP-sharded parameters will produce **incorrect** diagnostics on rank 0. Multi-process support lands in v0.next; see `docs/design.md` §11 for the upgrade path.
- Benchmark numbers (overhead at default settings on a 50M-param transformer) will be filled in alongside the M2 mendu cutover. The harness is in `tests/perf/` if you want to run it yourself.

## License

MIT.
