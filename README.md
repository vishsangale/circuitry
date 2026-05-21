# circuitry

> **Disambiguation:** `circuitry` is statistical diagnostics on neural-network weights / activations / gradients during training. It is **not** causal-pathway discovery in the Anthropic "circuits" research sense. The name is borrowed from electronics.

Training-time mechanistic-interpretability diagnostics for PyTorch — works across LLMs, vision (CNNs / ViTs), and recommender (two-tower) models with a single API.

**Status:** pre-release. Design contract is in [`docs/design.md`](docs/design.md). Implementation has not started.

## What it does

- **Primitives** — pure functions for weight / activation / gradient / spectral diagnostics (`effective_rank`, `heavy_tail_alpha`, `dead_fraction`, `kurtosis`, `participation_ratio`, …).
- **Recorder** — opinionated training-time workflow that attaches hooks per recipe, writes TensorBoard events at configurable intervals, and produces a markdown report.
- **Recipes** — modality adapters that know which submodules matter for LLMs, vision, or two-tower models.

## Quick reference

See `docs/design.md` for the full API surface and design rationale.

## License

MIT (added at v0.1.0).
