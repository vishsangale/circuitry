# scripts/parity_check.py
"""Numerical parity between circuitry and the in-tree mendu inspector.

Run in M2 (mendu cutover). The script trains a tiny canonical model under
both pipelines, captures all TB scalars from each, and asserts they agree
within the tolerances from docs/design.md §7 Phase M2:

  - Most metrics:        rtol=1e-5, atol=1e-7
  - SVD-derived metrics: rtol=1e-4   (effective_rank, condition_number,
                                       heavy_tail_alpha, singular_values)

M1 ships the harness; M2 wires it up against ~/workspace/mendu and ratchets
tolerances down if a metric is empirically tighter than expected.
"""

from __future__ import annotations

import argparse
import sys

DEFAULT_TOLERANCES = {
    "default": {"rtol": 1e-5, "atol": 1e-7},
    "svd": {"rtol": 1e-4, "atol": 1e-6},
}
SVD_METRICS = {
    "effective_rank",
    "condition_number",
    "heavy_tail_alpha",
    "singular_values",
    "stable_rank",
}


def _build_tiny_llama(seed: int = 0):
    """Build a tiny LLaMA-shaped model with fixed seed.

    Uses LLaMA naming conventions (layers.N.attention.wq.weight etc.) so that
    both mendu's arch_discovery and circuitry's _discovery.discover() can classify
    params into per-role buckets.  bias=False throughout — mendu's arch_discovery
    only matches `.weight` suffixes.
    """
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)

    class TinyAttn(nn.Module):
        def __init__(self, dim: int = 64) -> None:
            super().__init__()
            self.wq = nn.Linear(dim, dim, bias=False)
            self.wk = nn.Linear(dim, dim, bias=False)
            self.wv = nn.Linear(dim, dim, bias=False)
            self.wo = nn.Linear(dim, dim, bias=False)

        def forward(self, x):
            # Simplified (no causal mask, no rotary) — we only need gradients
            # flowing through the named params.
            return self.wo(self.wv(x) + self.wq(x) + self.wk(x))

    class TinyFFN(nn.Module):
        def __init__(self, dim: int = 64, hidden: int = 128) -> None:
            super().__init__()
            self.w1 = nn.Linear(dim, hidden, bias=False)
            self.w2 = nn.Linear(hidden, dim, bias=False)
            self.w3 = nn.Linear(dim, hidden, bias=False)

        def forward(self, x):
            return self.w2(self.w1(x) * self.w3(x))

    class TinyLayer(nn.Module):
        def __init__(self, dim: int = 64) -> None:
            super().__init__()
            self.attention = TinyAttn(dim)
            self.feed_forward = TinyFFN(dim)
            self.attention_norm = nn.LayerNorm(dim)
            self.ffn_norm = nn.LayerNorm(dim)

        def forward(self, x):
            x = x + self.attention(self.attention_norm(x))
            x = x + self.feed_forward(self.ffn_norm(x))
            return x

    class TinyLLaMA(nn.Module):
        def __init__(
            self, vocab: int = 100, dim: int = 64, n_layers: int = 2
        ) -> None:
            super().__init__()
            self.tok_embeddings = nn.Embedding(vocab, dim)
            self.layers = nn.ModuleList(
                [TinyLayer(dim) for _ in range(n_layers)]
            )
            self.norm = nn.LayerNorm(dim)
            self.output = nn.Linear(dim, vocab, bias=False)

        def forward(self, ids):
            x = self.tok_embeddings(ids)
            for layer in self.layers:
                x = layer(x)
            return self.output(self.norm(x))

    return TinyLLaMA()


def _make_batch(seed: int = 0, batch: int = 2, seq: int = 8, vocab: int = 100):
    """Return a fixed random integer batch for reproducible gradient traces."""
    import torch

    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, vocab, (batch, seq), generator=g, dtype=torch.long)


def _run_mendu(mendu_root, steps: int, out_dir) -> None:
    """Run the tiny canonical training under mendu's InspectionRecorder."""
    import torch

    sys.path.insert(0, str(mendu_root))
    from tools.inspect_checkpoint.live import Cadence, InspectionRecorder  # noqa: PLC0415

    model = _build_tiny_llama()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    batch = _make_batch()
    rec = InspectionRecorder(
        run_dir=str(out_dir),
        model=model,
        optimizer=optimizer,
        cadence=Cadence(step_scalars=1, weight_cheap=1),
    )
    for step in range(steps):
        logits = model(batch)
        loss = logits.sum()
        loss.backward()
        rec.on_step_pre_optim(
            step,
            total=loss.item(),
            lm_loss=loss.item(),
            aux_loss=0.0,
            lr=1e-3,
        )
        if step % 5 == 0:
            rec.on_checkpoint(step)
        optimizer.step()
        optimizer.zero_grad()
    rec.close()


def _run_circuitry(steps: int, out_dir) -> None:
    """Run the tiny canonical training under circuitry's Recorder."""
    import torch

    from circuitry import Recorder  # noqa: PLC0415

    model = _build_tiny_llama()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    batch = _make_batch()
    rec = Recorder(
        model,
        run_dir=out_dir,
        recipe="llm",
        writer="tensorboard",
        every_n_steps=1,
    )
    rec.attach()
    for step in range(steps):
        logits = model(batch)
        loss = logits.sum()
        loss.backward()
        rec.step(
            step,
            loss=loss.item(),
            loss_components={"lm_loss": loss.item(), "lr": 1e-3},
        )
        optimizer.step()
        optimizer.zero_grad()
    rec.detach()


def main() -> int:
    p = argparse.ArgumentParser(
        description="Parity check: circuitry vs. mendu pre-cutover InspectionRecorder."
    )
    p.add_argument(
        "--mendu-root",
        required=True,
        help="path to ~/workspace/mendu",
    )
    p.add_argument("--steps", type=int, default=20)
    p.add_argument(
        "--out-dir",
        default=None,
        help="if set, persist run dirs here; else use tempdir",
    )
    args = p.parse_args()

    import pathlib
    import tempfile

    if args.out_dir:
        out_base = pathlib.Path(args.out_dir)
        out_base.mkdir(parents=True, exist_ok=True)
    else:
        out_base = pathlib.Path(tempfile.mkdtemp(prefix="circuitry_parity_"))

    mendu_dir = out_base / "mendu"
    circuitry_dir = out_base / "circuitry"

    print(
        f"parity_check.py: running canonical tiny LLaMA for {args.steps} steps"
    )
    print(f"  mendu     -> {mendu_dir}")
    print(f"  circuitry -> {circuitry_dir}")

    _run_mendu(pathlib.Path(args.mendu_root), args.steps, mendu_dir)
    _run_circuitry(args.steps, circuitry_dir)

    print("Both pipelines completed. Tolerance comparison wires up in P2.")
    print(f"Output: {out_base}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
