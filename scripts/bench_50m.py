"""Reference benchmark per docs/design.md §10.

Usage:
    venv/bin/python scripts/bench_50m.py --n-layers 8 --d-model 768 --steps 100

Reports wall-clock ratio with and without circuitry attached. The budget is
≤10% overhead at default settings on a 50M-param model.
"""

from __future__ import annotations

import argparse
import time

import torch
import torch.nn as nn

from circuitry import Recorder


class Attn(nn.Module):
    def __init__(self, d: int) -> None:
        super().__init__()
        for k in ("q_proj", "k_proj", "v_proj", "o_proj"):
            setattr(self, k, nn.Linear(d, d, bias=False))

    def forward(self, x):
        return self.o_proj(self.v_proj(x))


class Mlp(nn.Module):
    def __init__(self, d: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(d, 4 * d, bias=False)
        self.up_proj = nn.Linear(d, 4 * d, bias=False)
        self.down_proj = nn.Linear(4 * d, d, bias=False)

    def forward(self, x):
        return self.down_proj(self.gate_proj(x) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, d: int) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(d)
        self.ln_2 = nn.LayerNorm(d)
        self.attn = Attn(d)
        self.mlp = Mlp(d)

    def forward(self, x):
        x = self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class _Decoder(nn.Module):
    r"""Mirrors HF decoder nesting (`model.layers.N`, `model.norm`) so the stock
    llm recipe's block-output hook (`.*\.layers\.\d+$`) and lens layernorm
    lookup (`model.model.norm`) resolve as they would on a real HF model."""

    def __init__(self, n_layers: int, d: int, vocab: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.layers = nn.ModuleList([Block(d) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d)

    def forward(self, tokens, **kwargs):
        x = self.embed(tokens)
        for b in self.layers:
            x = b(x)
        return self.norm(x)


class TinyTransformer(nn.Module):
    def __init__(self, n_layers: int, d: int, vocab: int = 8192) -> None:
        super().__init__()
        self.model = _Decoder(n_layers, d, vocab)
        self.lm_head = nn.Linear(d, vocab, bias=False)

    def get_output_embeddings(self):
        # Lets logit_lens_kl resolve the unembed at attach time.
        return self.lm_head

    def forward(self, tokens, **kwargs):
        # **kwargs swallows output_attentions=True (injected by the
        # attention_pattern_entropy pre-hook); this synthetic attn has no
        # per-head weights to return, so that diagnostic no-ops by design.
        return self.lm_head(self.model(tokens))


def _run(model, steps: int, recorder: Recorder | None, device: str) -> float:
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    if recorder is not None:
        recorder.attach()
    if device == "cuda":
        torch.cuda.synchronize()  # mandatory: don't time async kernel launches
    t0 = time.perf_counter()
    for s in range(steps):
        tokens = torch.randint(0, 8192, (4, 64), device=device)
        logits = model(tokens)
        loss = logits.mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if recorder is not None:
            recorder.step(s, loss=float(loss.item()))
    if device == "cuda":
        torch.cuda.synchronize()  # flush before stopping the clock
    elapsed = time.perf_counter() - t0
    if recorder is not None:
        recorder.detach()
    return elapsed


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n-layers", type=int, default=8)
    p.add_argument("--d-model", type=int, default=768)
    p.add_argument("--steps", type=int, default=100)
    # Default cadence emits diagnostics every 25 steps (4 emissions over the
    # default 100 steps) — a realistic periodic-diagnostic schedule, not a
    # gate-cost-only measurement.
    p.add_argument("--every-n-steps", type=int, default=25)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                   choices=["cpu", "cuda"])
    p.add_argument("--run-dir", default="runs/bench")
    args = p.parse_args()

    torch.manual_seed(0)
    model = TinyTransformer(args.n_layers, args.d_model).to(args.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params/1e6:.1f}M  device={args.device}  "
          f"every_n_steps={args.every_n_steps} (~{args.steps // args.every_n_steps} emissions)")

    baseline = _run(model, args.steps, recorder=None, device=args.device)
    rec = Recorder(model, run_dir=args.run_dir, recipe="llm",
                   writer="null", every_n_steps=args.every_n_steps, strict=False)
    instrumented = _run(model, args.steps, recorder=rec, device=args.device)

    overhead = (instrumented / baseline) - 1.0
    print(f"baseline:     {baseline:7.2f}s")
    print(f"instrumented: {instrumented:7.2f}s")
    print(f"overhead:     {overhead*100:+5.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
