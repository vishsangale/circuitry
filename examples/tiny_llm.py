# examples/tiny_llm.py
"""Runnable LLM example. ``python examples/tiny_llm.py``."""

from __future__ import annotations

import pathlib
import sys

import torch
import torch.nn as nn

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from circuitry import Recorder  # noqa: E402


class TinyAttn(nn.Module):
    def __init__(self, d=16):
        super().__init__()
        for k in ("q_proj", "k_proj", "v_proj", "o_proj"):
            setattr(self, k, nn.Linear(d, d, bias=False))

    def forward(self, x):
        return self.o_proj(self.v_proj(x))


class TinyMlp(nn.Module):
    def __init__(self, d=16):
        super().__init__()
        self.gate_proj = nn.Linear(d, d * 2, bias=False)
        self.up_proj = nn.Linear(d, d * 2, bias=False)
        self.down_proj = nn.Linear(d * 2, d, bias=False)

    def forward(self, x):
        return self.down_proj(self.gate_proj(x) * self.up_proj(x))


class TinyBlock(nn.Module):
    def __init__(self, d=16):
        super().__init__()
        self.attn = TinyAttn(d)
        self.mlp = TinyMlp(d)
        self.ln_1 = nn.LayerNorm(d)
        self.ln_2 = nn.LayerNorm(d)

    def forward(self, x):
        x = self.attn(self.ln_1(x))
        x = self.mlp(self.ln_2(x))
        return x


class TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(64, 16)
        self.block = TinyBlock(16)
        self.lm_head = nn.Linear(16, 64, bias=False)

    def forward(self, tokens):
        x = self.embed(tokens)
        x = self.block(x)
        return self.lm_head(x)


def main():
    out = pathlib.Path("runs/tiny_llm")
    out.mkdir(parents=True, exist_ok=True)
    model = TinyLM()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    rec = Recorder(model, run_dir=out, recipe="llm",
                   writer="tensorboard", every_n_steps=5)
    rec.attach()
    for step in range(50):
        tokens = torch.randint(0, 64, (8, 16))
        loss = model(tokens).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
        rec.step(step, loss=float(loss.item()))
    rec.detach()
    print(f"tensorboard --logdir {out}")


if __name__ == "__main__":
    main()
