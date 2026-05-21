# examples/tiny_two_tower.py
"""Runnable two-tower example. ``python examples/tiny_two_tower.py``."""

from __future__ import annotations

import pathlib
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from circuitry import Recorder  # noqa: E402


class TwoTower(nn.Module):
    def __init__(self, d_in: int = 8, d_out: int = 16) -> None:
        super().__init__()
        self.query_tower = nn.Sequential(
            nn.Linear(d_in, 32), nn.ReLU(), nn.Linear(32, d_out),
        )
        self.item_tower = nn.Sequential(
            nn.Linear(d_in, 32), nn.ReLU(), nn.Linear(32, d_out),
        )

    def forward(self, q, i):
        return (self.query_tower(q) * self.item_tower(i)).sum(-1)


def main():
    out = pathlib.Path("runs/tiny_two_tower")
    out.mkdir(parents=True, exist_ok=True)
    model = TwoTower()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    rec = Recorder(model, run_dir=out, recipe="two_tower",
                   writer="tensorboard", every_n_steps=5)
    rec.attach()
    for step in range(50):
        q = torch.randn(16, 8)
        i_pos = torch.randn(16, 8)
        i_neg = torch.randn(16, 8)
        score_pos = model(q, i_pos)
        score_neg = model(q, i_neg)
        loss = F.softplus(score_neg - score_pos).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        rec.step(step, loss=float(loss.item()))
    rec.detach()
    print(f"tensorboard --logdir {out}")


if __name__ == "__main__":
    main()
