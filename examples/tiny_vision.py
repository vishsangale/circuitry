# examples/tiny_vision.py
"""Runnable vision example. ``python examples/tiny_vision.py``."""

from __future__ import annotations

import pathlib
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from circuitry import Recorder  # noqa: E402


class TinyCNN(nn.Module):
    def __init__(self, n_classes: int = 10) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 8 * 8, 64)
        self.fc2 = nn.Linear(64, n_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.relu(self.fc1(x.flatten(1)))
        return self.fc2(x)


def main():
    out = pathlib.Path("runs/tiny_vision")
    out.mkdir(parents=True, exist_ok=True)
    model = TinyCNN()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    rec = Recorder(model, run_dir=out, recipe="vision",
                   writer="tensorboard", every_n_steps=5)
    rec.attach()
    for step in range(50):
        x = torch.randn(8, 3, 16, 16)
        y = torch.randint(0, 10, (8,))
        loss = F.cross_entropy(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        rec.step(step, loss=float(loss.item()))
    rec.detach()
    print(f"tensorboard --logdir {out}")


if __name__ == "__main__":
    main()
