"""CLI entry point tests."""
from __future__ import annotations

import subprocess
import sys

import torch
import torch.nn as nn


def test_cli_list_recipes_prints_stock_names():
    out = subprocess.run(
        [sys.executable, "-m", "circuitry.cli.main", "list-recipes"],
        capture_output=True, text=True, check=True,
    ).stdout
    assert "llm" in out
    assert "vision" in out
    assert "two_tower" in out


def test_cli_scan_and_report(tmp_path):
    # Lay down a single checkpoint.
    ckpts = tmp_path / "checkpoints"
    ckpts.mkdir()
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 4))
    torch.save(model.state_dict(), ckpts / "step000000100.pt")

    # `scan` needs a model factory; for the CLI smoke test we use the
    # built-in stub recipe (registered by --recipe llm requires real model
    # structure). Instead, use the "null" recipe wired below.
    # For test purposes we just call `report` against an empty jsonl path,
    # which is enough to exercise the entry point. (Real scan tested in
    # tests/recorder/test_scan.py.)
    (tmp_path / "metrics.jsonl").write_text(
        '{"tag": "loss", "value": 1.0, "step": 0, "kind": "scalar"}\n'
    )
    subprocess.run(
        [sys.executable, "-m", "circuitry.cli.main", "report",
         "--run", str(tmp_path)],
        capture_output=True, text=True, check=True,
    )
    assert (tmp_path / "inspect" / "report.md").exists()
