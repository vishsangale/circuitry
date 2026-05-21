from __future__ import annotations

import json
import pathlib

import torch

from circuitry.writers.jsonl import JsonlWriter
from circuitry.writers.null import NullWriter


def test_null_writer_is_silent(tmp_path):
    w = NullWriter()
    w.add_scalar("loss", 1.0, 1)
    w.add_histogram("g", torch.zeros(4), 1)
    w.add_image("k", torch.zeros(3, 4, 4), 1)
    w.add_text("note", "x", 1)
    w.flush()
    w.close()
    assert list(tmp_path.iterdir()) == []


def test_jsonl_writer_writes_scalars_per_line(tmp_path):
    w = JsonlWriter(tmp_path)
    w.add_scalar("loss", 1.5, 1)
    w.add_scalar("loss", 1.2, 2)
    w.flush()
    w.close()
    path = tmp_path / "metrics.jsonl"
    assert path.exists()
    lines = path.read_text().splitlines()
    assert json.loads(lines[0]) == {"tag": "loss", "value": 1.5, "step": 1, "kind": "scalar"}
    assert json.loads(lines[1]) == {"tag": "loss", "value": 1.2, "step": 2, "kind": "scalar"}


def test_jsonl_writer_dumps_histogram_to_artifacts(tmp_path):
    w = JsonlWriter(tmp_path)
    w.add_histogram("grad", torch.arange(8.0), 1)
    w.close()
    art_dir = tmp_path / "circuitry" / "artifacts"
    assert any(p.name.startswith("grad") and p.suffix == ".npy" for p in art_dir.iterdir())
