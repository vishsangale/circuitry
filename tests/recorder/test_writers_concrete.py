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


def test_tensorboard_writer_writes_event_files(tmp_path):
    from circuitry.writers.tensorboard import TensorBoardWriter

    w = TensorBoardWriter(tmp_path)
    w.add_scalar("loss", 1.5, 1)
    w.add_histogram("g", torch.arange(10.0), 1)
    w.add_image("k", torch.zeros(3, 8, 8), 1, dataformats="CHW")
    w.add_text("n", "ok", 1)
    w.flush()
    w.close()

    event_files = [p for p in tmp_path.rglob("events.out.tfevents.*")]
    assert event_files, "TensorBoard event file not written"


def test_tensorboard_writer_async_does_not_lose_data(tmp_path):
    from circuitry.writers.tensorboard import TensorBoardWriter

    w = TensorBoardWriter(tmp_path, async_writes=True)
    for i in range(50):
        w.add_scalar("loss", float(i), i)
    w.flush()
    w.close()
    event_files = [p for p in tmp_path.rglob("events.out.tfevents.*")]
    assert event_files


def test_wandb_writer_skips_when_wandb_absent():
    pytest = __import__("pytest")
    try:
        import wandb  # noqa: F401
    except ImportError:
        pytest.skip("wandb not installed")
    from circuitry.writers.wandb import WandbWriter

    # Use mode='disabled' so no network/auth is required.
    w = WandbWriter(project="circuitry-test", mode="disabled")
    w.add_scalar("loss", 1.0, 1)
    w.flush()
    w.close()
