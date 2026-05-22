"""Retrospective scan over checkpoints: rehydrate model state, run the recipe's
weight diagnostics, emit metrics via the chosen ``MetricWriter``.

Checkpoint discovery defaults to ``<run_dir>/checkpoints/step*.pt`` sorted by
filename (which sorts by step under the conventional ``step000000100.pt`` form).
"""

from __future__ import annotations

import pathlib
import re
from collections.abc import Callable

import torch
import torch.nn as nn

from circuitry.recipes import Recipe, get_recipe
from circuitry.recorder.live import Recorder, _resolve_writer
from circuitry.writers.base import MetricWriter

_STEP_RX = re.compile(r"step(\d+)")


def _discover_checkpoints(run_dir: pathlib.Path) -> list[tuple[int, pathlib.Path]]:
    ckpts = sorted((run_dir / "checkpoints").glob("step*.pt"))
    out: list[tuple[int, pathlib.Path]] = []
    for p in ckpts:
        m = _STEP_RX.search(p.stem)
        out.append((int(m.group(1)) if m else 0, p))
    return out


def scan_run(
    run_dir: str | pathlib.Path,
    recipe: str | Recipe,
    out_dir: str | pathlib.Path,
    model_factory: Callable[[], nn.Module],
    writer: MetricWriter | str = "tensorboard",
) -> None:
    """Replay each checkpoint through the recipe's weight diagnostics.

    ``model_factory`` produces a fresh model whose architecture matches the
    checkpoint state-dict; the same model is reused with ``load_state_dict``
    across checkpoints (cheaper than rebuilding).

    ``writer`` defaults to TensorBoard for back-compatibility; pass
    ``"jsonl"`` (or a custom ``MetricWriter`` instance) to make the scan
    output consumable by ``build_report``.
    """
    run_dir = pathlib.Path(run_dir)
    out_dir = pathlib.Path(out_dir)
    ckpts = _discover_checkpoints(run_dir)
    if not ckpts:
        raise FileNotFoundError(
            f"no checkpoints found under {run_dir / 'checkpoints'}"
        )

    recipe = recipe if isinstance(recipe, Recipe) else get_recipe(recipe)
    model = model_factory()
    resolved_writer = _resolve_writer(writer, out_dir)
    rec = Recorder(model, run_dir=out_dir, recipe=recipe,
                   writer=resolved_writer, every_n_steps=1)
    rec.attach()
    try:
        for step, ckpt_path in ckpts:
            sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            model.load_state_dict(sd)
            rec.step(step)
    finally:
        rec.detach()
