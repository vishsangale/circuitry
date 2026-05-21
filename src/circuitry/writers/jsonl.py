"""JSONL writer — one JSON line per scalar/text; tensors and images dumped to
side files under <run_dir>/circuitry/artifacts/.

Zero non-stdlib deps beyond numpy / torch.
"""

from __future__ import annotations

import json
import pathlib
import re
from typing import IO

import numpy as np
import torch

_SAFE = re.compile(r"[^a-zA-Z0-9_.-]+")


def _slug(tag: str) -> str:
    return _SAFE.sub("_", tag).strip("_") or "tag"


class JsonlWriter:
    def __init__(self, run_dir: str | pathlib.Path) -> None:
        self._run_dir = pathlib.Path(run_dir)
        self._run_dir.mkdir(parents=True, exist_ok=True)
        self._artifacts = self._run_dir / "circuitry" / "artifacts"
        self._artifacts.mkdir(parents=True, exist_ok=True)
        self._fh: IO[str] = (self._run_dir / "metrics.jsonl").open("a", buffering=1)

    def _emit(self, record: dict) -> None:
        self._fh.write(json.dumps(record) + "\n")

    def add_scalar(self, tag: str, value: float, step: int) -> None:
        self._emit({"tag": tag, "value": float(value), "step": int(step), "kind": "scalar"})

    def add_histogram(self, tag: str, values: torch.Tensor, step: int) -> None:
        arr = torch.as_tensor(values).detach().cpu().numpy()
        out = self._artifacts / f"{_slug(tag)}-step{step:09d}.npy"
        np.save(out, arr)
        self._emit({"tag": tag, "path": str(out.relative_to(self._run_dir)),
                    "step": int(step), "kind": "histogram"})

    def add_image(self, tag: str, image: torch.Tensor, step: int,
                  dataformats: str = "CHW") -> None:
        arr = torch.as_tensor(image).detach().cpu().numpy()
        out = self._artifacts / f"{_slug(tag)}-step{step:09d}.npy"
        np.save(out, arr)
        self._emit({"tag": tag, "path": str(out.relative_to(self._run_dir)),
                    "step": int(step), "kind": "image", "dataformats": dataformats})

    def add_text(self, tag: str, text: str, step: int) -> None:
        self._emit({"tag": tag, "text": text, "step": int(step), "kind": "text"})

    def flush(self) -> None:
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()
