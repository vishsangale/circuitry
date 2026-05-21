"""TensorBoard MetricWriter (default).

Thin wrapper over torch.utils.tensorboard.SummaryWriter with an optional
background-thread queue (``async_writes=True``) so add_* never blocks the
training step on disk I/O. The async path drains the queue from a single
worker thread to preserve write order.
"""

from __future__ import annotations

import pathlib
import queue
import threading
from typing import Any

import torch
from torch.utils.tensorboard import SummaryWriter

_SENTINEL: Any = object()


class TensorBoardWriter:
    def __init__(self, run_dir: str | pathlib.Path, async_writes: bool = False) -> None:
        self._writer = SummaryWriter(log_dir=str(run_dir))
        self._async = async_writes
        if async_writes:
            self._q: queue.Queue[Any] = queue.Queue()
            self._worker = threading.Thread(target=self._drain, daemon=True)
            self._worker.start()

    def _drain(self) -> None:
        while True:
            item = self._q.get()
            if item is _SENTINEL:
                return
            method, args, kwargs = item
            getattr(self._writer, method)(*args, **kwargs)
            self._q.task_done()

    def _dispatch(self, method: str, *args: Any, **kwargs: Any) -> None:
        if self._async:
            self._q.put((method, args, kwargs))
        else:
            getattr(self._writer, method)(*args, **kwargs)

    def add_scalar(self, tag: str, value: float, step: int) -> None:
        self._dispatch("add_scalar", tag, float(value), int(step))

    def add_histogram(self, tag: str, values: torch.Tensor, step: int) -> None:
        self._dispatch("add_histogram", tag, torch.as_tensor(values), int(step))

    def add_image(self, tag: str, image: torch.Tensor, step: int,
                  dataformats: str = "CHW") -> None:
        self._dispatch("add_image", tag, torch.as_tensor(image), int(step),
                       dataformats=dataformats)

    def add_text(self, tag: str, text: str, step: int) -> None:
        self._dispatch("add_text", tag, text, int(step))

    def flush(self) -> None:
        if self._async:
            self._q.join()
        self._writer.flush()

    def close(self) -> None:
        if self._async:
            self._q.put(_SENTINEL)
            self._worker.join()
        self._writer.close()
