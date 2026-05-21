"""Optional wandb MetricWriter. Install with ``pip install circuitry[wandb]``."""

from __future__ import annotations

from typing import Any

import torch


class WandbWriter:
    def __init__(self, project: str | None = None, run_name: str | None = None,
                 mode: str = "online", **init_kwargs: Any) -> None:
        try:
            import wandb
        except ImportError as e:
            raise ImportError(
                "wandb writer requires the [wandb] extra: "
                "`pip install circuitry[wandb]`"
            ) from e
        self._wandb = wandb
        self._run = wandb.init(project=project, name=run_name, mode=mode, **init_kwargs)

    def add_scalar(self, tag: str, value: float, step: int) -> None:
        self._wandb.log({tag: float(value)}, step=int(step))

    def add_histogram(self, tag: str, values: torch.Tensor, step: int) -> None:
        arr = torch.as_tensor(values).detach().cpu().numpy()
        self._wandb.log({tag: self._wandb.Histogram(arr)}, step=int(step))

    def add_image(self, tag: str, image: torch.Tensor, step: int,
                  dataformats: str = "CHW") -> None:
        arr = torch.as_tensor(image).detach().cpu().numpy()
        if dataformats == "CHW" and arr.ndim == 3:
            arr = arr.transpose(1, 2, 0)
        self._wandb.log({tag: self._wandb.Image(arr)}, step=int(step))

    def add_text(self, tag: str, text: str, step: int) -> None:
        self._wandb.log({tag: text}, step=int(step))

    def flush(self) -> None:
        # wandb flushes asynchronously; nothing required.
        pass

    def close(self) -> None:
        self._wandb.finish()
