"""MetricWriter protocol + a recording test double used in recorder tests.

See docs/design.md §4.5 for the protocol contract.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import torch


@runtime_checkable
class MetricWriter(Protocol):
    """Protocol for metric writers.

    All implementations must support adding scalars, histograms, images, and text,
    with explicit flush and close lifecycle methods.
    """

    def add_scalar(self, tag: str, value: float, step: int) -> None:
        """Record a scalar metric."""
        ...

    def add_histogram(self, tag: str, values: torch.Tensor, step: int) -> None:
        """Record a histogram of tensor values."""
        ...

    def add_image(self, tag: str, image: torch.Tensor, step: int,
                  dataformats: str = "CHW") -> None:
        """Record an image tensor.

        Args:
            tag: Identifier for the image.
            image: Tensor containing image data.
            step: Global step / iteration.
            dataformats: Format string (e.g., "CHW" for channel-height-width).
        """
        ...

    def add_text(self, tag: str, text: str, step: int) -> None:
        """Record text content."""
        ...

    def flush(self) -> None:
        """Flush pending writes to storage."""
        ...

    def close(self) -> None:
        """Close the writer and clean up resources."""
        ...


class RecordingWriter:
    """Captures every call into in-memory lists. For tests only."""

    def __init__(self) -> None:
        self.scalars: list[tuple[str, float, int]] = []
        self.histograms: list[tuple[str, torch.Tensor, int]] = []
        self.images: list[tuple[str, Any, int, str]] = []
        self.texts: list[tuple[str, str, int]] = []
        self.flushed: int = 0
        self.closed: bool = False

    def add_scalar(self, tag: str, value: float, step: int) -> None:
        self.scalars.append((tag, float(value), int(step)))

    def add_histogram(self, tag: str, values: torch.Tensor, step: int) -> None:
        self.histograms.append((tag, values, int(step)))

    def add_image(self, tag: str, image: torch.Tensor, step: int,
                  dataformats: str = "CHW") -> None:
        # Store shape rather than the tensor to keep memory small in tests.
        self.images.append((tag, image.shape, int(step), dataformats))

    def add_text(self, tag: str, text: str, step: int) -> None:
        self.texts.append((tag, text, int(step)))

    def flush(self) -> None:
        self.flushed += 1

    def close(self) -> None:
        self.closed = True
