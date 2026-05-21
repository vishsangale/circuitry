"""Test MetricWriter protocol and RecordingWriter test double."""
from __future__ import annotations

import torch

from circuitry.writers.base import MetricWriter, RecordingWriter


def test_recording_writer_satisfies_protocol():
    w: MetricWriter = RecordingWriter()
    w.add_scalar("loss", 1.5, 1)
    w.add_scalar("loss", 1.2, 2)
    w.add_histogram("grad", torch.arange(10.0), 3)
    w.add_image("kernel", torch.zeros(3, 4, 4), 3, dataformats="CHW")
    w.add_text("note", "hi", 3)
    w.flush()
    w.close()

    assert w.scalars == [("loss", 1.5, 1), ("loss", 1.2, 2)]
    assert w.histograms[0][0] == "grad"
    assert w.images[0] == ("kernel", torch.zeros(3, 4, 4).shape, 3, "CHW")
    assert w.texts == [("note", "hi", 3)]
    assert w.flushed == 1
    assert w.closed
