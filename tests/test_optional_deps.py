"""fingerprint #2: sae-lens / tensorboard are optional extras, not hard deps.

A lean ``pip install circuitry`` must import the core, build a Recorder, and
record metrics via the no-dep jsonl writer without sae-lens or tensorboard
installed. These tests assert (a) the core import path never eagerly pulls
either package, (b) ``writer="auto"`` falls back to jsonl when tensorboard is
unimportable, and (c) the lazy sites raise a friendly, install-pointing error.
"""
from __future__ import annotations

import builtins
import subprocess
import sys

import pytest


def test_core_import_does_not_pull_optional_deps():
    """In a fresh interpreter, importing the core + constructing a Recorder /
    scan_run entry points must not import sae_lens or tensorboard."""
    code = (
        "import sys\n"
        "import circuitry\n"
        "from circuitry.recorder.live import Recorder\n"
        "from circuitry.recorder.scan import scan_run\n"
        "from circuitry.recipes import get_recipe\n"
        "get_recipe('llm'); get_recipe('recsys')\n"
        "bad = [m for m in sys.modules if 'sae_lens' in m or 'tensorboard' in m]\n"
        "assert not bad, bad\n"
        "print('OK')\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert out.returncode == 0, out.stderr
    assert "OK" in out.stdout


def test_auto_writer_falls_back_to_jsonl_when_tensorboard_missing(tmp_path, monkeypatch):
    """writer="auto" must degrade to the jsonl writer (not crash) when the
    tensorboard import fails."""
    real_import = builtins.__import__

    def _block_tensorboard(name, *args, **kwargs):
        if name.startswith("torch.utils.tensorboard") or name == "tensorboard":
            raise ImportError("simulated: tensorboard not installed")
        return real_import(name, *args, **kwargs)

    # Drop any cached tensorboard writer/module so the import re-runs.
    for mod in list(sys.modules):
        if "tensorboard" in mod:
            sys.modules.pop(mod, None)
    monkeypatch.setattr(builtins, "__import__", _block_tensorboard)

    from circuitry.recorder.live import _make_auto_writer
    from circuitry.writers.jsonl import JsonlWriter

    writer = _make_auto_writer(tmp_path)
    assert isinstance(writer, JsonlWriter)


def test_explicit_tensorboard_writer_raises_friendly_error(tmp_path, monkeypatch):
    """writer="tensorboard" (explicit) must raise a clear, install-pointing
    ImportError when the extra is absent — not a bare ModuleNotFoundError."""
    real_import = builtins.__import__

    def _block_tensorboard(name, *args, **kwargs):
        if name.startswith("torch.utils.tensorboard") or name == "tensorboard":
            raise ImportError("simulated: tensorboard not installed")
        return real_import(name, *args, **kwargs)

    for mod in list(sys.modules):
        if "tensorboard" in mod:
            sys.modules.pop(mod, None)
    monkeypatch.setattr(builtins, "__import__", _block_tensorboard)

    from circuitry.recorder.live import _make_tensorboard_writer

    with pytest.raises(ImportError, match=r"circuitry\[tensorboard\]"):
        _make_tensorboard_writer(tmp_path)


def test_load_sae_raises_friendly_error_when_missing(monkeypatch):
    real_import = builtins.__import__

    def _block_sae(name, *args, **kwargs):
        if name == "sae_lens" or name.startswith("sae_lens."):
            raise ImportError("simulated: sae-lens not installed")
        return real_import(name, *args, **kwargs)

    sys.modules.pop("sae_lens", None)
    monkeypatch.setattr(builtins, "__import__", _block_sae)

    from circuitry.sae.loader import load_sae

    with pytest.raises(ImportError, match=r"circuitry\[sae\]"):
        load_sae("some-release", "some-id")
