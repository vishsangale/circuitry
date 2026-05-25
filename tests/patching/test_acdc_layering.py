"""ACDC respects layering: no cli import, transformers/TL stay lazy."""
from __future__ import annotations

import sys


def test_importing_acdc_does_not_import_heavy_optional_deps():
    for mod in ("transformer_lens", "transformers"):
        sys.modules.pop(mod, None)
    import importlib
    importlib.import_module("circuitry.patching.acdc")
    # acdc must not eagerly import the optional backends at module import:
    assert "transformer_lens" not in sys.modules


def test_acdc_exported_from_package():
    from circuitry.patching import ACDCResult, ACDCRunner
    assert ACDCRunner is not None and ACDCResult is not None
