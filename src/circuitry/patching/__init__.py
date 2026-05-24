"""Activation patching — interventional diagnostics. See design spec §2.

This subsystem is opt-in and isolated: every intervention is scoped to a
context manager, model state is restored on exit (including on exception),
and the model stays frozen throughout.
"""
from __future__ import annotations

from circuitry.patching.sites import Site

__all__ = ["Site"]
