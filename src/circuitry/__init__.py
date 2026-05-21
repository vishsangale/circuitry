"""circuitry — training-time mechanistic-interpretability diagnostics for PyTorch.

Public surface re-exports below are the v0.1.0 stable API. Anything not re-exported
here is an internal implementation detail and may change without notice.
"""

from circuitry.recipes import Recipe, register_recipe
from circuitry.recorder.hooks import HookPoint, StepContext, TensorSource
from circuitry.recorder.live import Recorder
from circuitry.recorder.report import build_report
from circuitry.recorder.scan import scan_run
from circuitry.writers.base import MetricWriter

__version__ = "0.1.0.dev0"

__all__ = [
    "HookPoint",
    "MetricWriter",
    "Recipe",
    "Recorder",
    "StepContext",
    "TensorSource",
    "__version__",
    "build_report",
    "register_recipe",
    "scan_run",
]
