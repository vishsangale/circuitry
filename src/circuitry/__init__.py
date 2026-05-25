"""circuitry — mechanistic-interpretability diagnostics for PyTorch (live during training or post-hoc on a checkpoint).

Public surface re-exports below are the v0.6.0 stable API. Anything not re-exported
here is an internal implementation detail and may change without notice.
"""

from circuitry.core.activation import token_similarity
from circuitry.core.inventory import ModelInventory, ParameterRecord
from circuitry.core.weight import direction_cosine, update_delta
from circuitry.recipes import Recipe, register_recipe
from circuitry.recipes._discovery import discover
from circuitry.recorder.hooks import HookPoint, StepContext, TensorSource
from circuitry.recorder.live import Recorder
from circuitry.recorder.report import build_report
from circuitry.recorder.scan import scan_run
from circuitry.writers.base import MetricWriter

__version__ = "1.0.0"

__all__ = [
    "HookPoint",
    "MetricWriter",
    "ModelInventory",
    "ParameterRecord",
    "Recipe",
    "Recorder",
    "StepContext",
    "TensorSource",
    "__version__",
    "build_report",
    "direction_cosine",
    "discover",
    "register_recipe",
    "scan_run",
    "token_similarity",
    "update_delta",
]
