"""circuitry — mechanistic-interpretability diagnostics for PyTorch (live during training or post-hoc on a checkpoint).

Public surface re-exports below are the stable top-level API. Anything not re-exported
here is an internal implementation detail and may change without notice. (The patching
pillar — including SAEFeatureRunner (v1.5) and SAEFeatureEdgeRunner / FeatureACDCRunner
(v1.6) — is reached via ``circuitry.patching``. v1.7 extended SAE circuits to
``mlp_out`` / ``attn_out`` sites, enabled the TransformerLens backend, and added an
integrated-gradients variant; ``resid_post`` + HF results are byte-for-byte identical.)
"""

from circuitry.core.activation import repr_drift, token_similarity
from circuitry.core.inventory import ModelInventory, ParameterRecord
from circuitry.core.lens import LayerPrediction, logit_lens_distributions
from circuitry.core.steer import steer_vector
from circuitry.core.weight import direction_cosine, update_delta
from circuitry.patching.steer import apply_steer
from circuitry.recipes import Recipe, register_recipe
from circuitry.recipes._discovery import discover
from circuitry.recorder.hooks import HookPoint, StepContext, TensorSource
from circuitry.recorder.live import Recorder
from circuitry.recorder.report import build_report
from circuitry.recorder.scan import scan_run
from circuitry.writers.base import MetricWriter

__version__ = "1.22.0"

__all__ = [
    "HookPoint",
    "LayerPrediction",
    "MetricWriter",
    "ModelInventory",
    "ParameterRecord",
    "Recipe",
    "Recorder",
    "StepContext",
    "TensorSource",
    "__version__",
    "apply_steer",
    "build_report",
    "direction_cosine",
    "discover",
    "logit_lens_distributions",
    "register_recipe",
    "repr_drift",
    "scan_run",
    "steer_vector",
    "token_similarity",
    "update_delta",
]
