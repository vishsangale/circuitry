"""circuitry — mechanistic-interpretability diagnostics for PyTorch (live during training or post-hoc on a checkpoint).

Public surface re-exports below are the stable top-level API. Anything not re-exported
here is an internal implementation detail and may change without notice. (The patching
pillar — including SAEFeatureRunner (v1.5) and SAEFeatureEdgeRunner / FeatureACDCRunner
(v1.6) — is reached via ``circuitry.patching``. v1.7 extended SAE circuits to
``mlp_out`` / ``attn_out`` sites, enabled the TransformerLens backend, and added an
integrated-gradients variant; ``resid_post`` + HF results are byte-for-byte identical.)
"""

from circuitry.core.activation import repr_drift, token_similarity
from circuitry.core.dynamics import fourier_feature_alignment, information_bottleneck_score
from circuitry.patching.sae_features import CrosscoderWrapper
from circuitry.core.erase import EraseProjection, leace_erase
from circuitry.core.inventory import ModelInventory, ParameterRecord
from circuitry.core.lens import LayerPrediction, future_lens_kl, logit_lens_distributions
from circuitry.core.probe import LinearProbe, train_linear_probe
from circuitry.core.steer import steer_vector
from circuitry.core.weight import direction_cosine, update_delta
from circuitry.patching.das import DASResult, DASRunner
from circuitry.patching.edge_pruning import EdgePruningResult, EdgePruningRunner
from circuitry.patching.hap import HAPRunner
from circuitry.patching.scrubbing import CausalScrubResult, CausalScrubRunner, CircuitHypothesis
from circuitry.patching.steer import apply_steer
from circuitry.recipes import Recipe, register_recipe
from circuitry.recipes._discovery import discover
from circuitry.recorder.hooks import HookPoint, StepContext, TensorSource
from circuitry.recorder.live import Recorder
from circuitry.recorder.report import build_report
from circuitry.recorder.scan import scan_run
from circuitry.writers.base import MetricWriter

__version__ = "1.27.0"

__all__ = [
    "CausalScrubResult",
    "CausalScrubRunner",
    "CircuitHypothesis",
    "CrosscoderWrapper",
    "DASResult",
    "DASRunner",
    "EraseProjection",
    "HookPoint",
    "LayerPrediction",
    "LinearProbe",
    "MetricWriter",
    "ModelInventory",
    "ParameterRecord",
    "Recipe",
    "Recorder",
    "StepContext",
    "TensorSource",
    "__version__",
    "EdgePruningResult",
    "EdgePruningRunner",
    "HAPRunner",
    "apply_steer",
    "build_report",
    "direction_cosine",
    "discover",
    "fourier_feature_alignment",
    "future_lens_kl",
    "information_bottleneck_score",
    "leace_erase",
    "logit_lens_distributions",
    "register_recipe",
    "repr_drift",
    "scan_run",
    "steer_vector",
    "token_similarity",
    "train_linear_probe",
    "update_delta",
]
