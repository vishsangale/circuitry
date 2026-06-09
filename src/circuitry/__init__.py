"""circuitry — mechanistic-interpretability diagnostics for PyTorch (live during training or post-hoc on a checkpoint).

Public surface re-exports below are the stable top-level API. Anything not re-exported
here is an internal implementation detail and may change without notice. (The patching
pillar — including SAEFeatureRunner (v1.5) and SAEFeatureEdgeRunner / FeatureACDCRunner
(v1.6) — is reached via ``circuitry.patching``. v1.7 extended SAE circuits to
``mlp_out`` / ``attn_out`` sites, enabled the TransformerLens backend, and added an
integrated-gradients variant; ``resid_post`` + HF results are byte-for-byte identical.
v1.29 added probing & representation geometry primitives: MDL probing, mass-mean probe,
RepE direction, directional ablation, local intrinsic dimensionality, cross-model kernel
alignment, embedding uniformity, and superposition index.)
"""

from circuitry.core.activation import (
    embedding_uniformity,
    kernel_alignment,
    local_intrinsic_dim,
    repr_drift,
    token_similarity,
)
from circuitry.core.dynamics import fourier_feature_alignment, information_bottleneck_score
from circuitry.patching.sae_features import CrosscoderWrapper
from circuitry.core.erase import EraseProjection, leace_erase
from circuitry.core.inventory import ModelInventory, ParameterRecord
from circuitry.core.lens import LayerPrediction, future_lens_kl, logit_lens_distributions
from circuitry.core.probe import (
    LinearProbe,
    MDLResult,
    MassMeanProbe,
    mass_mean_probe,
    mdl_probe,
    train_linear_probe,
    verify_linear_representation,
)
from circuitry.core.steer import directional_ablation, repe_direction, steer_vector
from circuitry.core.weight import direction_cosine, update_delta
from circuitry.patching.das import DASResult, DASRunner
from circuitry.patching.edge_pruning import EdgePruningResult, EdgePruningRunner
from circuitry.patching.hap import HAPRunner
from circuitry.patching.scrubbing import CausalScrubResult, CausalScrubRunner, CircuitHypothesis
from circuitry.patching.steer import apply_ablation, apply_steer
from circuitry.recipes import Recipe, register_recipe
from circuitry.recipes._discovery import discover
from circuitry.recorder.hooks import HookPoint, StepContext, TensorSource
from circuitry.recorder.live import Recorder
from circuitry.recorder.report import build_report
from circuitry.recorder.scan import scan_run
from circuitry.sae.metrics import superposition_index
from circuitry.writers.base import MetricWriter

__version__ = "1.29.0"

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
    "MDLResult",
    "MassMeanProbe",
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
    "apply_ablation",
    "apply_steer",
    "build_report",
    "direction_cosine",
    "directional_ablation",
    "discover",
    "embedding_uniformity",
    "fourier_feature_alignment",
    "future_lens_kl",
    "information_bottleneck_score",
    "kernel_alignment",
    "leace_erase",
    "local_intrinsic_dim",
    "logit_lens_distributions",
    "mass_mean_probe",
    "mdl_probe",
    "register_recipe",
    "repe_direction",
    "repr_drift",
    "scan_run",
    "steer_vector",
    "superposition_index",
    "token_similarity",
    "train_linear_probe",
    "update_delta",
    "verify_linear_representation",
]
