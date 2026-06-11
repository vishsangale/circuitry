"""circuitry — mechanistic-interpretability diagnostics for PyTorch (live during training or post-hoc on a checkpoint).

Public surface re-exports below are the stable top-level API. Anything not re-exported
here is an internal implementation detail and may change without notice. (The patching
pillar — including SAEFeatureRunner (v1.5) and SAEFeatureEdgeRunner / FeatureACDCRunner
(v1.6) — is reached via ``circuitry.patching``. v1.7 extended SAE circuits to
``mlp_out`` / ``attn_out`` sites, enabled the TransformerLens backend, and added an
integrated-gradients variant; ``resid_post`` + HF results are byte-for-byte identical.
v1.29 added probing & representation geometry primitives: MDL probing, mass-mean probe,
RepE direction, directional ablation, local intrinsic dimensionality, cross-model kernel
alignment, embedding uniformity, and superposition index.
v1.30 added training diagnostics: update_weight_ratio, finetuning_delta_svd,
spectral_edge_gap, neural_collapse_score, spectral_collapse_rank, emergence_score,
and attention_rollout.
v1.31 added SAE quality & steering: sae_downstream_loss, sae_influence_scores,
fgaa_steering_vector, rlace_erase, p_anneal/hierarchical_topk arch support, and
UNRELIABLE_METRICS guard.
v1.32 added attribution quality: ReLPRunner (LRP-epsilon edge attribution, arXiv:2508.21258),
CertifiedCircuitRunner/CertifiedCircuitResult (randomised subsampling stability, arXiv:2602.22968),
and MIB benchmark additions: load_ravel, load_arithmetic, load_mcqa, mib_circuit_f1,
mib_iia_score (Mueller et al. ICML 2025).
v1.33 added: ITIConfig/fit_iti/apply_iti (Inference-Time Intervention, arXiv:2306.03341),
CDResult/cd_token_contributions (CD-T contextual decomposition, arXiv:2407.00886),
critical_sharpness (Hessian sharpness via HVP power iteration, arXiv:2601.16979),
gradient_subspace_saturation (gradient subspace saturation diagnostic, arXiv:2508.07370).
v1.34 added CLT Attribution Graphs: CLTNode, CLTEdge, CLTGraphResult, CLTGraphRunner
(feature-level EAP over lossless-spliced transcoders, arXiv:2503.10474).
v1.35 added: daam_attribution (DAAM cross-attention aggregation, arXiv:2210.04885),
HyperDASNet/HyperDASResult/HyperDASRunner (input-conditioned alignment search via
hypernetworks, arXiv:2503.10894).
"""

from circuitry.core.activation import (
    embedding_uniformity,
    kernel_alignment,
    local_intrinsic_dim,
    neural_collapse_score,
    repr_drift,
    spectral_collapse_rank,
    token_similarity,
)
from circuitry.core.attention import attention_rollout, daam_attribution
from circuitry.core.attribution import gradient_input_attribution, integrated_gradients
from circuitry.core.feature_geometry import (
    feature_coverage,
    feature_interference,
    feature_spread,
)
from circuitry.core.circuits import (
    composition_scores,
    feature_token_alignment,
    head_composition_score,
    ov_matrix,
    qk_matrix,
    top_embedding_tokens,
    top_logit_tokens,
    top_virtual_connections,
    transcoder_virtual_weights,
)
from circuitry.core.decompose import LogitDecompositionResult, logit_decomposition
from circuitry.core.dynamics import (
    emergence_score,
    fourier_feature_alignment,
    information_bottleneck_score,
)
from circuitry.patching.sae_features import CrosscoderWrapper
from circuitry.core.erase import EraseProjection, leace_erase, rlace_erase
from circuitry.core.inventory import ModelInventory, ParameterRecord
from circuitry.core.lens import LayerPrediction, future_lens_kl, logit_lens_distributions
from circuitry.core.neuron import NeuronStats, neuron_stats
from circuitry.core.probe import (
    LinearProbe,
    MDLResult,
    MassMeanProbe,
    mass_mean_probe,
    mdl_probe,
    train_linear_probe,
    verify_linear_representation,
)
from circuitry.core.spectral import spectral_edge_gap
from circuitry.core.steer import directional_ablation, repe_direction, steer_vector
from circuitry.core.weight import (
    FinetuningDeltaResult,
    critical_sharpness,
    direction_cosine,
    finetuning_delta_svd,
    gradient_subspace_saturation,
    update_delta,
    update_weight_ratio,
)
from circuitry.patching.causal_trace import CausalTraceResult, CausalTraceRunner
from circuitry.patching.head_knockout import HeadKnockoutResult, HeadKnockoutRunner
from circuitry.patching.mean_ablation import compute_mean_activation, mean_ablation
from circuitry.patching.cd import CDResult, cd_token_contributions
from circuitry.patching.patch_grid import PatchGridResult, PatchGridRunner
from circuitry.patching.clt import CLTEdge, CLTGraphResult, CLTGraphRunner, CLTNode
from circuitry.patching.hyperdas import HyperDASNet, HyperDASResult, HyperDASRunner
from circuitry.patching.iti import ITIConfig, apply_iti, fit_iti
from circuitry.patching.certified import CertifiedCircuitResult, CertifiedCircuitRunner
from circuitry.patching.das import DASResult, DASRunner
from circuitry.patching.edge_pruning import EdgePruningResult, EdgePruningRunner
from circuitry.patching.hap import HAPRunner
from circuitry.patching.relp import ReLPRunner
from circuitry.patching.scrubbing import CausalScrubResult, CausalScrubRunner, CircuitHypothesis
from circuitry.patching.consensus import CircuitConsensus
from circuitry.patching.export import (
    save_html,
    save_neuronpedia_graph,
    to_html,
    to_neuronpedia_graph,
)
from circuitry.patching.steer import apply_ablation, apply_steer
from circuitry.recipes import Recipe, register_recipe
from circuitry.recipes._discovery import discover
from circuitry.recorder.hooks import HookPoint, StepContext, TensorSource
from circuitry.recorder.live import Recorder
from circuitry.recorder.report import build_report
from circuitry.recorder.scan import scan_run
from circuitry.sae.metrics import sae_downstream_loss, superposition_index
from circuitry.sae.grad import sae_influence_scores
from circuitry.sae.labeling import FeatureEvidence, describe_features
from circuitry.sae.steer import fgaa_steering_vector
from circuitry.writers.base import MetricWriter

__version__ = "1.42.0"

__all__ = [
    "CDResult",
    "CLTEdge",
    "HyperDASNet",
    "HyperDASResult",
    "HyperDASRunner",
    "CLTGraphResult",
    "CLTGraphRunner",
    "CLTNode",
    "CausalScrubResult",
    "CausalScrubRunner",
    "CausalTraceResult",
    "CausalTraceRunner",
    "CertifiedCircuitResult",
    "CircuitConsensus",
    "FeatureEvidence",
    "composition_scores",
    "describe_features",
    "feature_token_alignment",
    "head_composition_score",
    "ov_matrix",
    "qk_matrix",
    "top_embedding_tokens",
    "top_logit_tokens",
    "top_virtual_connections",
    "transcoder_virtual_weights",
    "PatchGridResult",
    "PatchGridRunner",
    "CertifiedCircuitRunner",
    "ITIConfig",
    "CircuitHypothesis",
    "CrosscoderWrapper",
    "DASResult",
    "DASRunner",
    "EraseProjection",
    "FinetuningDeltaResult",
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
    "ReLPRunner",
    "StepContext",
    "TensorSource",
    "__version__",
    "EdgePruningResult",
    "EdgePruningRunner",
    "HAPRunner",
    "HeadKnockoutResult",
    "HeadKnockoutRunner",
    "NeuronStats",
    "neuron_stats",
    "compute_mean_activation",
    "feature_coverage",
    "feature_interference",
    "feature_spread",
    "mean_ablation",
    "apply_ablation",
    "apply_iti",
    "apply_steer",
    "cd_token_contributions",
    "critical_sharpness",
    "fit_iti",
    "gradient_subspace_saturation",
    "attention_rollout",
    "build_report",
    "LogitDecompositionResult",
    "daam_attribution",
    "gradient_input_attribution",
    "integrated_gradients",
    "logit_decomposition",
    "direction_cosine",
    "directional_ablation",
    "discover",
    "emergence_score",
    "embedding_uniformity",
    "finetuning_delta_svd",
    "fourier_feature_alignment",
    "future_lens_kl",
    "information_bottleneck_score",
    "kernel_alignment",
    "fgaa_steering_vector",
    "leace_erase",
    "rlace_erase",
    "sae_downstream_loss",
    "sae_influence_scores",
    "local_intrinsic_dim",
    "logit_lens_distributions",
    "mass_mean_probe",
    "mdl_probe",
    "neural_collapse_score",
    "register_recipe",
    "repe_direction",
    "repr_drift",
    "save_html",
    "save_neuronpedia_graph",
    "scan_run",
    "spectral_collapse_rank",
    "spectral_edge_gap",
    "steer_vector",
    "superposition_index",
    "to_html",
    "to_neuronpedia_graph",
    "token_similarity",
    "train_linear_probe",
    "update_delta",
    "update_weight_ratio",
    "verify_linear_representation",
]
