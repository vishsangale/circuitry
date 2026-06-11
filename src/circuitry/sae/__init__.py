"""SAE primitives. SAELens hard-deps wrapper.

See docs/design.md §4 and the v0.9 spec §4.3.
"""

from circuitry.sae.grad import (
    SUPPORTED_SAE_ARCHITECTURES,
    assert_supported_sae,
    decode_features,
    encode_features,
    sae_decompose,
    sae_influence_scores,
)
from circuitry.sae.labeling import FeatureEvidence, describe_features
from circuitry.sae.loader import load_gemma_scope, load_llama_scope, load_sae
from circuitry.sae.metrics import (
    UNRELIABLE_METRICS,
    sae_downstream_loss,
    sae_reconstruction_error,
    warn_if_unreliable,
)
from circuitry.sae.steer import fgaa_steering_vector

__all__ = [
    "SUPPORTED_SAE_ARCHITECTURES",
    "UNRELIABLE_METRICS",
    "assert_supported_sae",
    "FeatureEvidence",
    "decode_features",
    "describe_features",
    "encode_features",
    "fgaa_steering_vector",
    "load_gemma_scope",
    "load_llama_scope",
    "load_sae",
    "sae_decompose",
    "sae_downstream_loss",
    "sae_influence_scores",
    "sae_reconstruction_error",
    "warn_if_unreliable",
]
