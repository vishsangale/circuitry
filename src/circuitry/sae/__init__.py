"""SAE primitives. SAELens hard-deps wrapper.

See docs/design.md §4 and the v0.9 spec §4.3.
"""

from circuitry.sae.grad import (
    SUPPORTED_SAE_ARCHITECTURES,
    assert_supported_sae,
    decode_features,
    encode_features,
    sae_decompose,
)
from circuitry.sae.loader import load_gemma_scope, load_llama_scope, load_sae
from circuitry.sae.metrics import sae_reconstruction_error

__all__ = [
    "SUPPORTED_SAE_ARCHITECTURES",
    "assert_supported_sae",
    "decode_features",
    "encode_features",
    "load_gemma_scope",
    "load_llama_scope",
    "load_sae",
    "sae_decompose",
    "sae_reconstruction_error",
]
