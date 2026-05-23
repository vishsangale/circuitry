"""SAE primitives. SAELens hard-deps wrapper.

See docs/design.md §4 and the v0.9 spec §4.3.
"""

from circuitry.sae.loader import load_sae
from circuitry.sae.metrics import sae_reconstruction_error

__all__ = ["load_sae", "sae_reconstruction_error"]
