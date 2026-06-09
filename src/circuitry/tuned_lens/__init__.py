"""Tuned-lens workflow (v1.10).

Post-hoc fitting + serialization of per-layer affine translators for the tuned
lens (Belrose et al. 2023). The apply-time KL math is the pure primitive
``circuitry.core.lens.tuned_lens_kl``; the recorder applies a fitted lens
forward-only as the ``tuned_lens_kl`` diagnostic.

Layering: this package may import ``core/`` and be imported by ``recorder/`` /
``cli/``; it MUST NOT import ``recorder/`` / ``recipes/`` / ``cli/``.
"""

from __future__ import annotations

from circuitry.tuned_lens.container import TunedLens
from circuitry.tuned_lens.fit import fit_tuned_lens, model_fingerprint

__all__ = ["TunedLens", "fit_tuned_lens", "model_fingerprint"]
