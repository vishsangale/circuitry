"""``TunedLens`` — a serializable container of per-layer affine translators.

A tuned lens (Belrose et al. 2023) is a set of per-layer affine maps
``h ↦ A_ℓ h + b_ℓ`` (``A_ℓ`` init identity) trained to align each layer's
residual stream with the final-logit frame. This container holds the fitted
translators plus enough metadata to (a) map them back to the right blocks and
(b) refuse to apply a lens fitted on a different model.

Pure data + (de)serialization — no optimizer, no model. The trainer lives in
``circuitry.tuned_lens.fit``; the apply-time math is ``core.lens.tuned_lens_kl``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor


@dataclass
class TunedLens:
    """Fitted per-layer affine translators for a tuned lens.

    Attributes:
        translators: one ``(A, b)`` pair per fitted layer, in ``layers`` order.
            ``A`` is ``(d_model, d_model)``, ``b`` is ``(d_model,)``.
        layers: the block indices the translators map to (the integer ``N`` in
            ``...layers.N``), aligned with ``translators``.
        d_model: residual-stream width the translators were fitted for.
        model_fingerprint: a stable hash of the source model's parameter
            shapes; guards against applying the lens to a different model.
    """

    translators: list[tuple[Tensor, Tensor]]
    layers: list[int]
    d_model: int
    model_fingerprint: str

    def __post_init__(self) -> None:
        if len(self.translators) != len(self.layers):
            raise ValueError(
                f"TunedLens: {len(self.translators)} translators but "
                f"{len(self.layers)} layers — must match."
            )
        for i, (a, b) in enumerate(self.translators):
            if tuple(a.shape) != (self.d_model, self.d_model):
                raise ValueError(
                    f"TunedLens: translator[{i}] A has shape {tuple(a.shape)}, "
                    f"expected ({self.d_model}, {self.d_model})."
                )
            if tuple(b.shape) != (self.d_model,):
                raise ValueError(
                    f"TunedLens: translator[{i}] b has shape {tuple(b.shape)}, "
                    f"expected ({self.d_model},)."
                )

    def translator_for(self, layer: int) -> tuple[Tensor, Tensor] | None:
        """Return the ``(A, b)`` for block ``layer``, or ``None`` if unfitted."""
        try:
            return self.translators[self.layers.index(layer)]
        except ValueError:
            return None

    def save(self, path: str | Path) -> None:
        """Serialize to ``path`` via ``torch.save`` (CPU tensors)."""
        payload: dict[str, Any] = {
            "format": "circuitry.TunedLens.v1",
            "translators": [
                (a.detach().cpu(), b.detach().cpu()) for a, b in self.translators
            ],
            "layers": list(self.layers),
            "d_model": self.d_model,
            "model_fingerprint": self.model_fingerprint,
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str | Path) -> TunedLens:
        """Load a ``TunedLens`` previously written by :meth:`save`."""
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or payload.get("format") != "circuitry.TunedLens.v1":
            raise ValueError(
                f"{path}: not a circuitry.TunedLens.v1 file "
                f"(got {payload.get('format') if isinstance(payload, dict) else type(payload)})."
            )
        return cls(
            translators=[(a, b) for a, b in payload["translators"]],
            layers=list(payload["layers"]),
            d_model=int(payload["d_model"]),
            model_fingerprint=str(payload["model_fingerprint"]),
        )
