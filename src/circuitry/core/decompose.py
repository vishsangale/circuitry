"""Logit decomposition — per-component attribution over the residual stream.

logit_decomposition projects each residual-stream contribution onto the
logit-difference direction W_U[:,token_a] - W_U[:,token_b], giving a scalar
"how much did this component push the model toward token_a over token_b?"

See docs/design.md §4 and the Transformer Circuits framework:
Elhage et al. 2021, https://transformer-circuits.pub/2021/framework/index.html
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor


@dataclass
class LogitDecompositionResult:
    """Per-component logit contributions for a token pair at a given position.

    Attributes:
        scores:   Mapping component name → scalar contribution to
                  ``logit[token_a] − logit[token_b]``.
        token_a:  "Positive" token index.
        token_b:  "Negative" / reference token index.
        position: Sequence position that was analysed.
    """

    scores: dict[str, float]
    token_a: int
    token_b: int
    position: int

    def ranked(self) -> list[tuple[str, float]]:
        """Components sorted by absolute contribution, descending."""
        return sorted(self.scores.items(), key=lambda kv: abs(kv[1]), reverse=True)

    def top_k(self, k: int) -> list[tuple[str, float]]:
        """Top-k components by absolute contribution."""
        return self.ranked()[:k]

    def to_markdown(self, *, top_k: int = 20) -> str:
        """Render as a Markdown table (top-k rows)."""
        rows = self.top_k(top_k)
        lines = [
            f"## Logit Decomposition"
            f" (token_a={self.token_a}, token_b={self.token_b}, pos={self.position})",
            "",
            "| Component | Contribution |",
            "| --- | ---: |",
        ]
        for name, score in rows:
            lines.append(f"| {name} | {score:+.4f} |")
        return "\n".join(lines)


def logit_decomposition(
    components: dict[str, Any],
    unembed: Any,
    token_a: int,
    token_b: int,
    *,
    position: int = -1,
    ln_scale: Any | None = None,
    ln_bias: Any | None = None,
) -> LogitDecompositionResult:
    """Decompose logit[token_a] − logit[token_b] into per-component contributions.

    Each entry in *components* is a residual-stream contribution of shape
    ``(batch, seq, d_model)`` or ``(batch, d_model)`` (already position-selected).
    The entries should sum to the full residual stream at *position* (complete
    decomposition) so that ``Σ scores[c] ≈ logit[token_a] − logit[token_b]``.

    When ``ln_scale`` is provided the final layer-norm is approximated linearly:
    ``LN(x) ≈ (x − mean(x)) / σ_total · ln_scale + ln_bias``
    where ``σ_total`` is computed from the *sum* of all components.

    Args:
        components: Mapping ``name → Tensor`` of residual-stream contributions.
                    Each tensor: ``(batch, seq, d_model)`` or ``(batch, d_model)``.
                    Batch and (if present) sequence axes are mean-pooled.
        unembed:    ``(d_model, vocab_size)`` unembedding matrix W_U.
        token_a:    Vocabulary index — "positive" token.
        token_b:    Vocabulary index — "negative" / reference token.
        position:   Sequence position to analyse (default −1 = last token).
                    Ignored when component tensors are already 2-D.
        ln_scale:   Optional ``(d_model,)`` scale from the final LayerNorm.
                    When provided, the linear LN approximation is applied.
        ln_bias:    Optional ``(d_model,)`` bias from the final LayerNorm.
                    Used only when *ln_scale* is provided.

    Returns:
        :class:`LogitDecompositionResult` with a scalar score per component.

    Reference:
        Elhage et al. 2021 "A Mathematical Framework for Transformer Circuits".
        https://transformer-circuits.pub/2021/framework/index.html
    """
    W_U = torch.as_tensor(unembed).detach().to(torch.float32)   # (d_model, vocab)
    direction = W_U[:, token_a] - W_U[:, token_b]               # (d_model,)

    def _to_vec(t: Any) -> Tensor:
        v = torch.as_tensor(t).detach().to(torch.float32)
        if v.ndim == 3:
            v = v[:, position, :]   # (batch, d_model)
        return v.mean(dim=0)        # (d_model,)

    vecs: dict[str, Tensor] = {name: _to_vec(c) for name, c in components.items()}

    if not vecs:
        return LogitDecompositionResult(
            scores={}, token_a=token_a, token_b=token_b, position=position
        )

    if ln_scale is not None:
        scale = torch.as_tensor(ln_scale).detach().to(torch.float32)  # (d_model,)
        bias_vec = (
            torch.as_tensor(ln_bias).detach().to(torch.float32)
            if ln_bias is not None
            else torch.zeros_like(scale)
        )

        # σ_total from the full residual (sum of all components)
        full = sum(vecs.values())
        centered_full = full - full.mean()
        d = float(full.shape[0])
        sigma_total = centered_full.norm() / (d ** 0.5)
        sigma_total = sigma_total.clamp_min(1e-8)

        # Effective projection direction after folding LN scale
        eff_dir = scale * direction / sigma_total  # (d_model,)

        scores: dict[str, float] = {}
        for name, v in vecs.items():
            centered = v - v.mean()
            scores[name] = (centered @ eff_dir).item()
    else:
        scores = {name: (v @ direction).item() for name, v in vecs.items()}

    return LogitDecompositionResult(
        scores=scores,
        token_a=token_a,
        token_b=token_b,
        position=position,
    )
