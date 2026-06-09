"""Attention head knockout — (layer × head) importance heatmap.

Ablates individual attention heads by zeroing each head module's output and
measures the drop in the target metric.  Returns an (n_layers, n_heads)
importance matrix where importance[l, h] = clean_score − knockout_score[l, h].

Based on: Michel et al. 2019, NeurIPS "Are Sixteen Heads Really Better Than One?"
          Voita et al. 2019, ACL "Analyzing Multi-Head Self-Attention: Specialized
          Heads Do the Heavy Lifting, the Rest Can Be Pruned".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.nn as nn
from torch import Tensor

__all__ = ["HeadKnockoutResult", "HeadKnockoutRunner"]

_Inputs = Any


@dataclass
class HeadKnockoutResult:
    """Per-head importance scores from a knockout experiment.

    Attributes:
        importance:      ``(n_layers, n_heads)`` tensor where each cell is
                         ``clean_score − knockout_score``.  Positive = head
                         was helping; negative = head was hurting.
        layer_names:     Human-readable name per layer.
        clean_score:     Metric on the unablated model.
        knockout_scores: ``(n_layers, n_heads)`` metric scores with each head
                         individually zeroed.
    """

    importance: Tensor
    layer_names: list[str]
    clean_score: float
    knockout_scores: Tensor

    def top_heads(self, k: int = 5) -> list[tuple[str, int, float]]:
        """Top-k most important (layer_name, head_idx, importance) triples."""
        n_layers, n_heads = self.importance.shape
        triples: list[tuple[str, int, float]] = [
            (self.layer_names[l], h, self.importance[l, h].item())
            for l in range(n_layers)
            for h in range(n_heads)
        ]
        return sorted(triples, key=lambda t: t[2], reverse=True)[:k]

    def to_markdown(self, *, top_k: int = 20) -> str:
        """Render top-k heads as a Markdown table."""
        rows = self.top_heads(k=top_k)
        lines = [
            f"## Head Knockout (clean={self.clean_score:.4f})",
            "",
            "| Layer | Head | Importance |",
            "| --- | ---: | ---: |",
        ]
        for name, head, score in rows:
            lines.append(f"| {name} | {head} | {score:+.4f} |")
        return "\n".join(lines)


class HeadKnockoutRunner:
    """Ablate individual attention heads and measure metric importance.

    For each ``(layer, head)`` pair, registers a forward hook that zeros the
    head module's output, runs the model on the given inputs, and records the
    resulting metric value.  The importance matrix is
    ``clean_score − knockout_scores``.

    Args:
        model:        PyTorch model (frozen during run).
        head_modules: ``head_modules[layer][head]`` — the ``nn.Module`` whose
                      output represents that head's contribution.  Zeroing its
                      output effectively removes the head.  Any shape is
                      accepted; the hook replaces the output with zeros of the
                      same shape.
        layer_names:  Optional human-readable names for each layer.
                      Defaults to ``"layer_0"``, ``"layer_1"``, etc.
    """

    def __init__(
        self,
        model: nn.Module,
        head_modules: list[list[nn.Module]],
        *,
        layer_names: list[str] | None = None,
    ) -> None:
        if not head_modules:
            raise ValueError("HeadKnockoutRunner: head_modules must be non-empty")
        n_layers = len(head_modules)
        self._model = model
        self._head_modules = head_modules
        self._layer_names = (
            list(layer_names)
            if layer_names is not None
            else [f"layer_{i}" for i in range(n_layers)]
        )
        if len(self._layer_names) != n_layers:
            raise ValueError(
                f"HeadKnockoutRunner: layer_names length {len(self._layer_names)} "
                f"!= n_layers {n_layers}"
            )

    def run(
        self,
        inputs: _Inputs,
        metric: Callable[[Any], float],
    ) -> HeadKnockoutResult:
        """Run the head knockout experiment.

        Args:
            inputs: Model inputs (Tensor or dict).
            metric: ``(model_output) → float`` — the quantity to track.
                    ``importance[l, h]`` is positive when head (l, h)
                    contributes positively to the metric.

        Returns:
            :class:`HeadKnockoutResult` with ``(n_layers, n_heads)`` scores.
        """
        model = self._model
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

        n_layers = len(self._head_modules)
        n_heads = max(len(row) for row in self._head_modules)

        # ── Baseline: clean forward pass ─────────────────────────────────────
        with torch.no_grad():
            clean_out = _run_model(model, inputs)
        clean_score = metric(clean_out)

        # ── Grid: zero each head and measure ─────────────────────────────────
        knockout_scores = torch.full((n_layers, n_heads), float("nan"))

        for l_idx, heads in enumerate(self._head_modules):
            for h_idx, module in enumerate(heads):

                def _make_zero_hook() -> Callable:
                    def hook(mod: nn.Module, inp: tuple, out: Any) -> Any:  # noqa: ARG001
                        if isinstance(out, tuple):
                            return (torch.zeros_like(out[0]),) + out[1:]
                        return torch.zeros_like(out)
                    return hook

                h = module.register_forward_hook(_make_zero_hook())
                with torch.no_grad():
                    ko_out = _run_model(model, inputs)
                h.remove()
                knockout_scores[l_idx, h_idx] = metric(ko_out)

        # Replace NaN cells (unset heads when rows differ in length) with clean_score
        # so importance = 0 for missing heads
        knockout_scores = torch.nan_to_num(knockout_scores, nan=clean_score)

        importance = clean_score - knockout_scores

        return HeadKnockoutResult(
            importance=importance,
            layer_names=self._layer_names,
            clean_score=clean_score,
            knockout_scores=knockout_scores,
        )


def _run_model(model: nn.Module, inputs: _Inputs) -> Any:
    if isinstance(inputs, Tensor):
        return model(inputs)
    if isinstance(inputs, dict):
        return model(**inputs)
    return model(inputs)
