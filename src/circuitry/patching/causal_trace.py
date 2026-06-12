"""Causal tracing — layer-by-layer causal effect localisation.

Implements the causal tracing experiment from ROME (Meng et al. 2022):
run the model on clean and corrupted inputs; for each candidate module,
patch the clean hidden state back into the corrupted forward pass and measure
metric recovery.  The resulting recovery curve pinpoints which module stores
the information needed to recover the clean prediction.

arXiv:2202.05262
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

__all__ = ["CausalTraceResult", "CausalTraceRunner"]

_Inputs = Any


@dataclass
class CausalTraceResult:
    """Per-layer causal effect of restoring clean activations into a corrupted run.

    Attributes:
        recovery:        ``(n_layers,)`` recovery scores, normalised so that
                         0 = no effect (patch doesn't help) and 1 = full recovery.
                         Values outside [0, 1] are possible when patching
                         overshoots — they are not clipped.
        layer_names:     Human-readable name for each patched module.
        clean_score:     Metric value on the clean (unperturbed) inputs.
        corrupted_score: Metric value on the corrupted inputs (no patching).
    """

    recovery: Tensor
    layer_names: list[str]
    clean_score: float
    corrupted_score: float

    def top_layers(self, k: int = 3) -> list[tuple[str, float]]:
        """Return the top-k layers by recovery score, descending."""
        pairs = list(zip(self.layer_names, self.recovery.tolist(), strict=True))
        return sorted(pairs, key=lambda kv: kv[1], reverse=True)[:k]

    def to_markdown(self, *, top_k: int = 20) -> str:
        """Render as a Markdown table."""
        rows = list(zip(self.layer_names, self.recovery.tolist(), strict=True))[:top_k]
        lines = [
            f"## Causal Trace"
            f" (clean={self.clean_score:.4f}, corrupted={self.corrupted_score:.4f})",
            "",
            "| Layer | Recovery |",
            "| --- | ---: |",
        ]
        for name, score in rows:
            lines.append(f"| {name} | {score:.4f} |")
        return "\n".join(lines)


class CausalTraceRunner:
    """Layer-by-layer causal tracing: which module restores the corrupted prediction?

    Runs the ROME causal tracing experiment: for each candidate module, patch the
    clean hidden state back into the corrupted run and measure logit (or metric)
    recovery.  Recovery is normalised to [0, 1]:
    ``recovery = (patched_score − corrupted_score) / (clean_score − corrupted_score)``.

    Supply exactly one of *modules* or *module_pattern*.

    Args:
        model:          PyTorch model (set to eval and frozen during run).
        modules:        Explicit list of ``nn.Module`` instances to probe.
        module_names:   Optional human-readable names parallel to *modules*.
                        Auto-derived when omitted.
        module_pattern: Regex matched against ``model.named_modules()`` names
                        (alternative to explicit *modules*).

    Reference:
        Meng et al. 2022, NeurIPS "Locating and Editing Factual Associations in GPT".
        https://arxiv.org/abs/2202.05262
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        modules: list[nn.Module] | None = None,
        module_names: list[str] | None = None,
        module_pattern: str | None = None,
    ) -> None:
        if modules is None and module_pattern is None:
            raise ValueError(
                "CausalTraceRunner: supply either modules= or module_pattern="
            )
        if modules is not None and module_pattern is not None:
            raise ValueError(
                "CausalTraceRunner: supply either modules= or module_pattern=, not both"
            )
        self._model = model
        if modules is not None:
            self._modules = list(modules)
            self._names = (
                list(module_names)
                if module_names is not None
                else [f"module_{i}" for i in range(len(modules))]
            )
        else:
            pat = re.compile(module_pattern)  # type: ignore[arg-type]
            matched = [
                (name, mod)
                for name, mod in model.named_modules()
                if pat.search(name)
            ]
            if not matched:
                raise ValueError(
                    f"CausalTraceRunner: no modules matched pattern {module_pattern!r}"
                )
            names, mods = zip(*matched, strict=True)
            self._modules = list(mods)
            self._names = list(names)

    def run(
        self,
        clean_inputs: _Inputs,
        corrupted_inputs: _Inputs,
        metric: Callable[[Any], float],
    ) -> CausalTraceResult:
        """Run the causal tracing experiment.

        Args:
            clean_inputs:     Inputs for the unperturbed run (Tensor or dict).
            corrupted_inputs: Inputs for the corrupted run (Tensor or dict).
            metric:           ``(model_output) → float`` — higher = more
                              recovered.  Common: ``lambda out: out[:, -1, id].mean().item()``.

        Returns:
            :class:`CausalTraceResult` with per-layer recovery scores.
        """
        model = self._model
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

        n = len(self._modules)

        # ── Step 1: clean forward — cache each module's output ──────────────
        clean_cache: list[Tensor | None] = [None] * n

        def _make_capture(idx: int) -> Callable:
            def hook(mod: nn.Module, inp: tuple, out: Any) -> None:  # noqa: ARG001
                val = out[0] if isinstance(out, tuple) else out
                clean_cache[idx] = val.detach().clone()
            return hook

        handles = [
            self._modules[i].register_forward_hook(_make_capture(i))
            for i in range(n)
        ]
        with torch.no_grad():
            clean_out = _run_model(model, clean_inputs)
        for h in handles:
            h.remove()
        clean_score = metric(clean_out)

        # ── Step 2: corrupted forward — baseline score ───────────────────────
        with torch.no_grad():
            corrupted_out = _run_model(model, corrupted_inputs)
        corrupted_score = metric(corrupted_out)

        denom = clean_score - corrupted_score

        # ── Step 3: for each module patch clean state into corrupted run ─────
        recovery_scores: list[float] = []

        for mod, cached in zip(self._modules, clean_cache, strict=True):
            if cached is None:
                recovery_scores.append(0.0)
                continue

            def _make_inject(val: Tensor) -> Callable:
                def hook(m: nn.Module, inp: tuple, out: Any) -> Any:  # noqa: ARG001
                    if isinstance(out, tuple):
                        return (val,) + out[1:]
                    return val
                return hook

            h = mod.register_forward_hook(_make_inject(cached))
            with torch.no_grad():
                patched_out = _run_model(model, corrupted_inputs)
            h.remove()
            patched_score = metric(patched_out)

            if abs(denom) < 1e-8:
                # Clean ≈ corrupted — recovery is 1 by convention
                recovery_scores.append(1.0)
            else:
                recovery_scores.append((patched_score - corrupted_score) / denom)

        return CausalTraceResult(
            recovery=torch.tensor(recovery_scores, dtype=torch.float32),
            layer_names=self._names,
            clean_score=clean_score,
            corrupted_score=corrupted_score,
        )


def _run_model(model: nn.Module, inputs: _Inputs) -> Any:
    if isinstance(inputs, Tensor):
        return model(inputs)
    if isinstance(inputs, dict):
        return model(**inputs)
    return model(inputs)
