"""Activation patch grid — per-(layer, position) causal attribution heatmap.

Runs activation patching across all (layer, position) pairs in a single
experiment, returning a (n_layers, seq_len) recovery heatmap.  This is the
standard "residual stream patching" grid used in circuit-analysis papers
(Wang et al. 2022 IOI, Hanna et al. 2023 greater-than) to identify which
(layer, position) combination stores the information needed for a behaviour.

For each cell (l, p), the runner:
  1. Replaces position p of layer l's output with the clean activation.
  2. Runs the corrupted forward pass with this single patch.
  3. Records normalised recovery = (patched − corrupted) / (clean − corrupted).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.nn as nn
from torch import Tensor

__all__ = ["PatchGridResult", "PatchGridRunner"]

_Inputs = Any


@dataclass
class PatchGridResult:
    """Per-(layer, position) causal recovery heatmap.

    Attributes:
        recovery:        ``(n_layers, seq_len)`` recovery scores, normalised so
                         0 = no effect and 1 = full recovery.
        layer_names:     Human-readable name for each patched module.
        clean_score:     Metric value on the unperturbed inputs.
        corrupted_score: Metric value on the corrupted inputs (no patch).
    """

    recovery: Tensor
    layer_names: list[str]
    clean_score: float
    corrupted_score: float

    def top_sites(self, k: int = 5) -> list[tuple[str, int, float]]:
        """Top-k (layer_name, position, score) triples by recovery, descending."""
        n_layers, seq_len = self.recovery.shape
        triples: list[tuple[str, int, float]] = []
        for l_idx in range(n_layers):
            for p_idx in range(seq_len):
                triples.append((
                    self.layer_names[l_idx],
                    p_idx,
                    self.recovery[l_idx, p_idx].item(),
                ))
        return sorted(triples, key=lambda t: t[2], reverse=True)[:k]

    def to_markdown(self, *, top_k: int = 20) -> str:
        """Render top-k sites as a Markdown table."""
        rows = self.top_sites(k=top_k)
        lines = [
            f"## Patch Grid"
            f" (clean={self.clean_score:.4f}, corrupted={self.corrupted_score:.4f})",
            "",
            "| Layer | Position | Recovery |",
            "| --- | ---: | ---: |",
        ]
        for name, pos, score in rows:
            lines.append(f"| {name} | {pos} | {score:.4f} |")
        return "\n".join(lines)


class PatchGridRunner:
    """Runs the (layer × position) activation-patching grid experiment.

    For each module, patches only the activation at a single sequence position
    back from the clean run into the corrupted run, then measures how much the
    target metric recovers.  The result is a ``(n_layers, seq_len)`` heatmap
    identifying which (layer, position) pair encodes the relevant information.

    Supply exactly one of *modules* or *module_pattern*.

    Args:
        model:          PyTorch model (frozen during run).
        modules:        Explicit list of ``nn.Module`` instances to probe.
        module_names:   Optional human-readable names parallel to *modules*.
        module_pattern: Regex matched against ``model.named_modules()`` names.
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
                "PatchGridRunner: supply either modules= or module_pattern="
            )
        if modules is not None and module_pattern is not None:
            raise ValueError(
                "PatchGridRunner: supply either modules= or module_pattern=, not both"
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
                    f"PatchGridRunner: no modules matched pattern {module_pattern!r}"
                )
            names, mods = zip(*matched)
            self._modules = list(mods)
            self._names = list(names)

    def run(
        self,
        clean_inputs: _Inputs,
        corrupted_inputs: _Inputs,
        metric: Callable[[Any], float],
    ) -> PatchGridResult:
        """Run the full (layer × position) patching grid.

        Args:
            clean_inputs:     Unperturbed inputs (Tensor or dict).
            corrupted_inputs: Corrupted inputs (Tensor or dict).
            metric:           ``(model_output) → float`` — higher = more
                              recovered.

        Returns:
            :class:`PatchGridResult` with ``(n_layers, seq_len)`` recovery.
        """
        model = self._model
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

        n_layers = len(self._modules)

        # ── Step 1: clean forward — cache each module's full activation ──────
        clean_cache: list[Tensor | None] = [None] * n_layers

        def _make_capture(idx: int) -> Callable:
            def hook(mod: nn.Module, inp: tuple, out: Any) -> None:  # noqa: ARG001
                val = out[0] if isinstance(out, tuple) else out
                clean_cache[idx] = val.detach().clone()
            return hook

        handles = [
            self._modules[i].register_forward_hook(_make_capture(i))
            for i in range(n_layers)
        ]
        with torch.no_grad():
            clean_out = _run_model(model, clean_inputs)
        for h in handles:
            h.remove()
        clean_score = metric(clean_out)

        # ── Step 2: corrupted baseline ────────────────────────────────────────
        with torch.no_grad():
            corrupted_out = _run_model(model, corrupted_inputs)
        corrupted_score = metric(corrupted_out)
        denom = clean_score - corrupted_score

        # Determine seq_len from cached activations (use first non-None)
        seq_len = 1
        for cached in clean_cache:
            if cached is not None and cached.ndim == 3:
                seq_len = cached.shape[1]
                break

        # ── Step 3: grid: for each (layer, position) patch and measure ────────
        grid = torch.zeros(n_layers, seq_len, dtype=torch.float32)

        for l_idx, (mod, cached) in enumerate(zip(self._modules, clean_cache)):
            if cached is None:
                continue

            for p_idx in range(seq_len):

                def _make_pos_inject(act: Tensor, pos: int) -> Callable:
                    def hook(m: nn.Module, inp: tuple, out: Any) -> Any:  # noqa: ARG001
                        val = out[0] if isinstance(out, tuple) else out
                        patched = val.clone()
                        if act.ndim == 3 and patched.ndim == 3:
                            patched[:, pos, :] = act[:, pos, :]
                        else:
                            patched = act
                        if isinstance(out, tuple):
                            return (patched,) + out[1:]
                        return patched
                    return hook

                h = mod.register_forward_hook(_make_pos_inject(cached, p_idx))
                with torch.no_grad():
                    patched_out = _run_model(model, corrupted_inputs)
                h.remove()
                patched_score = metric(patched_out)

                if abs(denom) < 1e-8:
                    grid[l_idx, p_idx] = 1.0
                else:
                    grid[l_idx, p_idx] = (patched_score - corrupted_score) / denom

        return PatchGridResult(
            recovery=grid,
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
