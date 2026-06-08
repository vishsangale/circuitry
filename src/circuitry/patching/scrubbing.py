"""Causal Scrubbing — faithfulness scoring via resampling ablations.

Conmy et al. / Redwood Research 2022.
https://www.lesswrong.com/posts/JvZhhzycHu2Yd57RN/causal-scrubbing-a-method-for-rigorously-testing

Given a :class:`CircuitHypothesis` (a set of "circuit" modules + per-module
variable assignments), :class:`CausalScrubRunner` measures how faithfully
the hypothesis explains the model's behaviour:

- Circuit modules keep their **clean** activations (they implement their
  assigned causal variable correctly).
- Non-circuit modules are replaced with **corrupted** activations (resampled
  from a distribution that should not affect the output if the hypothesis is
  correct).

Faithfulness = (metric(scrubbed) − metric(corrupted)) /
               (metric(clean)    − metric(corrupted))

A score near 1.0 means the circuit hypothesis fully explains the behaviour.
A score near 0.0 means the circuit is no better than random ablation.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

_Inputs = Tensor | dict[str, Any]


@dataclass
class CircuitHypothesis:
    """A circuit hypothesis: which modules implement the behaviour of interest.

    Attributes:
        circuit_modules: Modules whose activations should be **kept** (they
            correctly implement their causal variable).
        node_labels: Optional human-readable labels per module (for display /
            per-node scores).
    """

    circuit_modules: list[nn.Module]
    node_labels: dict[nn.Module, str] = field(default_factory=dict)


@dataclass
class CausalScrubResult:
    """Result of a causal-scrubbing run.

    Attributes:
        faithfulness: Primary score in [0, 1] (higher = more faithful).
        scrubbed_metric: Metric value after scrubbing.
        clean_metric: Metric value on the clean run.
        corrupted_metric: Metric value on the corrupted run.
        per_module_delta: Per-module metric drop when that module is
            individually ablated (removed from the circuit).
    """

    faithfulness: float
    scrubbed_metric: float
    clean_metric: float
    corrupted_metric: float
    per_module_delta: dict[str, float] = field(default_factory=dict)


class CausalScrubRunner:
    """Compute a faithfulness score for a :class:`CircuitHypothesis`.

    Usage::

        hyp = CircuitHypothesis(circuit_modules=[model.layers[0].mlp])
        runner = CausalScrubRunner(model)
        result = runner.run(clean_inputs, corrupted_inputs, metric, hyp)
        print(result.faithfulness)   # ~1.0 for a correct circuit
    """

    def __init__(self, model: nn.Module) -> None:
        self._model = model

    def run(
        self,
        clean_inputs: _Inputs,
        corrupted_inputs: _Inputs,
        metric: Callable[[Tensor], Tensor],
        hypothesis: CircuitHypothesis,
        *,
        compute_per_module: bool = True,
    ) -> CausalScrubResult:
        """Run causal scrubbing.

        Args:
            clean_inputs: Inputs for the "clean" (correct) distribution.
            corrupted_inputs: Inputs for the "corrupted" (resampled) distribution.
            metric: ``logits → scalar tensor`` (must be differentiable; use
                ``logit_diff_t`` / ``kl_divergence_t`` or any scalar-returning
                callable).
            hypothesis: The :class:`CircuitHypothesis` to test.
            compute_per_module: If ``True``, additionally compute per-module
                delta scores (cost: one extra forward pass per circuit module).

        Returns:
            :class:`CausalScrubResult` with faithfulness and supporting metrics.
        """
        model = self._model
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

        circuit_set = set(id(m) for m in hypothesis.circuit_modules)

        # ── baseline metrics ───────────────────────────────────────────────
        with torch.no_grad():
            clean_metric = metric(_run_model(model, clean_inputs)).item()
            corrupted_metric = metric(_run_model(model, corrupted_inputs)).item()

        # ── capture corrupted activations for all modules ──────────────────
        corrupted_acts = _capture_all(model, corrupted_inputs)

        # ── scrubbing pass: circuit modules keep clean acts, rest get corrupted
        scrubbed_metric = _run_scrubbed(
            model, clean_inputs, corrupted_acts, circuit_set, metric
        )

        denom = clean_metric - corrupted_metric
        if abs(denom) < 1e-8:
            faithfulness = 1.0 if abs(scrubbed_metric - clean_metric) < 1e-6 else 0.0
        else:
            faithfulness = float((scrubbed_metric - corrupted_metric) / denom)
        faithfulness = max(-1.0, min(2.0, faithfulness))  # soft clamp for display

        # ── per-module delta (optional) ────────────────────────────────────
        per_module_delta: dict[str, float] = {}
        if compute_per_module and hypothesis.circuit_modules:
            for mod in hypothesis.circuit_modules:
                # Remove this one module from the circuit
                reduced_circuit = set(circuit_set) - {id(mod)}
                m_metric = _run_scrubbed(
                    model, clean_inputs, corrupted_acts, reduced_circuit, metric
                )
                label = hypothesis.node_labels.get(mod, repr(mod)[:40])
                per_module_delta[label] = float(scrubbed_metric - m_metric)

        return CausalScrubResult(
            faithfulness=faithfulness,
            scrubbed_metric=float(scrubbed_metric),
            clean_metric=float(clean_metric),
            corrupted_metric=float(corrupted_metric),
            per_module_delta=per_module_delta,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_model(model: nn.Module, inputs: _Inputs) -> Tensor:
    if isinstance(inputs, Tensor):
        return model(inputs)
    return model(**inputs)


def _capture_all(model: nn.Module, inputs: _Inputs) -> dict[int, Tensor]:
    """Capture the output of every nn.Module in a single forward pass."""
    acts: dict[int, Tensor] = {}
    handles = []

    def _hook(mod: nn.Module, inp: tuple, output: object) -> None:  # noqa: ARG001
        out = output[0] if isinstance(output, tuple) else output
        if isinstance(out, Tensor):
            acts[id(mod)] = out.detach()

    for mod in model.modules():
        handles.append(mod.register_forward_hook(_hook))

    with torch.no_grad():
        _run_model(model, inputs)

    for h in handles:
        h.remove()
    return acts


def _run_scrubbed(
    model: nn.Module,
    clean_inputs: _Inputs,
    corrupted_acts: dict[int, Tensor],
    circuit_ids: set[int],
    metric: Callable[[Tensor], Tensor],
) -> float:
    """Run model on clean_inputs, replacing non-circuit modules' outputs
    with the pre-captured corrupted activations."""
    handles = []

    def _make_hook(mod_id: int) -> Callable:
        def hook(mod: nn.Module, inp: tuple, output: object) -> object:  # noqa: ARG001
            corrupted = corrupted_acts.get(mod_id)
            if corrupted is None:
                return output
            if isinstance(output, tuple):
                return (corrupted,) + output[1:]
            return corrupted
        return hook

    # Install hooks only for non-circuit modules that have cached activations
    for mod in model.modules():
        if id(mod) not in circuit_ids and id(mod) in corrupted_acts:
            handles.append(mod.register_forward_hook(_make_hook(id(mod))))

    with torch.no_grad():
        logits = _run_model(model, clean_inputs)

    for h in handles:
        h.remove()

    return metric(logits).item()
