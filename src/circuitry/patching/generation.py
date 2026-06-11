"""Generation-time analysis — trace, steer, and patch across decode steps.

The single-forward runners in this package analyse one forward pass; most
multi-token behaviour (chain-of-thought faithfulness, refusal dynamics,
entropy collapse during decoding) only shows up *across* decode steps.  This
module adds:

  trace_generation       — drive a greedy/custom decode loop and record a
                           per-step trace (chosen token, top-k logits,
                           entropy, per-site activation stats).
  apply_steer_steps      — steer a site only on selected decode steps.
  patch_site_steps       — replace a site's last-position activation on
                           selected decode steps.
  prepare_generation_attribution — build the (clean, corrupted, metric)
                           triple that attributes a *generated* token at
                           step t back through the realized sequence;
                           feed it to any existing runner.
  generation_attribution — convenience wrapper running CausalTraceRunner
                           or PatchGridRunner on that triple.

Model contract: ``model(input_ids)`` returns ``(batch, seq, vocab)`` logits,
an object with a ``.logits`` attribute, or anything *logits_fn* can map to
logits.  The decode loop re-runs the full prefix each step (teacher-forced;
exact, KV-cache-free), so it works on any causal-LM-shaped ``nn.Module`` —
HF or hand-rolled.  The step-indexed context managers count *top-level model
forwards* instead, so they also work inside ``model.generate``-style loops
with a KV cache (step 0 = the prefill forward).
"""
from __future__ import annotations

import contextlib
import math
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

__all__ = [
    "StepRecord",
    "GenerationTrace",
    "trace_generation",
    "apply_steer_steps",
    "patch_site_steps",
    "GenerationAttributionSetup",
    "prepare_generation_attribution",
    "generation_attribution",
]


def _default_logits_fn(output: Any) -> Tensor:
    if isinstance(output, Tensor):
        return output
    logits = getattr(output, "logits", None)
    if logits is None:
        raise TypeError(
            "model output is neither a Tensor nor has .logits — "
            "pass logits_fn= to map the output to (batch, seq, vocab) logits"
        )
    return logits


def _first_tensor(output: Any) -> Tensor:
    return output[0] if isinstance(output, tuple) else output


def _entropy(last_logits: Tensor) -> float:
    logp = torch.log_softmax(last_logits.float(), dim=-1)
    return float(-(logp.exp() * logp).sum(dim=-1).mean().item())


# ---------------------------------------------------------------------------
# Trace
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StepRecord:
    """One decode step of a :class:`GenerationTrace`."""

    step: int
    token_id: int
    top_token_ids: tuple[int, ...]
    top_logits: tuple[float, ...]
    entropy: float
    site_stats: dict[str, dict[str, float]] = field(default_factory=dict)


@dataclass
class GenerationTrace:
    """Per-decode-step trace from :func:`trace_generation`.

    Attributes:
        records: one :class:`StepRecord` per generated token, in order.
        prompt_len: length of the prompt the generation started from.
    """

    records: list[StepRecord]
    prompt_len: int

    @property
    def token_ids(self) -> list[int]:
        """Generated token ids in decode order."""
        return [r.token_id for r in self.records]

    def entropy_series(self) -> list[float]:
        """Next-token-distribution entropy (nats) per decode step."""
        return [r.entropy for r in self.records]

    def site_series(self, site: str, stat: str = "norm") -> list[float]:
        """One site statistic across steps (e.g. ``site_series("blocks.0")``)."""
        return [r.site_stats[site][stat] for r in self.records]

    def to_markdown(self) -> str:
        lines = ["## Generation Trace", ""]
        lines.append(f"- Prompt length: {self.prompt_len}")
        lines.append(f"- Steps: {len(self.records)}")
        lines.append("")
        lines.append("| step | token | entropy | top-1 logit | margin |")
        lines.append("| ---: | ---: | ---: | ---: | ---: |")
        for r in self.records:
            margin = (
                r.top_logits[0] - r.top_logits[1]
                if len(r.top_logits) > 1 else math.nan
            )
            lines.append(
                f"| {r.step} | {r.token_id} | {r.entropy:.4g}"
                f" | {r.top_logits[0]:.4g} | {margin:.4g} |"
            )
        return "\n".join(lines)


def trace_generation(
    model: nn.Module,
    input_ids: Tensor,
    *,
    n_steps: int,
    modules: dict[str, nn.Module] | None = None,
    logits_fn: Callable[[Any], Tensor] | None = None,
    next_token_fn: Callable[[Tensor], int] | None = None,
    top_k: int = 5,
    stop_token_id: int | None = None,
) -> GenerationTrace:
    """Drive a decode loop and record a per-step trace.

    Each step re-runs the model on the full prefix (teacher-forced, exact)
    and appends the chosen token.  No KV cache is used or required.

    Args:
        model: causal-LM-shaped module — ``model(ids)`` → logits (see module
            docstring for the contract).
        input_ids: ``(1, prompt_len)`` int64 prompt. Batch size must be 1.
        n_steps: maximum number of tokens to generate.
        modules: optional ``{name: module}`` map; for each named module the
            trace records last-position output stats (``norm`` / ``mean`` /
            ``std``) at every step.
        logits_fn: maps the model output to ``(batch, seq, vocab)`` logits.
        next_token_fn: ``(vocab,) last-position logits → token id``
            (default: greedy argmax).
        top_k: number of top logits recorded per step.
        stop_token_id: stop early when this token is generated (the stop
            token's step is still recorded).

    Returns:
        :class:`GenerationTrace`.
    """
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError(
            f"trace_generation requires input_ids of shape (1, prompt_len), "
            f"got {tuple(input_ids.shape)}"
        )
    logits_fn = logits_fn or _default_logits_fn
    if next_token_fn is None:
        def next_token_fn(last: Tensor) -> int:  # greedy
            return int(last.argmax(dim=-1).item())

    site_capture: dict[str, dict[str, float]] = {}
    handles = []
    for name, module in (modules or {}).items():
        def _hook(mod: nn.Module, inputs: tuple, output: Any, _name: str = name) -> None:  # noqa: ARG001
            t = _first_tensor(output).detach().float()
            last = t[:, -1, :] if t.ndim >= 3 else t
            site_capture[_name] = {
                "norm": float(last.norm().item()),
                "mean": float(last.mean().item()),
                "std": float(last.std(unbiased=False).item()),
            }
        handles.append(module.register_forward_hook(_hook))

    was_training = model.training
    records: list[StepRecord] = []
    ids = input_ids
    try:
        model.eval()
        with torch.no_grad():
            for step in range(n_steps):
                site_capture.clear()
                last = logits_fn(model(ids))[:, -1, :]
                token = next_token_fn(last[0])
                k = min(top_k, last.shape[-1])
                topk = last[0].float().topk(k)
                records.append(StepRecord(
                    step=step,
                    token_id=token,
                    top_token_ids=tuple(topk.indices.tolist()),
                    top_logits=tuple(topk.values.tolist()),
                    entropy=_entropy(last),
                    site_stats=dict(site_capture),
                ))
                ids = torch.cat(
                    [ids, torch.tensor([[token]], dtype=ids.dtype, device=ids.device)],
                    dim=1,
                )
                if stop_token_id is not None and token == stop_token_id:
                    break
    finally:
        for h in handles:
            h.remove()
        model.train(was_training)
    return GenerationTrace(records=records, prompt_len=input_ids.shape[1])


# ---------------------------------------------------------------------------
# Step-indexed interventions
# ---------------------------------------------------------------------------


def _as_step_set(steps: Iterable[int]) -> Any:
    return steps if isinstance(steps, range) else set(steps)


@contextlib.contextmanager
def _step_counted_hook(
    model: nn.Module,
    module: nn.Module,
    hook: Callable[[Any, int], Any],
) -> Iterator[None]:
    """Register *hook* on *module*; it receives (output, current_step).

    The step counter increments on every top-level forward of *model*
    (step 0 = the first forward inside the context, i.e. the prefill in a
    KV-cached generate loop).  Both hooks are removed on exit.
    """
    counter = {"step": -1}

    def _pre(mod: nn.Module, inputs: tuple) -> None:  # noqa: ARG001
        counter["step"] += 1

    pre_handle = model.register_forward_pre_hook(_pre)

    def _post(mod: nn.Module, inputs: tuple, output: Any) -> Any:  # noqa: ARG001
        return hook(output, counter["step"])

    handle = module.register_forward_hook(_post)
    try:
        yield
    finally:
        pre_handle.remove()
        handle.remove()


@contextlib.contextmanager
def apply_steer_steps(
    model: nn.Module,
    module: nn.Module,
    vector: Tensor,
    *,
    steps: Iterable[int],
    coeff: float = 1.0,
) -> Iterator[None]:
    """Add ``coeff * vector`` to *module*'s output, only on selected decode steps.

    The step-indexed sibling of ``apply_steer``: e.g. ``steps=range(10, 999)``
    steers only after the 10th decode step.  Steps count top-level forwards
    of *model* (prefill = step 0), so this works both with
    :func:`trace_generation` and inside an external ``model.generate`` loop.

    Args:
        model: the model whose forwards define the step clock.
        module: the module to steer (resolve a ``Site`` yourself if needed).
        vector: ``(d_model,)`` steering direction.
        steps: decode steps on which to steer (any int iterable or ``range``).
        coeff: scale factor.
    """
    step_set = _as_step_set(steps)
    v = vector.detach()

    def _hook(output: Any, step: int) -> Any:
        if step not in step_set:
            return output
        t = _first_tensor(output)
        steered = t + coeff * v.to(dtype=t.dtype, device=t.device)
        if isinstance(output, tuple):
            return (steered,) + output[1:]
        return steered

    with _step_counted_hook(model, module, _hook):
        yield


@contextlib.contextmanager
def patch_site_steps(
    model: nn.Module,
    module: nn.Module,
    value: Tensor,
    *,
    steps: Iterable[int],
    position: int = -1,
) -> Iterator[None]:
    """Replace *module*'s output at *position* with *value* on selected steps.

    The step-indexed sibling of ``patch_site``, specialised to one sequence
    position (default: the last, i.e. the position generating the next
    token).  2-D ``(batch, d)`` outputs are replaced wholesale.

    Args:
        model: the model whose forwards define the step clock.
        module: the module to patch.
        value: ``(d_model,)`` replacement activation.
        steps: decode steps on which to patch.
        position: sequence position to replace (default ``-1``).
    """
    step_set = _as_step_set(steps)
    v = value.detach()

    def _hook(output: Any, step: int) -> Any:
        if step not in step_set:
            return output
        t = _first_tensor(output)
        patched = t.clone()
        cast = v.to(dtype=t.dtype, device=t.device)
        if t.ndim >= 3:
            patched[:, position, :] = cast
        else:
            patched[:] = cast
        if isinstance(output, tuple):
            return (patched,) + output[1:]
        return patched

    with _step_counted_hook(model, module, _hook):
        yield


# ---------------------------------------------------------------------------
# Generation attribution
# ---------------------------------------------------------------------------


@dataclass
class GenerationAttributionSetup:
    """Inputs + metric attributing a generated token; feed to any runner.

    ``runner.run(setup.clean_inputs, setup.corrupted_inputs, setup.metric)``
    works with ``CausalTraceRunner``, ``PatchGridRunner``, ``EAPRunner``-style
    metrics, etc.

    Attributes:
        clean_inputs: ``(1, prompt_len + target_step)`` — clean prompt plus
            the realized generated prefix (teacher-forced).
        corrupted_inputs: same generated prefix appended to the corrupted
            prompt.
        target_token_id: the realized token at *target_step* whose logit the
            metric reads.
        target_step: decode step being attributed.
    """

    clean_inputs: Tensor
    corrupted_inputs: Tensor
    target_token_id: int
    target_step: int
    _logits_fn: Callable[[Any], Tensor] = _default_logits_fn

    def metric(self, model_output: Any) -> float:
        """Logit of the realized target token at the last position."""
        logits = self._logits_fn(model_output)
        return float(logits[:, -1, self.target_token_id].mean().item())


def prepare_generation_attribution(
    model: nn.Module,
    clean_ids: Tensor,
    corrupted_ids: Tensor,
    *,
    target_step: int,
    logits_fn: Callable[[Any], Tensor] | None = None,
    next_token_fn: Callable[[Tensor], int] | None = None,
) -> GenerationAttributionSetup:
    """Attribute the token generated at *target_step* back through the prompt.

    Greedy-generates (or *next_token_fn*-generates) ``target_step + 1``
    tokens from *clean_ids*, then builds the teacher-forced input pair: the
    realized generated prefix ``y_0..y_{t-1}`` appended to both the clean and
    the corrupted prompt.  The metric is the last-position logit of the
    realized token ``y_t`` — exactly the single-forward shape every existing
    runner expects, so generation attribution reduces to a standard
    clean/corrupted run over the realized sequence.

    Args:
        model: causal-LM-shaped module.
        clean_ids: ``(1, prompt_len)`` clean prompt.
        corrupted_ids: ``(1, prompt_len)`` corrupted prompt — must have the
            same length as *clean_ids* so positions align.
        target_step: decode step to attribute (0 = first generated token).
        logits_fn / next_token_fn: as in :func:`trace_generation`.
    """
    if clean_ids.shape != corrupted_ids.shape:
        raise ValueError(
            f"clean and corrupted prompts must have the same shape, got "
            f"{tuple(clean_ids.shape)} vs {tuple(corrupted_ids.shape)}"
        )
    trace = trace_generation(
        model, clean_ids, n_steps=target_step + 1,
        logits_fn=logits_fn, next_token_fn=next_token_fn, top_k=1,
    )
    realized = trace.token_ids
    prefix = torch.tensor(
        [realized[:target_step]], dtype=clean_ids.dtype, device=clean_ids.device,
    )
    return GenerationAttributionSetup(
        clean_inputs=torch.cat([clean_ids, prefix], dim=1),
        corrupted_inputs=torch.cat([corrupted_ids, prefix], dim=1),
        target_token_id=realized[target_step],
        target_step=target_step,
        _logits_fn=logits_fn or _default_logits_fn,
    )


def generation_attribution(
    model: nn.Module,
    clean_ids: Tensor,
    corrupted_ids: Tensor,
    *,
    target_step: int,
    runner: str = "causal_trace",
    modules: list[nn.Module] | None = None,
    module_names: list[str] | None = None,
    module_pattern: str | None = None,
    logits_fn: Callable[[Any], Tensor] | None = None,
    next_token_fn: Callable[[Tensor], int] | None = None,
) -> Any:
    """Convenience wrapper: :func:`prepare_generation_attribution` + a runner.

    Args:
        runner: ``"causal_trace"`` (per-layer recovery; default) or
            ``"patch_grid"`` (``(layer, position)`` recovery heatmap).
        modules / module_names / module_pattern: forwarded to the runner.
        Other args as in :func:`prepare_generation_attribution`.

    Returns:
        ``CausalTraceResult`` or ``PatchGridResult``.
    """
    setup = prepare_generation_attribution(
        model, clean_ids, corrupted_ids, target_step=target_step,
        logits_fn=logits_fn, next_token_fn=next_token_fn,
    )
    if runner == "causal_trace":
        from circuitry.patching.causal_trace import CausalTraceRunner
        r: Any = CausalTraceRunner(
            model, modules=modules, module_names=module_names,
            module_pattern=module_pattern,
        )
    elif runner == "patch_grid":
        from circuitry.patching.patch_grid import PatchGridRunner
        r = PatchGridRunner(
            model, modules=modules, module_names=module_names,
            module_pattern=module_pattern,
        )
    else:
        raise ValueError(
            f"unknown runner {runner!r} (expected 'causal_trace' or 'patch_grid')"
        )
    return r.run(setup.clean_inputs, setup.corrupted_inputs, setup.metric)
