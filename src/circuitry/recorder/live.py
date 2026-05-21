"""LiveRecorder — attach hooks per recipe, snapshot tensors at emit steps,
run diagnostics, write scalars through a MetricWriter.

See docs/design.md §4.2, §4.4, §10, §11. Single-process v1 — non-zero ranks
no-op in attach().
"""

from __future__ import annotations

import logging
import pathlib
from typing import Any

import torch
import torch.nn as nn

from circuitry.core import activation as _act
from circuitry.core import gradient as _grad
from circuitry.core import weight as _w
from circuitry.recipes import Recipe, get_recipe
from circuitry.recorder.hooks import StepContext, TensorSource, match_modules
from circuitry.writers.base import MetricWriter

logger = logging.getLogger("circuitry")

_WEIGHT_DIAGS = {
    "effective_rank": _w.effective_rank,
    "stable_rank": _w.stable_rank,
    "condition_number": _w.condition_number,
    "heavy_tail_alpha": _w.heavy_tail_alpha,
}

_ACT_DIAGS = {
    "dead_fraction": _act.dead_fraction,
    "participation_ratio": _act.participation_ratio,
    "kurtosis": lambda x: float(_act.kurtosis(x).mean().item()),
}

_GRAD_DIAGS = {
    "layer_norm": _grad.layer_norm,  # dict in, dict out
}

_WRITERS: dict[str, Any] = {}  # name → factory; populated below


def _resolve_writer(writer: MetricWriter | str, run_dir: pathlib.Path) -> MetricWriter:
    if not isinstance(writer, str):
        return writer
    from circuitry.writers.jsonl import JsonlWriter
    from circuitry.writers.null import NullWriter
    table = {
        "tensorboard": lambda: _make_tensorboard_writer(run_dir),
        "jsonl": lambda: JsonlWriter(run_dir),
        "null": lambda: NullWriter(),
    }
    if writer == "wandb":
        from circuitry.writers.wandb import WandbWriter
        return WandbWriter()
    if writer not in table:
        raise ValueError(f"unknown writer {writer!r}; known: {sorted(table) + ['wandb']}")
    return table[writer]()


def _make_tensorboard_writer(run_dir: pathlib.Path):
    from circuitry.writers.tensorboard import TensorBoardWriter
    return TensorBoardWriter(run_dir)


class Recorder:
    """Single-process training-time diagnostics recorder.

    On a multi-rank setup (``torch.distributed.is_initialized()`` true and
    ``get_rank() != 0``), every method is a no-op so existing scripts don't
    crash. Multi-process support is the v0.next deliverable (design §11).
    """

    def __init__(
        self,
        model: nn.Module,
        run_dir: str | pathlib.Path,
        recipe: str | Recipe,
        writer: MetricWriter | str = "tensorboard",
        every_n_steps: int = 200,
        strict: bool = True,
    ) -> None:
        self.model = model
        self.run_dir = pathlib.Path(run_dir)
        self.recipe = recipe if isinstance(recipe, Recipe) else get_recipe(recipe)
        self.every_n_steps = int(every_n_steps)
        self.strict = bool(strict)
        self._writer_arg = writer
        self._writer: MetricWriter | None = None
        self._hook_handles: list[Any] = []
        # name → tensor captured by hook, refreshed each emit step
        self._captured_activations: dict[str, torch.Tensor] = {}
        self._matched: dict[int, list[str]] = {}  # hp index → names
        self._noop = False
        self._current_step: int = -1

    # ---- internal helpers ------------------------------------------------

    def _is_inactive_rank(self) -> bool:
        if not torch.distributed.is_available():
            return False
        try:
            if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
                return True
        except Exception:  # pragma: no cover — defensive
            return False
        return False

    def _should_capture(self, step: int) -> bool:
        return step % self.every_n_steps == 0

    # ---- lifecycle -------------------------------------------------------

    def attach(self) -> None:
        if self._is_inactive_rank():
            self._noop = True
            logger.info("circuitry: rank != 0 — Recorder is no-op")
            return

        (self.run_dir / "circuitry").mkdir(parents=True, exist_ok=True)
        self._writer = _resolve_writer(self._writer_arg, self.run_dir)

        matched_lines: list[str] = []
        for idx, hp in enumerate(self.recipe.hook_points):
            names = match_modules(self.model, hp)
            self._matched[idx] = names

            label = hp.pattern or "<modules>" if hp.modules is not None else "<selector>"
            matched_lines.append(f"# hook_point[{idx}] source={hp.source.value} target={label}")
            for n in names:
                matched_lines.append(n)
            matched_lines.append("")
            logger.info("circuitry: hook_point[%d] (%s) matched %d modules: %s",
                        idx, label, len(names), names)

            if len(names) == 0:
                raise RuntimeError(
                    f"HookPoint {idx} ({label}) matched 0 modules — refusing to attach"
                )
            expected = self.recipe.expected_min_matches.get(hp.pattern or "", 0)
            if expected and len(names) < expected:
                msg = (f"HookPoint {idx} ({label}) matched {len(names)} modules but "
                       f"expected at least {expected}")
                if self.strict:
                    raise RuntimeError(msg)
                logger.warning("circuitry: %s", msg)

        (self.run_dir / "circuitry" / "matched_modules.txt").write_text(
            "\n".join(matched_lines)
        )

        # Install hooks for INPUT / OUTPUT sources (WEIGHT/GRAD read directly at step time).
        name_to_mod = dict(self.model.named_modules())
        for idx, hp in enumerate(self.recipe.hook_points):
            if hp.source is TensorSource.OUTPUT:
                for n in self._matched[idx]:
                    handle = name_to_mod[n].register_forward_hook(self._mk_fwd_hook(n))
                    self._hook_handles.append(handle)
            elif hp.source is TensorSource.INPUT:
                for n in self._matched[idx]:
                    handle = name_to_mod[n].register_forward_pre_hook(self._mk_pre_hook(n))
                    self._hook_handles.append(handle)
            # WEIGHT / GRAD are pulled in step() directly from the module — no hook needed.

    def _mk_fwd_hook(self, name: str):
        # Hooks always capture (cheap detach); step() decides what to consume.
        # Gating the hook itself on `_current_step` is broken: forward runs
        # before step(), so the hook would see a stale step index.
        def _hook(_mod, _inp, out):
            t = out[0] if isinstance(out, tuple) else out
            self._captured_activations[name] = t.detach()
        return _hook

    def _mk_pre_hook(self, name: str):
        def _hook(_mod, inputs):
            t = inputs[0]
            self._captured_activations[name] = t.detach()
        return _hook

    def detach(self) -> None:
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()
        if self._writer is not None:
            self._writer.flush()
            self._writer.close()
            self._writer = None

    def step(self, step: int, loss: float | None = None, **user: Any) -> None:
        if self._noop:
            return
        self._current_step = int(step)
        assert self._writer is not None, "Recorder.step called before attach()"

        if loss is not None:
            self._writer.add_scalar("loss", float(loss), self._current_step)

        if not self._should_capture(self._current_step):
            return

        # Build StepContext from currently-captured tensors + read-on-demand weights/grads.
        name_to_mod = dict(self.model.named_modules())
        weights: dict[str, torch.Tensor] = {}
        gradients: dict[str, torch.Tensor] = {}
        for idx, hp in enumerate(self.recipe.hook_points):
            if hp.source is TensorSource.WEIGHT:
                for n in self._matched[idx]:
                    p = getattr(name_to_mod[n], "weight", None)
                    if isinstance(p, torch.Tensor):
                        weights[n] = p.detach()
            elif hp.source is TensorSource.GRAD:
                for n in self._matched[idx]:
                    p = getattr(name_to_mod[n], "weight", None)
                    if isinstance(p, torch.Tensor) and p.grad is not None:
                        gradients[n] = p.grad.detach()

        ctx = StepContext(
            step=self._current_step,
            model=self.model,
            activations=dict(self._captured_activations),
            gradients=gradients,
            weights=weights,
            loss=loss,
            user=dict(user),
        )

        self._run_diagnostics(ctx)
        # Discard activations now that we've consumed them.
        self._captured_activations.clear()

    def _enabled(self, name: str) -> bool:
        return self.recipe.enabled.get(name, True)

    def _run_diagnostics(self, ctx: StepContext) -> None:
        assert self._writer is not None
        for name in self.recipe.weight_diagnostics:
            if not self._enabled(name):
                continue
            fn = _WEIGHT_DIAGS.get(name)
            if fn is None:
                logger.warning("circuitry: unknown weight diagnostic %r — skipping", name)
                continue
            for mod_name, w in ctx.weights.items():
                self._writer.add_scalar(f"weight/{name}/{mod_name}", float(fn(w)), ctx.step)

        for name in self.recipe.activation_diagnostics:
            if not self._enabled(name):
                continue
            fn = _ACT_DIAGS.get(name)
            if fn is None:
                logger.warning("circuitry: unknown activation diagnostic %r — skipping", name)
                continue
            for mod_name, x in ctx.activations.items():
                self._writer.add_scalar(f"activation/{name}/{mod_name}", float(fn(x)), ctx.step)

        for name in self.recipe.gradient_diagnostics:
            if not self._enabled(name):
                continue
            if name == "layer_norm":
                for mod_name, val in _grad.layer_norm(ctx.gradients).items():
                    self._writer.add_scalar(f"gradient/layer_norm/{mod_name}", val, ctx.step)
            else:
                logger.warning("circuitry: unknown gradient diagnostic %r — skipping", name)

        for fn in self.recipe.custom:
            out = fn(ctx)
            for tag, val in out.items():
                if isinstance(val, torch.Tensor):
                    if val.numel() == 1:
                        self._writer.add_scalar(f"custom/{tag}", float(val.item()), ctx.step)
                    else:
                        self._writer.add_histogram(f"custom/{tag}", val, ctx.step)
                else:
                    self._writer.add_scalar(f"custom/{tag}", float(val), ctx.step)
