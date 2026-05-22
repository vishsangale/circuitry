"""LiveRecorder — attach hooks per recipe, snapshot tensors at emit steps,
run diagnostics, write scalars through a MetricWriter.

See docs/design.md §4.2, §4.4, §10, §11. Single-process — non-zero ranks
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
from circuitry.core.inventory import ModelInventory
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


def _scalarize(value: float | torch.Tensor) -> float:
    """Coerce a Python number or 0-d Tensor to a plain Python ``float``.

    Detaches Tensors before ``.item()`` so a ``requires_grad=True`` loss is
    accepted without the PyTorch UserWarning. Non-scalar Tensors are
    rejected explicitly.
    """
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(
                f"scalar metric expected a 0-d / size-1 Tensor, got shape {tuple(value.shape)}"
            )
        return float(value.detach().item())
    return float(value)


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
    if writer not in table:
        raise ValueError(f"unknown writer {writer!r}; known: {sorted(table)}")
    return table[writer]()


def _make_tensorboard_writer(run_dir: pathlib.Path):
    from circuitry.writers.tensorboard import TensorBoardWriter
    return TensorBoardWriter(run_dir)


class Recorder:
    """Single-process training-time diagnostics recorder.

    On a multi-rank setup (``torch.distributed.is_initialized()`` true and
    ``get_rank() != 0``), every method is a no-op so existing scripts don't
    crash. Multi-process support is planned for a future release (design §11).
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
        # Inventory-derived module-name → parameter-name resolution for
        # WEIGHT / GRAD HookPoints. Built once at attach() time. Catches
        # weights hidden inside wrapper Linear classes (e.g.
        # ``Gemma4ClippableLinear`` → resolves to ``<module>.linear.weight``).
        self._inventory: ModelInventory | None = None
        self._param_for_module: dict[str, str] = {}
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

        # Inventory pass: enumerate every Parameter once. Source of truth for
        # WEIGHT/GRAD HookPoint resolution; replaces the old
        # ``getattr(module, "weight", None)`` heuristic that silently dropped
        # weights hidden behind wrapper Linear classes.
        self._inventory = ModelInventory.build(self.model)
        (self.run_dir / "circuitry" / "inventory.json").write_text(
            self._inventory.to_json()
        )

        name_to_mod = dict(self.model.named_modules())
        matched_lines: list[str] = []
        for idx, hp in enumerate(self.recipe.hook_points):
            names = match_modules(self.model, hp)
            self._matched[idx] = names

            if hp.pattern is not None:
                label = hp.pattern
            elif hp.modules is not None:
                label = "<modules>"
            else:
                label = "<selector>"
            matched_lines.append(f"# hook_point[{idx}] source={hp.source.value} target={label}")
            logger.info("circuitry: hook_point[%d] (%s) matched %d modules: %s",
                        idx, label, len(names), names)

            if len(names) == 0:
                msg = f"HookPoint {idx} ({label}) matched 0 modules"
                if self.strict:
                    raise RuntimeError(
                        msg + " — refusing to attach (pass strict=False to skip "
                        "unmatched HookPoints with a warning)"
                    )
                logger.warning("circuitry: %s — skipping this HookPoint", msg)
                matched_lines.append("")
                continue
            expected = self.recipe.expected_min_matches.get(hp.pattern or "", 0)
            if expected and len(names) < expected:
                msg = (f"HookPoint {idx} ({label}) matched {len(names)} modules but "
                       f"expected at least {expected}")
                if self.strict:
                    raise RuntimeError(msg)
                logger.warning("circuitry: %s", msg)

            # For WEIGHT / GRAD HookPoints, resolve each matched module name
            # to its primary weight Parameter via the inventory. Loud-on-fail:
            # unresolvable matches WARN per module so wrapper-Linear classes
            # don't silently drop diagnostics.
            if hp.source in (TensorSource.WEIGHT, TensorSource.GRAD):
                unresolved = 0
                for mn in names:
                    rec = self._inventory.find_primary_weight(mn)
                    if rec is None:
                        parent_cls = type(name_to_mod.get(mn, self.model)).__name__
                        logger.warning(
                            "circuitry: hook_point[%d] %s: module %r (%s) has no "
                            "resolvable 2-D+ weight in its subtree — skipping",
                            idx, hp.source.value, mn, parent_cls,
                        )
                        matched_lines.append(f"{mn} → UNRESOLVED ({parent_cls})")
                        unresolved += 1
                    else:
                        # Don't overwrite an earlier WEIGHT mapping with a GRAD
                        # one (they should be identical, but be explicit).
                        self._param_for_module.setdefault(mn, rec.name)
                        tail = rec.name[len(mn) + 1:] if mn else rec.name
                        matched_lines.append(f"{mn} → {tail} {tuple(rec.shape)}")
                if unresolved:
                    logger.warning(
                        "circuitry: hook_point[%d] (%s): %d of %d matched modules "
                        "had no resolvable primary weight",
                        idx, label, unresolved, len(names),
                    )
            else:
                # OUTPUT / INPUT: hooks attach to the module itself; nothing
                # to resolve.
                for n in names:
                    matched_lines.append(n)
            matched_lines.append("")

        (self.run_dir / "circuitry" / "matched_modules.txt").write_text(
            "\n".join(matched_lines)
        )

        # Install hooks for INPUT / OUTPUT sources (WEIGHT/GRAD read directly at step time).
        for idx, hp in enumerate(self.recipe.hook_points):
            if hp.source is TensorSource.OUTPUT:
                for n in self._matched[idx]:
                    handle = name_to_mod[n].register_forward_hook(self._mk_fwd_hook(n))
                    self._hook_handles.append(handle)
            elif hp.source is TensorSource.INPUT:
                for n in self._matched[idx]:
                    handle = name_to_mod[n].register_forward_pre_hook(self._mk_pre_hook(n))
                    self._hook_handles.append(handle)
            # WEIGHT / GRAD are pulled in step() directly from the inventory.

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

    def step(self, step: int, loss: float | torch.Tensor | None = None, *,
             loss_components: dict[str, float | torch.Tensor] | None = None,
             **user: Any) -> None:
        if self._noop:
            return
        self._current_step = int(step)
        assert self._writer is not None, "Recorder.step called before attach()"

        if loss is not None:
            self._writer.add_scalar("train/loss", _scalarize(loss), self._current_step)

        if loss_components is not None and self._writer is not None:
            for name, value in loss_components.items():
                self._writer.add_scalar(f"train/{name}", _scalarize(value),
                                        self._current_step)

        if not self._should_capture(self._current_step):
            return

        # Build StepContext from currently-captured tensors + read-on-demand
        # weights/grads. WEIGHT/GRAD HookPoints use the inventory-derived
        # module→param mapping from attach() so wrapper Linear classes
        # (e.g. ``Gemma4ClippableLinear`` → ``<module>.linear.weight``) are
        # handled without silent drops.
        name_to_param = dict(self.model.named_parameters())
        weights: dict[str, torch.Tensor] = {}
        gradients: dict[str, torch.Tensor] = {}
        for idx, hp in enumerate(self.recipe.hook_points):
            if hp.source not in (TensorSource.WEIGHT, TensorSource.GRAD):
                continue
            for mod_name in self._matched[idx]:
                param_name = self._param_for_module.get(mod_name)
                if param_name is None:
                    continue  # already WARNed at attach()
                p = name_to_param.get(param_name)
                if not isinstance(p, torch.Tensor):
                    continue
                if hp.source is TensorSource.WEIGHT:
                    weights[mod_name] = p.detach()
                else:  # GRAD
                    if p.grad is not None:
                        gradients[mod_name] = p.grad.detach()

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
            if name == "sv_histogram":
                if ctx.weights:
                    from circuitry.core.weight import singular_values
                    for mod_name, w in ctx.weights.items():
                        sv = singular_values(w)
                        self._writer.add_histogram(f"spectral/per_param/{mod_name}/sv_histogram", sv, ctx.step)
            else:
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
            elif name == "norms_per_param":
                if ctx.gradients:
                    global_sq = 0.0
                    for mod_name, g in ctx.gradients.items():
                        n = float(g.detach().to(torch.float32).norm().item())
                        self._writer.add_scalar(f"grad/per_param/{mod_name}/norm", n, ctx.step)
                        global_sq += n * n
                    self._writer.add_scalar("grad/global/total_norm", global_sq ** 0.5, ctx.step)
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
