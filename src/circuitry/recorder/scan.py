"""Retrospective scan over checkpoints: rehydrate model state, run the recipe's
weight diagnostics, emit metrics via the chosen ``MetricWriter``.

Checkpoint discovery defaults to ``<run_dir>/checkpoints/step*.pt`` sorted by
filename (which sorts by step under the conventional ``step000000100.pt`` form).
"""

from __future__ import annotations

import glob as _glob
import pathlib
import re
import warnings
from collections.abc import Callable, Sequence

import torch
import torch.nn as nn

from circuitry.recipes import Recipe, get_recipe
from circuitry.recorder.hooks import TensorSource
from circuitry.recorder.live import Recorder, _resolve_writer
from circuitry.writers.base import MetricWriter

_STEP_RX = re.compile(r"step(\d+)")

# Cross-step "trajectory" weight diagnostics: each compares consecutive emitted
# snapshots, so it produces nothing until >= 2 (>= 3 for direction_cosine) steps
# have been emitted. On a single-snapshot retrospective scan they are silently
# absent; scan_run warns once when that is the case (see the static-vs-trajectory
# note in its docstring).
_TRAJECTORY_DIAGNOSTICS = frozenset(
    {"update_delta", "rank_trajectory", "direction_cosine"}
)

# Accepted forms for the explicit ``checkpoints`` argument of scan_run.
CheckpointsArg = (
    str
    | pathlib.Path
    | Sequence[str | pathlib.Path]
    | Sequence[tuple[int, str | pathlib.Path]]
)


def _step_of(p: pathlib.Path) -> int:
    m = _STEP_RX.search(p.stem)
    return int(m.group(1)) if m else 0


def _discover_checkpoints(run_dir: pathlib.Path) -> list[tuple[int, pathlib.Path]]:
    ckpts = sorted((run_dir / "checkpoints").glob("step*.pt"))
    return [(_step_of(p), p) for p in ckpts]


def _coerce_checkpoints(
    checkpoints: CheckpointsArg, run_dir: pathlib.Path
) -> list[tuple[int, pathlib.Path]]:
    """Resolve an explicit ``checkpoints`` argument into ``(step, path)`` pairs.

    Accepts: a single file path; a glob string (matched against cwd, then under
    ``run_dir``); a list of paths; or a list of explicit ``(step, path)`` pairs.
    Steps are parsed from ``stepNNN`` in the filename when not given (0 if
    absent). The result is sorted by step for deterministic emission order.
    """
    # List/tuple of explicit (step, path) pairs.
    if (
        isinstance(checkpoints, (list, tuple))
        and checkpoints
        and isinstance(checkpoints[0], (list, tuple))
    ):
        out = [(int(s), pathlib.Path(p)) for s, p in checkpoints]
    # List/tuple of paths.
    elif isinstance(checkpoints, (list, tuple)):
        paths = [pathlib.Path(p) for p in checkpoints]
        out = [(_step_of(p), p) for p in paths]
    else:
        # Single string/Path: a concrete file or a glob pattern.
        s = str(checkpoints)
        if any(ch in s for ch in "*?["):
            matched = sorted(_glob.glob(s))
            if not matched:
                matched = sorted(str(p) for p in run_dir.glob(s))
            out = [(_step_of(pathlib.Path(m)), pathlib.Path(m)) for m in matched]
        else:
            out = [(_step_of(pathlib.Path(s)), pathlib.Path(s))]
    return sorted(out, key=lambda sp: sp[0])


def scan_run(
    run_dir: str | pathlib.Path,
    recipe: str | Recipe,
    out_dir: str | pathlib.Path,
    model_factory: Callable[[], nn.Module],
    writer: MetricWriter | str = "auto",
    strict: bool = True,
    forward_fn: Callable[[nn.Module], None] | None = None,
    checkpoints: CheckpointsArg | None = None,
) -> None:
    """Replay each checkpoint through the recipe's weight diagnostics.

    ``model_factory`` produces a fresh model whose architecture matches the
    checkpoint state-dict; the same model is reused with ``load_state_dict``
    across checkpoints (cheaper than rebuilding).

    ``checkpoints`` overrides discovery. By default scan_run globs
    ``<run_dir>/checkpoints/step*.pt`` and parses ``stepNNN``. Pass any of: a
    single checkpoint file; a glob string (e.g. ``"runs/*/ckpt_*.pt"``); a list
    of paths; or a list of explicit ``(step, path)`` pairs (for arbitrarily
    named checkpoints). Steps are parsed from the filename when not given.

    **Static vs trajectory diagnostics.** *Static* weight diagnostics
    (``effective_rank``, ``stable_rank``, ``condition_number``,
    ``heavy_tail_alpha``, ``sv_histogram``) are computed per checkpoint and work
    on a single snapshot. *Trajectory* diagnostics (``update_delta``,
    ``rank_trajectory``, ``direction_cosine``) compare consecutive emitted
    snapshots and produce nothing until >= 2 (>= 3 for ``direction_cosine``)
    checkpoints have been scanned. A single-snapshot scan emits only the static
    families; scan_run warns once when trajectory diagnostics are requested but
    fewer than two checkpoints are available.

    ``writer`` defaults to ``"auto"`` (TensorBoard when the optional
    ``tensorboard`` extra is installed, else the no-dep jsonl writer); pass
    ``"jsonl"`` (or a custom ``MetricWriter`` instance) to make the scan
    output consumable by ``build_report``, or ``"tensorboard"`` to require it.

    ``strict`` controls whether unmatched HookPoints cause an error; see
    ``Recorder`` for details.

    ``forward_fn`` is an optional callable ``(model) -> None`` that performs a
    forward pass on the model (e.g. ``lambda m: m(dummy_input)``).  When the
    recipe contains activation diagnostics and ``forward_fn`` is provided,
    scan_run calls it before each ``rec.step()`` so that OUTPUT hooks are fired
    and ``ctx.activations`` is populated.  When activation diagnostics are
    requested but ``forward_fn`` is not provided, a ``UserWarning`` is emitted
    once per run and activation families are silently absent from the output.
    """
    run_dir = pathlib.Path(run_dir)
    out_dir = pathlib.Path(out_dir)
    if checkpoints is None:
        ckpts = _discover_checkpoints(run_dir)
        if not ckpts:
            raise FileNotFoundError(
                f"no checkpoints found under {run_dir / 'checkpoints'}"
            )
    else:
        ckpts = _coerce_checkpoints(checkpoints, run_dir)
        if not ckpts:
            raise FileNotFoundError(
                f"explicit checkpoints argument resolved to no files: {checkpoints!r}"
            )

    recipe = recipe if isinstance(recipe, Recipe) else get_recipe(recipe)

    # Static vs trajectory: a single-snapshot scan can only produce the static
    # weight diagnostics (effective_rank, stable_rank, ...). The cross-step
    # trajectory diagnostics need >= 2 emitted steps and will be silently absent.
    # Warn once so the gap isn't mistaken for a recorder failure.
    if len(ckpts) < 2:
        traj = [
            d for d in recipe.weight_diagnostics
            if d in _TRAJECTORY_DIAGNOSTICS and recipe.enabled.get(d, True)
        ]
        if traj:
            warnings.warn(
                f"scan_run: {len(ckpts)} checkpoint(s) found, but the recipe "
                f"requests trajectory diagnostics {traj} which compare consecutive "
                f"snapshots and need >= 2 emitted steps (>= 3 for direction_cosine). "
                f"They will emit nothing on this scan; the static weight "
                f"diagnostics still run. Pass >= 2 checkpoints for trajectory output.",
                UserWarning,
                stacklevel=2,
            )

    # Warn once per run when the recipe requests activation diagnostics but no
    # forward_fn was supplied — the activation families will be absent from the
    # output (ctx.activations stays empty without a forward pass).
    has_output_hooks = any(
        hp.source is TensorSource.OUTPUT for hp in recipe.hook_points
    )
    if recipe.activation_diagnostics and has_output_hooks and forward_fn is None:
        warnings.warn(
            "scan_run: recipe contains activation_diagnostics "
            f"{recipe.activation_diagnostics!r} but no forward_fn was supplied. "
            "Activation families will be absent from the scan output. "
            "Pass forward_fn=lambda m: m(dummy_input) to enable them.",
            UserWarning,
            stacklevel=2,
        )

    model = model_factory()
    resolved_writer = _resolve_writer(writer, out_dir)
    rec = Recorder(model, run_dir=out_dir, recipe=recipe,
                   writer=resolved_writer, every_n_steps=1, strict=strict)
    rec.attach()
    try:
        for step, ckpt_path in ckpts:
            sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            model.load_state_dict(sd)
            if forward_fn is not None:
                forward_fn(model)
            rec.step(step)
    finally:
        rec.detach()
