"""``circuitry`` CLI entry point. See docs/design.md §4.3."""

from __future__ import annotations

import argparse
import importlib
import sys
from typing import Any

from circuitry import __version__
from circuitry.recipes import list_recipes
from circuitry.recorder.compare import build_compare_report
from circuitry.recorder.report import build_report


def _load_entrypoint(spec: str) -> Any:
    """Resolve a ``package.module:attr`` entry point to the attribute.

    The attribute is typically a zero-arg factory (e.g. a function returning a
    model or an iterable of batches). Raises ``ValueError`` with a clear message
    on a malformed spec or a missing module / attribute.
    """
    if ":" not in spec:
        raise ValueError(
            f"entry point {spec!r} must be 'package.module:attr' "
            "(e.g. 'mypkg.lens:make_model')"
        )
    mod_path, attr = spec.split(":", 1)
    try:
        mod = importlib.import_module(mod_path)
    except ImportError as e:
        raise ValueError(f"could not import module {mod_path!r}: {e}") from e
    try:
        return getattr(mod, attr)
    except AttributeError as e:
        raise ValueError(f"module {mod_path!r} has no attribute {attr!r}") from e


def _cmd_list_recipes(_: argparse.Namespace) -> int:
    for name in list_recipes():
        print(name)
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    out = build_report(run_dir=args.run, out_path=args.out, compact=args.compact)
    print(f"wrote {out}")
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    out = build_compare_report(
        run_a=args.run_a,
        run_b=args.run_b,
        out_path=args.out,
        compact=args.compact,
    )
    print(f"wrote {out}")
    return 0


def _cmd_scan(args: argparse.Namespace) -> int:
    # scan_run needs a model_factory which the CLI cannot conjure without a
    # user-supplied import path. Surface a clear error pointing to the
    # programmatic API. The `--model-factory dotted.path:fn` flag is planned
    # for a future release.
    print(
        "circuitry scan: requires a model factory not yet exposed via the CLI.\n"
        "  Use circuitry.recorder.scan.scan_run(...) programmatically for now.\n"
        f"  Discovered checkpoints: {args.run}/checkpoints",
        file=sys.stderr,
    )
    return 2


def _cmd_fit_tuned_lens(args: argparse.Namespace) -> int:
    from circuitry.tuned_lens import fit_tuned_lens

    model_factory = _load_entrypoint(args.model)
    batches_factory = _load_entrypoint(args.batches)
    model = model_factory() if callable(model_factory) else model_factory
    batches = batches_factory() if callable(batches_factory) else batches_factory

    lens = fit_tuned_lens(
        model, batches,
        layers=args.layers,
        steps=args.steps,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=args.device,
    )
    lens.save(args.out)
    print(
        f"wrote {args.out} — tuned lens for layers {lens.layers} "
        f"(d_model={lens.d_model}, fingerprint={lens.model_fingerprint})"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="circuitry")
    parser.add_argument(
        "--version", action="version", version=f"circuitry {__version__}",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list-recipes", help="list registered recipes")
    p_list.set_defaults(func=_cmd_list_recipes)

    p_report = sub.add_parser("report", help="build markdown report from a run")
    p_report.add_argument("--run", required=True, help="run directory")
    p_report.add_argument("--out", default=None, help="report output path")
    p_report.add_argument(
        "--compact", action="store_true", default=False,
        help="emit only Summary + Flags, suppress per-tag tables",
    )
    p_report.set_defaults(func=_cmd_report)

    p_compare = sub.add_parser(
        "compare", help="compare two runs at family/diagnostic granularity"
    )
    p_compare.add_argument("run_a", help="first run directory")
    p_compare.add_argument("run_b", help="second run directory")
    p_compare.add_argument("--out", default=None, help="output path (default: run_a/../compare.md)")
    p_compare.add_argument(
        "--compact", action="store_true", default=False,
        help="accepted for API symmetry; currently a no-op for compare",
    )
    p_compare.set_defaults(func=_cmd_compare)

    p_scan = sub.add_parser("scan", help="retrospective scan of checkpoints")
    p_scan.add_argument("--run", required=True)
    p_scan.add_argument("--recipe", required=True)
    p_scan.set_defaults(func=_cmd_scan)

    p_fit = sub.add_parser(
        "fit-tuned-lens",
        help="fit per-layer tuned-lens translators and save them (v1.10)",
    )
    p_fit.add_argument(
        "--model", required=True,
        help="entry point 'pkg.module:factory' returning the model to fit a lens for",
    )
    p_fit.add_argument(
        "--batches", required=True,
        help="entry point 'pkg.module:factory' returning an iterable of model inputs",
    )
    p_fit.add_argument("--out", required=True, help="output path for the TunedLens (.pt)")
    p_fit.add_argument(
        "--layers", type=int, nargs="*", default=None,
        help="block indices to fit (default: all blocks except the final frame)",
    )
    p_fit.add_argument("--steps", type=int, default=250, help="optimizer iterations")
    p_fit.add_argument("--lr", type=float, default=1e-3, help="AdamW learning rate")
    p_fit.add_argument("--weight-decay", type=float, default=1e-3, dest="weight_decay")
    p_fit.add_argument("--device", default=None, help="device to fit on (default: model's)")
    p_fit.set_defaults(func=_cmd_fit_tuned_lens)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
