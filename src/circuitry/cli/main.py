"""``circuitry`` CLI entry point. See docs/design.md §4.3."""

from __future__ import annotations

import argparse
import sys

from circuitry import __version__
from circuitry.recipes import list_recipes
from circuitry.recorder.compare import build_compare_report
from circuitry.recorder.report import build_report


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

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
