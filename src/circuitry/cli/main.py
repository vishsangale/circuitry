"""``circuitry`` CLI entry point. See docs/design.md §4.3."""

from __future__ import annotations

import argparse
import importlib
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
    import pathlib

    from circuitry.recorder.scan import scan_run
    factory = _load_entrypoint(args.model_factory)
    out_dir = args.out or str(pathlib.Path(args.run) / "scan_report")
    scan_run(
        run_dir=args.run,
        recipe=args.recipe,
        out_dir=out_dir,
        model_factory=factory,
    )
    return 0


def _circuit_edge_set(data: dict) -> frozenset:
    """Extract a frozenset of (writer_str, slot, reader_str) tuples from circuit JSON."""
    from circuitry.patching.graph import _node_from_dict, _node_str
    kind = data.get("kind", "")
    if kind == "acdc":
        rows = data.get("kept_edges", [])
    elif kind == "eap":
        rows = data.get("scores", [])
    else:
        raise ValueError(
            f"unsupported circuit kind {kind!r} — expected 'eap' or 'acdc'"
        )
    result = set()
    for row in rows:
        writer = _node_str(_node_from_dict(row["writer"]))
        reader = _node_str(_node_from_dict(row["reader"]))
        result.add((writer, row["slot"], reader))
    return frozenset(result)


def _cmd_circuit_compare(args: argparse.Namespace) -> int:
    import json
    import pathlib

    a_data = json.loads(pathlib.Path(args.a).read_text())
    b_data = json.loads(pathlib.Path(args.b).read_text())

    edges_a = _circuit_edge_set(a_data)
    edges_b = _circuit_edge_set(b_data)

    only_a = sorted(edges_a - edges_b)
    only_b = sorted(edges_b - edges_a)
    n_both = len(edges_a & edges_b)

    lines = [
        "## Circuit Comparison",
        "",
        f"| | `{args.a}` | `{args.b}` |",
        "| --- | ---: | ---: |",
        f"| total edges | {len(edges_a)} | {len(edges_b)} |",
        f"| edges in both | {n_both} | {n_both} |",
        f"| unique to this file | {len(only_a)} | {len(only_b)} |",
        "",
    ]

    def _edge_row(writer: str, slot: str, reader: str) -> str:
        return f"| `{writer}` | {slot} | `{reader}` |"

    if only_a:
        lines.append(f"### Only in `{args.a}` ({len(only_a)} edges)")
        lines.append("")
        lines.append("| writer | slot | reader |")
        lines.append("| --- | --- | --- |")
        for writer, slot, reader in only_a:
            lines.append(_edge_row(writer, slot, reader))
        lines.append("")

    if only_b:
        lines.append(f"### Only in `{args.b}` ({len(only_b)} edges)")
        lines.append("")
        lines.append("| writer | slot | reader |")
        lines.append("| --- | --- | --- |")
        for writer, slot, reader in only_b:
            lines.append(_edge_row(writer, slot, reader))
        lines.append("")

    output = "\n".join(lines)
    if args.out:
        out_path = pathlib.Path(args.out)
        out_path.write_text(output)
        print(f"wrote {out_path}")
    else:
        print(output)
    return 0


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


def _cmd_export_graph(args: argparse.Namespace) -> int:
    from circuitry.patching.eap import EAPResult
    from circuitry.patching.export import save_html, save_neuronpedia_graph

    result = EAPResult.load(args.input)
    if args.format == "neuronpedia":
        out = save_neuronpedia_graph(
            result, args.out, slug=args.slug, scan=args.scan, top_k=args.top_k,
        )
    else:
        # Omit top_k when unset so save_html keeps its default (50).
        kwargs = {} if args.top_k is None else {"top_k": args.top_k}
        out = save_html(result, args.out, **kwargs)
    print(f"wrote {out}")
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
    p_scan.add_argument("--run", required=True, help="run directory (must contain checkpoints/)")
    p_scan.add_argument("--recipe", required=True, help="recipe name or dotted entry point")
    p_scan.add_argument(
        "--model-factory", required=True, dest="model_factory",
        help="entry point 'pkg.module:factory' returning a fresh nn.Module",
    )
    p_scan.add_argument(
        "--out", default=None,
        help="output directory for the scan report (default: <run>/scan_report)",
    )
    p_scan.set_defaults(func=_cmd_scan)

    p_cc = sub.add_parser(
        "circuit-compare",
        help="diff two circuit JSON files (EAP or ACDC) by edge-set",
    )
    p_cc.add_argument("a", help="first circuit JSON file")
    p_cc.add_argument("b", help="second circuit JSON file")
    p_cc.add_argument("--out", default=None, help="write markdown diff to file (default: stdout)")
    p_cc.set_defaults(func=_cmd_circuit_compare)

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

    p_export = sub.add_parser(
        "export-graph",
        help="export a saved circuit JSON (EAPResult.save) to Neuronpedia JSON or HTML (v1.41)",
    )
    p_export.add_argument("input", help="circuit JSON file written by EAPResult.save()")
    p_export.add_argument(
        "--format", choices=["neuronpedia", "html"], default="html",
        help="output format (default: html)",
    )
    p_export.add_argument("--out", required=True, help="output file path")
    p_export.add_argument("--slug", default="circuitry-graph", help="Neuronpedia graph slug")
    p_export.add_argument("--scan", default="custom", help="Neuronpedia model id (scan)")
    p_export.add_argument(
        "--top-k", type=int, default=None, dest="top_k",
        help="keep only the top-k edges by |score| (default: all for neuronpedia, 50 for html)",
    )
    p_export.set_defaults(func=_cmd_export_graph)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
