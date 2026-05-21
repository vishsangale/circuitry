"""Markdown report builder.

Reads ``<run_dir>/metrics.jsonl`` (produced by ``JsonlWriter``) and
``<run_dir>/circuitry/matched_modules.txt`` (produced by Recorder.attach()),
emits a single-file markdown summary suitable for committing alongside a run.

The report intentionally avoids plots — point users at TensorBoard for visuals.
"""

from __future__ import annotations

import json
import pathlib
from collections import defaultdict


def _group(rows: list[dict]) -> dict[str, list[tuple[int, float]]]:
    by_tag: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for r in rows:
        if r.get("kind") != "scalar":
            continue
        by_tag[r["tag"]].append((int(r["step"]), float(r["value"])))
    for v in by_tag.values():
        v.sort()
    return by_tag


def build_report(
    run_dir: str | pathlib.Path,
    out_path: str | pathlib.Path | None = None,
) -> pathlib.Path:
    run_dir = pathlib.Path(run_dir)
    out_path = pathlib.Path(out_path) if out_path else run_dir / "inspect" / "report.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    metrics_path = run_dir / "metrics.jsonl"
    rows: list[dict] = []
    if metrics_path.exists():
        for line in metrics_path.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))

    lines: list[str] = ["# circuitry report", ""]
    lines.append(f"Source run: `{run_dir}`")
    lines.append("")

    matched_path = run_dir / "circuitry" / "matched_modules.txt"
    if matched_path.exists():
        lines.append("## Matched modules")
        lines.append("")
        lines.append("```")
        lines.append(matched_path.read_text().rstrip())
        lines.append("```")
        lines.append("")

    if not rows:
        lines.append("_no metrics found_")
        out_path.write_text("\n".join(lines))
        return out_path

    grouped = _group(rows)
    families: dict[str, list[str]] = defaultdict(list)
    for tag in grouped:
        family = tag.split("/", 1)[0] if "/" in tag else "scalar"
        families[family].append(tag)

    for family in sorted(families):
        lines.append(f"## {family}")
        lines.append("")
        lines.append("| tag | first | last | min | max |")
        lines.append("| --- | --- | --- | --- | --- |")
        for tag in sorted(families[family]):
            series = grouped[tag]
            vals = [v for _, v in series]
            lines.append(
                f"| `{tag}` | {vals[0]:.4g} | {vals[-1]:.4g} | "
                f"{min(vals):.4g} | {max(vals):.4g} |"
            )
        lines.append("")

    out_path.write_text("\n".join(lines))
    return out_path
