"""Markdown report builder.

Reads ``<run_dir>/metrics.jsonl`` (produced by ``JsonlWriter``) and
``<run_dir>/circuitry/matched_modules.txt`` (produced by Recorder.attach()),
emits a single-file markdown summary suitable for committing alongside a run.

The report intentionally avoids plots — point users at TensorBoard for visuals.

Layout:
1. Source-run path + a one-line "N tags, M moving, K static, S emit steps" summary.
2. Matched-modules block, copied verbatim from ``circuitry/matched_modules.txt``.
3. One section per (family, diagnostic) pair (e.g. ``weight/effective_rank``,
   ``activation/dead_fraction``). Within each section, rows are sorted so
   metrics that moved over the emit window appear first.
4. Each row has columns: tag tail | first | last | min | max | Δ.
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


def _section_and_row(tag: str) -> tuple[str, str]:
    """Split a tag into (section header, row identifier).

    - 2-segment tags (e.g. ``train/loss``): section = first segment, row = second.
    - 3+ segments: section = first two joined (e.g. ``weight/effective_rank``),
      row = the rest joined (typically a dotted module name).
    """
    parts = tag.split("/")
    if len(parts) <= 1:
        return "scalar", tag
    if len(parts) == 2:
        return parts[0], parts[1]
    return "/".join(parts[:2]), "/".join(parts[2:])


def _stats(series: list[tuple[int, float]]) -> tuple[float, float, float, float, float]:
    """Return (first, last, vmin, vmax, delta) over a sorted time series."""
    vals = [v for _, v in series]
    vmin, vmax = min(vals), max(vals)
    return vals[0], vals[-1], vmin, vmax, vmax - vmin


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

    # Top-of-report summary: total / moving / static / emit-step count.
    moving = 0
    static = 0
    all_steps: set[int] = set()
    for series in grouped.values():
        _, _, vmin, vmax, _ = _stats(series)
        if vmax > vmin:
            moving += 1
        else:
            static += 1
        for s, _ in series:
            all_steps.add(s)

    lines.append("## Summary")
    lines.append("")
    lines.append(
        f"- **{len(grouped)}** scalar tags · **{moving}** moving "
        f"(Δ > 0) · **{static}** static · **{len(all_steps)}** emit step(s) "
        f"observed."
    )
    lines.append("")

    # Group tags by (section header). Within each section, sort moving-first
    # then alphabetical.
    sections: dict[str, list[str]] = defaultdict(list)
    for tag in grouped:
        section, _ = _section_and_row(tag)
        sections[section].append(tag)

    def _sort_key(tag: str) -> tuple[int, str]:
        _, _, vmin, vmax, _ = _stats(grouped[tag])
        return (0 if vmax > vmin else 1, tag)

    for section in sorted(sections):
        lines.append(f"## {section}")
        lines.append("")
        lines.append("| tag | first | last | min | max | Δ |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
        for tag in sorted(sections[section], key=_sort_key):
            _, row_id = _section_and_row(tag)
            first, last, vmin, vmax, delta = _stats(grouped[tag])
            delta_cell = f"{delta:.4g}" if delta > 0 else "—"
            lines.append(
                f"| `{row_id}` | {first:.4g} | {last:.4g} | "
                f"{vmin:.4g} | {vmax:.4g} | {delta_cell} |"
            )
        lines.append("")

    out_path.write_text("\n".join(lines))
    return out_path
