"""Compare two Recorder runs at family/diagnostic granularity.

Public API
----------
FamilyDelta   — dataclass with per-family comparison fields
compare_runs  — load and compare two runs, return list[FamilyDelta]
build_compare_report — render markdown and write to disk

Layering
--------
Imports only stdlib + circuitry.recorder._metrics (peer private module).
No imports from cli/, core/, or recipes/.

Missing-family sentinel
-----------------------
A family present in one run but not the other receives ``float('nan')`` for
the absent side's ``last`` value and ``"flat"`` for the absent side's trend.
``delta`` is computed as ``last_b - last_a``; when either side is NaN, delta
is also NaN.
"""

from __future__ import annotations

import dataclasses
import math
import pathlib

from circuitry.recorder._metrics import group, load_rows, stats


def _section_from_tag(tag: str) -> tuple[str, str]:
    """Return (section, diagnostic) from a tag.

    Rules mirror report.py's ``_section_and_row``:
    - 1-segment tag: section = "scalar", diagnostic = tag
    - 2-segment tag: section = first segment, diagnostic = second
    - 3+ segments: section = first two joined, diagnostic = second segment
    """
    parts = tag.split("/")
    if len(parts) <= 1:
        return "scalar", tag
    if len(parts) == 2:
        return parts[0], parts[1]
    return "/".join(parts[:2]), parts[1]


def _fmt(v: float) -> str:
    """Format a float for a markdown table cell; renders ``—`` for NaN or inf."""
    if math.isnan(v) or math.isinf(v):
        return "—"
    return f"{v:.4g}"


def _trend(signed: float) -> str:
    """Return 'up', 'down', or 'flat' from a signed last−first value."""
    if signed > 0:
        return "up"
    if signed < 0:
        return "down"
    return "flat"


def _family_stats(
    grouped: dict[str, list[tuple[int, float]]],
) -> dict[str, tuple[float, str]]:
    """Aggregate per-family mean-last and trend from a grouped tag dict.

    Returns a mapping of section -> (mean_last, trend_str).
    Trend is derived from the sign of the mean of (last - first) across tags.
    """
    # Collect per-tag contributions, keyed by section.
    section_lasts: dict[str, list[float]] = {}
    section_signeds: dict[str, list[float]] = {}

    for tag, series in grouped.items():
        section, _ = _section_from_tag(tag)
        first, last, _vmin, _vmax, _delta = stats(series)
        signed = last - first  # signed trend (last − first), NOT the range delta

        section_lasts.setdefault(section, []).append(last)
        section_signeds.setdefault(section, []).append(signed)

    result: dict[str, tuple[float, str]] = {}
    for section in section_lasts:
        mean_last = sum(section_lasts[section]) / len(section_lasts[section])
        mean_signed = sum(section_signeds[section]) / len(section_signeds[section])
        result[section] = (mean_last, _trend(mean_signed))

    return result


@dataclasses.dataclass
class FamilyDelta:
    """Per-family/diagnostic comparison between two runs.

    Fields
    ------
    section      : Full two-segment family key, e.g. ``"weight/effective_rank"``.
    diagnostic   : Second segment of section, e.g. ``"effective_rank"``.
    last_a       : Mean of last-value across all tags in the family for run_a.
                   ``float('nan')`` if the family is absent in run_a.
    last_b       : Same for run_b.
    delta        : ``last_b - last_a``. NaN if either side is absent.
    trend_a      : ``"up"`` | ``"down"`` | ``"flat"`` for run_a. ``"flat"`` if absent.
    trend_b      : Same for run_b.
    trend_agrees : ``True`` if ``trend_a == trend_b`` (both flat counts as agrees).
    """

    section: str
    diagnostic: str
    last_a: float
    last_b: float
    delta: float
    trend_a: str
    trend_b: str
    trend_agrees: bool


def compare_runs(
    run_a: str | pathlib.Path,
    run_b: str | pathlib.Path,
) -> list[FamilyDelta]:
    """Compare two runs at family/diagnostic granularity.

    Reads ``metrics.jsonl`` from each run directory. Returns one
    :class:`FamilyDelta` per (section, diagnostic) present in EITHER run.

    Granularity is the first two ``/``-segments of each tag (e.g.
    ``weight/effective_rank``), NOT per-module. Per-module comparison is
    ill-posed across architectures with different module names.

    Parameters
    ----------
    run_a, run_b
        Run directories. Each must contain ``metrics.jsonl``.

    Raises
    ------
    FileNotFoundError
        If ``metrics.jsonl`` is missing from either run directory.

    Missing families
    ----------------
    A section present in only one run produces a :class:`FamilyDelta` with
    ``float('nan')`` for the absent side's ``last`` value, ``"flat"`` for the
    absent side's trend, and ``float('nan')`` for ``delta``.
    """
    run_a = pathlib.Path(run_a)
    run_b = pathlib.Path(run_b)

    for run_dir in (run_a, run_b):
        if not (run_dir / "metrics.jsonl").exists():
            raise FileNotFoundError(
                f"compare_runs: no metrics.jsonl found in {run_dir}"
            )

    grouped_a = group(load_rows(run_a))
    grouped_b = group(load_rows(run_b))

    if not grouped_a:
        raise ValueError(
            f"compare_runs: {run_a}/metrics.jsonl has no scalar metrics to compare"
        )
    if not grouped_b:
        raise ValueError(
            f"compare_runs: {run_b}/metrics.jsonl has no scalar metrics to compare"
        )

    fam_a = _family_stats(grouped_a)  # section -> (mean_last, trend)
    fam_b = _family_stats(grouped_b)

    all_sections = sorted(set(fam_a) | set(fam_b))

    result: list[FamilyDelta] = []
    for section in all_sections:
        # Derive diagnostic from the section string (second segment).
        parts = section.split("/")
        diagnostic = parts[1] if len(parts) >= 2 else section

        if section in fam_a:
            last_a, trend_a = fam_a[section]
        else:
            last_a, trend_a = float("nan"), "flat"

        if section in fam_b:
            last_b, trend_b = fam_b[section]
        else:
            last_b, trend_b = float("nan"), "flat"

        delta = last_b - last_a  # NaN propagates if either side is NaN

        trend_agrees = (
            not math.isnan(last_a)
            and not math.isnan(last_b)
            and (trend_a == trend_b)
        )

        result.append(
            FamilyDelta(
                section=section,
                diagnostic=diagnostic,
                last_a=last_a,
                last_b=last_b,
                delta=delta,
                trend_a=trend_a,
                trend_b=trend_b,
                trend_agrees=trend_agrees,
            )
        )

    return result


def build_compare_report(
    run_a: str | pathlib.Path,
    run_b: str | pathlib.Path,
    out_path: str | pathlib.Path | None = None,
    *,
    compact: bool = False,
) -> pathlib.Path:
    """Write a markdown compare report.

    Parameters
    ----------
    run_a, run_b
        Run directories to compare.
    out_path
        Output path for the markdown file. Defaults to
        ``Path(run_a).parent / "compare.md"``.
    compact
        Accepted for API symmetry with ``build_report``; currently a no-op
        (the compare report has no extra sections to suppress beyond the single
        table already rendered).

    Returns
    -------
    pathlib.Path
        Absolute path of the written file.

    Raises
    ------
    FileNotFoundError
        Propagated from :func:`compare_runs` when ``metrics.jsonl`` is missing.
    """
    run_a = pathlib.Path(run_a)
    run_b = pathlib.Path(run_b)

    if out_path is None:
        out_path = run_a.parent / "compare.md"
    out_path = pathlib.Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    deltas = compare_runs(run_a, run_b)

    lines: list[str] = [
        "# circuitry compare",
        "",
        f"A: `{run_a}`",
        f"B: `{run_b}`",
        "",
        "| family/diagnostic | last_a | last_b | Δ (b−a) | trend_a | trend_b | agrees |",
        "| --- | ---: | ---: | ---: | --- | --- | --- |",
    ]

    for fd in deltas:
        last_a_cell = _fmt(fd.last_a)
        last_b_cell = _fmt(fd.last_b)
        delta_cell = _fmt(fd.delta)
        agrees_cell = "yes" if fd.trend_agrees else "no"
        lines.append(
            f"| {fd.section} | {last_a_cell} | {last_b_cell} "
            f"| {delta_cell} | {fd.trend_a} | {fd.trend_b} | {agrees_cell} |"
        )

    lines.append("")
    out_path.write_text("\n".join(lines))
    return out_path
