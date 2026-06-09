"""Shared grouping/stats helpers for report.py and compare.py (private).

Do not import this module from outside recorder/. Not part of the public API.
"""
from __future__ import annotations

import json
import pathlib
from collections import defaultdict


def group(rows: list[dict]) -> dict[str, list[tuple[int, float]]]:
    """Group JSONL scalar rows by tag; sort each series by step."""
    by_tag: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for r in rows:
        if r.get("kind") != "scalar":
            continue
        by_tag[r["tag"]].append((int(r["step"]), float(r["value"])))
    for v in by_tag.values():
        v.sort()
    return by_tag


def stats(series: list[tuple[int, float]]) -> tuple[float, float, float, float, float]:
    """Return (first, last, vmin, vmax, delta) over a sorted time series.

    ``delta`` is the **signed trend** ``last - first`` (v1.10) — a monotonically
    decreasing metric reports a negative delta. (Through v1.9 ``delta`` was the
    unsigned range ``vmax - vmin``, which rendered a falling metric as a positive
    Δ in the report table, reading like an increase.) Callers that need the range
    use ``vmax - vmin`` from the returned bounds.
    """
    vals = [v for _, v in series]
    vmin, vmax = min(vals), max(vals)
    return vals[0], vals[-1], vmin, vmax, vals[-1] - vals[0]


def load_rows(run_dir: pathlib.Path) -> list[dict]:
    """Load JSONL rows from run_dir/metrics.jsonl; return [] if absent."""
    p = run_dir / "metrics.jsonl"
    if not p.exists():
        return []
    rows = []
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise ValueError(f"malformed JSONL in {p}: {e}") from e
    return rows
