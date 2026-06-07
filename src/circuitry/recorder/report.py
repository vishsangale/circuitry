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
from collections.abc import Callable

from circuitry.recorder._metrics import group as _group
from circuitry.recorder._metrics import stats as _stats

HERO_SECTIONS = frozenset({
    "weight/effective_rank",
    "weight/attention_head_rank",
    "activation/dead_fraction",
    "activation/gate_stats",
    "grad/global",
    # v0.9 additions:
    "activation/logit_lens_kl",
    "activation/induction_score",
    # v1.11 copy-suppression heads:
    "activation/copy_suppression_score",
    "activation/attention_pattern_entropy",
    # v1.12 attention sinks:
    "activation/attention_sink_score",
    "activation/sae",
    # v1.10 tuned lens:
    "activation/tuned_lens_kl",
    # v1.3 training-dynamics:
    "weight/update_delta",
    "weight/rank_trajectory",
    "weight/direction_cosine",
    # v1.10 scale-invariant update size:
    "weight/update_delta_rel",
    # v1.16 representational drift:
    "activation/repr_drift",
})

GRAD_PER_PARAM_TOP_K = 10  # Show top K and bottom K; hide the middle.

# Declarative flag rules: (section_prefix, flag_label, predicate(last, signed), message_template)
# NOTE: ``signed = last - first`` (signed trend). Since v1.10 the _stats ``delta``
# field is also signed (last - first), so the table Δ and the flag trend agree; the
# explicit ``signed`` below documents intent and is robust to future _stats changes.
FLAG_RULES: list[tuple[str, str, Callable[[float, float], bool], str]] = [
    (
        "activation/dead_fraction",
        "dead_rising",
        lambda last, signed: signed > 0 and last > 0.05,
        "dead_fraction rising (last={last:.3f}, Δ={signed:+.4g})",
    ),
    (
        "weight/effective_rank",
        "rank_collapsing",
        lambda last, signed: signed < 0 and last < 10.0,
        "effective_rank collapsing (last={last:.2f}, Δ={signed:+.4g})",
    ),
    (
        "grad/global",
        "grad_norm_spiking",
        lambda last, signed: signed > 0 and last > 10.0,
        "grad_norm spiking (last={last:.4g}, Δ={signed:+.4g})",
    ),
    (
        "weight/attention_head_rank",
        "attn_rank_low",
        lambda last, signed: last < 2.0,
        "attention_head_rank critically low (last={last:.2f})",
    ),
    (
        "weight/rank_trajectory",
        "rank_collapse_trend",
        lambda last, signed: signed < -1.0 and last < 8.0,
        "rank_trajectory declining (last={last:.2f}, Δ={signed:+.4g})",
    ),
    (
        # Keys on the scale-invariant ||ΔW||/||W|| companion (v1.10) so the
        # threshold is dimensionless and means the same across parameter sizes —
        # the absolute weight/update_delta was scale-dependent (v1.3 review).
        "weight/update_delta_rel",
        "update_delta_vanishing",
        lambda last, signed: last < 1e-5,
        "relative update_delta near zero — possible gradient vanishing (last={last:.2g})",
    ),
    (
        "weight/direction_cosine",
        "direction_reversal",
        lambda last, signed: last < -0.5,
        "direction_cosine strongly negative — update direction reversal (last={last:.3f})",
    ),
    (
        "activation/attention_sink_score",
        "attention_sink_detected",
        # A per-head mean > 0.5 on the live training-forward attention means
        # the head is directing more than half its attention weight to the sink
        # position (position 0 / BOS by default) — the diagnostic signature of
        # attention sink heads (Xiao et al. 2023).
        lambda last, signed: last > 0.5,
        "attention_sink_score high — potential attention sink head (last={last:.3f})",
    ),
    (
        "activation/repr_drift",
        "repr_drift_high",
        # Linear CKA drift > 0.5 from the reference snapshot means the layer's
        # representational geometry has shifted substantially — worth investigating
        # whether this is intentional (e.g. post-grokking alignment) or a sign of
        # instability.
        lambda last, signed: last > 0.5,
        "repr_drift high — significant representational shift from reference (last={last:.3f})",
    ),
    (
        "activation/copy_suppression_score",
        "copy_suppression_detected",
        # A per-head mean > 0.3 on the repeated-token probe indicates the head is
        # consistently attending to same-token positions — the hallmark pattern of
        # copy-suppression heads (McDougall et al. 2023).  Scores well above this
        # threshold warrant closer inspection (the head may be suppressing token
        # copying across the full training distribution).
        lambda last, signed: last > 0.3,
        "copy_suppression_score high — potential copy-suppression head (last={last:.3f})",
    ),
    (
        "activation/tuned_lens_kl",
        "tuned_lens_not_forming",
        # A fitted tuned lens drives per-layer KL toward ~0; a tuned-lens KL
        # that stays high (> 1 nat) means the prediction is not forming in the
        # residual stream where the lens expects it (or the lens is stale for
        # this checkpoint).
        lambda last, signed: last > 1.0,
        "tuned_lens_kl still high — prediction not forming / stale lens (last={last:.3f} nats)",
    ),
]


def _build_flags(
    grouped: dict[str, list[tuple[int, float]]],
    step_count: int,
) -> list[str]:
    """Return markdown lines for the ## Flags block, or [] if step_count <= 1."""
    if step_count <= 1:
        return []

    fired: list[tuple[str, str, str]] = []  # (prefix, flag_label, detail)
    for prefix, flag_label, predicate, msg_template in FLAG_RULES:
        # Collect all tags whose section matches this prefix.
        for tag, series in grouped.items():
            section, _ = _section_and_row(tag)
            if section != prefix:
                continue
            first, last, _vmin, _vmax, _delta = _stats(series)
            signed = last - first  # signed trend (last − first), NOT the range delta
            if predicate(last, signed):
                detail = msg_template.format(last=last, signed=signed)
                fired.append((prefix, flag_label, detail))
                break  # one flag per rule is enough; stop scanning tags for this rule

    lines: list[str] = ["## Flags", "", "| family | flag | detail |", "| --- | --- | --- |"]
    if fired:
        for prefix, flag_label, detail in fired:
            lines.append(f"| {prefix} | {flag_label} | {detail} |")
    else:
        lines.append("| — | — | no flags |")
    lines.append("")
    return lines


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


def _render_section(
    name: str,
    tags: list[str],
    grouped: dict[str, list[tuple[int, float]]],
) -> list[str]:
    """Render one (section, tags) into markdown lines (table + spacer).

    For ``grad/per_param``-style sections with many rows, trim to top-K
    and bottom-K by max-magnitude with an elision label between.
    """
    out: list[str] = [
        f"## {name}",
        "",
        "| tag | first | last | min | max | Δ |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]

    def _key(tag: str) -> tuple[int, float, str]:
        _, _, vmin, vmax, _ = _stats(grouped[tag])
        return (0 if vmax > vmin else 1, -vmax, tag)

    sorted_tags = sorted(tags, key=_key)

    # Heuristic: grad/per_param sections are the only ones that benefit
    # from top/bottom-K trimming; everything else renders in full.
    if name.startswith("grad/per_param") and len(sorted_tags) > 2 * GRAD_PER_PARAM_TOP_K:
        # Re-sort by absolute magnitude for top/bottom selection.
        def _mag(tag: str) -> float:
            _, _, _, vmax, _ = _stats(grouped[tag])
            return abs(vmax)

        by_mag = sorted(sorted_tags, key=_mag, reverse=True)
        top = by_mag[:GRAD_PER_PARAM_TOP_K]
        bot = by_mag[-GRAD_PER_PARAM_TOP_K:]
        hidden = len(sorted_tags) - 2 * GRAD_PER_PARAM_TOP_K
        rendered_set = set(top + bot)
        rows_written = 0
        for tag in by_mag:
            if tag not in rendered_set:
                continue
            _, row_id = _section_and_row(tag)
            first, last, vmin, vmax, delta = _stats(grouped[tag])
            delta_cell = f"{delta:+.4g}" if delta != 0 else "—"
            out.append(
                f"| `{row_id}` | {first:.4g} | {last:.4g} | "
                f"{vmin:.4g} | {vmax:.4g} | {delta_cell} |"
            )
            rows_written += 1
            if rows_written == GRAD_PER_PARAM_TOP_K and hidden > 0:
                out.append(f"| _… {hidden} rows hidden …_ | | | | | |")
    else:
        for tag in sorted_tags:
            _, row_id = _section_and_row(tag)
            first, last, vmin, vmax, delta = _stats(grouped[tag])
            delta_cell = f"{delta:+.4g}" if delta != 0 else "—"
            out.append(
                f"| `{row_id}` | {first:.4g} | {last:.4g} | "
                f"{vmin:.4g} | {vmax:.4g} | {delta_cell} |"
            )

    out.append("")
    return out


_FORMATION_THRESHOLDS: dict[str, float] = {
    "activation/induction_score": 0.4,
    "activation/copy_suppression_score": 0.3,
    "activation/attention_sink_score": 0.5,
}

_TRANSITION_PREFIXES: frozenset[str] = frozenset({
    "weight/effective_rank",
    "weight/rank_trajectory",
    "weight/update_delta_rel",
})


def _transition_direction(section: str, before: float, after: float) -> str:
    """Human-readable direction label for a phase transition."""
    rising = after > before
    if section == "weight/update_delta_rel":
        return "↑ norm growth" if rising else "↓ norm collapse"
    return "↑ growth" if rising else "↓ collapse"


def _drift_trend(
    vals: list[float],
    series: list[tuple[int, float]],
    pts_fn: object,
) -> str:
    """Trend label for a repr_drift series: phase-transition > rising > stable > falling."""
    pts = pts_fn(series)  # type: ignore[operator]
    if pts:
        return f"⚡ step {pts[0]}"
    mid = max(1, len(vals) // 2)
    first_half = sum(vals[:mid]) / mid
    second_half = sum(vals[mid:]) / max(1, len(vals) - mid)
    if second_half > first_half + 0.05:
        return "↑ rising"
    if first_half > second_half + 0.05:
        return "↓ falling"
    return "→ stable"


def _build_training_dynamics_section(
    grouped: dict[str, list[tuple[int, float]]],
    step_count: int,
) -> list[str]:
    """Render a ## Training Dynamics section with four sub-tables.

    - **Head Formation Events**: heads whose attention score crossed its threshold
      *during* the recording window (formation step > series' first recorded step).
    - **Phase Transitions**: sharp change-points in rank/health metrics
      (``weight/effective_rank``, ``weight/rank_trajectory``, ``weight/update_delta_rel``)
      with a direction label (↑ growth / ↓ collapse; ↑ norm growth / ↓ norm collapse for
      ``update_delta_rel``).
    - **Grokking Signals**: first sharp transition in loss / accuracy / error /
      perplexity series via :func:`~circuitry.core.dynamics.grokking_step`.
    - **Representation Drift**: per-layer CKA drift from the reference snapshot, with a
      trend label (↑ rising / → stable / ↓ falling / ⚡ step N for sharp transitions).

    Returns ``[]`` when ``step_count <= 1`` or when no events are detected.
    """
    if step_count <= 1:
        return []

    from circuitry.core.dynamics import grokking_step as _gs
    from circuitry.core.dynamics import head_formation_step as _hfs
    from circuitry.core.dynamics import phase_transition_steps as _pts

    # --- Head formation events ---
    # (module, head_idx, score_type, formation_step, last_score)
    formation_events: list[tuple[str, int, str, int, float]] = []

    for tag, series in grouped.items():
        for prefix, thr in _FORMATION_THRESHOLDS.items():
            if not tag.startswith(prefix + "/"):
                continue
            rest = tag[len(prefix) + 1:]
            parts = rest.rsplit("/", 1)
            if len(parts) != 2 or not parts[1].startswith("head_"):
                continue
            module = parts[0]
            try:
                head_idx = int(parts[1][5:])  # len("head_") == 5
            except ValueError:
                continue
            if len(series) < 2:
                continue  # need dynamics (multiple steps)

            step = _hfs(series, threshold=thr)
            if step is None:
                continue
            first_step = min(s for s, _ in series)
            if step == first_step:
                continue  # pre-formed before recording started

            last_score = max(series, key=lambda x: x[0])[1]
            score_type = prefix.split("/", 1)[1]  # e.g. "induction_score"
            formation_events.append((module, head_idx, score_type, step, last_score))

    # --- Phase transitions on rank/health metrics ---
    # (section, row_id, pt_step, before_val, after_val)
    transition_events: list[tuple[str, str, int, float, float]] = []

    for tag, series in grouped.items():
        section, row_id = _section_and_row(tag)
        if section not in _TRANSITION_PREFIXES:
            continue
        if len(series) < 4:
            continue  # too few points for reliable smoothing
        pts = _pts(series)
        if not pts:
            continue
        sorted_s = sorted(series)
        step_vals = {s: v for s, v in sorted_s}
        all_steps = [s for s, _ in sorted_s]
        for pt_step in pts:
            before_steps = [s for s in all_steps if s < pt_step]
            after_steps = [s for s in all_steps if s >= pt_step]
            if not before_steps or not after_steps:
                continue
            transition_events.append(
                (section, row_id, pt_step, step_vals[before_steps[-1]], step_vals[after_steps[0]])
            )

    # --- Grokking signals on loss / accuracy series ---
    # (tag_row_id, grokking_step_val, value_at_step)
    _GROKKING_KEYWORDS = ("loss", "acc", "accuracy", "error", "perplexity")
    grokking_events: list[tuple[str, int, float]] = []
    for tag, series in grouped.items():
        if len(series) < 4:
            continue
        _, row_id = _section_and_row(tag)
        if not any(kw in row_id.lower() for kw in _GROKKING_KEYWORDS):
            continue
        gs = _gs(series)
        if gs is None:
            continue
        val_at = next((v for s, v in sorted(series) if s == gs), None)
        if val_at is not None:
            grokking_events.append((row_id, gs, val_at))

    # --- Representation drift trending ---
    # (row_id, first_drift, last_drift, trend_label)
    drift_trends: list[tuple[str, float, float, str]] = []
    for tag, series in grouped.items():
        section, row_id = _section_and_row(tag)
        if section != "activation/repr_drift":
            continue
        if len(series) < 2:
            continue
        sorted_s = sorted(series)
        vals = [v for _, v in sorted_s]
        drift_trends.append((row_id, vals[0], vals[-1], _drift_trend(vals, sorted_s, _pts)))
    drift_trends.sort(key=lambda x: -x[2])  # most-drifted layers first

    if not formation_events and not transition_events and not grokking_events and not drift_trends:
        return []

    lines: list[str] = ["## Training Dynamics", ""]

    if formation_events:
        formation_events.sort(key=lambda e: (e[0], e[1], e[3]))
        lines.append("### Head Formation Events")
        lines.append("")
        lines.append("| module | head | score | formed at step | final score |")
        lines.append("| --- | --- | --- | ---: | ---: |")
        for module, head_idx, score_type, step, last_score in formation_events:
            lines.append(
                f"| `{module}` | head_{head_idx} | {score_type} | {step} | {last_score:.3f} |"
            )
        lines.append("")

    if transition_events:
        transition_events.sort(key=lambda e: (e[2], e[1]))  # sort by pt_step then row_id
        lines.append("### Phase Transitions")
        lines.append("")
        lines.append("| metric | step | before | after | direction |")
        lines.append("| --- | ---: | ---: | ---: | --- |")
        for section, row_id, pt_step, before, after in transition_events:
            direction = _transition_direction(section, before, after)
            lines.append(
                f"| `{row_id}` | {pt_step} | {before:.4g} | {after:.4g} | {direction} |"
            )
        lines.append("")

    if grokking_events:
        grokking_events.sort(key=lambda e: e[1])
        lines.append("### Grokking Signals")
        lines.append("")
        lines.append("| metric | step | value at transition |")
        lines.append("| --- | ---: | ---: |")
        for row_id, step, val in grokking_events:
            lines.append(f"| `{row_id}` | {step} | {val:.4g} |")
        lines.append("")

    if drift_trends:
        lines.append("### Representation Drift")
        lines.append("")
        lines.append("| module | start drift | end drift | Δ | trend |")
        lines.append("| --- | ---: | ---: | ---: | --- |")
        for row_id, first_d, last_d, trend in drift_trends:
            delta = last_d - first_d
            lines.append(
                f"| `{row_id}` | {first_d:.3f} | {last_d:.3f} | {delta:+.3f} | {trend} |"
            )
        lines.append("")

    return lines


def _build_head_specialization_section(
    grouped: dict[str, list[tuple[int, float]]],
) -> list[str]:
    """Render a ## Head Specialization table from the last-step attention scores.

    Reads ``activation/induction_score``, ``activation/copy_suppression_score``,
    and ``activation/attention_sink_score`` tags; classifies each head; returns
    markdown lines.  Returns ``[]`` when none of those tags are present.
    """
    from circuitry.core.attention import head_specialization as _hs

    _PREFIXES = {
        "activation/induction_score": "ind",
        "activation/copy_suppression_score": "css",
        "activation/attention_sink_score": "snk",
    }

    # module -> head_idx -> {key: last_value}
    scores: dict[str, dict[int, dict[str, float]]] = {}
    for tag, series in grouped.items():
        for prefix, key in _PREFIXES.items():
            if not tag.startswith(prefix + "/"):
                continue
            rest = tag[len(prefix) + 1:]
            parts = rest.rsplit("/", 1)
            if len(parts) != 2 or not parts[1].startswith("head_"):
                continue
            module = parts[0]
            try:
                head_idx = int(parts[1][5:])  # len("head_") == 5
            except ValueError:
                continue
            last_val = max(series, key=lambda x: x[0])[1]
            scores.setdefault(module, {}).setdefault(head_idx, {})[key] = last_val

    if not scores:
        return []

    lines = ["## Head Specialization", ""]
    lines.append("| module | head | type | induction | copy_suppression | sink |")
    lines.append("| --- | --- | --- | ---: | ---: | ---: |")

    for module in sorted(scores):
        heads = scores[module]
        n = max(heads) + 1
        ind = [heads.get(i, {}).get("ind") for i in range(n)]
        css = [heads.get(i, {}).get("css") for i in range(n)]
        snk = [heads.get(i, {}).get("snk") for i in range(n)]

        all_present = all(v is not None for v in ind + css + snk)
        if all_present:
            types = _hs(ind, css, snk)  # type: ignore[arg-type]
        else:
            types = ["—"] * n

        for i in range(n):
            t = types[i]
            type_cell = f"**{t}**" if t not in ("uniform", "—") else t
            ind_cell = f"{ind[i]:.3f}" if ind[i] is not None else "—"
            css_cell = f"{css[i]:.3f}" if css[i] is not None else "—"
            snk_cell = f"{snk[i]:.3f}" if snk[i] is not None else "—"
            lines.append(
                f"| `{module}` | head_{i} | {type_cell} "
                f"| {ind_cell} | {css_cell} | {snk_cell} |"
            )
    lines.append("")
    return lines


def build_report(
    run_dir: str | pathlib.Path,
    out_path: str | pathlib.Path | None = None,
    *,
    compact: bool = False,
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

    # Pre-compute step count so the header can be subtitled.
    grouped = _group(rows) if rows else {}
    step_count = len({s for series in grouped.values() for s, _ in series})
    if step_count == 0:
        subtitle = ""
    elif step_count == 1:
        subtitle = " — static (1 step)"
    else:
        subtitle = f" — dynamic ({step_count} steps)"

    lines: list[str] = [f"# circuitry report{subtitle}", ""]
    lines.append(f"Source run: `{run_dir}`")
    lines.append("")

    matched_path = run_dir / "circuitry" / "matched_modules.txt"
    if not compact and matched_path.exists():
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

    # Top-of-report summary: total / moving / static / emit-step count.
    moving = 0
    static = 0
    for series in grouped.values():
        _, _, vmin, vmax, _ = _stats(series)
        if vmax > vmin:
            moving += 1
        else:
            static += 1

    lines.append("## Summary")
    lines.append("")
    lines.append(
        f"- **{len(grouped)}** scalar tags · **{moving}** moving "
        f"(changed) · **{static}** static · **{step_count}** emit step(s) "
        f"observed."
    )
    # Per-top-level-family tag counts (e.g. "weight: 24 · activation: 12").
    family_counts: dict[str, int] = {}
    for tag in grouped:
        fam = tag.split("/")[0]
        family_counts[fam] = family_counts.get(fam, 0) + 1
    if family_counts:
        fam_parts = " · ".join(
            f"**{fam}**: {cnt}" for fam, cnt in sorted(family_counts.items())
        )
        lines.append(f"- Tags by family: {fam_parts}.")
    if step_count == 1:
        lines.append("")
        lines.append(
            "> **Note:** Single-step run; Δ uniformly zero. For "
            "training-dynamics signal, run multiple steps."
        )
    lines.append("")

    # ## Flags verdict block — inserted after ## Summary, gated on step_count > 1.
    lines.extend(_build_flags(grouped, step_count))

    if compact:
        out_path.write_text("\n".join(lines))
        return out_path

    # Attach summary — written by Recorder.attach(); skip silently if absent (older runs).
    attach_summary_path = run_dir / "circuitry" / "attach_summary.json"
    if attach_summary_path.exists():
        import json as _json
        attach_data = _json.loads(attach_summary_path.read_text())
        lines.append("## Attach summary")
        lines.append("")
        lines.append("| hp | source | target | matched | resolved | unresolved |")
        lines.append("| --- | --- | --- | ---: | ---: | ---: |")
        for entry in attach_data.get("hook_points", []):
            hp_label = f"`{entry['label']}`"
            lines.append(
                f"| {entry['idx']} | {entry['source']} | {hp_label} "
                f"| {entry['matched']} | {entry['resolved']} | {entry['unresolved']} |"
            )
        t = attach_data.get("totals", {})
        lines.append(
            f"| — | — | **total** | {t.get('matched', 0)} "
            f"| {t.get('resolved', 0)} | {t.get('unresolved', 0)} |"
        )
        lines.append("")

    # Group tags by (section header).
    sections: dict[str, list[str]] = defaultdict(list)
    for tag in grouped:
        section, _ = _section_and_row(tag)
        sections[section].append(tag)

    hero = sorted(s for s in sections if s in HERO_SECTIONS)
    advanced = sorted(s for s in sections if s not in HERO_SECTIONS)

    for section in hero:
        lines.extend(_render_section(section, sections[section], grouped))

    # v1.13 head-specialization table: synthesises induction / copy-suppression /
    # sink scores into a per-head type label; only rendered when those tags exist.
    lines.extend(_build_head_specialization_section(grouped))

    # v1.14 training-dynamics: head formation events + phase transitions.
    lines.extend(_build_training_dynamics_section(grouped, step_count))

    if advanced:
        lines.append("<details>")
        lines.append("<summary>Advanced metrics</summary>")
        lines.append("")
        for section in advanced:
            lines.extend(_render_section(section, sections[section], grouped))
        lines.append("</details>")
        lines.append("")

    out_path.write_text("\n".join(lines))
    return out_path
