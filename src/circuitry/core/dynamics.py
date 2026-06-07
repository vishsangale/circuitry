"""Training-dynamics primitives for detecting phase transitions in scalar time series.

phase_transition_steps — detect steps where a recorded metric undergoes a sharp
change (curvature spike relative to the local noise floor).

head_formation_step — detect the first training step at which a per-head
attention score crosses a specialization threshold and sustains above it.

These primitives operate on the ``list[tuple[int, float]]`` series format produced
by ``circuitry.recorder._metrics.group``.  They are pure-Python (no torch) so they
can be called during post-hoc report building without a GPU.
"""

from __future__ import annotations


def phase_transition_steps(
    series: list[tuple[int, float]],
    *,
    window: int = 5,
    z_threshold: float = 2.0,
    min_gap: int = 5,
) -> list[int]:
    """Return training steps where the series has a sharp change.

    Algorithm: bilateral mean change — at each candidate center position ``i``,
    compute ``|mean(right_half) - mean(left_half)|`` using half-windows of
    ``window // 2`` points.  Positions whose bilateral change exceeds
    ``mean + z_threshold * std`` across all centers are flagged as transitions.
    Nearby detections within ``min_gap`` index positions are collapsed (keeping
    the largest-magnitude one per cluster).

    Unlike a first-difference of a rolling mean (which spreads a step change
    over the window and depresses the z-score), the bilateral approach
    concentrates all transition energy at the crossing point.  A monotone
    linear ramp produces identical bilateral changes at every center (std = 0)
    and returns ``[]``.

    Returns a sorted list of step values (integers from the series) at which
    transitions were detected.  Returns ``[]`` for series that are too short,
    constant, or have no outlier bilateral changes.

    Args:
        series: Sorted ``(step, value)`` pairs — the format from
            ``circuitry.recorder._metrics.group``.
        window: Controls the half-window size (``half = max(1, window // 2)``);
            centers require ``half`` points on each side.
        z_threshold: Number of standard deviations above the mean bilateral
            change to treat as a transition.
        min_gap: Minimum index distance between two kept detections.
    """
    if len(series) < 2:
        return []

    sorted_s = sorted(series)
    steps = [s for s, _ in sorted_s]
    vals = [v for _, v in sorted_s]

    half = max(1, window // 2)
    n = len(vals)
    if n < 2 * half + 1:
        return []

    # Compute bilateral change at each valid center position.
    changes: list[float] = []
    center_idxs: list[int] = []
    for i in range(half, n - half):
        left_mean = sum(vals[i - half: i]) / half
        right_mean = sum(vals[i: i + half]) / half
        changes.append(abs(right_mean - left_mean))
        center_idxs.append(i)

    if not changes:
        return []

    mean_c = sum(changes) / len(changes)
    var_c = sum((c - mean_c) ** 2 for c in changes) / len(changes)
    std_c = var_c ** 0.5

    if std_c == 0.0:
        return []

    cutoff = mean_c + z_threshold * std_c
    candidates = [
        (center_idxs[i], changes[i])
        for i in range(len(changes))
        if changes[i] > cutoff
    ]
    if not candidates:
        return []

    # Greedy collapse: sort by magnitude descending, keep non-overlapping detections.
    candidates.sort(key=lambda x: -x[1])
    kept_idxs: list[int] = []
    for idx, _ in candidates:
        if any(abs(idx - ki) < min_gap for ki in kept_idxs):
            continue
        kept_idxs.append(idx)

    return sorted(steps[idx] for idx in kept_idxs)


def grokking_step(
    series: list[tuple[int, float]],
    *,
    z_threshold: float = 2.5,
) -> int | None:
    """Return the first sharp transition step in a loss or accuracy time series.

    A convenience wrapper around :func:`phase_transition_steps` that returns
    only the *first* detected transition — the earliest grokking event — rather
    than all of them.  The default ``z_threshold`` is raised to 2.5 (vs 2.0 for
    the general-purpose primitive) to reduce false positives on the noisier loss
    and accuracy curves typically logged during training.

    Returns ``None`` if the series has no statistically sharp change.

    Args:
        series: Sorted ``(step, value)`` pairs — same format as
            :func:`phase_transition_steps`.
        z_threshold: Bilateral-change cutoff in standard deviations above the
            mean.  Default 2.5 is intentionally conservative for noisy loss
            curves.
    """
    pts = phase_transition_steps(series, z_threshold=z_threshold)
    return pts[0] if pts else None


def head_formation_step(
    series: list[tuple[int, float]],
    *,
    threshold: float,
    n_sustain: int = 2,
) -> int | None:
    """Return the first step where the score crosses ``threshold`` and sustains.

    Scans the sorted series and returns the step value at the first index ``i``
    where all values in ``vals[i : i + n_sustain]`` are ``>= threshold``.  If
    the end of the series is reached before ``n_sustain`` confirmations, any
    remaining tail points that are all above the threshold still count (end-of-
    series tolerance — we cannot observe future steps).

    Returns ``None`` if the score never crosses ``threshold``.

    Args:
        series: Sorted ``(step, value)`` pairs.
        threshold: Specialization threshold (e.g. 0.4 for induction heads).
        n_sustain: Number of consecutive points (including the crossing) that
            must all be ``>= threshold`` to confirm formation.
    """
    if not series:
        return None

    sorted_s = sorted(series)
    steps = [s for s, _ in sorted_s]
    vals = [v for _, v in sorted_s]

    for i in range(len(vals)):
        if vals[i] < threshold:
            continue
        window = vals[i: i + n_sustain]
        if all(v >= threshold for v in window):
            return steps[i]

    return None
