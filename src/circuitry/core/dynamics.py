"""Training-dynamics primitives for detecting phase transitions in scalar time series.

phase_transition_steps — detect steps where a recorded metric undergoes a sharp
change (curvature spike relative to the local noise floor).

head_formation_step — detect the first training step at which a per-head
attention score crosses a specialization threshold and sustains above it.

fourier_feature_alignment — measure how much a weight matrix's column spectrum
aligns with task-relevant Fourier modes (Nanda et al., ICLR 2024).

information_bottleneck_score — normalized mutual information proxy: how much
activations predict labels (generalisation / compression progress measure).

These primitives operate on the ``list[tuple[int, float]]`` series format produced
by ``circuitry.recorder._metrics.group``.  They are pure-Python (no torch) so they
can be called during post-hoc report building without a GPU.

``fourier_feature_alignment`` and ``information_bottleneck_score`` additionally
accept ``torch.Tensor`` arguments and require PyTorch.
"""

from __future__ import annotations

import torch


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


# ---------------------------------------------------------------------------
# Fourier-feature alignment (Nanda et al., ICLR 2024)
# ---------------------------------------------------------------------------

def fourier_feature_alignment(
    W: torch.Tensor,
    task_freqs: list[int],
    *,
    n_freqs: int | None = None,
) -> float:
    """Measure alignment between a weight matrix's column spectrum and Fourier modes.

    Computes how much of W's variance is concentrated in the task-relevant
    Fourier frequency components.  High values (→1.0) indicate the weight
    matrix has learned to represent the task-relevant frequencies; near-zero
    values indicate random or non-Fourier structure.

    Algorithm:
    1. For each column of W (shape d_out × d_in), compute the 1-D DFT along d_in.
    2. Sum the power spectral density across columns: PSD[k] = sum_j |DFT(W_j)[k]|²
    3. Normalize to a probability distribution over frequencies.
    4. Alignment = sum of PSD at task_freqs (fraction of total power in task modes).

    Args:
        W:          (d_out, d_in) weight tensor. d_in is the "input" axis over which
                    the Fourier transform is computed.
        task_freqs: list of integer frequency indices (0-based) that correspond to
                    task-relevant Fourier modes.  E.g. for modular-arithmetic
                    grokking at modulus 113, pass the top-K active frequencies
                    found by analyzing the embedding matrix.
        n_freqs:    if set, only keep the first n_freqs frequency bins (truncates
                    the spectrum).  Useful for filtering noise at high frequencies.
                    Defaults to d_in // 2 + 1 (the one-sided spectrum).

    Returns:
        float in [0, 1]: fraction of spectral power in task_freqs.
        Returns 0.0 if task_freqs is empty or W has fewer than 2 columns.
    """
    if not task_freqs:
        return 0.0

    if W.shape[-1] < 2:
        return 0.0

    W_f = W.float()

    # 1-D real FFT along the input axis (last dim); shape → (d_out, d_in // 2 + 1)
    fft_W = torch.fft.rfft(W_f, dim=-1)

    # Power spectral density summed across output channels → (d_in // 2 + 1,)
    psd = fft_W.abs().pow(2).sum(0)

    if n_freqs is not None:
        psd = psd[:n_freqs]

    # Normalize to a probability distribution
    psd = psd / psd.sum().clamp(min=1e-12)

    # Keep only indices that fall within the (possibly truncated) spectrum
    valid_task_freqs = [f for f in task_freqs if f < len(psd)]
    if not valid_task_freqs:
        return 0.0

    return psd[valid_task_freqs].sum().item()


# ---------------------------------------------------------------------------
# Information-bottleneck progress score
# ---------------------------------------------------------------------------

def information_bottleneck_score(
    acts_train: torch.Tensor,
    acts_val: torch.Tensor,
    labels_train: torch.Tensor,
    labels_val: torch.Tensor,
    *,
    n_bins: int = 20,
    eps: float = 1e-10,
) -> float:
    """Information bottleneck progress measure: normalized I(T;Y) proxy.

    Estimates the "label predictability from activations" score as:

        score = I(acts_val, labels_val) / (H(labels_val) + eps)

    where I(T;Y) is the mutual information between the binned activation
    representation and the class labels (estimated via histogram counts), and
    H(Y) is the label entropy.  A score near 1 means the representation almost
    perfectly predicts the label; a score near 0 means the representation carries
    little information about the label.

    Activations are projected onto the first principal component (highest
    variance direction across the combined train+val set) before binning, giving
    a cheap 1-D summary that is robust for high-dimensional hidden states.

    Args:
        acts_train: (n_train, d) activation tensor (training set).
        acts_val:   (n_val, d) activation tensor (validation set).
        labels_train: (n_train,) integer labels.
        labels_val:   (n_val,) integer labels.
        n_bins:     number of bins for the activation histogram (per dimension).
                    Uses the first principal component (highest variance direction)
                    to reduce to 1-D before binning — cheap and robust.
        eps:        small constant for numerical stability.

    Returns:
        float in [0, 1] — the IB progress score.
        Returns 0.0 if either input has fewer than 2 samples or fewer than 2 classes.
    """
    n_train = acts_train.shape[0]
    n_val = acts_val.shape[0]

    if n_train < 2 or n_val < 2:
        return 0.0

    # Need at least 2 distinct classes
    unique_train = labels_train.unique()
    unique_val = labels_val.unique()
    if unique_train.numel() < 2 or unique_val.numel() < 2:
        return 0.0

    # Cast to float32 for PCA
    acts_train_f = acts_train.float()
    acts_val_f = acts_val.float()

    # Project onto first principal component (computed on combined set)
    combined = torch.cat([acts_train_f, acts_val_f], dim=0)
    mean = combined.mean(0)
    std = combined.std(0).clamp(min=1e-8)
    combined_std = (combined - mean) / std

    # pca_lowrank returns (U, S, V); V has shape (d, q); first PC is V[:, 0]
    _, _, V = torch.pca_lowrank(combined_std, q=1)
    pc_dir = V[:, 0]  # shape (d,)

    z_train = ((acts_train_f - mean) / std) @ pc_dir  # (n_train,)
    z_val = ((acts_val_f - mean) / std) @ pc_dir       # (n_val,)

    # Bin boundaries based on combined range
    z_all = torch.cat([z_train, z_val])
    z_min = z_all.min().item()
    z_max = z_all.max().item()
    if z_min == z_max:
        # No variance — activations are constant; score is 0
        return 0.0

    # Map to integer bin indices in [0, n_bins-1]
    def to_bins(z: torch.Tensor) -> torch.Tensor:
        scaled = (z - z_min) / (z_max - z_min) * n_bins
        return scaled.long().clamp(0, n_bins - 1)

    bins_val = to_bins(z_val)           # (n_val,)
    labels_val_long = labels_val.long()

    # Compute I(T; Y) on validation set via joint histogram
    n_classes = int(max(unique_val.max().item(), unique_train.max().item())) + 1

    # Joint distribution p(t, y) — shape (n_bins, n_classes)
    joint = torch.zeros(n_bins, n_classes, dtype=torch.float64)
    for idx in range(n_val):
        b = bins_val[idx].item()
        y = labels_val_long[idx].item()
        if 0 <= y < n_classes:
            joint[b, y] += 1.0
    joint = joint / joint.sum().clamp(min=eps)

    p_t = joint.sum(dim=1)   # marginal over bins
    p_y = joint.sum(dim=0)   # marginal over labels

    # I(T;Y) = sum_{t,y} p(t,y) log( p(t,y) / (p(t)*p(y)) )
    outer = p_t.unsqueeze(1) * p_y.unsqueeze(0)  # (n_bins, n_classes)
    ratio = joint / (outer + eps)
    # Only sum where joint > 0 to avoid log(0)
    mask = joint > 0
    mi = (joint[mask] * ratio[mask].log()).sum().item()
    mi = max(0.0, mi)  # numerical guard

    # H(Y) = -sum_y p(y) log p(y)
    p_y_pos = p_y[p_y > 0]
    h_y = -(p_y_pos * p_y_pos.log()).sum().item()
    h_y = max(h_y, 0.0)

    score = mi / (h_y + eps)
    # Clip to [0, 1]
    return float(min(max(score, 0.0), 1.0))



def emergence_score(
    series: list[tuple[int, float]],
    *,
    window: int = 5,
    eps: float = 1e-12,
) -> float:
    """Smoothed second log-derivative of a scalar training series.

    Computes ``d²M / d(log step)²`` — the second derivative of the metric with
    respect to log-step.  A sharp spike indicates a sudden non-linear
    acceleration: the emergence signature of a capability appearing abruptly.

    Contrast with :func:`phase_transition_steps` (which detects *any* sharp
    change, smooth or abrupt, via a bilateral-mean filter): this function
    specifically quantifies how *curved* the metric is on a log-step axis,
    which is the appropriate axis for emergent-capability detection (Wei et al.
    2022 found emergence on log-FLOPs scales).

    Args:
        series: List of ``(step, value)`` pairs, sorted ascending by step.
        window: Smoothing half-width (triangular window); reduces noise.
        eps:    Guard for log(0) and division by zero.

    Returns:
        Maximum absolute value of the smoothed second log-derivative.
        Returns 0.0 for series shorter than 3 points.

    Reference: arXiv:2508.04401 "Measuring Emergence"; Wei et al. 2022
               "Emergent Abilities of Large Language Models".
    """
    import math as _math

    if len(series) < 3:
        return 0.0

    steps, values = zip(*sorted(series, key=lambda x: x[0]), strict=True)
    log_steps = [_math.log(max(s, 1)) for s in steps]
    vals = list(values)
    n = len(vals)

    # Triangular moving average
    def _smooth(arr: list[float]) -> list[float]:
        out = []
        for i in range(len(arr)):
            lo, hi = max(0, i - window), min(len(arr), i + window + 1)
            out.append(sum(arr[lo:hi]) / (hi - lo))
        return out

    sv = _smooth(vals)
    ls = log_steps  # already smooth (log is monotone)

    # First derivative: dM / d(log step) via finite differences
    d1 = [(sv[i + 1] - sv[i]) / (ls[i + 1] - ls[i] + eps) for i in range(n - 1)]

    # Second derivative: d²M / d(log step)²
    ls_mid = [(ls[i] + ls[i + 1]) / 2 for i in range(n - 1)]
    d2 = [(d1[i + 1] - d1[i]) / (ls_mid[i + 1] - ls_mid[i] + eps) for i in range(len(d1) - 1)]

    if not d2:
        return 0.0
    return float(max(abs(v) for v in d2))
