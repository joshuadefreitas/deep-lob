"""
Deterministic splitting strategies for overlapping sliding-window datasets.

Background
----------
`deep_lob.data.build_lob_windows` produces overlapping windows with stride 1:
window ``i`` covers raw rows ``[i, i + window_size)`` and its label depends on
row ``i + window_size - 1 + horizon``. Two windows ``i`` and ``j`` therefore
touch (and can leak information through) the same raw rows whenever

    |i - j| < window_size + horizon

Randomly shuffling window indices before a train/validation split (as
``torch.utils.data.random_split`` does) all but guarantees that many
validation windows sit within this distance of a training window, so the
model can partially "see" validation rows during training. This module
provides that leaky baseline alongside two increasingly rigorous
alternatives so the effect can be measured, not just asserted.

References: see docs/references/ (Bergmeir & Benitez 2012; Cerqueira et al.
2020; Lopez de Prado 2018, ch. 7 "Cross-Validation in Finance").
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def n_windows_for(n_rows: int, window_size: int, horizon: int) -> int:
    """Number of valid window start indices, matching build_lob_windows."""
    return max(0, n_rows - window_size - horizon + 1)


def window_span(window_size: int, horizon: int) -> int:
    """Number of raw rows a single window "touches" (features + label row)."""
    return window_size + horizon


@dataclass(frozen=True)
class SplitResult:
    name: str
    train_idx: np.ndarray
    val_idx: np.ndarray


def random_overlap_split(
    n_windows: int,
    train_frac: float = 0.8,
    seed: int = 0,
) -> SplitResult:
    """
    Reproduces the current/default behaviour of ``random_split`` in
    train.py / train_tcn.py: window indices are shuffled uniformly at
    random before the train/val cut. This is the leaky baseline.
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_windows)
    cut = int(train_frac * n_windows)
    train_idx = np.sort(perm[:cut])
    val_idx = np.sort(perm[cut:])
    return SplitResult("random_overlap", train_idx, val_idx)


def chronological_split(
    n_windows: int,
    train_frac: float = 0.8,
) -> SplitResult:
    """
    Train on the first `train_frac` windows in time order, validate on the
    remainder. No purge/embargo: windows immediately adjacent to the cut
    point can still share raw rows across the boundary.
    """
    cut = int(train_frac * n_windows)
    train_idx = np.arange(0, cut)
    val_idx = np.arange(cut, n_windows)
    return SplitResult("chronological", train_idx, val_idx)


def purged_embargoed_split(
    n_windows: int,
    window_size: int,
    horizon: int,
    train_frac: float = 0.8,
    embargo: int | None = None,
) -> SplitResult:
    """
    Chronological split with purging + embargo (Lopez de Prado, 2018):

    - Purge: drop training windows near the cut whose raw-row span would
      overlap the validation region.
    - Embargo: additionally drop the first `embargo` validation windows
      after the cut, to absorb residual serial dependence at the boundary.

    Default embargo = horizon (the minimum buffer needed so that no
    validation window's label lookback crosses back into training data).
    """
    if embargo is None:
        embargo = horizon

    cut = int(train_frac * n_windows)
    purge_distance = window_span(window_size, horizon) - 1

    train_end = max(0, cut - purge_distance)
    val_start = min(n_windows, cut + embargo)

    train_idx = np.arange(0, train_end)
    val_idx = np.arange(val_start, n_windows)
    return SplitResult("purged_embargoed", train_idx, val_idx)


def overlap_fraction(
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    window_size: int,
    horizon: int,
) -> float:
    """
    Fraction of validation windows whose raw-row span intersects the
    raw-row span of at least one training window. 0.0 means the split is
    free of window-overlap leakage; values near 1.0 mean nearly every
    validation window shares data with some training window.
    """
    if len(val_idx) == 0:
        return 0.0
    if len(train_idx) == 0:
        return 0.0

    span = window_span(window_size, horizon)
    train_sorted = np.sort(train_idx)

    # Window i touches raw rows [i, i + span). Windows i, j overlap iff
    # |i - j| < span. For each val index, check whether any train index
    # falls within `span` using searchsorted (both idx arrays are sorted).
    n_overlapping = 0
    for j in np.sort(val_idx):
        lo = np.searchsorted(train_sorted, j - span + 1, side="left")
        hi = np.searchsorted(train_sorted, j + span - 1, side="right")
        if hi > lo:
            n_overlapping += 1

    return n_overlapping / len(val_idx)
