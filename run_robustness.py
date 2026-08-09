"""
Robustness of the closed-form ceiling.

Everything published so far rests on one random-walk realisation (seed 42),
one window size (100), one threshold, and one train fraction. This script
attacks all four, plus the theory's own falsifiable prediction.

The prediction. The ceiling

    C = pi * A(h, s, theta, sigma) + (1 - pi) * M

contains stride, horizon, threshold, per-step volatility and train fraction.
It does NOT contain the window size W. So the theory claims W is irrelevant to
how much leakage a split makes available: widening the feature window changes
what the model sees but not how correlated the LABELS of neighbouring windows
are, and the leak lives in the labels.

That is falsifiable and cheap to test, because the oracle memoriser needs no
training at all. If the ceiling predicts the oracle across window sizes, the
formula is right for a reason rather than by coincidence at W=100.

Four sweeps, all against the same closed form:

  paths      20 independent generator seeds       -> is the result one path's quirk?
  window     W in {25, 50, 100, 200, 400}         -> the W-invariance prediction
  threshold  theta in {1e-4 ... 2e-3}             -> does it hold off the calibrated point?
  trainfrac  f in {0.5, 0.6, 0.7, 0.8, 0.9}       -> pi = 1-(1-f)^2 is part of the claim

Output: results/robustness/robustness.csv and .json
Run:    .venv/bin/python run_robustness.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deep_lob.data import build_lob_windows  # noqa: E402
from deep_lob.simulator import simulate_lob  # noqa: E402
from deep_lob.splits import random_overlap_split  # noqa: E402
from run_ceiling import (  # noqa: E402
    analytic_ceiling,
    STEP_SD,
    THRESHOLD,
    TRAIN_FRAC,
    WINDOW_SIZE,
    _bivariate_normal_cdf,
    _norm_cdf,
    majority_frac,
    oracle_twin_accuracy,
)

N_ROWS = 5000
HORIZONS = (5, 10, 20)
SPLIT_SEEDS = tuple(range(10))


def ceiling(horizon: int, threshold: float = THRESHOLD, train_frac: float = TRAIN_FRAC,
            stride: int = 1) -> float:
    """Defer to the single derivation in run_ceiling."""
    return analytic_ceiling(horizon, train_frac=train_frac, stride=stride,
                            threshold=threshold)["ceiling"]


def oracle(y: np.ndarray, train_frac: float = TRAIN_FRAC) -> float:
    accs = []
    for s in SPLIT_SEEDS:
        sp = random_overlap_split(len(y), train_frac, seed=s)
        accs.append(oracle_twin_accuracy(sp.train_idx, sp.val_idx, y))
    return float(np.mean(accs))


def labels(horizon: int, sim_seed: int = 42, window: int = WINDOW_SIZE,
           threshold: float = THRESHOLD) -> np.ndarray:
    df = simulate_lob(n_rows=N_ROWS, seed=sim_seed)
    _, y = build_lob_windows(df, window_size=window, horizon=horizon, threshold=threshold)
    return y


def main() -> None:
    out = ROOT / "results" / "robustness"
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    def record(sweep, param, value, h, pred, obs, maj, n):
        rows.append({"sweep": sweep, "param": param, "value": value, "horizon": h,
                     "ceiling": pred, "oracle": obs, "error": obs - pred,
                     "majority": maj, "n_windows": int(n)})

    # ---- 1. independent generator paths ------------------------------------
    print("=== 1. independent generator paths (is this one path's quirk?) ===")
    print(f"{'h':>3}{'ceiling':>10}{'oracle mean':>13}{'sd':>8}{'min':>8}{'max':>8}{'|max err|':>11}")
    for h in HORIZONS:
        pred = ceiling(h)
        obs = []
        for sd_seed in range(20):
            y = labels(h, sim_seed=sd_seed)
            obs.append(oracle(y))
            record("paths", "sim_seed", sd_seed, h, pred, obs[-1], majority_frac(y), len(y))
        a = np.array(obs)
        print(f"{h:>3}{pred:>10.4f}{a.mean():>13.4f}{a.std(ddof=1):>8.4f}"
              f"{a.min():>8.4f}{a.max():>8.4f}{np.abs(a-pred).max():>11.4f}")

    # ---- 2. THE PREDICTION: window size should not matter -------------------
    print("\n=== 2. window size (the formula says W is irrelevant) ===")
    print(f"{'h':>3}{'W':>6}{'ceiling':>10}{'oracle':>9}{'error':>9}{'n_win':>8}")
    for h in HORIZONS:
        pred = ceiling(h)
        for W in (25, 50, 100, 200, 400):
            y = labels(h, window=W)
            obs = oracle(y)
            record("window", "window_size", W, h, pred, obs, majority_frac(y), len(y))
            print(f"{h:>3}{W:>6}{pred:>10.4f}{obs:>9.4f}{obs-pred:>+9.4f}{len(y):>8}")
        print()

    # ---- 3. threshold -------------------------------------------------------
    print("=== 3. labelling threshold ===")
    print(f"{'h':>3}{'theta':>10}{'ceiling':>10}{'oracle':>9}{'error':>9}")
    for h in HORIZONS:
        for th in (1e-4, 2.5e-4, 5e-4, 1e-3, 2e-3):
            y = labels(h, threshold=th)
            pred, obs = ceiling(h, threshold=th), oracle(y)
            record("threshold", "threshold", th, h, pred, obs, majority_frac(y), len(y))
            print(f"{h:>3}{th:>10.1e}{pred:>10.4f}{obs:>9.4f}{obs-pred:>+9.4f}")
        print()

    # ---- 4. train fraction --------------------------------------------------
    print("=== 4. train fraction (pi = 1-(1-f)^2 is part of the claim) ===")
    print(f"{'h':>3}{'f':>7}{'ceiling':>10}{'oracle':>9}{'error':>9}")
    for h in HORIZONS:
        y = labels(h)
        for f in (0.5, 0.6, 0.7, 0.8, 0.9):
            pred, obs = ceiling(h, train_frac=f), oracle(y, train_frac=f)
            record("trainfrac", "train_frac", f, h, pred, obs, majority_frac(y), len(y))
            print(f"{h:>3}{f:>7.1f}{pred:>10.4f}{obs:>9.4f}{obs-pred:>+9.4f}")
        print()

    with (out / "robustness.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    with (out / "robustness.json").open("w") as fh:
        json.dump(rows, fh, indent=2)

    print("=== worst absolute error, by sweep ===")
    for sweep in ("paths", "window", "threshold", "trainfrac"):
        sel = [abs(r["error"]) for r in rows if r["sweep"] == sweep]
        print(f"  {sweep:<12} max |error| = {max(sel):.4f}   mean = {np.mean(sel):.4f}  (n={len(sel)})")
    print(f"\nwrote {out/'robustness.csv'}")


if __name__ == "__main__":
    main()
