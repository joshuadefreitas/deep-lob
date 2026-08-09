"""
The leakage ceiling: how much accuracy a split geometry makes available.

The study (run_study.py) measures what a CNN achieves under each split
protocol. This script asks a different question, one that does not involve a
neural network at all:

    Given only the geometry of the windows and the correlation structure of
    the labels, how much apparent accuracy is AVAILABLE to be manufactured?

The argument
------------
Windows are built at stride 1, so window t and window t+1 share
window_size - 1 raw rows. Their labels are the sign of the forward return
over horizon h, thresholded at +/- theta:

    r_t   = (p_{t+h}   - p_t)   / p_t        spans increments t+1 .. t+h
    r_{t+1}                                   spans increments t+2 .. t+h+1

For an i.i.d. random walk these are sums of h increments sharing h-1 of them,
so

    corr(r_t, r_{t+1}) = (h - 1) / h

which rises with the horizon. The probability that two adjacent windows carry
the SAME label follows from the bivariate normal with that correlation and the
threshold expressed in units of the h-step return's standard deviation.

A model that copies the label of its NEAREST training window therefore scores,
in expectation, the agreement probability averaged over the distance to that
neighbour. Under a random split at train fraction f the distance d is
two-sided geometric,

    P(d = k) = (1-f)^(2(k-1)) - (1-f)^(2k)

and a neighbour d windows away shares h - d*stride increments, so

    C = sum_d P(d) * A(max(0, (h - d*stride)/h))

An earlier version of this file used P(twin)*A(adjacent) + P(no twin)*majority,
which assumes a memoriser with no adjacent twin gives up and guesses. It does
not; it copies its second-nearest neighbour. That error was invisible at f=0.8
(0.015) and large at f=0.5 (0.094).

This is an upper bound on what leakage can produce. Nothing about the model
enters it. An oracle handed its nearest training neighbour attains it. A
1-nearest-neighbour classifier in raw FEATURE space does not, and scores near
chance - Euclidean distance over price-level features finds windows at similar
price levels rather than temporal twins.

Three estimators are compared against the bound:

  analytic        the closed form above
  oracle_twin     predict the label of the temporally nearest TRAINING window.
                  This is the memoriser the derivation describes, with the
                  neighbour handed to it for free. It isolates the label
                  correlation model from any question of whether a learner
                  could FIND the neighbour.
  knn1            1-nearest-neighbour in feature space. A real memoriser that
                  has to locate its twin from the features alone.

Output: results/ceiling/ceiling.csv and ceiling.json.
Run:    .venv/bin/python run_ceiling.py
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

from deep_lob.data import build_lob_windows  # noqa: E402
from deep_lob.simulator import simulate_lob  # noqa: E402
from deep_lob.splits import (  # noqa: E402
    chronological_split,
    purged_embargoed_split,
    random_overlap_split,
)

# Mirrors run_study.py and run_experiment.py exactly.
WINDOW_SIZE = 100
N_ROWS = 5000
SIM_SEED = 42
THRESHOLD = 5e-4
TRAIN_FRAC = 0.8
HORIZONS = (5, 10, 20)
SPLIT_SEEDS = tuple(range(20))
SIM_SEEDS = (0, 1, 2, 3, 4, 42)  # rule 11: state the span, and make it > 1

# simulator.py: mids[t] = mids[t-1] * (1 + N(0, 0.03)/100)
# so the per-step log return has standard deviation 0.03/100.
STEP_SD = 0.03 / 100.0


# ----------------------------------------------------------------------
# the closed form
# ----------------------------------------------------------------------
def _bivariate_normal_cdf(a: float, b: float, rho: float, n: int = 4001) -> float:
    """
    P(X <= a, Y <= b) for a standard bivariate normal with correlation rho.

    Computed by conditioning on X and integrating numerically rather than
    pulling in scipy: the repository has no scipy dependency and adding one
    for a single function would be a poor trade. Simpson's rule over a
    truncated domain; the truncation error is far below the precision this
    result is quoted to.
    """
    lo = -8.0
    if a <= lo:
        return 0.0
    xs = np.linspace(lo, a, n)
    s = np.sqrt(max(1.0 - rho * rho, 1e-15))
    inner = _norm_cdf((b - rho * xs) / s)
    pdf = np.exp(-0.5 * xs * xs) / np.sqrt(2.0 * np.pi)
    return float(np.trapezoid(inner * pdf, xs))


def _norm_cdf(z):
    from math import erf, sqrt

    z = np.asarray(z, dtype="float64")
    return np.vectorize(lambda v: 0.5 * (1.0 + erf(v / sqrt(2.0))))(z)


def _agreement_at_distance(z: float, rho: float, p_flat: float) -> float:
    """P(two windows d apart carry the same label), given their label correlation."""
    if rho <= 0.0:
        p_up = (1.0 - p_flat) / 2.0
        return p_flat**2 + 2.0 * p_up**2
    p_dd = _bivariate_normal_cdf(-z, -z, rho)
    p_le = _bivariate_normal_cdf(z, z, rho)
    p_ff = p_le - 2.0 * _bivariate_normal_cdf(-z, z, rho) + p_dd
    p_uu = 1.0 - 2.0 * float(_norm_cdf(z)) + p_le
    return p_ff + p_dd + p_uu


def analytic_ceiling(horizon: int, train_frac: float = TRAIN_FRAC,
                     stride: int = 1, threshold: float = THRESHOLD) -> dict:
    """
    Closed-form ceiling on manufactured accuracy for a random split.

    A memoriser copies the label of its NEAREST training window, which is not
    always the adjacent one. Under a random split at train fraction f, the
    distance d to the nearest training neighbour has

        P(d > k)   = (1-f)^(2k)          both sides validation out to k
        P(d = k)   = (1-f)^(2(k-1)) - (1-f)^(2k)

    and a neighbour d windows away shares h - d*stride of h increments, so its
    label correlation is rho_d = max(0, (h - d*stride)/h). The ceiling is the
    expectation of the agreement probability over that distance distribution:

        C = sum_d P(d) * A(rho_d)

    An earlier version collapsed everything beyond d=1 to the majority class.
    That under-predicted badly at low train fractions - by 0.094 at h=20,
    f=0.5, where a quarter of validation windows have no adjacent twin - and
    the error vanished as f rose, which is the signature of exactly this
    mis-specification.
    """
    sd = STEP_SD * np.sqrt(horizon)
    z = threshold / sd
    p_flat = float(2.0 * _norm_cdf(z) - 1.0)
    p_up = p_down = (1.0 - p_flat) / 2.0
    majority = max(p_flat, p_up)

    q = 1.0 - train_frac
    ceiling = 0.0
    tail = 1.0
    distances = {}
    d = 1
    while tail > 1e-9 and d < 2000:
        p_d = q ** (2 * (d - 1)) - q ** (2 * d)
        rho_d = max(0.0, (horizon - d * stride) / horizon)
        ceiling += p_d * _agreement_at_distance(z, rho_d, p_flat)
        distances[d] = p_d
        tail -= p_d
        d += 1
    ceiling += tail * _agreement_at_distance(z, 0.0, p_flat)

    rho1 = max(0.0, (horizon - stride) / horizon)
    return {
        "horizon": horizon,
        "stride": stride,
        "label_sd": sd,
        "z": z,
        "p_flat": p_flat,
        "majority_predicted": majority,
        "rho_adjacent": rho1,
        "p_adjacent_agree": _agreement_at_distance(z, rho1, p_flat),
        "p_has_training_twin": 1.0 - (1.0 - train_frac) ** 2,
        "ceiling": ceiling,
    }


# ----------------------------------------------------------------------
# empirical estimators — no neural network involved
# ----------------------------------------------------------------------
def oracle_twin_accuracy(train_idx, val_idx, y) -> float:
    """
    For each validation window, copy the label of the temporally nearest
    training window. The neighbour is handed over for free, so this measures
    the label-correlation model alone.
    """
    train_sorted = np.sort(train_idx)
    pos = np.searchsorted(train_sorted, val_idx)
    left = np.clip(pos - 1, 0, len(train_sorted) - 1)
    right = np.clip(pos, 0, len(train_sorted) - 1)
    d_left = np.abs(val_idx - train_sorted[left])
    d_right = np.abs(train_sorted[right] - val_idx)
    nearest = np.where(d_left <= d_right, train_sorted[left], train_sorted[right])
    return float((y[nearest] == y[val_idx]).mean())


def knn1_accuracy(train_idx, val_idx, X, y, chunk: int = 256) -> float:
    """
    1-nearest-neighbour in flattened feature space, Euclidean, exact.
    Chunked so the distance matrix never materialises in full.
    """
    Xtr = X[train_idx].reshape(len(train_idx), -1).astype("float32")
    Xva = X[val_idx].reshape(len(val_idx), -1).astype("float32")
    tr_sq = (Xtr * Xtr).sum(axis=1)
    correct = 0
    for start in range(0, len(Xva), chunk):
        q = Xva[start : start + chunk]
        d = tr_sq[None, :] - 2.0 * (q @ Xtr.T)  # + |q|^2, constant per row
        nearest = d.argmin(axis=1)
        correct += int((y[train_idx[nearest]] == y[val_idx[start : start + chunk]]).sum())
    return correct / len(Xva)


def majority_frac(y) -> float:
    _, counts = np.unique(y, return_counts=True)
    return float(counts.max() / counts.sum())


# ----------------------------------------------------------------------
def main() -> None:
    out_dir = ROOT / "results" / "ceiling"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    analytics: dict[str, dict] = {}

    for sim_seed in SIM_SEEDS:
      df = simulate_lob(n_rows=N_ROWS, seed=sim_seed)
      for h in HORIZONS:
        X, y = build_lob_windows(
            df, window_size=WINDOW_SIZE, horizon=h, threshold=THRESHOLD
        )
        n = len(y)
        a = analytic_ceiling(h)
        a["majority_measured"] = majority_frac(y)
        a["n_windows"] = int(n)
        a["sim_seed"] = sim_seed
        analytics[f"s{sim_seed}|h{h}"] = a

        print(
            f"h={h:<3} windows={n:<5} "
            f"P(flat)={a['p_flat']:.4f} maj_pred={a['majority_predicted']:.4f} "
            f"maj_meas={a['majority_measured']:.4f} rho={a['rho_adjacent']:.3f} "
            f"agree={a['p_adjacent_agree']:.4f} CEILING={a['ceiling']:.4f}"
        )

        for protocol in ("random_split", "chronological", "purged_embargoed"):
            seeds = SPLIT_SEEDS if protocol == "random_split" else (0,)
            for s in seeds:
                if protocol == "random_split":
                    sp = random_overlap_split(n, TRAIN_FRAC, seed=s)
                elif protocol == "chronological":
                    sp = chronological_split(n, TRAIN_FRAC)
                else:
                    sp = purged_embargoed_split(n, WINDOW_SIZE, h, TRAIN_FRAC)

                rows.append(
                    {
                        "sim_seed": sim_seed,
                        "protocol": protocol,
                        "horizon": h,
                        "split_seed": s,
                        "n_train": int(len(sp.train_idx)),
                        "n_val": int(len(sp.val_idx)),
                        "val_majority": majority_frac(y[sp.val_idx]),
                        "oracle_twin_acc": oracle_twin_accuracy(sp.train_idx, sp.val_idx, y),
                        "knn1_acc": knn1_accuracy(sp.train_idx, sp.val_idx, X, y),
                        "analytic_ceiling": a["ceiling"],
                    }
                )
        print(f"      ...{len([r for r in rows if r['horizon'] == h])} split evaluations done")

    with (out_dir / "ceiling.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    summary: dict = {"analytic": analytics, "empirical": {}}
    print(f"\n{'protocol':<20}{'h':>3}{'oracle':>10}{'knn1':>10}{'ceiling':>10}{'val maj':>10}   (pooled over %d paths)" % len(SIM_SEEDS))
    print("-" * 78)
    for protocol in ("random_split", "chronological", "purged_embargoed"):
        for h in HORIZONS:
            sel = [r for r in rows if r["protocol"] == protocol and r["horizon"] == h]
            o = float(np.mean([r["oracle_twin_acc"] for r in sel]))
            k = float(np.mean([r["knn1_acc"] for r in sel]))
            mj = float(np.mean([r["val_majority"] for r in sel]))
            c = sel[0]["analytic_ceiling"]
            summary["empirical"][f"{protocol}|h{h}"] = {
                "oracle_twin_acc": o,
                "knn1_acc": k,
                "val_majority": mj,
                "analytic_ceiling": c,
                "n_splits": len(sel),
            }
            print(f"{protocol:<20}{h:>3}{o:>10.4f}{k:>10.4f}{c:>10.4f}{mj:>10.4f}")

    with (out_dir / "ceiling.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {out_dir/'ceiling.csv'} and {out_dir/'ceiling.json'}")


if __name__ == "__main__":
    main()
