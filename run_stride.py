"""
The remedy: stride, and the collapse of the leakage ceiling.

run_ceiling.py derives an upper bound on manufactured accuracy for stride-1
windows. That derivation generalises, and the generalisation is prescriptive.

With stride s, consecutive retained windows are s rows apart, so their forward
returns over horizon h share h - s of h i.i.d. increments:

    rho(s) = max(0, (h - s) / h)

At s = 1 this is (h-1)/h, close to 1 — adjacent labels are near-duplicates.
At s >= h it is exactly 0 — retained windows share no increments at all, their
labels are independent, and a memoriser that copies its nearest training
neighbour can do no better than the majority baseline.

So the closed form predicts that leakage from window overlap disappears once
the stride reaches the label horizon. That is a design rule a practitioner can
apply before running anything, and it is falsifiable.

This script tests it. Windows are built at stride 1 as the pipeline does, then
subsampled at stride s — which is exactly a stride-s window set — and the
oracle memoriser from run_ceiling.py is run against the predicted ceiling at
each stride.

Note the trade: raising the stride also shrinks the dataset by a factor of s.
The columns report n_windows so the cost is visible alongside the benefit.

Output: results/stride/stride.csv and stride.json
Run:    .venv/bin/python run_stride.py
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

from deep_lob.data import build_lob_windows, load_raw_lob  # noqa: E402
from deep_lob.simulator import save_simulated_lob_csv  # noqa: E402
from deep_lob.splits import random_overlap_split  # noqa: E402
from run_ceiling import (  # noqa: E402
    STEP_SD,
    THRESHOLD,
    TRAIN_FRAC,
    WINDOW_SIZE,
    N_ROWS,
    SIM_SEED,
    _bivariate_normal_cdf,
    _norm_cdf,
    majority_frac,
    oracle_twin_accuracy,
)

HORIZONS = (5, 10, 20)
STRIDES = (1, 2, 3, 5, 8, 10, 15, 20, 30)
SPLIT_SEEDS = tuple(range(10))


def ceiling_at_stride(horizon: int, stride: int, train_frac: float = TRAIN_FRAC) -> dict:
    """Closed-form ceiling with the stride-generalised correlation."""
    sd = STEP_SD * np.sqrt(horizon)
    z = THRESHOLD / sd

    p_flat = float(2.0 * _norm_cdf(z) - 1.0)
    majority = max(p_flat, (1.0 - p_flat) / 2.0)

    rho = max(0.0, (horizon - stride) / horizon)

    if rho <= 0.0:
        # independent labels: agreement is chance under the marginal distribution
        p_up = p_down = (1.0 - p_flat) / 2.0
        agree = p_flat**2 + p_up**2 + p_down**2
    else:
        p_dd = _bivariate_normal_cdf(-z, -z, rho)
        p_le = _bivariate_normal_cdf(z, z, rho)
        p_ff = p_le - 2.0 * _bivariate_normal_cdf(-z, z, rho) + p_dd
        p_uu = 1.0 - 2.0 * float(_norm_cdf(z)) + p_le
        agree = p_ff + p_dd + p_uu

    p_twin = 1.0 - (1.0 - train_frac) ** 2
    return {
        "rho": rho,
        "p_adjacent_agree": agree,
        "majority_predicted": majority,
        "ceiling": p_twin * agree + (1.0 - p_twin) * majority,
    }


def main() -> None:
    out_dir = ROOT / "results" / "stride"
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_csv = ROOT / "data" / "raw" / "simulated_lob.csv"
    if not raw_csv.exists():
        save_simulated_lob_csv(out_path=raw_csv, n_rows=N_ROWS, seed=SIM_SEED)
    df = load_raw_lob(raw_csv)

    rows: list[dict] = []
    print(f"{'h':>3}{'stride':>8}{'rho':>7}{'ceiling':>10}{'oracle':>9}{'excess':>9}{'n_win':>8}")
    print("-" * 54)

    for h in HORIZONS:
        X_all, y_all = build_lob_windows(
            df, window_size=WINDOW_SIZE, horizon=h, threshold=THRESHOLD
        )
        for s in STRIDES:
            keep = np.arange(0, len(y_all), s)
            y = y_all[keep]
            n = len(y)
            if n < 200:
                continue
            c = ceiling_at_stride(h, s)

            oracles = []
            for seed in SPLIT_SEEDS:
                sp = random_overlap_split(n, TRAIN_FRAC, seed=seed)
                oracles.append(oracle_twin_accuracy(sp.train_idx, sp.val_idx, y))
            oracle = float(np.mean(oracles))
            maj = majority_frac(y)

            rows.append(
                {
                    "horizon": h,
                    "stride": s,
                    "rho": c["rho"],
                    "p_adjacent_agree": c["p_adjacent_agree"],
                    "ceiling": c["ceiling"],
                    "oracle_twin_acc": oracle,
                    "oracle_excess_over_majority": oracle - maj,
                    "majority": maj,
                    "n_windows": int(n),
                }
            )
            print(
                f"{h:>3}{s:>8}{c['rho']:>7.3f}{c['ceiling']:>10.4f}"
                f"{oracle:>9.4f}{oracle-maj:>+9.4f}{n:>8}"
            )
        print()

    with (out_dir / "stride.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with (out_dir / "stride.json").open("w") as f:
        json.dump(rows, f, indent=2)

    # the headline: does the leak vanish at stride >= horizon?
    print("=== leak at stride 1 vs stride >= horizon ===")
    for h in HORIZONS:
        a = next(r for r in rows if r["horizon"] == h and r["stride"] == 1)
        b = next(
            (r for r in rows if r["horizon"] == h and r["stride"] >= h), None
        )
        if b:
            print(
                f"h={h:<3} stride 1: oracle {a['oracle_twin_acc']:.4f} "
                f"(+{a['oracle_excess_over_majority']:.4f} over majority, n={a['n_windows']})"
                f"   ->   stride {b['stride']}: oracle {b['oracle_twin_acc']:.4f} "
                f"({b['oracle_excess_over_majority']:+.4f}, n={b['n_windows']})"
            )
    print(f"\nwrote {out_dir/'stride.csv'}")


if __name__ == "__main__":
    main()
