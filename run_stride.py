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

Configuration span
------------------
Measured across: 5 independent generator paths (sim seeds 0-4), 3 horizons,
9 strides, 10 split seeds per cell.
NOT varied across: window size (100), threshold (5e-4), train fraction (0.8),
number of rows (5000), model (none - the oracle needs no model).

An earlier version measured ONE generator path. The remedy is a headline claim
and it was verified on a single random walk; that is the failure protocol rule
11 exists to prevent.

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

from deep_lob.data import build_lob_windows  # noqa: E402
from deep_lob.simulator import simulate_lob  # noqa: E402
from deep_lob.splits import random_overlap_split  # noqa: E402
from run_ceiling import (  # noqa: E402
    analytic_ceiling,
    STEP_SD,
    THRESHOLD,
    TRAIN_FRAC,
    WINDOW_SIZE,
    N_ROWS,
    _bivariate_normal_cdf,
    _norm_cdf,
    majority_frac,
    oracle_twin_accuracy,
)

HORIZONS = (5, 10, 20)
STRIDES = (1, 2, 3, 5, 8, 10, 15, 20, 30)
SPLIT_SEEDS = tuple(range(10))
SIM_SEEDS = (0, 1, 2, 3, 4)
# Two generator lengths. 5000 matches the rest of the study; at stride = horizon
# it leaves ~49 validation windows, so one window is worth 2 accuracy points and
# the verification of the remedy is underpowered exactly where the remedy bites.
# 20000 gives ~199 validation windows at s=h. Both are reported.
ROW_COUNTS = (5000, 20000)


def ceiling_at_stride(horizon: int, stride: int, train_frac: float = TRAIN_FRAC) -> dict:
    """
    Thin wrapper over run_ceiling.analytic_ceiling.

    This module used to carry its own copy of the ceiling algebra. It now
    defers, so the repository has exactly one derivation and the two scripts
    cannot drift apart.
    """
    a = analytic_ceiling(horizon, train_frac=train_frac, stride=stride)
    return {
        "rho": a["rho_adjacent"],
        "p_adjacent_agree": a["p_adjacent_agree"],
        "majority_predicted": a["majority_predicted"],
        "ceiling": a["ceiling"],
    }


def main() -> None:
    out_dir = ROOT / "results" / "stride"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for n_rows in ROW_COUNTS:
      for sim_seed in SIM_SEEDS:
        df = simulate_lob(n_rows=n_rows, seed=sim_seed)
        for h in HORIZONS:
            X_all, y_all = build_lob_windows(
                df, window_size=WINDOW_SIZE, horizon=h, threshold=THRESHOLD
            )
            for st in STRIDES:
                y = y_all[::st]
                n = len(y)
                if n < 200:
                    continue
                c = ceiling_at_stride(h, st)
                oracles = [
                    oracle_twin_accuracy(sp.train_idx, sp.val_idx, y)
                    for sp in (random_overlap_split(n, TRAIN_FRAC, seed=s) for s in SPLIT_SEEDS)
                ]
                oracle = float(np.mean(oracles))
                maj = majority_frac(y)
                rows.append({
                    "n_rows": n_rows, "sim_seed": sim_seed, "horizon": h, "stride": st,
                    "rho": c["rho"], "p_adjacent_agree": c["p_adjacent_agree"],
                    "ceiling": c["ceiling"], "oracle_twin_acc": oracle,
                    "oracle_excess_over_majority": oracle - maj,
                    "majority": maj, "n_windows": int(n),
                })
        print(f"  rows={n_rows} path={sim_seed} done")

    with (out_dir / "stride.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    with (out_dir / "stride.json").open("w") as f:
        json.dump(rows, f, indent=2)

    # ---- the claim, tested on every path separately -----------------------
    print("\n=== DOES THE LEAK VANISH AT STRIDE >= HORIZON, ON EVERY PATH? ===")
    print(f"{'rows':>7}{'h':>4}{'path':>6}{'n_val':>7}{'s=1':>10}{'s=h':>10}   verdict")
    print("-" * 56)
    violations = []
    for n_rows in ROW_COUNTS:
      for h in HORIZONS:
        for sim_seed in SIM_SEEDS:
            a = next(r for r in rows if r["n_rows"]==n_rows and r["sim_seed"] == sim_seed and r["horizon"] == h and r["stride"] == 1)
            b = next((r for r in rows if r["n_rows"]==n_rows and r["sim_seed"] == sim_seed and r["horizon"] == h and r["stride"] >= h), None)
            if b is None:
                continue
            ok = a["oracle_excess_over_majority"] > 0 and b["oracle_excess_over_majority"] <= 0
            if not ok:
                violations.append((n_rows, h, sim_seed, b["oracle_excess_over_majority"]))
            print(f"{n_rows:>7}{h:>4}{sim_seed:>6}{int(0.2*b['n_windows']):>7}"
                  f"{a['oracle_excess_over_majority']:>+10.4f}{b['oracle_excess_over_majority']:>+10.4f}"
                  f"   {'OK' if ok else 'VIOLATION'}")
        print()

    print("=== closed form vs oracle, pooled over paths ===")
    import statistics as st
    errs = [abs(r["oracle_twin_acc"] - r["ceiling"]) for r in rows]
    print(f"  n = {len(errs)}   mean |error| = {st.mean(errs):.4f}   max = {max(errs):.4f}")

    print("\nVIOLATIONS:", violations if violations else "none - the remedy holds on all 5 paths")
    print(f"wrote {out_dir/'stride.csv'}")


if __name__ == "__main__":
    main()
