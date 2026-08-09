"""
Reproducible study harness for the DeepLOB leakage study.

One command, a stranger can regenerate the paper's main table:

    python run_study.py

Sweep: seeds 0..19 x protocols {random_split, chronological, purged_embargoed}
x horizons {5, 10, 20}, window fixed at 100. Fully seeded per run:

    torch.manual_seed(s)                       -> model init, any global RNG use
    random_split(..., generator=Generator(s))  -> the split draw
    DataLoader(..., generator=Generator(s))    -> training batch order

Data generation mirrors run_experiment.py exactly (simulator seed 42,
build_lob_windows defaults, window 100, 5 training epochs, batch 64, Adam lr
1e-3). Scoring is deliberately dual: validation-only accuracy (windows the
model never trained on) and all-windows accuracy (what evaluate.py reports,
including training windows). The leaky pipeline is the subject of study; this
harness measures it, it does not fix it.

Outputs:
    results/study/runs.csv      one row per (protocol, horizon, seed)
    results/study/summary.json  aggregates: mean, sd, min, max, 95% interval
                                (t-based CI of the mean across seeds, and the
                                2.5/97.5 percentiles of the observed spread)

Runtime note: 180 runs, each training a small DeepLOB for 5 epochs on CPU.
Sequential (--workers 1) this takes on the order of 1.5 hours; --workers N
parallelises across processes. Per-run results are bit-deterministic across
processes and thread counts (verified), so --workers does not change numbers.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
import time
from datetime import datetime, timezone
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, TensorDataset, random_split

# ---------------------------------------------------------------------------
# Make src/ importable so we can use deep_lob.*
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from deep_lob.data import build_lob_windows, load_raw_lob  # noqa: E402
from deep_lob.models import DeepLOBModel  # noqa: E402
from deep_lob.simulator import save_simulated_lob_csv  # noqa: E402
from deep_lob.splits import (  # noqa: E402
    chronological_split,
    overlap_fraction,
    purged_embargoed_split,
)

WINDOW_SIZE = 100
N_ROWS = 5000
N_LEVELS = 3
THRESHOLD = 5e-4
BATCH = 64
EPOCHS = 5
LR = 1e-3

PROTOCOLS = ["random_split", "chronological", "purged_embargoed"]
HORIZONS = [5, 10, 20]
DEFAULT_SEEDS = list(range(20))

# t_{0.975} for df = 19 (20 seeds), for the 95% CI of the mean.
T_975_DF19 = 2.0930240544082634

_DF: Any = None


def _init_worker(df) -> None:
    global _DF
    _DF = df


def majority_frac(labels: np.ndarray) -> float:
    counts = np.bincount(np.asarray(labels) + 1, minlength=3)
    return float(counts.max() / counts.sum())


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for Xb, yb in loader:
        Xb, yb = Xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(Xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * Xb.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == yb).sum().item()
        total += yb.size(0)
    return total_loss / total, correct / total


def eval_epoch(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            logits = model(Xb)
            preds = logits.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)
    return correct / total


def run_cell(protocol: str, horizon: int, seed: int, threads: int) -> dict:
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    gen_split = torch.Generator().manual_seed(seed)
    gen_loader = torch.Generator().manual_seed(seed)

    df = _DF
    X, y = build_lob_windows(
        df,
        window_size=WINDOW_SIZE,
        horizon=horizon,
        n_levels=N_LEVELS,
        threshold=THRESHOLD,
    )

    Xt = torch.from_numpy(X).float()
    yt = torch.from_numpy(y).long() + 1
    ds = TensorDataset(Xt, yt)
    n = len(ds)
    n_train = int(0.8 * n)

    if protocol == "random_split":
        train_ds, val_ds = random_split(ds, [n_train, n - n_train], generator=gen_split)
        train_idx = np.asarray(train_ds.indices)
        val_idx = np.asarray(val_ds.indices)
    else:
        if protocol == "chronological":
            sp = chronological_split(n, 0.8)
        else:
            sp = purged_embargoed_split(n, WINDOW_SIZE, horizon, 0.8)
        train_idx = np.asarray(sp.train_idx)
        val_idx = np.asarray(sp.val_idx)
        train_ds = Subset(ds, train_idx)
        val_ds = Subset(ds, val_idx)

    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, generator=gen_loader)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)
    full_loader = DataLoader(ds, batch_size=256, shuffle=False)

    device = torch.device("cpu")
    model = DeepLOBModel(num_features=X.shape[2]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    for _ in range(EPOCHS):
        train_epoch(model, train_loader, optimizer, criterion, device)
    val_acc = eval_epoch(model, val_loader, device)
    full_acc = eval_epoch(model, full_loader, device)

    return {
        "protocol": protocol,
        "horizon": int(horizon),
        "seed": int(seed),
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "val_acc": float(val_acc),
        "full_acc": float(full_acc),
        "val_majority": majority_frac(y[val_idx]),
        "full_majority": majority_frac(y),
        "overlap_fraction": float(overlap_fraction(train_idx, val_idx, WINDOW_SIZE, horizon)),
    }


def aggregate(values: list[float]) -> dict:
    arr = np.asarray(values, dtype="float64")
    mean = float(arr.mean())
    sd = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    se = sd / np.sqrt(len(arr))
    return {
        "mean": mean,
        "sd": sd,
        "min": float(arr.min()),
        "max": float(arr.max()),
        "ci95_of_mean_low": mean - T_975_DF19 * se,
        "ci95_of_mean_high": mean + T_975_DF19 * se,
        "pct2_5": float(np.percentile(arr, 2.5)),
        "pct97_5": float(np.percentile(arr, 97.5)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=4, help="parallel processes")
    ap.add_argument("--threads", type=int, default=2, help="torch threads per worker")
    ap.add_argument("--protocols", nargs="+", default=PROTOCOLS)
    ap.add_argument("--horizons", nargs="+", type=int, default=HORIZONS)
    ap.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    ap.add_argument("--out", type=str, default="results/study")
    args = ap.parse_args()

    protocols = args.protocols
    horizons = args.horizons
    seeds = args.seeds

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()

    with tempfile.TemporaryDirectory() as tmp:
        raw_csv = Path(tmp) / "simulated_lob.csv"
        save_simulated_lob_csv(out_path=raw_csv, n_rows=N_ROWS, seed=42)
        df = load_raw_lob(raw_csv)

    tasks = [
        (p, h, s)
        for p in protocols
        for h in horizons
        for s in seeds
    ]
    n_tasks = len(tasks)
    print(f"Running {n_tasks} cells: {len(protocols)} protocols x {len(horizons)} "
          f"horizons x {len(seeds)} seeds, window={WINDOW_SIZE}, workers={args.workers}")

    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker, initargs=(df,)) as ex:
        futures = [ex.submit(run_cell, p, h, s, args.threads) for (p, h, s) in tasks]
        for i, fut in enumerate(futures, 1):
            rows.append(fut.result())
            if i % 20 == 0 or i == n_tasks:
                print(f"  {i}/{n_tasks} done ({time.perf_counter() - t0:.0f}s)")

    rows.sort(key=lambda r: (r["protocol"], r["horizon"], r["seed"]))

    runs_csv = out_dir / "runs.csv"
    with runs_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "protocol", "horizon", "seed", "val_acc", "full_acc",
                "val_majority", "full_majority", "overlap_fraction",
                "n_train", "n_val",
            ],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    cells: dict[str, dict[str, dict]] = {}
    for p in protocols:
        for h in horizons:
            cell_rows = [r for r in rows if r["protocol"] == p and r["horizon"] == h]
            key = f"{p}|h{h}"
            cells[key] = {
                m: aggregate([r[m] for r in cell_rows])
                for m in ("val_acc", "full_acc", "val_majority", "full_majority",
                          "overlap_fraction")
            }
            cells[key]["n_train"] = {
                "mean": float(np.mean([r["n_train"] for r in cell_rows]))
            }
            cells[key]["n_val"] = {
                "mean": float(np.mean([r["n_val"] for r in cell_rows]))
            }

    wall_clock_seconds = time.perf_counter() - t0

    import torch as _torch
    import numpy as _np
    import pandas as _pd

    summary = {
        "description": (
            "Main table of the DeepLOB leakage study: accuracy by protocol "
            "(random_split / chronological / purged_embargoed) and horizon "
            "(5/10/20), window 100. val_acc is scored on windows the model "
            "never trained on; full_acc includes training windows (as "
            "evaluate.py does). Seeds 0..19."
        ),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "wall_clock_seconds": wall_clock_seconds,
        "n_runs": n_tasks,
        "environment": {
            "python": sys.version.split()[0],
            "torch": _torch.__version__,
            "numpy": _np.__version__,
            "pandas": _pd.__version__,
        },
        "pipeline": {
            "simulator_seed": 42,
            "window_size": WINDOW_SIZE,
            "n_levels": N_LEVELS,
            "threshold": THRESHOLD,
            "epochs": EPOCHS,
            "batch": BATCH,
            "lr": LR,
            "size_normalization": "global_max (leaky legacy default)",
            "scoring": "val_acc = validation-only; full_acc = all windows incl. train",
        },
        "seeding": (
            "torch.manual_seed(s); random_split(generator=Generator(s)); "
            "train DataLoader(generator=Generator(s))"
        ),
        "cells": cells,
    }

    summary_path = out_dir / "summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nWrote {runs_csv}")
    print(f"Wrote {summary_path}")
    print(f"Wall clock: {wall_clock_seconds:.1f}s for {n_tasks} runs")

    print("\n=== aggregate table ===")
    hdr = (f"{'protocol':16} {'h':>3} | {'val mean':>9} {'val sd':>8} {'val min':>8} "
           f"{'val max':>8} | {'full mean':>9} {'full sd':>8} | {'val maj':>8} "
           f"{'full maj':>8} {'overlap':>8}")
    print(hdr)
    print("-" * len(hdr))
    for p in protocols:
        for h in horizons:
            c = cells[f"{p}|h{h}"]
            print(
                f"{p:16} {h:>3} | "
                f"{c['val_acc']['mean']:9.6f} {c['val_acc']['sd']:8.6f} "
                f"{c['val_acc']['min']:8.6f} {c['val_acc']['max']:8.6f} | "
                f"{c['full_acc']['mean']:9.6f} {c['full_acc']['sd']:8.6f} | "
                f"{c['val_majority']['mean']:8.6f} {c['full_majority']['mean']:8.6f} "
                f"{c['overlap_fraction']['mean']:8.6f}"
            )


if __name__ == "__main__":
    main()
