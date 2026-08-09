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
    results/study/runs.csv      one row per (protocol, horizon, sim_seed, seed)
    results/study/summary.json  aggregates: mean, sd, min, max, 95% interval
                                (t-based CI of the mean across seeds, and the
                                2.5/97.5 percentiles of the observed spread)

Multi-generator sweep
---------------------

    python run_study.py --sim-seeds 0 1 2 3 4 --out results/study-multiseed

Runs the same 3 protocols x 3 horizons x 20 split seeds on every generator
seed in --sim-seeds, so the trained-model result can be checked on independent
realisations, not only the seed-42 path. The summary adds, per cell:

    across_paths   mean/sd of the per-path means of val_acc (paths = the
                   per-path mean over the 20 split seeds)
    per_path       the same aggregates per generator seed
    conclusion_check  for each (cell, path): whether the qualitative claim
                   holds on that path individually --- honest protocols below
                   their majority baseline at every horizon, random_split above
                   it at h >= 10. Contradictions are reported loudly, not
                   averaged away.

--max-minutes caps the wall clock (default 180); on timeout the run aborts
without writing results.

Unseeded arm
------------

    python run_study.py --unseeded-cell random_split|h20 --unseeded-reps 6

Runs one (protocol, horizon) cell 6 times with NO seeding anywhere: no
torch.manual_seed, no split generator, no loader generator, same data
(simulator seed 42). This reproduces the setup of the pre-harness runs the
paper's reproducibility section cites ("six runs ... 0.674 to 0.749"), which
are not reproducible from this repository, and records the spread as an
artifact.

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

# t_{0.975} quantiles for the 95% CI of the mean. Cells hold 20 seeds per
# (protocol, horizon, generator seed), so df is a multiple of 19.
# df=19 is exact (matches the committed study); the larger df are close
# approximations good to ~1e-5, used only by the multiseed cells.
T_975_DF19 = 2.0930240544082634
_T_975 = {
    19: T_975_DF19,
    39: 2.0226909200,
    59: 2.0009954409,
    79: 1.9904495237,
    99: 1.9842169519,
}


def _t975(df: int) -> float:
    if df in _T_975:
        return _T_975[df]
    if df < 30:
        return T_975_DF19
    return 1.96


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


def run_cell(protocol: str, horizon: int, seed: int, threads: int,
             sim_seed: int = 42) -> dict:
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
        "sim_seed": int(sim_seed),
        "seed": int(seed),
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "val_acc": float(val_acc),
        "full_acc": float(full_acc),
        "val_majority": majority_frac(y[val_idx]),
        "full_majority": majority_frac(y),
        "overlap_fraction": float(overlap_fraction(train_idx, val_idx, WINDOW_SIZE, horizon)),
    }


def run_cell_unseeded(protocol: str, horizon: int, threads: int, rep: int, df) -> dict:
    """The same cell with no seeding at all: global torch RNG drives the split,
    the model init and the batch order. This is the setup the pre-harness runs
    used. `rep` is only a label; nothing is seeded from it."""
    torch.set_num_threads(threads)

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
        train_ds, val_ds = random_split(ds, [n_train, n - n_train])
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

    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
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
        "rep": int(rep),
        "protocol": protocol,
        "horizon": int(horizon),
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "val_acc": float(val_acc),
        "full_acc": float(full_acc),
        "val_majority": majority_frac(y[val_idx]),
        "full_majority": majority_frac(y),
    }


def aggregate(values: list[float], df: int | None = None) -> dict:
    arr = np.asarray(values, dtype="float64")
    mean = float(arr.mean())
    sd = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    se = sd / np.sqrt(len(arr))
    t = _t975(df if df is not None else len(arr) - 1)
    return {
        "mean": mean,
        "sd": sd,
        "min": float(arr.min()),
        "max": float(arr.max()),
        "ci95_of_mean_low": mean - t * se,
        "ci95_of_mean_high": mean + t * se,
        "pct2_5": float(np.percentile(arr, 2.5)),
        "pct97_5": float(np.percentile(arr, 97.5)),
    }


def spread(values: list[float]) -> dict:
    arr = np.asarray(values, dtype="float64")
    sd = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    return {
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "sd": sd,
        "range": float(arr.max() - arr.min()),
    }


def load_dataframe(sim_seed: int):
    with tempfile.TemporaryDirectory() as tmp:
        raw_csv = Path(tmp) / "simulated_lob.csv"
        save_simulated_lob_csv(out_path=raw_csv, n_rows=N_ROWS, seed=sim_seed)
        return load_raw_lob(raw_csv)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=4, help="parallel processes")
    ap.add_argument("--threads", type=int, default=2, help="torch threads per worker")
    ap.add_argument("--protocols", nargs="+", default=PROTOCOLS)
    ap.add_argument("--horizons", nargs="+", type=int, default=HORIZONS)
    ap.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    ap.add_argument("--sim-seeds", nargs="+", type=int, default=[42],
                    help="generator (simulator) seeds; one full sweep each")
    ap.add_argument("--out", type=str, default="results/study")
    ap.add_argument("--max-minutes", type=float, default=180.0,
                    help="abort (write nothing) once the wall clock exceeds this")
    ap.add_argument("--unseeded-cell", type=str, default=None,
                    help="run an unseeded arm for one cell, e.g. random_split|h20")
    ap.add_argument("--unseeded-reps", type=int, default=6)
    args = ap.parse_args()

    protocols = args.protocols
    horizons = args.horizons
    seeds = args.seeds
    sim_seeds = args.sim_seeds

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_tasks = len(protocols) * len(horizons) * len(seeds) * len(sim_seeds)
    cap_seconds = args.max_minutes * 60.0

    baseline = None
    study_sum = ROOT / "results" / "study" / "summary.json"
    if study_sum.exists():
        try:
            baseline = json.loads(study_sum.read_text()).get("wall_clock_seconds")
        except Exception:
            baseline = None

    t0 = time.perf_counter()
    start_utc = datetime.now(timezone.utc)
    print(f"Starting {n_tasks} runs at {start_utc.isoformat()} UTC")
    print(f"  workers={args.workers} threads/worker={args.threads} "
          f"sim_seeds={sim_seeds} protocols={protocols} horizons={horizons} seeds={len(seeds)}")
    if baseline:
        est_min = n_tasks * baseline / 180.0 / 60.0
        print(f"  estimated wall clock: ~{est_min:.0f} min "
              f"(baseline {baseline:.0f}s for 180 runs at {args.workers} workers)")
    else:
        print("  estimated wall clock: unknown (no results/study/summary.json baseline)")
    print(f"  hard stop at {args.max_minutes:.0f} minutes; on abort nothing is written")

    rows: list[dict] = []
    done = 0

    for sim_seed in sim_seeds:
        df = load_dataframe(sim_seed)
        block_tasks = [(p, h, s) for p in protocols for h in horizons for s in seeds]
        print(f"\n--- generator path sim_seed={sim_seed} "
              f"({len(block_tasks)} cells) ---")
        with ProcessPoolExecutor(
            max_workers=args.workers, initializer=_init_worker, initargs=(df,)
        ) as ex:
            futures = [ex.submit(run_cell, p, h, s, args.threads, sim_seed)
                       for (p, h, s) in block_tasks]
            for i, fut in enumerate(futures, 1):
                rows.append(fut.result())
                done += 1
                if done % 20 == 0 or done == n_tasks:
                    elapsed = time.perf_counter() - t0
                    print(f"  {done}/{n_tasks} done ({elapsed:.0f}s elapsed)")
                if time.perf_counter() - t0 > cap_seconds:
                    print(f"\nABORT: wall clock {(time.perf_counter()-t0)/60:.1f} min "
                          f"exceeds the {args.max_minutes:.0f} min cap after {done}/{n_tasks} "
                          f"cells. Nothing written.", flush=True)
                    ex.shutdown(wait=False, cancel_futures=True)
                    sys.exit(1)

    unseeded_arm = None
    if args.unseeded_cell:
        if "|" not in args.unseeded_cell:
            sys.exit("--unseeded-cell must look like random_split|h20")
        up, uh = args.unseeded_cell.split("|")
        uh = int(uh.removeprefix("h"))
        df = load_dataframe(42)
        print(f"\n--- unseeded arm: {args.unseeded_cell}, {args.unseeded_reps} reps, "
              f"no seeding anywhere ---")
        reps = []
        for i in range(args.unseeded_reps):
            r = run_cell_unseeded(up, uh, args.threads, i + 1, df)
            reps.append(r)
            print(f"  rep {i+1}: val_acc={r['val_acc']:.4f} full_acc={r['full_acc']:.4f}")
        unseeded_arm = {
            "cell": args.unseeded_cell,
            "n_reps": args.unseeded_reps,
            "simulator_seed": 42,
            "seeding": (
                "none: no torch.manual_seed, no split generator, no loader "
                "generator; the global torch RNG drives init, split and batch order"
            ),
            "context": (
                "The paper's reproducibility section cites six pre-harness runs "
                "spanning 0.674-0.749 that predate this harness and are not "
                "reproducible from the repository. This arm measures the same "
                "setup (same data, no seeding) with the harness, as an artifact."
            ),
            "reps": reps,
            "val_acc": spread([r["val_acc"] for r in reps]),
            "full_acc": spread([r["full_acc"] for r in reps]),
        }
        print(f"  val_acc spread: {unseeded_arm['val_acc']['min']:.4f}.."
              f"{unseeded_arm['val_acc']['max']:.4f} "
              f"(mean {unseeded_arm['val_acc']['mean']:.4f}, "
              f"sd {unseeded_arm['val_acc']['sd']:.4f})")
        print(f"  full_acc spread: {unseeded_arm['full_acc']['min']:.4f}.."
              f"{unseeded_arm['full_acc']['max']:.4f} "
              f"(mean {unseeded_arm['full_acc']['mean']:.4f}, "
              f"sd {unseeded_arm['full_acc']['sd']:.4f})")

    rows.sort(key=lambda r: (r["protocol"], r["horizon"], r["sim_seed"], r["seed"]))

    runs_csv = out_dir / "runs.csv"
    with runs_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "protocol", "horizon", "sim_seed", "seed", "val_acc", "full_acc",
                "val_majority", "full_majority", "overlap_fraction",
                "n_train", "n_val",
            ],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    multi = len(sim_seeds) > 1
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
            if multi:
                per_path: dict[str, dict] = {}
                for s in sim_seeds:
                    pr = [r for r in cell_rows if r["sim_seed"] == s]
                    per_path[str(s)] = {
                        m: aggregate([r[m] for r in pr])
                        for m in ("val_acc", "val_majority")
                    }
                cells[key]["per_path"] = per_path
                path_means = {str(s): per_path[str(s)]["val_acc"]["mean"] for s in sim_seeds}
                pm = np.asarray(list(path_means.values()))
                cells[key]["across_paths"] = {
                    "metric": "per-path mean of val_acc over the 20 split seeds; "
                              "aggregated across generator paths",
                    "means": path_means,
                    "mean": float(pm.mean()),
                    "sd": float(pm.std(ddof=1)) if len(pm) > 1 else 0.0,
                    "min": float(pm.min()),
                    "max": float(pm.max()),
                }

    wall_clock_seconds = time.perf_counter() - t0

    conclusion_check = None
    contradictions: list[dict] = []
    if multi:
        checks: dict[str, dict] = {}
        print("\n=== per-path conclusion check (path mean over the 20 split seeds) ===")
        print(f"{'cell':24} {'sim':>3} {'val_acc':>9} {'val_maj':>9} {'diff':>9}  check")
        for p in protocols:
            for h in horizons:
                for s in sim_seeds:
                    pr = [r for r in rows if r["protocol"] == p and r["horizon"] == h
                          and r["sim_seed"] == s]
                    acc = float(np.mean([r["val_acc"] for r in pr]))
                    maj = float(np.mean([r["val_majority"] for r in pr]))
                    if p == "random_split":
                        if h >= 10:
                            holds = acc > maj
                            label = "leaky above baseline"
                        else:
                            holds = None
                            label = "informational (h=5 excluded)"
                    else:
                        holds = acc < maj
                        label = "honest below baseline"
                    key = f"{p}|h{h}"
                    checks[f"{key}|sim{s}"] = {
                        "cell": key,
                        "sim_seed": s,
                        "val_acc_mean": acc,
                        "val_majority_mean": maj,
                        "diff": acc - maj,
                        "rule": label,
                        "holds": holds,
                    }
                    flag = "TRUE" if holds else ("-" if holds is None else "FALSE <<")
                    print(f"{key:24} {s:>3} {acc:9.4f} {maj:9.4f} {acc-maj:+9.4f}  {flag}")
                    if holds is False:
                        contradictions.append(checks[f"{key}|sim{s}"])
        conclusion_check = {
            "rule": (
                "for each generator path individually: honest protocols "
                "(chronological, purged_embargoed) score below their own "
                "validation majority baseline at every horizon; random_split "
                "scores above it at h >= 10. Means are over the 20 split seeds."
            ),
            "per_cell": checks,
            "contradictions": contradictions,
            "all_hold": len(contradictions) == 0,
        }
        if contradictions:
            print("\n*** CONTRADICTIONS (a single path contradicts the conclusion) ***")
            for c in contradictions:
                print(f"  {c['cell']} sim_seed={c['sim_seed']}: val_acc {c['val_acc_mean']:.4f} "
                      f"vs majority {c['val_majority_mean']:.4f} "
                      f"(diff {c['diff']:+.4f}) — rule: {c['rule']}")
        else:
            print("\nNo path contradicts the conclusion.")

    import torch as _torch
    import numpy as _np
    import pandas as _pd

    summary = {
        "description": (
            "DeepLOB leakage study: accuracy by protocol "
            "(random_split / chronological / purged_embargoed) and horizon "
            "(5/10/20), window 100. val_acc is scored on windows the model "
            "never trained on; full_acc includes training windows (as "
            "evaluate.py does)."
        ),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "wall_clock_seconds": wall_clock_seconds,
        "n_runs": n_tasks,
        "sweep": {
            "sim_seeds": sim_seeds,
            "split_seeds": seeds,
            "protocols": protocols,
            "horizons": horizons,
            "per_cell_runs": len(seeds),
        },
        "environment": {
            "python": sys.version.split()[0],
            "torch": _torch.__version__,
            "numpy": _np.__version__,
            "pandas": _pd.__version__,
        },
        "pipeline": {
            "simulator_seed": sim_seeds if multi else 42,
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
    if conclusion_check is not None:
        summary["conclusion_check"] = conclusion_check
    if unseeded_arm is not None:
        summary["unseeded_arm"] = unseeded_arm

    summary_path = out_dir / "summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nWrote {runs_csv}")
    print(f"Wrote {summary_path}")
    print(f"Wall clock: {wall_clock_seconds:.1f}s for {n_tasks} runs")

    print("\n=== aggregate table (across all runs in the cell) ===")
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

    if multi:
        print("\n=== val_acc across generator paths (mean of 20 split seeds per path) ===")
        print(f"{'cell':24} {'mean':>9} {'sd':>9} {'min':>8} {'max':>8}   per-path means")
        for p in protocols:
            for h in horizons:
                ap = cells[f"{p}|h{h}"]["across_paths"]
                print(f"{p + '|h' + str(h):24} {ap['mean']:9.4f} {ap['sd']:9.4f} "
                      f"{ap['min']:8.4f} {ap['max']:8.4f}   "
                      + " ".join(f"{k}:{v:.4f}" for k, v in ap["means"].items()))


if __name__ == "__main__":
    main()
