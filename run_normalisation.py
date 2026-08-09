"""Paired measurement of the legacy full-data normalisation leak.

The sweep holds the study harness fixed and changes only feature scaling:

* ``global_max`` calls :func:`run_study.run_cell` directly.
* ``train_only`` builds raw-size windows, creates the identical split, calls
  ``TrainOnlyScaler.fit(X[train_idx])``, and uses that fitted scaler for every
  train, validation, and full-dataset feature value.

Default sweep: 3 protocols x 3 horizons x 2 scaling arms x 20 paired seeds =
360 trained models. Results are written only after the complete sweep succeeds.
The artifact is specific to the synthetic simulator seed 42 and CPU DeepLOB
configuration inherited from ``run_study.py``; it makes no claim about real
exchange data or other models.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, TensorDataset, random_split

import run_study as study
from deep_lob.data import build_lob_windows
from deep_lob.models import DeepLOBModel
from deep_lob.scaling import TrainOnlyScaler
from deep_lob.splits import chronological_split, overlap_fraction, purged_embargoed_split

ROOT = Path(__file__).resolve().parent
SCALINGS = ("global_max", "train_only")
METRICS = ("val_acc", "full_acc", "val_majority", "full_majority", "overlap_fraction")
DEFAULT_OUT = ROOT / "results" / "normalisation"


def _init_worker(df: pd.DataFrame) -> None:
    study._DF = df


def _split_indices(
    protocol: str, n: int, horizon: int, gen_split: torch.Generator
) -> tuple[np.ndarray, np.ndarray]:
    n_train = int(0.8 * n)
    if protocol == "random_split":
        index_ds = TensorDataset(torch.arange(n))
        train_ds, val_ds = random_split(
            index_ds, [n_train, n - n_train], generator=gen_split
        )
        return np.asarray(train_ds.indices), np.asarray(val_ds.indices)
    if protocol == "chronological":
        split = chronological_split(n, 0.8)
    elif protocol == "purged_embargoed":
        split = purged_embargoed_split(n, study.WINDOW_SIZE, horizon, 0.8)
    else:
        raise ValueError(f"unknown protocol: {protocol}")
    return np.asarray(split.train_idx), np.asarray(split.val_idx)


def scale_train_only(X: np.ndarray, train_idx: np.ndarray) -> np.ndarray:
    """Fit on exactly ``X[train_idx]`` and transform the complete X array."""
    scaler = TrainOnlyScaler().fit(X[train_idx])
    return scaler.transform(X).astype("float32", copy=False)


def run_train_only_cell(
    protocol: str, horizon: int, seed: int, threads: int, sim_seed: int = 42
) -> dict[str, Any]:
    """Run the study model with raw windows and train-only feature scaling."""
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    gen_split = torch.Generator().manual_seed(seed)
    gen_loader = torch.Generator().manual_seed(seed)

    X, y = build_lob_windows(
        study._DF,
        window_size=study.WINDOW_SIZE,
        horizon=horizon,
        n_levels=study.N_LEVELS,
        threshold=study.THRESHOLD,
        size_normalization="none",
    )
    train_idx, val_idx = _split_indices(protocol, len(X), horizon, gen_split)
    X_scaled = scale_train_only(X, train_idx)

    Xt = torch.from_numpy(X_scaled).float()
    yt = torch.from_numpy(y).long() + 1
    ds = TensorDataset(Xt, yt)
    train_ds = Subset(ds, train_idx)
    val_ds = Subset(ds, val_idx)

    train_loader = DataLoader(
        train_ds,
        batch_size=study.BATCH,
        shuffle=True,
        generator=gen_loader,
    )
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)
    full_loader = DataLoader(ds, batch_size=256, shuffle=False)

    device = torch.device("cpu")
    model = DeepLOBModel(num_features=X.shape[2]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=study.LR)
    criterion = nn.CrossEntropyLoss()
    for _ in range(study.EPOCHS):
        study.train_epoch(model, train_loader, optimizer, criterion, device)

    return {
        "protocol": protocol,
        "horizon": int(horizon),
        "scaling": "train_only",
        "sim_seed": int(sim_seed),
        "seed": int(seed),
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "val_acc": float(study.eval_epoch(model, val_loader, device)),
        "full_acc": float(study.eval_epoch(model, full_loader, device)),
        "val_majority": study.majority_frac(y[val_idx]),
        "full_majority": study.majority_frac(y),
        "overlap_fraction": float(
            overlap_fraction(train_idx, val_idx, study.WINDOW_SIZE, horizon)
        ),
    }


def run_pair(protocol: str, horizon: int, seed: int, threads: int) -> list[dict[str, Any]]:
    """Run both arms, resetting every seeded RNG inside each arm."""
    global_row = study.run_cell(protocol, horizon, seed, threads, sim_seed=42)
    global_row["scaling"] = "global_max"
    train_only_row = run_train_only_cell(protocol, horizon, seed, threads)
    return [global_row, train_only_row]


def _paired_difference(global_values: list[float], train_values: list[float]) -> dict[str, float]:
    differences = np.asarray(global_values, dtype="float64") - np.asarray(
        train_values, dtype="float64"
    )
    n = len(differences)
    mean = float(differences.mean())
    sd = float(differences.std(ddof=1)) if n > 1 else 0.0
    half_width = study._t975(n - 1) * sd / math.sqrt(n)
    return {
        "definition": "global_max val_acc minus train_only val_acc, paired by seed",
        "n_pairs": n,
        "mean": mean,
        "mean_percentage_points": 100.0 * mean,
        "sd": sd,
        "ci95_low": mean - half_width,
        "ci95_high": mean + half_width,
        "ci95_low_percentage_points": 100.0 * (mean - half_width),
        "ci95_high_percentage_points": 100.0 * (mean + half_width),
    }


def _baseline_sign(excess: float, tolerance: float = 1e-15) -> str:
    if excess > tolerance:
        return "above"
    if excess < -tolerance:
        return "below"
    return "equal"


def _load_study_rows(path: Path) -> dict[tuple[str, int, int], dict[str, str]]:
    with path.open(newline="") as handle:
        return {
            (row["protocol"], int(row["horizon"]), int(row["seed"])): row
            for row in csv.DictReader(handle)
        }


def compare_global_to_study(
    rows: list[dict[str, Any]], study_csv: Path
) -> dict[str, Any]:
    """Require exact numeric reproduction for every shared study row."""
    expected = _load_study_rows(study_csv)
    observed = {
        (r["protocol"], r["horizon"], r["seed"]): r
        for r in rows
        if r["scaling"] == "global_max"
    }
    shared = sorted(expected.keys() & observed.keys())
    numeric_fields = (*METRICS, "n_train", "n_val")
    mismatches: list[dict[str, Any]] = []
    max_abs_delta = {field: 0.0 for field in numeric_fields}
    for key in shared:
        for field in numeric_fields:
            exp = float(expected[key][field])
            got = float(observed[key][field])
            delta = abs(got - exp)
            max_abs_delta[field] = max(max_abs_delta[field], delta)
            if delta != 0.0:
                mismatches.append(
                    {
                        "protocol": key[0],
                        "horizon": key[1],
                        "seed": key[2],
                        "field": field,
                        "expected": exp,
                        "observed": got,
                        "absolute_delta": delta,
                    }
                )
    return {
        "reference": str(study_csv.relative_to(ROOT)),
        "comparison": "exact float equality after CSV parsing",
        "shared_rows": len(shared),
        "reference_rows_not_run": len(expected.keys() - observed.keys()),
        "run_rows_not_in_reference": len(observed.keys() - expected.keys()),
        "reproduced_exactly": len(shared) > 0 and not mismatches,
        "mismatch_count": len(mismatches),
        "max_absolute_delta_by_field": max_abs_delta,
        "mismatches": mismatches,
    }


def build_summary(
    rows: list[dict[str, Any]], wall_clock_seconds: float, reproduction: dict[str, Any]
) -> dict[str, Any]:
    cells: dict[str, Any] = {}
    sign_changes: list[dict[str, Any]] = []
    for protocol in study.PROTOCOLS:
        for horizon in study.HORIZONS:
            key = f"{protocol}|h{horizon}"
            arm_rows = {
                scaling: sorted(
                    (
                        r
                        for r in rows
                        if r["protocol"] == protocol
                        and r["horizon"] == horizon
                        and r["scaling"] == scaling
                    ),
                    key=lambda r: r["seed"],
                )
                for scaling in SCALINGS
            }
            if not arm_rows["global_max"] and not arm_rows["train_only"]:
                continue
            arms = {
                scaling: {metric: study.aggregate([r[metric] for r in arm_rows[scaling]])
                          for metric in METRICS}
                for scaling in SCALINGS
            }
            global_excess = arms["global_max"]["val_acc"]["mean"] - arms["global_max"][
                "val_majority"
            ]["mean"]
            train_excess = arms["train_only"]["val_acc"]["mean"] - arms["train_only"][
                "val_majority"
            ]["mean"]
            global_sign = _baseline_sign(global_excess)
            train_sign = _baseline_sign(train_excess)
            changed = global_sign != train_sign
            sign_check = {
                "baseline": "mean val_majority for the same arm and cell",
                "global_max_excess": global_excess,
                "global_max_sign": global_sign,
                "train_only_excess": train_excess,
                "train_only_sign": train_sign,
                "changed_sign": changed,
            }
            if changed:
                sign_changes.append({"cell": key, **sign_check})
            paired = _paired_difference(
                [r["val_acc"] for r in arm_rows["global_max"]],
                [r["val_acc"] for r in arm_rows["train_only"]],
            )
            cells[key] = {"arms": arms, "paired_val_acc_difference": paired,
                          "baseline_sign_check": sign_check}

    purged = {
        key: {
            "mean_difference_percentage_points": value["paired_val_acc_difference"][
                "mean_percentage_points"
            ],
            "absolute_mean_move_exceeds_2_points": abs(
                value["paired_val_acc_difference"]["mean_percentage_points"]
            ) > 2.0,
        }
        for key, value in cells.items()
        if key.startswith("purged_embargoed|")
    }

    return {
        "description": "Paired measurement of full-data global-max normalisation versus train-only scaling.",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "wall_clock_seconds": wall_clock_seconds,
        "n_runs": len(rows),
        "n_paired_seeds": len(rows) // 2,
        "environment": {
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "device": "cpu",
        },
        "pipeline": {
            "source_harness": "run_study.py",
            "simulator_seed": 42,
            "window_size": study.WINDOW_SIZE,
            "n_levels": study.N_LEVELS,
            "threshold": study.THRESHOLD,
            "epochs": study.EPOCHS,
            "batch": study.BATCH,
            "optimizer": "torch.optim.Adam",
            "lr": study.LR,
            "global_max": "run_study.run_cell called directly",
            "train_only": "build_lob_windows(size_normalization='none'); TrainOnlyScaler.fit(X[train_idx]); scaler.transform(X)",
            "scaler_fit_array": "X[train_idx] ONLY",
            "seeding": "torch.manual_seed(seed); random_split(generator=Generator(seed)); train DataLoader(generator=Generator(seed))",
        },
        "scope_limits": [
            "Synthetic simulator seed 42 only.",
            "DeepLOBModel on CPU with the run_study.py configuration only.",
            "No real exchange data, trading, PnL, Sharpe, or causal-performance claim.",
        ],
        "global_max_reproduction": reproduction,
        "cells": cells,
        "purged_embargoed_assessment": purged,
        "baseline_sign_changes": sign_changes,
        "any_baseline_sign_change": bool(sign_changes),
        "not_verified": [
            "Independent verification by a separate reviewer was not performed by this runner.",
            "GPU execution was not tested; the study and this measurement run on CPU.",
            "Generalisability beyond simulator seed 42 was not tested.",
        ],
    }


def _write_results(rows: list[dict[str, Any]], summary: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "runs.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "protocol", "horizon", "scaling", "sim_seed", "seed", "val_acc",
                "full_acc", "val_majority", "full_majority", "overlap_fraction",
                "n_train", "n_val",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--protocols", nargs="+", default=study.PROTOCOLS)
    parser.add_argument("--horizons", nargs="+", type=int, default=study.HORIZONS)
    parser.add_argument("--seeds", nargs="+", type=int, default=study.DEFAULT_SEEDS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-minutes", type=float, default=90.0)
    args = parser.parse_args()

    n_pairs = len(args.protocols) * len(args.horizons) * len(args.seeds)
    n_runs = 2 * n_pairs
    study_summary_path = ROOT / "results" / "study" / "summary.json"
    study_seconds = None
    if study_summary_path.exists():
        study_seconds = json.loads(study_summary_path.read_text()).get("wall_clock_seconds")
    estimate_seconds = (n_runs / 180.0) * study_seconds if study_seconds else None
    print(f"Starting paired normalisation measurement: {n_runs} runs ({n_pairs} pairs)")
    if estimate_seconds is None:
        print("Projected wall clock: unknown (no committed study timing available)")
    else:
        print(
            f"Projected wall clock: {estimate_seconds / 60:.1f} minutes "
            f"from results/study baseline ({study_seconds:.1f}s for 180 runs)"
        )
    print(f"Hard stop: {args.max_minutes:.1f} minutes; incomplete sweeps write nothing")
    if estimate_seconds is not None and estimate_seconds > args.max_minutes * 60:
        raise SystemExit(
            "STOP: projected wall clock exceeds the configured limit; full sweep not started."
        )

    df = study.load_dataframe(42)
    tasks = [(p, h, s) for p in args.protocols for h in args.horizons for s in args.seeds]
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    with ProcessPoolExecutor(
        max_workers=args.workers, initializer=_init_worker, initargs=(df,)
    ) as executor:
        futures = [executor.submit(run_pair, p, h, s, args.threads) for p, h, s in tasks]
        for index, future in enumerate(futures, 1):
            rows.extend(future.result())
            elapsed = time.perf_counter() - started
            if index % 10 == 0 or index == n_pairs:
                print(f"  {index}/{n_pairs} pairs ({2 * index}/{n_runs} runs), {elapsed:.1f}s")
            if elapsed > args.max_minutes * 60:
                print(
                    f"ABORT: {elapsed / 60:.1f} minutes exceeded the cap after "
                    f"{index}/{n_pairs} pairs. Nothing written.",
                    flush=True,
                )
                executor.shutdown(wait=False, cancel_futures=True)
                raise SystemExit(1)

    wall_clock_seconds = time.perf_counter() - started
    rows.sort(key=lambda r: (r["protocol"], r["horizon"], r["seed"], r["scaling"]))
    reproduction = compare_global_to_study(rows, ROOT / "results" / "study" / "runs.csv")
    summary = build_summary(rows, wall_clock_seconds, reproduction)
    _write_results(rows, summary, args.out)
    print(f"Completed in {wall_clock_seconds / 60:.1f} minutes")
    print(
        "Global arm exact reproduction: "
        f"{reproduction['reproduced_exactly']} "
        f"({reproduction['shared_rows']} shared rows, {reproduction['mismatch_count']} mismatches)"
    )
    for key, cell in summary["cells"].items():
        diff = cell["paired_val_acc_difference"]
        sign = cell["baseline_sign_check"]
        print(
            f"{key:24} delta={diff['mean_percentage_points']:+.3f} pp "
            f"95% CI [{diff['ci95_low_percentage_points']:+.3f}, "
            f"{diff['ci95_high_percentage_points']:+.3f}] "
            f"sign {sign['global_max_sign']}->{sign['train_only_sign']}"
        )
    print(f"Wrote {args.out / 'runs.csv'} and {args.out / 'summary.json'}")


if __name__ == "__main__":
    main()
