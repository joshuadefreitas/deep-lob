"""Paired measurement of the legacy full-data normalisation leak.

The sweep holds the study harness fixed and changes only feature scaling:

* ``global_max`` calls :func:`run_study.run_cell` directly.
* ``train_only`` builds raw-size windows, creates the identical split, calls
  ``TrainOnlyScaler.fit(X[train_idx])``, and uses that fitted scaler for every
  train, validation, and full-dataset feature value.

The default remains the original simulator-seed-42 sweep. ``--sim-seeds``
extends the same paired design across independent generator paths, matching
``run_study.py``. Results are written only after the complete sweep succeeds.
The artifact is specific to the requested synthetic simulator paths and CPU
DeepLOB configuration inherited from ``run_study.py``; it makes no claim about
real exchange data or other models.
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


def run_pair(
    protocol: str, horizon: int, seed: int, threads: int, sim_seed: int = 42
) -> list[dict[str, Any]]:
    """Run both arms, resetting every seeded RNG inside each arm."""
    global_row = study.run_cell(protocol, horizon, seed, threads, sim_seed=sim_seed)
    global_row["scaling"] = "global_max"
    train_only_row = run_train_only_cell(
        protocol, horizon, seed, threads, sim_seed=sim_seed
    )
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


def _row_key(row: dict[str, Any]) -> tuple[str, int, int, int]:
    return (
        row["protocol"],
        int(row["horizon"]),
        int(row.get("sim_seed", 42)),
        int(row["seed"]),
    )


def _load_study_rows(path: Path) -> dict[tuple[str, int, int, int], dict[str, str]]:
    with path.open(newline="") as handle:
        return {_row_key(row): row for row in csv.DictReader(handle)}


def compare_global_to_study(
    rows: list[dict[str, Any]], study_csv: Path
) -> dict[str, Any]:
    """Require exact numeric reproduction for every shared study row."""
    expected = _load_study_rows(study_csv)
    observed = {
        _row_key(r): r for r in rows if r["scaling"] == "global_max"
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
                        "sim_seed": key[2],
                        "seed": key[3],
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


def _summarise_path(path_rows: list[dict[str, Any]]) -> dict[str, Any]:
    cells: dict[str, Any] = {}
    for protocol in study.PROTOCOLS:
        for horizon in study.HORIZONS:
            key = f"{protocol}|h{horizon}"
            arm_rows = {
                scaling: sorted(
                    (
                        row
                        for row in path_rows
                        if row["protocol"] == protocol
                        and row["horizon"] == horizon
                        and row["scaling"] == scaling
                    ),
                    key=lambda row: row["seed"],
                )
                for scaling in SCALINGS
            }
            if not arm_rows["global_max"] and not arm_rows["train_only"]:
                continue
            global_seeds = [row["seed"] for row in arm_rows["global_max"]]
            train_only_seeds = [row["seed"] for row in arm_rows["train_only"]]
            if global_seeds != train_only_seeds:
                raise RuntimeError(
                    f"unpaired scaling arms for {key}: "
                    f"global_max={global_seeds}, train_only={train_only_seeds}"
                )
            arms: dict[str, Any] = {}
            for scaling in SCALINGS:
                aggregates = {
                    metric: study.aggregate([row[metric] for row in arm_rows[scaling]])
                    for metric in METRICS
                }
                excess = (
                    aggregates["val_acc"]["mean"]
                    - aggregates["val_majority"]["mean"]
                )
                aggregates["excess_over_baseline"] = excess
                aggregates["baseline_sign"] = _baseline_sign(excess)
                aggregates["question_3_verdict"] = (
                    "PASS_BELOW_BASELINE"
                    if protocol == "purged_embargoed" and excess < 0.0
                    else "FAIL_AT_OR_ABOVE_BASELINE"
                    if protocol == "purged_embargoed"
                    else "NOT_APPLICABLE"
                )
                arms[scaling] = aggregates
            cells[key] = {
                "split_seeds": global_seeds,
                "arms": arms,
                "paired_val_acc_difference": _paired_difference(
                    [row["val_acc"] for row in arm_rows["global_max"]],
                    [row["val_acc"] for row in arm_rows["train_only"]],
                ),
                "baseline_sign_change": (
                    arms["global_max"]["baseline_sign"]
                    != arms["train_only"]["baseline_sign"]
                ),
            }
    return {"cells": cells}


def _load_normalisation_rows(path: Path) -> list[dict[str, Any]]:
    numeric_float = set(METRICS)
    numeric_int = {"horizon", "sim_seed", "seed", "n_train", "n_val"}
    with path.open(newline="") as handle:
        rows: list[dict[str, Any]] = []
        for raw in csv.DictReader(handle):
            row: dict[str, Any] = dict(raw)
            for field in numeric_float:
                row[field] = float(row[field])
            for field in numeric_int:
                row[field] = int(row[field])
            rows.append(row)
        return rows


def build_summary(
    measured_rows: list[dict[str, Any]],
    report_rows: list[dict[str, Any]],
    wall_clock_seconds: float,
    reproduction: dict[str, Any],
    measured_sim_seeds: list[int],
    reference_sim_seeds: list[int],
) -> dict[str, Any]:
    sim_seeds = sorted({int(row["sim_seed"]) for row in report_rows})
    paths = {
        str(sim_seed): _summarise_path(
            [row for row in report_rows if int(row["sim_seed"]) == sim_seed]
        )
        for sim_seed in sim_seeds
    }

    direction_checks: list[dict[str, Any]] = []
    h5_sign_checks: list[dict[str, Any]] = []
    purged_checks: list[dict[str, Any]] = []
    for sim_seed in sim_seeds:
        cells = paths[str(sim_seed)]["cells"]
        for horizon in (10, 20):
            cell = cells[f"random_split|h{horizon}"]
            global_acc = cell["arms"]["global_max"]["val_acc"]["mean"]
            train_acc = cell["arms"]["train_only"]["val_acc"]["mean"]
            direction_checks.append(
                {
                    "sim_seed": sim_seed,
                    "horizon": horizon,
                    "global_max_val_acc": global_acc,
                    "train_only_val_acc": train_acc,
                    "train_only_higher": train_acc > global_acc,
                }
            )
        h5 = cells["random_split|h5"]
        h5_sign_checks.append(
            {
                "sim_seed": sim_seed,
                "global_max_excess": h5["arms"]["global_max"][
                    "excess_over_baseline"
                ],
                "global_max_sign": h5["arms"]["global_max"]["baseline_sign"],
                "train_only_excess": h5["arms"]["train_only"][
                    "excess_over_baseline"
                ],
                "train_only_sign": h5["arms"]["train_only"]["baseline_sign"],
                "changed_sign": h5["baseline_sign_change"],
            }
        )
        for horizon in study.HORIZONS:
            cell = cells[f"purged_embargoed|h{horizon}"]
            for scaling in SCALINGS:
                arm = cell["arms"][scaling]
                purged_checks.append(
                    {
                        "sim_seed": sim_seed,
                        "horizon": horizon,
                        "scaling": scaling,
                        "val_acc": arm["val_acc"]["mean"],
                        "val_majority": arm["val_majority"]["mean"],
                        "excess_over_baseline": arm["excess_over_baseline"],
                        "below_baseline": arm["excess_over_baseline"] < 0.0,
                        "verdict": arm["question_3_verdict"],
                    }
                )
    measured_direction = [
        check for check in direction_checks if check["sim_seed"] in measured_sim_seeds
    ]
    purged_violations = [check for check in purged_checks if not check["below_baseline"]]

    return {
        "description": "Paired measurement of full-data global-max normalisation versus train-only scaling.",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "wall_clock_seconds": wall_clock_seconds,
        "n_runs": len(measured_rows),
        "n_paired_runs": len(measured_rows) // 2,
        "sweep": {
            "measured_sim_seeds": measured_sim_seeds,
            "reference_sim_seeds": reference_sim_seeds,
            "reported_sim_seeds": sim_seeds,
            "split_seeds_by_path": {
                str(sim_seed): paths[str(sim_seed)]["cells"]["random_split|h5"][
                    "split_seeds"
                ]
                for sim_seed in sim_seeds
            },
            "protocols": study.PROTOCOLS,
            "horizons": study.HORIZONS,
            "scalings": list(SCALINGS),
        },
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
            "simulator_seeds_measured": measured_sim_seeds,
            "simulator_seeds_reported": sim_seeds,
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
            "Synthetic simulator paths only.",
            "DeepLOBModel on CPU with the run_study.py configuration only.",
            "No real exchange data, trading, PnL, Sharpe, or causal-performance claim.",
        ],
        "global_max_reproduction": reproduction,
        "paths": paths,
        "questions": {
            "1_random_h10_h20_direction": {
                "rule": "train_only mean val_acc is higher than global_max on every measured path at h10 and h20",
                "per_path": direction_checks,
                "all_five_measured_paths_hold": all(
                    check["train_only_higher"] for check in measured_direction
                ),
            },
            "2_random_h5_sign_change": {
                "rule": "compare each path's mean val_acc excess over its own mean validation-majority baseline",
                "per_path": h5_sign_checks,
                "paths_with_sign_change": [
                    check["sim_seed"] for check in h5_sign_checks if check["changed_sign"]
                ],
            },
            "3_purged_below_baseline": {
                "rule": "every purged path/horizon/scaling mean val_acc must be strictly below its own mean validation-majority baseline",
                "per_path": purged_checks,
                "violations": purged_violations,
                "all_reported_paths_hold": not purged_violations,
            },
        },
        "not_verified": [
            "Independent verification by a separate reviewer was not performed by this runner.",
            "GPU execution was not tested; the study and this measurement run on CPU.",
            "Generalisability beyond the reported synthetic simulator paths was not tested.",
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
    parser.add_argument("--sim-seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-minutes", type=float, default=120.0)
    args = parser.parse_args()

    n_pairs = (
        len(args.sim_seeds)
        * len(args.protocols)
        * len(args.horizons)
        * len(args.seeds)
    )
    n_runs = 2 * n_pairs
    timing_path = ROOT / "results" / "normalisation" / "summary.json"
    timing = json.loads(timing_path.read_text()) if timing_path.exists() else None
    estimate_seconds = (
        n_runs * timing["wall_clock_seconds"] / timing["n_runs"] if timing else None
    )
    print(f"Starting paired normalisation measurement: {n_runs} runs ({n_pairs} pairs)")
    if estimate_seconds is None:
        print("Projected wall clock: unknown (no committed study timing available)")
    else:
        print(
            f"Projected wall clock: {estimate_seconds / 60:.1f} minutes "
            f"from results/normalisation baseline "
            f"({timing['wall_clock_seconds']:.1f}s for {timing['n_runs']} runs)"
        )
    print(f"Hard stop: {args.max_minutes:.1f} minutes; incomplete sweeps write nothing")
    if estimate_seconds is not None and estimate_seconds > args.max_minutes * 60:
        raise SystemExit(
            "STOP: projected wall clock exceeds the configured limit; full sweep not started."
        )

    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    completed_pairs = 0
    for sim_seed in args.sim_seeds:
        df = study.load_dataframe(sim_seed)
        tasks = [
            (protocol, horizon, seed)
            for protocol in args.protocols
            for horizon in args.horizons
            for seed in args.seeds
        ]
        with ProcessPoolExecutor(
            max_workers=args.workers, initializer=_init_worker, initargs=(df,)
        ) as executor:
            futures = [
                executor.submit(
                    run_pair, protocol, horizon, seed, args.threads, sim_seed
                )
                for protocol, horizon, seed in tasks
            ]
            for future in futures:
                rows.extend(future.result())
                completed_pairs += 1
                elapsed = time.perf_counter() - started
                if completed_pairs % 10 == 0 or completed_pairs == n_pairs:
                    print(
                        f"  {completed_pairs}/{n_pairs} pairs "
                        f"({2 * completed_pairs}/{n_runs} runs), {elapsed:.1f}s"
                    )
                if elapsed > args.max_minutes * 60:
                    print(
                        f"ABORT: {elapsed / 60:.1f} minutes exceeded the cap after "
                        f"{completed_pairs}/{n_pairs} pairs. Nothing written.",
                        flush=True,
                    )
                    executor.shutdown(wait=False, cancel_futures=True)
                    raise SystemExit(1)

    wall_clock_seconds = time.perf_counter() - started
    rows.sort(
        key=lambda row: (
            row["sim_seed"], row["protocol"], row["horizon"], row["seed"], row["scaling"]
        )
    )
    multiseed_reference = ROOT / "results" / "study-multiseed" / "runs.csv"
    study_reference = (
        multiseed_reference
        if multiseed_reference.exists() and any(seed != 42 for seed in args.sim_seeds)
        else ROOT / "results" / "study" / "runs.csv"
    )
    reproduction = compare_global_to_study(rows, study_reference)
    reference_rows: list[dict[str, Any]] = []
    reference_sim_seeds: list[int] = []
    seed_42_path = ROOT / "results" / "normalisation" / "runs.csv"
    if 42 not in args.sim_seeds and seed_42_path.exists():
        reference_rows = _load_normalisation_rows(seed_42_path)
        reference_sim_seeds = [42]
    summary = build_summary(
        rows,
        rows + reference_rows,
        wall_clock_seconds,
        reproduction,
        list(args.sim_seeds),
        reference_sim_seeds,
    )
    _write_results(rows, summary, args.out)
    print(f"Completed in {wall_clock_seconds / 60:.1f} minutes")
    print(
        "Global arm exact reproduction: "
        f"{reproduction['reproduced_exactly']} "
        f"({reproduction['shared_rows']} shared rows, {reproduction['mismatch_count']} mismatches)"
    )
    questions = summary["questions"]
    print(
        "Q1 all five measured paths hold: "
        f"{questions['1_random_h10_h20_direction']['all_five_measured_paths_hold']}"
    )
    print(
        "Q2 paths with random h5 sign change: "
        f"{questions['2_random_h5_sign_change']['paths_with_sign_change']}"
    )
    print(
        "Q3 all reported paths hold: "
        f"{questions['3_purged_below_baseline']['all_reported_paths_hold']}"
    )
    print(f"Wrote {args.out / 'runs.csv'} and {args.out / 'summary.json'}")


if __name__ == "__main__":
    main()
