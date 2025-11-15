from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import yaml

# --- Make src/ importable so we can use deep_lob.* ---
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from deep_lob.simulator import save_simulated_lob_csv
from deep_lob.data import load_raw_lob, build_lob_windows
from deep_lob.train import main as train_deeplob
from deep_lob.evaluate import evaluate_model
from deep_lob.backtest import run_backtest


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def run_experiment(config_path: str) -> None:
    # 1. Load config
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    exp_name = cfg["experiment_name"]
    paths_cfg = cfg["paths"]
    sim_cfg = cfg["simulation"]
    lob_cfg = cfg["lob"]
    eval_cfg = cfg["evaluation"]
    bt_cfg = cfg["backtest"]

    raw_csv = Path(paths_cfg["raw_csv"])
    lob_npz = Path(paths_cfg["lob_npz"])
    model_path = Path(paths_cfg["model"])
    results_dir = Path(paths_cfg["results_dir"])

    print(f"\n=== Running experiment: {exp_name} ===")

    # Make sure directories exist
    for p in [raw_csv, lob_npz, model_path, results_dir]:
        ensure_parent(p)

    # 2. Simulate synthetic LOB data (optional)
    if sim_cfg.get("enabled", True):
        print("\n[Step 1] Simulating LOB data ...")
        save_simulated_lob_csv(
            out_path=raw_csv,
            n_rows=sim_cfg.get("n_rows", 5000),
            seed=sim_cfg.get("seed", 42),
        )
    else:
        print("\n[Step 1] Skipping simulation, using existing CSV:", raw_csv)

    # 3. Build windows and save NPZ
    print("\n[Step 2] Building windows and saving NPZ ...")
    df = load_raw_lob(raw_csv)
    X, y = build_lob_windows(
        df=df,
        window_size=lob_cfg["window_size"],
        horizon=lob_cfg["horizon"],
        n_levels=lob_cfg["n_levels"],
        threshold=lob_cfg["threshold"],
    )
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    classes, counts = np.unique(y, return_counts=True)
    print("Label distribution:", {int(c): int(n) for c, n in zip(classes, counts)})

    np.savez(lob_npz, X=X, y=y)
    print(f"Saved windows to {lob_npz}")

    # 4. Train DeepLOB baseline (uses data/processed/lob_windows.npz)
    print("\n[Step 3] Training DeepLOB baseline model ...")
    train_deeplob()
    print(f"Expected model weights at {model_path}")

    # 5. Evaluate model: writes reports/metrics.json
    print("\n[Step 4] Evaluating model (classification metrics) ...")
    evaluate_model(
        data_path=lob_npz,
        model_path=model_path,
        batch_size=eval_cfg["batch_size"],
        device_str=eval_cfg["device"],
    )

    metrics_path = Path("reports") / "metrics.json"
    if metrics_path.exists():
        with metrics_path.open("r") as f:
            metrics = json.load(f)
    else:
        print(f"[Warning] {metrics_path} not found, using empty metrics.")
        metrics = {}

    # 6. Backtest simple strategy
    print("\n[Step 5] Running backtest ...")
    stats = run_backtest(
        data_npz=lob_npz,
        raw_csv=raw_csv,
        model_path=model_path,
        window_size=lob_cfg["window_size"],
        horizon=lob_cfg["horizon"],
        tc_bps=bt_cfg["tc_bps"],
        slippage_bps=bt_cfg["slippage_bps"],
        device_str=bt_cfg["device"],
    )

    # 7. Save combined summary for this experiment
    results_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "experiment_name": exp_name,
        "config": cfg,
        "classification_metrics": metrics,
        "backtest_stats": stats,
    }

    summary_path = results_dir / "summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[Done] Saved experiment summary to {summary_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/baseline_a.yaml",
        help="Path to YAML config.",
    )
    args = parser.parse_args()
    run_experiment(args.config)


if __name__ == "__main__":
    main()