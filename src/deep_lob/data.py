"""
Data utilities for DeepLOB:
- load LOB data from CSV
- build sliding-window tensors
- compute mid-price movement labels
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd


# -----------------------------
# 1. Basic config / feature set
# -----------------------------

# Adjust this list if your simulator / real data has different columns
DEFAULT_FEATURE_COLUMNS: List[str] = [
    "mid",
    "bid_px_1", "bid_sz_1",
    "bid_px_2", "bid_sz_2",
    "bid_px_3", "bid_sz_3",
    "ask_px_1", "ask_sz_1",
    "ask_px_2", "ask_sz_2",
    "ask_px_3", "ask_sz_3",
]


@dataclass
class WindowConfig:
    window_size: int = 100   # number of LOB events per sample
    horizon: int = 10        # how far ahead to look for the label
    up_threshold: float = 0.0002   # relative move up (+0.02%)
    down_threshold: float = 0.0002 # relative move down (-0.02%)
    feature_columns: List[str] = None

    def __post_init__(self) -> None:
        if self.feature_columns is None:
            self.feature_columns = DEFAULT_FEATURE_COLUMNS


# -----------------------------
# 2. Loading LOB data
# -----------------------------

def load_lob_csv(path: str | Path) -> pd.DataFrame:
    """
    Load a limit-order-book time series from CSV.

    Assumes:
    - each row is one event / snapshot
    - has at least the columns in DEFAULT_FEATURE_COLUMNS
    - index order represents time
    """
    path = Path(path)
    df = pd.read_csv(path)

    # Ensure the expected columns are present
    missing = [col for col in DEFAULT_FEATURE_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns in {path}: {missing}")

    # Ensure a deterministic order
    df = df.reset_index(drop=True)
    return df


# -----------------------------
# 3. Label computation
# -----------------------------

def compute_labels_from_mid(
    mid: np.ndarray,
    window_size: int,
    horizon: int,
    up_threshold: float,
    down_threshold: float,
) -> np.ndarray:
    """
    Compute classification labels based on future mid-price movement.

    For each starting index i:
    - we take the last mid-price in the window (at i + window_size - 1)
    - compare it to the mid-price at i + window_size - 1 + horizon
    - assign:
        +1 if relative change > up_threshold
         0 if in [-down_threshold, up_threshold]
        -1 if relative change < -down_threshold
    """
    n = len(mid)
    max_start = n - window_size - horizon + 1
    if max_start <= 0:
        raise ValueError(
            f"Not enough data points (n={n}) for window_size={window_size}, "
            f"horizon={horizon}"
        )

    labels = np.zeros(max_start, dtype=np.int8)

    for i in range(max_start):
        idx_now = i + window_size - 1
        idx_future = idx_now + horizon

        p_now = mid[idx_now]
        p_future = mid[idx_future]

        if p_now <= 0:
            rel_change = 0.0
        else:
            rel_change = (p_future - p_now) / p_now

        if rel_change > up_threshold:
            labels[i] = 1
        elif rel_change < -down_threshold:
            labels[i] = -1
        else:
            labels[i] = 0

    return labels


# -----------------------------
# 4. Sliding-window tensor builder
# -----------------------------

def build_windows(
    df: pd.DataFrame,
    config: WindowConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build sliding-window tensors from a LOB dataframe.

    Returns:
    - X: numpy array of shape (num_samples, window_size, num_features)
    - y: numpy array of shape (num_samples,) with labels in {-1, 0, +1}
    """
    feats = df[config.feature_columns].to_numpy(dtype=np.float32)
    mid = df["mid"].to_numpy(dtype=np.float32)

    n, num_features = feats.shape
    window_size = config.window_size
    horizon = config.horizon

    max_start = n - window_size - horizon + 1
    if max_start <= 0:
        raise ValueError(
            f"Not enough rows for window_size={window_size}, horizon={horizon}, "
            f"n={n}"
        )

    X = np.zeros((max_start, window_size, num_features), dtype=np.float32)

    for i in range(max_start):
        X[i] = feats[i : i + window_size, :]

    y = compute_labels_from_mid(
        mid=mid,
        window_size=window_size,
        horizon=horizon,
        up_threshold=config.up_threshold,
        down_threshold=config.down_threshold,
    )

    return X, y


# -----------------------------
# 5. Convenience: end-to-end API
# -----------------------------

def lob_csv_to_tensors(
    csv_path: str | Path,
    config: WindowConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    High-level helper: load CSV, build windows, return X, y.
    """
    df = load_lob_csv(csv_path)
    return build_windows(df, config)


def save_tensors_npz(
    X: np.ndarray,
    y: np.ndarray,
    out_path: str | Path,
) -> None:
    """
    Save tensors in compressed .npz format.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, X=X, y=y)


# -----------------------------
# 6. CLI example
# -----------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build DeepLOB tensors from CSV.")
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to LOB CSV file (e.g., simulated or real).",
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output .npz path under data/processed/",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=100,
        help="Number of events per window.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=10,
        help="Prediction horizon (in events).",
    )
    args = parser.parse_args()

    cfg = WindowConfig(
        window_size=args.window_size,
        horizon=args.horizon,
    )

    print(f"Loading {args.csv} ...")
    df_lob = load_lob_csv(args.csv)
    print(f"Data shape: {df_lob.shape}")

    print("Building windows ...")
    X, y = build_windows(df_lob, cfg)
    print(f"X shape: {X.shape}, y shape: {y.shape}")

    print(f"Saving tensors to {args.out} ...")
    save_tensors_npz(X, y, args.out)
    print("Done.")