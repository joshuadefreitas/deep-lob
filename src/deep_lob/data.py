"""
Data loading, feature engineering, and window construction for DeepLOB.

This module handles:
- Reading raw LOB data (CSV)
- Engineering microstructure features
- Building sliding-window tensors [samples, timesteps, features]
- Generating -1 / 0 / 1 labels based on future mid-price moves
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import argparse
import numpy as np
import pandas as pd


# -----------------------------
# 1. Raw data loading
# -----------------------------


def load_raw_lob(path: Path) -> pd.DataFrame:
    """
    Load raw limit order book data from disk.

    Expected columns (from simulator.py):
    - mid
    - bid_px_1, bid_sz_1, ..., bid_px_N, bid_sz_N
    - ask_px_1, ask_sz_1, ..., ask_px_N, ask_sz_N

    Parameters
    ----------
    path : Path
        Path to the raw data file.

    Returns
    -------
    pd.DataFrame
        Raw LOB data.
    """
    df = pd.read_csv(path)
    if "mid" not in df.columns:
        raise ValueError("Expected a 'mid' column in the LOB data.")
    return df


# -----------------------------
# 2. Feature engineering
# -----------------------------


def prepare_features(df: pd.DataFrame, n_levels: int = 3) -> pd.DataFrame:
    """
    Engineer normalized microstructure features from raw LOB.

    We compute:
    - Price levels relative to mid-price: (px - mid) / mid
    - Volume levels normalized by global max
    - Bid/ask spread relative to mid
    - Bid/ask volume imbalance
    - One-step mid-price return

    Parameters
    ----------
    df : pd.DataFrame
        Raw LOB data with 'mid' and bid/ask levels.
    n_levels : int, optional
        Number of book levels per side (default: 3).

    Returns
    -------
    pd.DataFrame
        Engineered feature matrix with float32 columns.
    """
    if "mid" not in df.columns:
        raise ValueError("prepare_features expects a 'mid' column.")

    mid = df["mid"].astype("float32")
    features = {}

    # 2.1 Price features relative to mid
    for side in ("bid", "ask"):
        for level in range(1, n_levels + 1):
            px_col = f"{side}_px_{level}"
            if px_col in df.columns:
                rel_name = f"{px_col}_rel"
                features[rel_name] = ((df[px_col].astype("float32") - mid) / mid)

    # 2.2 Volume features normalized by global max per level
    for side in ("bid", "ask"):
        for level in range(1, n_levels + 1):
            sz_col = f"{side}_sz_{level}"
            if sz_col in df.columns:
                max_sz = df[sz_col].max()
                if max_sz is None or max_sz == 0:
                    max_sz = 1.0
                norm_name = f"{sz_col}_norm"
                features[norm_name] = df[sz_col].astype("float32") / float(max_sz)

    # 2.3 Spread (best ask - best bid) relative to mid
    if "ask_px_1" in df.columns and "bid_px_1" in df.columns:
        spread_rel = (df["ask_px_1"].astype("float32") - df["bid_px_1"].astype("float32")) / mid
        features["spread_rel"] = spread_rel

    # 2.4 Volume imbalance across the top n_levels
    bid_total = None
    ask_total = None
    for level in range(1, n_levels + 1):
        b_col = f"bid_sz_{level}"
        a_col = f"ask_sz_{level}"
        if b_col in df.columns:
            bid_total = df[b_col].astype("float32") if bid_total is None else bid_total + df[b_col].astype("float32")
        if a_col in df.columns:
            ask_total = df[a_col].astype("float32") if ask_total is None else ask_total + df[a_col].astype("float32")

    if bid_total is not None and ask_total is not None:
        denom = bid_total + ask_total
        denom = denom.replace(0, 1.0)
        features["imbalance"] = (bid_total - ask_total) / denom

    # 2.5 One-step mid-price return
    mid_ret_1 = mid.pct_change().fillna(0.0)
    features["mid_ret_1"] = mid_ret_1

    # Build DataFrame
    feat_df = pd.DataFrame(features, index=df.index)
    # Ensure float32 dtype
    feat_df = feat_df.astype("float32")

    return feat_df


# -----------------------------
# 3. Window builder + labels
# -----------------------------


def build_lob_windows(
    df: pd.DataFrame,
    window_size: int,
    horizon: int,
    n_levels: int = 3,
    threshold: float = 5e-4,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build sliding windows of engineered features and labels.

    Each window:
      X_t = [features at time t ... t+window_size-1]
    Label:
      based on mid-price move from time (t+window_size-1) to (t+window_size-1 + horizon)

    We keep the label convention:
      -1 -> mid-price down
       0 -> flat
       1 -> up

    Parameters
    ----------
    df : pd.DataFrame
        Raw LOB data including 'mid'.
    window_size : int
        Number of timesteps per window.
    horizon : int
        Lookahead steps for label.
    n_levels : int, optional
        Number of levels for feature engineering.
    threshold : float, optional
        Relative return threshold to classify up / down.

    Returns
    -------
    X : np.ndarray
        Shape (n_samples, window_size, n_features)
    y : np.ndarray
        Shape (n_samples,), values in {-1, 0, 1}
    """
    mid = df["mid"].astype("float32").to_numpy()
    features = prepare_features(df, n_levels=n_levels)
    feat_values = features.to_numpy(dtype="float32")

    n = len(df)
    max_start = n - window_size - horizon + 1
    if max_start <= 0:
        raise ValueError(
            f"Not enough rows ({n}) for window_size={window_size} and horizon={horizon}"
        )

    X_list = []
    y_list = []

    for start in range(max_start):
        end = start + window_size
        last_idx = end - 1
        future_idx = last_idx + horizon

        m_now = mid[last_idx]
        m_fut = mid[future_idx]

        ret = (m_fut - m_now) / m_now

        if ret > threshold:
            label = 1
        elif ret < -threshold:
            label = -1
        else:
            label = 0

        window = feat_values[start:end, :]
        X_list.append(window)
        y_list.append(label)

    X = np.stack(X_list, axis=0).astype("float32")
    y = np.array(y_list, dtype=np.int8)

    return X, y


# -----------------------------
# 4. CLI entrypoint
# -----------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Build LOB windows for DeepLOB.")
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to raw LOB CSV (e.g. data/raw/simulated_lob.csv)",
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output .npz file path (e.g. data/processed/lob_windows.npz)",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=100,
        help="Number of timesteps per window (default: 100)",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=10,
        help="Lookahead steps for label (default: 10)",
    )
    parser.add_argument(
        "--levels",
        type=int,
        default=3,
        help="Number of order book levels per side to use (default: 3)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=5e-4,
        help="Relative return threshold to classify up/down (default: 5e-4)",
    )

    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_path = Path(args.out)

    print(f"Loading {csv_path} ...")
    df = load_raw_lob(csv_path)
    print(f"Data shape: {df.shape}")

    print("Building windows with engineered features ...")
    X, y = build_lob_windows(
        df,
        window_size=args.window_size,
        horizon=args.horizon,
        n_levels=args.levels,
        threshold=args.threshold,
    )
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    print(f"Label distribution: { {int(v): int((y == v).sum()) for v in sorted(set(y))} }")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving tensors to {out_path} ...")
    np.savez(out_path, X=X, y=y)
    print("Done.")


if __name__ == "__main__":
    main()