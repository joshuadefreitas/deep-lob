"""
Synthetic limit order book simulator.

Generates a toy LOB with:
- mid price
- 3 bid levels (price, size)
- 3 ask levels (price, size)

This is just to exercise the DeepLOB pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def simulate_lob(
    n_rows: int = 5000,
    seed: Optional[int] = 42,
    mid_start: float = 100.0,
    tick_size: float = 0.01,
    n_levels: int = 3,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    mids = [mid_start]
    for _ in range(n_rows - 1):
        # small random walk in mid price
        step = rng.normal(loc=0.0, scale=0.03)
        mids.append(mids[-1] * (1.0 + step / 100.0))

    mids = np.array(mids, dtype=np.float32)

    rows = []
    for mid in mids:
        # spread ~ 1–3 ticks
        spread_ticks = rng.integers(1, 4)
        best_bid = mid - spread_ticks * tick_size / 2
        best_ask = mid + spread_ticks * tick_size / 2

        # construct levels around best bid/ask
        level = {
            "mid": mid,
        }

        bid_px = best_bid
        ask_px = best_ask

        for lvl in range(1, n_levels + 1):
            bid_size = rng.integers(1, 10)
            ask_size = rng.integers(1, 10)

            level[f"bid_px_{lvl}"] = bid_px
            level[f"bid_sz_{lvl}"] = float(bid_size)

            level[f"ask_px_{lvl}"] = ask_px
            level[f"ask_sz_{lvl}"] = float(ask_size)

            # move deeper into the book
            bid_px -= tick_size
            ask_px += tick_size

        rows.append(level)

    df = pd.DataFrame(rows)
    return df


def save_simulated_lob_csv(
    out_path: str | Path,
    n_rows: int = 5000,
    seed: Optional[int] = 42,
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = simulate_lob(n_rows=n_rows, seed=seed)
    df.to_csv(out_path, index=False)
    print(f"Saved simulated LOB: {len(df)} rows -> {out_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic LOB CSV.")
    parser.add_argument(
        "--out",
        type=str,
        default="data/raw/simulated_lob.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--n-rows",
        type=int,
        default=5000,
        help="Number of LOB events to simulate.",
    )
    args = parser.parse_args()

    save_simulated_lob_csv(out_path=args.out, n_rows=args.n_rows)