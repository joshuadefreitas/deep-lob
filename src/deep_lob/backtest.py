import argparse
import json
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from deep_lob.dataset import DeepLOBDataset
from deep_lob.models import DeepLOBModel


def max_drawdown(equity_curve: np.ndarray) -> float:
    """
    Compute max drawdown of a cumulative PnL curve.

    equity_curve: 1D array of cumulative PnL (not returns).
    """
    peak = -np.inf
    mdd = 0.0
    for x in equity_curve:
        if x > peak:
            peak = x
        drawdown = peak - x
        if drawdown > mdd:
            mdd = drawdown
    return float(mdd)


def sharpe_ratio(returns: np.ndarray, eps: float = 1e-9) -> float:
    """
    Naive per-trade Sharpe ratio: mean / std of trade returns.
    """
    if returns.size == 0:
        return 0.0
    mu = returns.mean()
    sigma = returns.std()
    if sigma < eps:
        return 0.0
    return float(mu / sigma)


def run_backtest(
    data_npz: Path,
    raw_csv: Path,
    model_path: Path,
    window_size: int,
    horizon: int,
    tc_bps: float = 1.0,
    slippage_bps: float = 0.0,
    device_str: str = "auto",
) -> Dict[str, Any]:
    """
    Backtest a simple directional strategy:

    - Predictions are in {0,1,2} -> {down, flat, up}
    - Map to positions: {0: -1, 1: 0, 2: +1}
    - Trade horizon = `horizon` steps ahead in mid-price.
    - Return per trade = position * (mid_future - mid_now) / mid_now - cost

    Where cost is (tc_bps + slippage_bps) in basis points.
    """
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    print(f"Using device: {device}")

    # Load processed windows (X, y)
    npz = np.load(data_npz)
    X = npz["X"]
    y_true = npz["y"]
    n_samples, win_T, n_features = X.shape
    print(f"Loaded X with shape {X.shape}, labels shape {y_true.shape}")

    if win_T != window_size:
        print(f"[Warning] window_size={window_size} but X.shape[1]={win_T}. Using X.shape[1].")
        window_size = win_T

    # Dataset + loader (reuse your Dataset but we won't use y from it)
    dataset = DeepLOBDataset(str(data_npz))
    loader = DataLoader(dataset, batch_size=256, shuffle=False)

    # Model
    num_features = dataset[0][0].shape[1]
    model = DeepLOBModel(num_features=num_features)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    # Get predictions for all samples
    preds = []
    with torch.no_grad():
        for X_batch, _ in loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            batch_preds = torch.argmax(logits, dim=1)
            preds.append(batch_preds.cpu().numpy())
    y_pred = np.concatenate(preds)
    assert y_pred.shape[0] == n_samples, "Prediction length mismatch"

    # Map class -> position
    # 0: DOWN -> -1, 1: FLAT -> 0, 2: UP -> +1
    pos_map = {0: -1, 1: 0, 2: 1}
    positions = np.vectorize(lambda c: pos_map[int(c)])(y_pred)

    # Load raw mid-price series
    df = pd.read_csv(raw_csv)
    if "mid" not in df.columns:
        raise ValueError("raw CSV must contain a 'mid' column for backtesting.")
    mid = df["mid"].values

    # Align windows with mid-price:
    # In data.py we effectively used:
    #   for i in range(end):
    #       window = df.iloc[i : i + window_size]
    #       mid_now = mid[i + window_size - 1]
    #       mid_future = mid[i + window_size - 1 + horizon]
    #
    # So we replicate that logic here.
    max_valid_samples = len(mid) - window_size - horizon
    n_trades = min(n_samples, max_valid_samples)
    if n_trades <= 0:
        raise ValueError("Not enough data for backtest with given window_size and horizon.")

    positions = positions[:n_trades]
    y_true = y_true[:n_trades]  # for reporting

    trade_returns = []
    trade_directions = []  # whether prediction sign matched actual sign
    tc = (tc_bps + slippage_bps) / 10000.0  # basis points -> fraction

    for i in range(n_trades):
        entry_idx = i + window_size - 1
        exit_idx = entry_idx + horizon

        mid_now = mid[entry_idx]
        mid_future = mid[exit_idx]

        raw_ret = (mid_future - mid_now) / mid_now  # underlying move
        pos = positions[i]
        pnl_ret = pos * raw_ret - tc * abs(pos)  # subtract cost only if nonzero position
        trade_returns.append(pnl_ret)

        # actual movement sign: -1, 0, 1
        if raw_ret > 0:
            true_dir = 1
        elif raw_ret < 0:
            true_dir = -1
        else:
            true_dir = 0

        pred_dir = np.sign(pos)
        trade_directions.append(int(pred_dir == true_dir) if true_dir != 0 else int(pred_dir == 0))

    trade_returns = np.array(trade_returns, dtype=float)
    trade_directions = np.array(trade_directions, dtype=int)
    equity_curve = np.cumsum(trade_returns)

    # Stats
    total_pnl = float(equity_curve[-1])
    avg_ret = float(trade_returns.mean()) if trade_returns.size > 0 else 0.0
    win_rate = float(trade_directions.mean()) if trade_directions.size > 0 else 0.0
    mdd = max_drawdown(equity_curve)
    sharpe = sharpe_ratio(trade_returns)

    stats = {
        "n_trades": int(n_trades),
        "total_pnl": total_pnl,
        "avg_return_per_trade": avg_ret,
        "win_rate": win_rate,
        "max_drawdown": mdd,
        "sharpe_ratio": sharpe,
        "tc_bps": tc_bps,
        "slippage_bps": slippage_bps,
    }

    print("\n=== Backtest Summary ===")
    print(f"Trades        : {n_trades}")
    print(f"Total PnL     : {total_pnl:.6f}")
    print(f"Avg ret/trade : {avg_ret:.6f}")
    print(f"Win rate      : {win_rate:.3f}")
    print(f"Max drawdown  : {mdd:.6f}")
    print(f"Sharpe (per trade): {sharpe:.3f}")

    # Save stats + equity curve
    out_dir = Path("backtest")
    out_dir.mkdir(exist_ok=True)

    stats_path = out_dir / "stats.json"
    with stats_path.open("w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nSaved backtest stats to {stats_path}")

    # Optional: equity curve plot
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(equity_curve)
        ax.set_title("Equity Curve (Cumulative PnL)")
        ax.set_xlabel("Trade #")
        ax.set_ylabel("Cumulative PnL (fraction)")

        fig.tight_layout()
        curve_path = out_dir / "equity_curve.png"
        fig.savefig(curve_path)
        plt.close(fig)
        print(f"Saved equity curve plot to {curve_path}")
    except ImportError:
        print("matplotlib not installed; skipping equity curve plot.")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Backtest DeepLOB predictions as a simple directional strategy.")
    parser.add_argument(
        "--data",
        type=str,
        default="data/processed/lob_windows.npz",
        help="Path to NPZ file with X and y.",
    )
    parser.add_argument(
        "--raw",
        type=str,
        default="data/raw/simulated_lob.csv",
        help="Path to raw CSV with at least a 'mid' column.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/deeplob_synthetic.pt",
        help="Path to trained model weights.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=100,
        help="Window size used when building tensors.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=10,
        help="Prediction horizon (steps ahead) used for labeling.",
    )
    parser.add_argument(
        "--tc-bps",
        type=float,
        default=1.0,
        help="Transaction cost in basis points (per trade side).",
    )
    parser.add_argument(
        "--slippage-bps",
        type=float,
        default=0.0,
        help="Additional slippage in basis points.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: 'auto', 'cpu', or 'cuda'.",
    )

    args = parser.parse_args()

    run_backtest(
        data_npz=Path(args.data),
        raw_csv=Path(args.raw),
        model_path=Path(args.model),
        window_size=args.window_size,
        horizon=args.horizon,
        tc_bps=args.tc_bps,
        slippage_bps=args.slippage_bps,
        device_str=args.device,
    )


if __name__ == "__main__":
    main()