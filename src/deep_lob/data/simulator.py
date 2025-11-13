from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd

@dataclass
class LobSimulatorConfig:
    n_steps: int = 1_000          # number of time steps
    mid_price_start: float = 100.0
    drift: float = 0.0            # average directional move
    vol: float = 0.1              # volatility scale
    mean_reversion: float = 0.01  # pull-back strength to long-term mean
    mean_price: float = 100.0     # long-term mean level
    n_levels: int = 5             # number of bid/ask levels
    base_spread: float = 0.01     # minimum spread between best bid/ask
    tick_size: float = 0.01       # price granularity
    
def simulate_mid_price(cfg: LobSimulatorConfig) -> np.ndarray:
    """
    Simulate a simple mid-price process with drift + mean reversion.

    Returns
    -------
    np.ndarray
        Array of shape (n_steps,) with mid prices.
    """
    mid = np.zeros(cfg.n_steps)
    mid[0] = cfg.mid_price_start

    for t in range(1, cfg.n_steps):
        # white noise shock
        eps = np.random.randn()
        # mean reversion term towards cfg.mean_price
        mr_term = cfg.mean_reversion * (cfg.mean_price - mid[t - 1])
        # standard random walk with drift + mean reversion
        mid[t] = mid[t - 1] + cfg.drift + cfg.vol * eps + mr_term

        # enforce positivity and tick grid
        mid[t] = max(cfg.tick_size, round(mid[t] / cfg.tick_size) * cfg.tick_size)

    return mid


def build_lob_from_mid(
    mid: np.ndarray,
    cfg: LobSimulatorConfig,
) -> pd.DataFrame:
    """
    Given a mid-price series, construct a synthetic limit order book with
    multiple bid/ask levels.

    Returns
    -------
    pd.DataFrame
        Columns like:
        - mid
        - bid_px_1, bid_sz_1, ..., bid_px_n, bid_sz_n
        - ask_px_1, ask_sz_1, ..., ask_px_n, ask_sz_n
    """
    n = len(mid)
    levels = np.arange(1, cfg.n_levels + 1)

    # price offsets per level (in ticks)
    bid_offsets = -levels
    ask_offsets = levels

    rows = []

    for t in range(n):
        m = mid[t]

        # random spread noise
        spread_ticks = 1 + np.random.binomial(1, 0.3)  # sometimes 1 tick, sometimes 2
        best_bid = m - (cfg.base_spread * spread_ticks) / 2
        best_ask = m + (cfg.base_spread * spread_ticks) / 2

        # align to tick grid
        best_bid = round(best_bid / cfg.tick_size) * cfg.tick_size
        best_ask = round(best_ask / cfg.tick_size) * cfg.tick_size

        row = {"mid": m}

        # build bid side
        for lvl, offset in zip(levels, bid_offsets):
            px = best_bid + offset * cfg.tick_size
            sz = np.random.randint(1, 10)  # toy size
            row[f"bid_px_{lvl}"] = px
            row[f"bid_sz_{lvl}"] = sz

        # build ask side
        for lvl, offset in zip(levels, ask_offsets):
            px = best_ask + offset * cfg.tick_size
            sz = np.random.randint(1, 10)
            row[f"ask_px_{lvl}"] = px
            row[f"ask_sz_{lvl}"] = sz

        rows.append(row)

    return pd.DataFrame(rows)

def simulate_lob(cfg: LobSimulatorConfig) -> pd.DataFrame:
    """
    High-level convenience function:
    simulate mid-price, then build a synthetic LOB around it.
    """
    mid = simulate_mid_price(cfg)
    lob = build_lob_from_mid(mid, cfg)
    return lob

if __name__ == "__main__":
    cfg = LobSimulatorConfig(n_steps=10, n_levels=3)
    df = simulate_lob(cfg)
    print(df.head())
    
    




