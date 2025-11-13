"""
Data loading and preprocessing utilities for DeepLOB.

This module will handle:
- Reading raw LOB data (csv/parquet)
- Building LOB tensors (bid/ask levels over time)
- Train/validation/test splits
"""

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


def load_raw_lob(path: Path) -> pd.DataFrame:
    """
    Load raw limit order book data from disk.

    Parameters
    ----------
    path : Path
        Path to the raw data file.

    Returns
    -------
    pd.DataFrame
        Raw LOB data.
    """
    # Placeholder: adapt based on your raw data format
    return pd.read_csv(path)


def build_lob_tensor(df: pd.DataFrame) -> np.ndarray:
    """
    Transform a raw LOB dataframe into a 3D tensor:
    [samples, timesteps, features].

    This is where you define:
    - number of levels (bid/ask)
    - features per level (price, volume, etc.)
    - lookback window length

    Returns
    -------
    np.ndarray
        LOB tensor.
    """
    # TODO: implement actual tensor construction
    raise NotImplementedError("LOB tensor construction not implemented yet.")
