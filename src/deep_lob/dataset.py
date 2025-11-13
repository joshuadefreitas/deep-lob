"""
PyTorch Dataset for DeepLOB windows.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class DeepLOBDataset(Dataset):
    """
    Dataset wrapping precomputed LOB windows stored in a .npz file.

    Expects an .npz with arrays:
      - X: (num_samples, window_size, num_features)
      - y: (num_samples,) with labels in {-1, 0, 1}
    """

    def __init__(self, npz_path: str | Path) -> None:
        npz_path = Path(npz_path)
        data = np.load(npz_path)

        X = data["X"]
        y = data["y"]

        # Store as torch tensors
        # X: (N, T, F)
        self.X = torch.from_numpy(X).float()

        # y is currently in {-1, 0, 1}
        # For cross-entropy we map it to {0, 1, 2}
        self.y = torch.from_numpy(y).long() + 1

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]