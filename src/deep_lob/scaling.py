"""
Train-only feature scaling.

`deep_lob.data.prepare_features` normalizes order-book sizes by the global
max across the *entire* dataframe (train + validation + test combined) by
default, which leaks distributional information about held-out data into
the training-time feature representation. `TrainOnlyScaler` fits mean/std
strictly on a provided training array; callers are responsible for passing
only the training split. Note that fitting `TrainOnlyScaler` on features
that were already global-max-normalized does NOT remove that leak, since
the leak is baked into the values before `TrainOnlyScaler` ever sees them
-- callers that want a genuinely train-only pipeline must also pass
`size_normalization="none"` to `prepare_features` / `build_lob_windows`.
`deep_lob.audit` does this; see `docs/leakage_audit.md`.
"""

from __future__ import annotations

import numpy as np


class TrainOnlyScaler:
    """Per-feature z-score scaler fit on a training array only.

    Expects arrays shaped ``(n_samples, ..., n_features)``; statistics are
    computed over every axis except the last.
    """

    def __init__(self, eps: float = 1e-8) -> None:
        self.eps = eps
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, X_train: np.ndarray) -> "TrainOnlyScaler":
        flat = X_train.reshape(-1, X_train.shape[-1])
        self.mean_ = flat.mean(axis=0)
        std = flat.std(axis=0)
        std[std < self.eps] = 1.0
        self.std_ = std
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("TrainOnlyScaler.fit() must be called before transform().")
        return (X - self.mean_) / self.std_

    def fit_transform(self, X_train: np.ndarray) -> np.ndarray:
        return self.fit(X_train).transform(X_train)
