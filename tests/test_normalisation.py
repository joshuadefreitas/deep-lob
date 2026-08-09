from __future__ import annotations

import numpy as np
import pytest

# run_normalisation imports torch at module scope. Without this guard a missing
# torch is a COLLECTION error, which aborts the entire suite -- not just this
# file. One absent optional dependency silently disabling every test is a worse
# failure than the skip.
pytest.importorskip("torch", reason="run_normalisation requires torch")

from run_normalisation import (
    _paired_difference,
    _summarise_path,
    scale_train_only,
)


def test_scale_train_only_fit_is_unaffected_by_validation_values() -> None:
    rng = np.random.default_rng(7)
    X = rng.normal(size=(8, 4, 3)).astype("float32")
    train_idx = np.array([0, 1, 2, 3, 4])
    changed_validation = X.copy()
    changed_validation[5:] += 10_000

    original_scaled = scale_train_only(X, train_idx)
    changed_scaled = scale_train_only(changed_validation, train_idx)

    np.testing.assert_allclose(original_scaled[train_idx], changed_scaled[train_idx])
    np.testing.assert_allclose(
        original_scaled[train_idx].reshape(-1, 3).mean(axis=0), 0.0, atol=1e-6
    )


def test_paired_ci_is_computed_from_seedwise_differences() -> None:
    global_values = [0.7, 0.5, 0.9]
    train_values = [0.6, 0.6, 0.7]
    result = _paired_difference(global_values, train_values)

    expected = np.asarray(global_values) - np.asarray(train_values)
    assert result["mean"] == expected.mean()
    assert result["sd"] == expected.std(ddof=1)
    assert result["n_pairs"] == 3


def test_path_summary_rejects_unpaired_scaling_seeds() -> None:
    common = {
        "protocol": "random_split",
        "horizon": 5,
        "sim_seed": 0,
        "n_train": 8,
        "n_val": 2,
        "val_acc": 0.5,
        "full_acc": 0.5,
        "val_majority": 0.6,
        "full_majority": 0.6,
        "overlap_fraction": 1.0,
    }
    rows = [
        {**common, "scaling": "global_max", "seed": 0},
        {**common, "scaling": "train_only", "seed": 1},
    ]

    try:
        _summarise_path(rows)
        assert False, "expected RuntimeError"
    except RuntimeError as exc:
        assert "unpaired scaling arms" in str(exc)
