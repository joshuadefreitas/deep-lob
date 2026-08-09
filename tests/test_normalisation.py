from __future__ import annotations

import numpy as np

from run_normalisation import _paired_difference, scale_train_only


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
