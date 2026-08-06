import numpy as np
import pytest

from deep_lob.data import prepare_features
from deep_lob.simulator import simulate_lob

N_ROWS = 100
SEED = 0


def test_prepare_features_default_is_unchanged_legacy_global_max():
    # Legacy default: size columns divided by their max over the whole
    # dataframe. This is left intentionally unchanged for the production
    # training path (deep_lob.train / deep_lob.train_tcn).
    df = simulate_lob(n_rows=N_ROWS, seed=SEED)
    features = prepare_features(df, n_levels=1)

    max_sz = df["bid_sz_1"].max()
    expected = (df["bid_sz_1"].astype("float32") / float(max_sz)).to_numpy()
    np.testing.assert_allclose(features["bid_sz_1_norm"].to_numpy(), expected, rtol=1e-6)
    assert np.isclose(features["bid_sz_1_norm"].max(), 1.0)


def test_prepare_features_none_leaves_size_columns_raw():
    df = simulate_lob(n_rows=N_ROWS, seed=SEED)
    features = prepare_features(df, n_levels=1, size_normalization="none")

    np.testing.assert_array_equal(
        features["bid_sz_1_norm"].to_numpy(), df["bid_sz_1"].astype("float32").to_numpy()
    )


def test_prepare_features_rejects_unknown_size_normalization():
    df = simulate_lob(n_rows=N_ROWS, seed=SEED)
    with pytest.raises(ValueError):
        prepare_features(df, size_normalization="bogus")


def test_prepare_features_none_is_row_local_for_size_columns():
    # Under "none" mode, a size column's engineered value for a given row
    # must depend only on that row -- perturbing any other row's size
    # values must leave every other row's engineered size feature
    # byte-identical. This is the property that makes TrainOnlyScaler,
    # fit only on training-window rows, immune to held-out row values.
    df = simulate_lob(n_rows=N_ROWS, seed=SEED)
    df_perturbed = df.copy()
    df_perturbed.loc[N_ROWS - 1, "bid_sz_1"] = 1e9

    before = prepare_features(df, n_levels=1, size_normalization="none")
    after = prepare_features(df_perturbed, n_levels=1, size_normalization="none")

    np.testing.assert_array_equal(
        before["bid_sz_1_norm"].to_numpy()[:-1], after["bid_sz_1_norm"].to_numpy()[:-1]
    )
