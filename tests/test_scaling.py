import numpy as np

from deep_lob.scaling import TrainOnlyScaler


def test_fit_uses_only_training_statistics():
    rng = np.random.default_rng(0)
    X_train = rng.normal(loc=0.0, scale=1.0, size=(50, 10, 4)).astype("float32")
    # Validation data has a wildly different distribution; fit() must not
    # be affected by it since it is never passed in.
    X_val = rng.normal(loc=1000.0, scale=50.0, size=(20, 10, 4)).astype("float32")

    scaler = TrainOnlyScaler().fit(X_train)

    expected_mean = X_train.reshape(-1, 4).mean(axis=0)
    expected_std = X_train.reshape(-1, 4).std(axis=0)

    np.testing.assert_allclose(scaler.mean_, expected_mean, rtol=1e-5)
    np.testing.assert_allclose(scaler.std_, expected_std, rtol=1e-5)

    # Transformed training data should be ~standardized.
    X_train_scaled = scaler.transform(X_train)
    assert abs(X_train_scaled.mean()) < 0.1
    assert abs(X_train_scaled.std() - 1.0) < 0.1

    # Transformed validation data uses train statistics, so it will NOT be
    # standardized (this is expected and correct: it proves val stats were
    # never used to fit the scaler).
    X_val_scaled = scaler.transform(X_val)
    assert X_val_scaled.mean() > 10


def test_transform_before_fit_raises():
    scaler = TrainOnlyScaler()
    try:
        scaler.transform(np.zeros((2, 2, 2)))
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass


def test_constant_feature_does_not_produce_nan():
    X_train = np.ones((10, 5, 3), dtype="float32")
    scaler = TrainOnlyScaler().fit(X_train)
    out = scaler.transform(X_train)
    assert np.isfinite(out).all()
