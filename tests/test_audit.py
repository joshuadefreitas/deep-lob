import numpy as np

from deep_lob.audit import AUDIT_SIZE_NORMALIZATION, run_full_audit, run_signal_audit, run_split_audit
from deep_lob.data import build_lob_windows
from deep_lob.scaling import TrainOnlyScaler
from deep_lob.simulator import simulate_lob
from deep_lob.splits import purged_embargoed_split

# Small, fast configuration used across this test module.
N_ROWS = 400
WINDOW_SIZE = 15
HORIZON = 5
SEED = 0


def test_run_full_audit_is_deterministic():
    r1 = run_full_audit(n_rows=N_ROWS, window_size=WINDOW_SIZE, horizon=HORIZON, seed=SEED,
                         n_permutations=100, logreg_iters=100)
    r2 = run_full_audit(n_rows=N_ROWS, window_size=WINDOW_SIZE, horizon=HORIZON, seed=SEED,
                         n_permutations=100, logreg_iters=100)

    assert r1["split_audit"] == r2["split_audit"]
    assert r1["signal_audit"]["per_feature"] == r2["signal_audit"]["per_feature"]


def test_run_full_audit_makes_no_pnl_or_sharpe_claims():
    # The report is allowed to *disclaim* Sharpe/PnL claims by name (that's
    # the point of `claims_policy`); it must never report or claim any such
    # figure in the actual result data.
    r = run_full_audit(n_rows=N_ROWS, window_size=WINDOW_SIZE, horizon=HORIZON, seed=SEED,
                        n_permutations=50, logreg_iters=50)
    data = {k: v for k, v in r.items() if k != "claims_policy"}
    blob = str(data).lower()
    for forbidden in ["sharpe", "pnl", "alpha signal", "profitable"]:
        assert forbidden not in blob
    assert "sharpe" in r["claims_policy"].lower()
    assert "pnl" in r["claims_policy"].lower()


def test_split_audit_purged_embargoed_has_zero_overlap():
    df = simulate_lob(n_rows=N_ROWS, seed=SEED)
    X, y = build_lob_windows(df, window_size=WINDOW_SIZE, horizon=HORIZON)
    result = run_split_audit(X, y, WINDOW_SIZE, HORIZON, seed=SEED, iters=50)

    assert result["purged_embargoed"]["overlap_fraction"] == 0.0
    assert result["random_overlap"]["overlap_fraction"] > result["purged_embargoed"]["overlap_fraction"]
    assert 0.0 <= result["majority_class_baseline_accuracy"] <= 1.0

    null = result["shuffled_label_null_baseline"]
    # With labels independent of X, validation accuracy under the strictest
    # split should not wildly exceed the majority-class baseline.
    assert null["val_accuracy"] <= result["majority_class_baseline_accuracy"] + 0.35


def test_split_audit_reports_a_leakage_gap_field():
    df = simulate_lob(n_rows=N_ROWS, seed=SEED)
    X, y = build_lob_windows(df, window_size=WINDOW_SIZE, horizon=HORIZON)
    result = run_split_audit(X, y, WINDOW_SIZE, HORIZON, seed=SEED, iters=50)
    assert "leakage_gap_random_minus_purged" in result
    assert isinstance(result["leakage_gap_random_minus_purged"], float)


def test_signal_audit_structure_and_bounds():
    df = simulate_lob(n_rows=N_ROWS, seed=SEED)
    result = run_signal_audit(df, WINDOW_SIZE, HORIZON, n_permutations=100, seed=SEED)

    assert result["n_samples"] > 0
    assert result["verdict"] in {
        "no_causal_signal_detected",
        "signal_detected_investigate_before_trusting_it",
    }
    for name, stats in result["per_feature"].items():
        assert 0.0 <= stats["p_value_permutation"] <= 1.0
        assert -1.0 <= stats["pearson_r"] <= 1.0
        assert isinstance(stats["significant_at_alpha"], bool)


def test_signal_audit_uses_non_overlapping_windows():
    # Non-overlapping stride means n_samples should be roughly n_rows / window_size,
    # not n_rows - window_size - horizon + 1 (which is what the overlapping
    # window builder would produce).
    df = simulate_lob(n_rows=N_ROWS, seed=SEED)
    result = run_signal_audit(df, WINDOW_SIZE, HORIZON, n_permutations=50, seed=SEED)
    expected_upper_bound = N_ROWS // WINDOW_SIZE
    assert result["n_samples"] <= expected_upper_bound


def test_audit_uses_leakage_free_size_normalization():
    assert AUDIT_SIZE_NORMALIZATION == "none"


def test_audit_train_only_scaler_stats_immune_to_holdout_size_perturbation():
    # This is the release-blocking leakage this test guards against:
    # `prepare_features`'s legacy "global_max" mode divides size columns by
    # their max over the *entire* dataframe, computed before any train/val
    # split exists. Fitting TrainOnlyScaler afterward cannot undo that,
    # since a held-out row's magnitude already shifted the normalized
    # value of every training row. The audit must instead build features
    # with size_normalization="none" so TrainOnlyScaler -- fit strictly on
    # `split.train_idx` -- is the only source of size-column statistics.
    df = simulate_lob(n_rows=N_ROWS, seed=SEED)

    X_leaky_before, _ = build_lob_windows(df, window_size=WINDOW_SIZE, horizon=HORIZON)
    X_audit_before, _ = build_lob_windows(
        df, window_size=WINDOW_SIZE, horizon=HORIZON, size_normalization=AUDIT_SIZE_NORMALIZATION
    )
    n_windows = X_audit_before.shape[0]
    split = purged_embargoed_split(n_windows, WINDOW_SIZE, HORIZON, train_frac=0.8)
    assert len(split.train_idx) > 0 and len(split.val_idx) > 0

    # Perturb a raw order-book row that belongs only to the tail of the
    # dataframe, i.e. only to validation-region windows under a
    # chronological (train-comes-first) split -- never to a training
    # window's span.
    df_perturbed = df.copy()
    last_row = len(df_perturbed) - 1
    for level in (1, 2, 3):
        df_perturbed.loc[last_row, f"bid_sz_{level}"] = 1e9
        df_perturbed.loc[last_row, f"ask_sz_{level}"] = 1e9

    # Sanity check the perturbed row is indeed held-out: no training
    # window's raw-row span reaches it.
    last_train_span_end = max(
        int(idx) + WINDOW_SIZE - 1 + HORIZON for idx in split.train_idx
    )
    assert last_train_span_end < last_row

    # Sanity check the legacy default ("global_max") is in fact leaky:
    # perturbing this single held-out row changes the shared max
    # denominator, and therefore every row's normalized size features,
    # including training rows.
    X_leaky_after, _ = build_lob_windows(df_perturbed, window_size=WINDOW_SIZE, horizon=HORIZON)
    assert not np.array_equal(X_leaky_before[split.train_idx], X_leaky_after[split.train_idx])

    # The audit's mode ("none") must be completely unaffected.
    X_audit_after, _ = build_lob_windows(
        df_perturbed,
        window_size=WINDOW_SIZE,
        horizon=HORIZON,
        size_normalization=AUDIT_SIZE_NORMALIZATION,
    )
    np.testing.assert_array_equal(X_audit_before[split.train_idx], X_audit_after[split.train_idx])

    scaler_before = TrainOnlyScaler().fit(X_audit_before[split.train_idx])
    scaler_after = TrainOnlyScaler().fit(X_audit_after[split.train_idx])
    np.testing.assert_array_equal(scaler_before.mean_, scaler_after.mean_)
    np.testing.assert_array_equal(scaler_before.std_, scaler_after.std_)


def test_signal_audit_sample_spans_cannot_overlap():
    # Each sample consumes rows [start, start + window_size - 1 + horizon]:
    # the feature window plus the future row used for the label. The stride
    # between consecutive samples must be at least window_size + horizon,
    # otherwise one sample's label row falls inside the next sample's
    # feature window (or vice versa), leaking information across samples.
    df = simulate_lob(n_rows=N_ROWS, seed=SEED)
    result = run_signal_audit(df, WINDOW_SIZE, HORIZON, n_permutations=50, seed=SEED)

    n_samples = result["n_samples"]
    span = WINDOW_SIZE - 1 + HORIZON  # last feature index to label index, inclusive
    full_span_length = WINDOW_SIZE + HORIZON  # stride required for non-overlap

    # The number of samples the audit reports must match what a stride of
    # exactly (window_size + horizon) would produce -- a smaller stride
    # (e.g. the old buggy window_size stride) would yield strictly more
    # samples for the same data.
    expected_n_samples = len(
        range(0, N_ROWS - WINDOW_SIZE - HORIZON + 1, full_span_length)
    )
    assert n_samples == expected_n_samples

    # Directly reconstruct the start indices the audit must have used and
    # confirm consecutive full feature-plus-label spans never overlap.
    starts = [i * full_span_length for i in range(n_samples)]
    for start_prev, start_next in zip(starts, starts[1:]):
        prev_span_end = start_prev + span
        assert prev_span_end < start_next
