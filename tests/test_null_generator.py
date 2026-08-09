from deep_lob.audit import run_signal_audit
from deep_lob.simulator import simulate_lob


def test_synthetic_generator_has_no_detectable_forward_signal():
    window_size = 100
    horizon = 10
    n_permutations = 1_000

    df = simulate_lob(n_rows=20_000, seed=42)
    result = run_signal_audit(
        df,
        window_size=window_size,
        horizon=horizon,
        n_permutations=n_permutations,
        seed=42,
    )

    assert result["n_samples"] == len(
        range(0, len(df) - window_size - horizon + 1, window_size + horizon)
    )
    assert result["n_permutations"] >= 1_000
    assert result["n_significant_bonferroni"] == 0, result["per_feature"]
