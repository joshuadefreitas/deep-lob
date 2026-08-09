"""
Enforcement for the closed-form leakage ceiling.

The paper's second claim is that manufactured accuracy has an upper bound
computable from the window geometry alone, and that an oracle memoriser attains
it. That claim is currently supported by a table in a markdown file, which is
not enforcement.

These tests fail if the derivation stops matching the measurement — which is
what would happen if the simulator's step size changed, the label threshold
moved, the window builder's alignment shifted, or the algebra were edited.

Tolerances are deliberately loose (3-4 points). The claim being defended is
"the closed form predicts the oracle", not "it predicts it to four decimals";
a tight tolerance here would produce a brittle test that fails on unrelated
changes and gets deleted.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deep_lob.data import build_lob_windows  # noqa: E402
from deep_lob.simulator import simulate_lob  # noqa: E402
from deep_lob.splits import random_overlap_split  # noqa: E402
from run_ceiling import (  # noqa: E402
    THRESHOLD,
    TRAIN_FRAC,
    WINDOW_SIZE,
    analytic_ceiling,
    majority_frac,
    oracle_twin_accuracy,
)
from run_stride import ceiling_at_stride  # noqa: E402

HORIZONS = (5, 10, 20)
SEEDS = tuple(range(5))
TOL = 0.04
# The distance-weighted sum and the old d=1-collapse form differ by ~0.09 at
# f=0.5 and by only ~0.015 at f=0.8. Any tolerance that admits the latter must
# reject the former, or the enforcement is decorative.
TOL_LOW_F = 0.02


def _oracle(y: np.ndarray) -> float:
    accs = []
    for s in SEEDS:
        sp = random_overlap_split(len(y), TRAIN_FRAC, seed=s)
        accs.append(oracle_twin_accuracy(sp.train_idx, sp.val_idx, y))
    return float(np.mean(accs))


def _labels(horizon: int, stride: int = 1) -> np.ndarray:
    df = simulate_lob(n_rows=5000, seed=42)
    _, y = build_lob_windows(
        df, window_size=WINDOW_SIZE, horizon=horizon, threshold=THRESHOLD
    )
    return y[::stride]


@pytest.mark.parametrize("horizon", HORIZONS)
def test_closed_form_ceiling_predicts_the_oracle_at_stride_one(horizon: int) -> None:
    """The stride-1 ceiling should match what a temporal-twin memoriser scores."""
    y = _labels(horizon)
    predicted = analytic_ceiling(horizon)["ceiling"]
    observed = _oracle(y)
    assert abs(predicted - observed) < TOL, (
        f"h={horizon}: closed form {predicted:.4f} vs oracle {observed:.4f} "
        f"(delta {observed - predicted:+.4f}). The derivation and the "
        f"measurement have diverged."
    )


@pytest.mark.parametrize("horizon", HORIZONS)
def test_leak_vanishes_once_stride_reaches_the_horizon(horizon: int) -> None:
    """
    rho(s) = max(0, (h-s)/h), so at s = h the labels of retained windows share
    no increments and copying a neighbour must not beat the majority baseline.
    This is the paper's design rule; if it stops holding, the rule is wrong.
    """
    y = _labels(horizon, stride=horizon)
    observed = _oracle(y)
    majority = majority_frac(y)
    assert observed <= majority + 0.02, (
        f"h={horizon}, stride={horizon}: oracle {observed:.4f} exceeds majority "
        f"{majority:.4f}. Overlap leakage was expected to be gone at stride >= horizon."
    )


@pytest.mark.parametrize("horizon", HORIZONS)
def test_ceiling_is_monotone_decreasing_in_stride(horizon: int) -> None:
    """More stride, less shared increments, lower ceiling. No exceptions."""
    ceilings = [ceiling_at_stride(horizon, s)["ceiling"] for s in range(1, horizon + 1)]
    diffs = np.diff(ceilings)
    assert (diffs <= 1e-9).all(), (
        f"h={horizon}: ceiling not monotone in stride: {ceilings}"
    )


@pytest.mark.parametrize("horizon", HORIZONS)
def test_predicted_adjacent_agreement_matches_the_measured_rate(horizon: int) -> None:
    """
    The core of the derivation is P(adjacent windows carry the same label),
    obtained from the bivariate normal at rho = (h-1)/h. That quantity is
    directly measurable — just compare each label to its neighbour — so it can
    be checked without any model, split, or oracle in the way.

    This is the sensitive test. The ceiling assertion above is dominated by rho
    at long horizons and will happily pass on a mis-specified step size; this
    one will not, because P(agree) depends on the threshold in units of the
    h-step return's standard deviation.
    """
    y = _labels(horizon)
    measured = float((y[1:] == y[:-1]).mean())
    predicted = analytic_ceiling(horizon)["p_adjacent_agree"]
    assert abs(predicted - measured) < 0.03, (
        f"h={horizon}: predicted P(agree) {predicted:.4f} vs measured "
        f"{measured:.4f}. The bivariate-normal model of label correlation is "
        f"no longer describing the data."
    )


@pytest.mark.parametrize("horizon", HORIZONS)
def test_predicted_class_balance_matches_the_generator(horizon: int) -> None:
    """
    P(flat) = 2*Phi(theta / (sigma*sqrt(h))) - 1 depends directly on the
    simulator's per-step volatility and the label threshold. This is the
    assertion that catches a changed step size, which the ceiling test cannot.

    Tolerance is 0.04 because a single 5000-row path carries real sampling
    variation in class frequencies — measured across 20 independent paths the
    standard deviation is 0.010 to 0.019, and the h20 case sits about 1.3 sd
    from prediction. See the threats-to-validity section of the write-up.
    """
    y = _labels(horizon)
    measured_flat = float((y == 0).mean())
    predicted_flat = analytic_ceiling(horizon)["p_flat"]
    assert abs(predicted_flat - measured_flat) < 0.04, (
        f"h={horizon}: predicted P(flat) {predicted_flat:.4f} vs measured "
        f"{measured_flat:.4f}. Either the step size, the threshold, or the "
        f"label convention has changed."
    )


def test_ceiling_exceeds_what_the_trained_model_achieves() -> None:
    """
    It is a CEILING. The measured CNN accuracy from results/study must sit below
    it at every horizon, or the derivation is not bounding what it claims to.
    """
    import json

    summary_path = ROOT / "results" / "study" / "summary.json"
    if not summary_path.exists():
        pytest.skip("results/study/summary.json not present; run run_study.py")

    cells = json.loads(summary_path.read_text())["cells"]
    for horizon in HORIZONS:
        cnn = cells[f"random_split|h{horizon}"]["val_acc"]["mean"]
        ceiling = analytic_ceiling(horizon)["ceiling"]
        assert cnn < ceiling + TOL, (
            f"h={horizon}: model scored {cnn:.4f} against a ceiling of "
            f"{ceiling:.4f}. A model exceeding the bound means the bound is wrong."
        )


@pytest.mark.parametrize("horizon", HORIZONS)
def test_ceiling_holds_at_low_train_fraction(horizon: int) -> None:
    """
    The assertion that actually enforces the distance-weighted derivation.

    An earlier version of analytic_ceiling collapsed every case beyond the
    adjacent neighbour to the majority class. At f=0.8 that error is only
    ~0.015 and slips under a 0.04 tolerance, so the rest of this file passes
    on the wrong algebra - which an independent review demonstrated by
    swapping the old function back in and watching all tests stay green.

    At f=0.5 a quarter of validation windows have no adjacent training twin,
    the two forms diverge by ~0.09, and the tolerance below rejects the wrong
    one. This test is the discriminator; the others are not.
    """
    y = _labels(horizon)
    predicted = analytic_ceiling(horizon, train_frac=0.5)["ceiling"]
    observed = float(np.mean([
        oracle_twin_accuracy(sp.train_idx, sp.val_idx, y)
        for sp in (random_overlap_split(len(y), 0.5, seed=s) for s in SEEDS)
    ]))
    assert abs(predicted - observed) < TOL_LOW_F, (
        f"h={horizon}, f=0.5: closed form {predicted:.4f} vs oracle {observed:.4f} "
        f"(delta {observed - predicted:+.4f}). At low train fraction the nearest "
        f"training neighbour is often NOT adjacent; a derivation that ignores "
        f"that fails here."
    )


@pytest.mark.parametrize("train_frac", (0.5, 0.6, 0.7, 0.8, 0.9))
def test_ceiling_tracks_the_oracle_across_train_fractions(train_frac: float) -> None:
    """The error must not grow as f falls - that growth is the signature of the
    collapse mis-specification, and it is what the review used to find it."""
    y = _labels(20)
    predicted = analytic_ceiling(20, train_frac=train_frac)["ceiling"]
    observed = float(np.mean([
        oracle_twin_accuracy(sp.train_idx, sp.val_idx, y)
        for sp in (random_overlap_split(len(y), train_frac, seed=s) for s in SEEDS)
    ]))
    assert abs(predicted - observed) < TOL_LOW_F, (
        f"f={train_frac}: {predicted:.4f} vs {observed:.4f} ({observed-predicted:+.4f})"
    )
