"""
Leakage / predictability audit for the synthetic DeepLOB pipeline.

This module answers two narrow, falsifiable questions and nothing more:

1. Signal audit: do current-timestep synthetic order-book features carry
   any statistically detectable relationship with the *future* mid-price
   return, once windows are made non-overlapping? (correlation +
   permutation test)

2. Split audit: how much does validation accuracy change purely as a
   function of *how overlapping windows are split* into train/val, holding
   the model, data, and labels fixed? (random-overlap vs. chronological vs.
   purged+embargoed, plus a shuffled-label null baseline)

Explicit scope limits
----------------------
- This module makes NO trading, alpha, PnL, or Sharpe claims.
- It does not certify any model as "profitable" or "production ready".
- All results are specific to the synthetic random-walk generator in
  simulator.py; they say nothing about real exchange data.
- The classifier used here is a small deterministic (zero-init,
  full-batch gradient descent) multinomial logistic regression, chosen
  for reproducibility and speed, not predictive power. It is a
  methodology probe, not a benchmark model.

See docs/leakage_audit.md for the write-up and docs/references/ for
primary sources on purged/embargoed cross-validation and time-series
evaluation pitfalls.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from deep_lob.data import build_lob_windows, prepare_features
from deep_lob.scaling import TrainOnlyScaler
from deep_lob.simulator import simulate_lob
from deep_lob.splits import (
    SplitResult,
    chronological_split,
    n_windows_for,
    overlap_fraction,
    purged_embargoed_split,
    random_overlap_split,
)

NUM_CLASSES = 3  # {-1, 0, 1} -> {0, 1, 2}

# `prepare_features`'s legacy "global_max" mode normalizes size columns by
# the max over the *entire* input dataframe, before any train/val split
# exists -- so a val-only row can shift the normalized value of every train
# row. Fitting `TrainOnlyScaler` afterward cannot undo that: the leak is
# already baked into the numbers TrainOnlyScaler sees. The audit therefore
# always requests "none" (raw, unnormalized size columns) so that
# TrainOnlyScaler, fit strictly on `split.train_idx`, is the only source of
# size-column statistics. See docs/leakage_audit.md.
AUDIT_SIZE_NORMALIZATION = "none"


# ---------------------------------------------------------------------------
# Deterministic multinomial logistic regression (numpy only)
# ---------------------------------------------------------------------------


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def train_logreg(
    X: np.ndarray,
    y: np.ndarray,
    num_classes: int = NUM_CLASSES,
    iters: int = 200,
    lr: float = 0.5,
    l2: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Full-batch gradient descent from zero-initialized weights. Given fixed
    X, y, and hyperparameters, this is exactly reproducible with no RNG
    involved in the optimization itself.
    """
    n, d = X.shape
    W = np.zeros((d, num_classes))
    b = np.zeros(num_classes)
    Y = np.eye(num_classes)[y]

    for _ in range(iters):
        logits = X @ W + b
        probs = _softmax(logits)
        grad_logits = (probs - Y) / n
        grad_W = X.T @ grad_logits + l2 * W
        grad_b = grad_logits.sum(axis=0)
        W -= lr * grad_W
        b -= lr * grad_b

    return W, b


def predict_logreg(X: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    return _softmax(X @ W + b).argmax(axis=1)


# ---------------------------------------------------------------------------
# Split audit
# ---------------------------------------------------------------------------


def _fit_eval_logreg(
    X: np.ndarray,
    y_shifted: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    iters: int,
    lr: float,
) -> dict[str, float]:
    if len(train_idx) == 0 or len(val_idx) == 0:
        return {"train_accuracy": float("nan"), "val_accuracy": float("nan")}

    scaler = TrainOnlyScaler().fit(X[train_idx])
    X_train = scaler.transform(X[train_idx]).reshape(len(train_idx), -1)
    X_val = scaler.transform(X[val_idx]).reshape(len(val_idx), -1)

    W, b = train_logreg(X_train, y_shifted[train_idx], iters=iters, lr=lr)

    train_acc = float((predict_logreg(X_train, W, b) == y_shifted[train_idx]).mean())
    val_acc = float((predict_logreg(X_val, W, b) == y_shifted[val_idx]).mean())
    return {"train_accuracy": train_acc, "val_accuracy": val_acc}


def evaluate_split(
    X: np.ndarray,
    y: np.ndarray,
    split: SplitResult,
    window_size: int,
    horizon: int,
    iters: int = 200,
    lr: float = 0.5,
) -> dict[str, Any]:
    y_shifted = (y + 1).astype(int)  # {-1,0,1} -> {0,1,2}

    metrics = _fit_eval_logreg(X, y_shifted, split.train_idx, split.val_idx, iters, lr)
    metrics["n_train"] = int(len(split.train_idx))
    metrics["n_val"] = int(len(split.val_idx))
    metrics["overlap_fraction"] = overlap_fraction(
        split.train_idx, split.val_idx, window_size, horizon
    )
    return metrics


def run_split_audit(
    X: np.ndarray,
    y: np.ndarray,
    window_size: int,
    horizon: int,
    train_frac: float = 0.8,
    seed: int = 0,
    iters: int = 200,
    lr: float = 0.5,
) -> dict[str, Any]:
    """
    Compare random-overlap, chronological, and purged+embargoed splits on
    the *same* features/labels/model, and report a shuffled-label null
    baseline for the purged+embargoed split as a sanity check that the
    evaluation pipeline reports chance-level accuracy when there is no
    learnable relationship at all.
    """
    n_windows = X.shape[0]

    splits = {
        "random_overlap": random_overlap_split(n_windows, train_frac, seed=seed),
        "chronological": chronological_split(n_windows, train_frac),
        "purged_embargoed": purged_embargoed_split(
            n_windows, window_size, horizon, train_frac
        ),
    }

    results: dict[str, Any] = {}
    for name, split in splits.items():
        results[name] = evaluate_split(X, y, split, window_size, horizon, iters, lr)

    # Majority-class baseline (trivial reference point).
    y_shifted = (y + 1).astype(int)
    counts = np.bincount(y_shifted, minlength=NUM_CLASSES)
    majority_acc = float(counts.max() / counts.sum())
    results["majority_class_baseline_accuracy"] = majority_acc

    # Shuffled-label null baseline on the strictest (purged+embargoed) split:
    # labels are globally permuted, independent of X, before splitting.
    rng = np.random.default_rng(seed)
    y_shuffled = y.copy()
    rng.shuffle(y_shuffled)
    pe_split = splits["purged_embargoed"]
    null_metrics = evaluate_split(X, y_shuffled, pe_split, window_size, horizon, iters, lr)
    results["shuffled_label_null_baseline"] = {
        "description": (
            "purged_embargoed split re-run with y globally permuted "
            "(independent of X) before splitting; val_accuracy here should "
            "sit near the majority-class baseline if the pipeline has no "
            "hidden bugs inflating accuracy."
        ),
        **null_metrics,
    }

    results["leakage_gap_random_minus_purged"] = (
        results["random_overlap"]["val_accuracy"] - results["purged_embargoed"]["val_accuracy"]
    )

    return results


# ---------------------------------------------------------------------------
# Signal audit: correlation + permutation test on non-overlapping windows
# ---------------------------------------------------------------------------


def run_signal_audit(
    df: pd.DataFrame,
    window_size: int,
    horizon: int,
    n_levels: int = 3,
    threshold: float = 5e-4,
    n_permutations: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """
    Tests whether last-timestep feature values carry a linear relationship
    with the forward mid-price return, using *non-overlapping* windows
    so that autocorrelation between adjacent overlapping windows cannot
    manufacture spurious significance.

    Each sample consumes rows [start, start + window_size - 1 + horizon]:
    the feature window itself (window_size rows) plus the future row used
    to compute the forward return (horizon rows further out). The stride
    is therefore window_size + horizon, not window_size, so that
    consecutive samples' full feature-plus-label spans cannot overlap.

    Uses a permutation test (shuffle the forward return relative to the
    features) rather than a parametric p-value, since the underlying
    process is not assumed Gaussian.
    """
    mid = df["mid"].astype("float64").to_numpy()
    features = prepare_features(df, n_levels=n_levels, size_normalization=AUDIT_SIZE_NORMALIZATION)
    feat_names = list(features.columns)
    feat_values = features.to_numpy(dtype="float64")

    n = len(df)
    stride = window_size + horizon
    starts = list(range(0, n - window_size - horizon + 1, stride))
    if len(starts) < 3:
        raise ValueError("Not enough non-overlapping windows for a signal audit.")

    last_idx = np.array([s + window_size - 1 for s in starts])
    future_idx = last_idx + horizon

    x_last = feat_values[last_idx, :]  # (n_samples, n_features)
    forward_return = (mid[future_idx] - mid[last_idx]) / mid[last_idx]

    rng = np.random.default_rng(seed)
    per_feature: dict[str, Any] = {}
    n_significant_raw = 0

    for j, name in enumerate(feat_names):
        col = x_last[:, j]
        if np.std(col) < 1e-12 or np.std(forward_return) < 1e-12:
            per_feature[name] = {
                "pearson_r": 0.0,
                "p_value_permutation": 1.0,
                "significant_at_alpha": False,
            }
            continue

        observed_r = float(np.corrcoef(col, forward_return)[0, 1])

        perm_r = np.empty(n_permutations)
        shuffled = forward_return.copy()
        for k in range(n_permutations):
            rng.shuffle(shuffled)
            perm_r[k] = np.corrcoef(col, shuffled)[0, 1]

        p_value = float((np.abs(perm_r) >= abs(observed_r)).mean())
        if p_value == 0.0:
            p_value = 1.0 / (n_permutations + 1)  # conservative floor

        is_sig = p_value < alpha
        n_significant_raw += int(is_sig)

        per_feature[name] = {
            "pearson_r": observed_r,
            "p_value_permutation": p_value,
            "significant_at_alpha": is_sig,
        }

    # Bonferroni correction across all features tested.
    n_features = len(feat_names)
    bonferroni_alpha = alpha / max(1, n_features)
    n_significant_bonferroni = sum(
        1 for v in per_feature.values() if v["p_value_permutation"] < bonferroni_alpha
    )

    return {
        "n_samples": len(starts),
        "n_features": n_features,
        "n_permutations": n_permutations,
        "alpha": alpha,
        "bonferroni_alpha": bonferroni_alpha,
        "n_significant_uncorrected": n_significant_raw,
        "n_significant_bonferroni": n_significant_bonferroni,
        "per_feature": per_feature,
        "verdict": (
            "no_causal_signal_detected"
            if n_significant_bonferroni == 0
            else "signal_detected_investigate_before_trusting_it"
        ),
    }


# ---------------------------------------------------------------------------
# Full audit + CLI
# ---------------------------------------------------------------------------


def run_full_audit(
    n_rows: int = 2000,
    window_size: int = 50,
    horizon: int = 10,
    n_levels: int = 3,
    threshold: float = 5e-4,
    train_frac: float = 0.8,
    seed: int = 0,
    n_permutations: int = 500,
    logreg_iters: int = 200,
    logreg_lr: float = 0.5,
) -> dict[str, Any]:
    df = simulate_lob(n_rows=n_rows, seed=seed)

    X, y = build_lob_windows(
        df,
        window_size=window_size,
        horizon=horizon,
        n_levels=n_levels,
        threshold=threshold,
        size_normalization=AUDIT_SIZE_NORMALIZATION,
    )
    assert X.shape[0] == n_windows_for(n_rows, window_size, horizon)

    split_audit = run_split_audit(
        X, y, window_size, horizon, train_frac=train_frac, seed=seed,
        iters=logreg_iters, lr=logreg_lr,
    )
    signal_audit = run_signal_audit(
        df, window_size, horizon, n_levels=n_levels, threshold=threshold,
        n_permutations=n_permutations, seed=seed,
    )

    return {
        "title": "When Overlapping Windows Invent Predictability",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "metadata": {
            "n_rows": n_rows,
            "window_size": window_size,
            "horizon": horizon,
            "n_levels": n_levels,
            "threshold": threshold,
            "train_frac": train_frac,
            "seed": seed,
            "n_windows": int(X.shape[0]),
            "n_permutations": n_permutations,
            "data_source": "synthetic random-walk simulator (deep_lob.simulator.simulate_lob)",
        },
        "signal_audit": signal_audit,
        "split_audit": split_audit,
        "claims_policy": (
            "This report makes no trading, alpha, PnL, or Sharpe claims. "
            "It measures (a) whether synthetic features show a statistically "
            "detectable causal relationship with future returns on "
            "non-overlapping data, and (b) how much validation accuracy "
            "changes purely from the choice of train/val splitting strategy "
            "on overlapping windows, holding data and model fixed. Results "
            "are specific to the synthetic generator in simulator.py and do "
            "not generalize to real exchange data without independent "
            "validation."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the leakage/predictability audit and write a JSON report."
    )
    parser.add_argument("--out", type=str, default="reports/leakage_audit.json")
    parser.add_argument("--n-rows", type=int, default=2000)
    parser.add_argument("--window-size", type=int, default=50)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--levels", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=5e-4)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-permutations", type=int, default=500)
    parser.add_argument("--logreg-iters", type=int, default=200)
    parser.add_argument("--logreg-lr", type=float, default=0.5)
    args = parser.parse_args()

    report = run_full_audit(
        n_rows=args.n_rows,
        window_size=args.window_size,
        horizon=args.horizon,
        n_levels=args.levels,
        threshold=args.threshold,
        train_frac=args.train_frac,
        seed=args.seed,
        n_permutations=args.n_permutations,
        logreg_iters=args.logreg_iters,
        logreg_lr=args.logreg_lr,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(report, f, indent=2)

    print(f"Wrote leakage audit report to {out_path}")
    print(f"Signal audit verdict: {report['signal_audit']['verdict']}")
    sa = report["split_audit"]
    print(
        "Val accuracy — random_overlap: "
        f"{sa['random_overlap']['val_accuracy']:.3f} | "
        f"chronological: {sa['chronological']['val_accuracy']:.3f} | "
        f"purged_embargoed: {sa['purged_embargoed']['val_accuracy']:.3f} | "
        f"majority_baseline: {sa['majority_class_baseline_accuracy']:.3f}"
    )


if __name__ == "__main__":
    main()
