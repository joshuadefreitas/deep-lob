---
type: data-and-evaluation
status: active
scope: synthetic DeepLOB validation study
---

# Data and Evaluation

See [`docs/README.md`](./README.md) for how this fits with the other docs.
This is a summary aimed at someone about to run or modify the pipeline; for
the full validation study and how to reproduce it, see
[`leakage_audit.md`](./leakage_audit.md) — that document is authoritative
and this one does not duplicate its methodology in detail.

## Synthetic generator assumptions (`simulator.py`)

`simulate_lob` generates:

- **Mid price**: an i.i.d. multiplicative random walk,
  `mids[t] = mids[t-1] * (1 + N(0, 0.03) / 100)`. Each step is independent
  of all previous steps — there is no drift, momentum, mean reversion, or
  regime structure.
- **Spread**: 1–3 ticks, drawn uniformly at random each row, independent of
  the mid-price path.
- **Bid/ask sizes** (3 levels by default): independent discrete-uniform
  integers in `[1, 10)`, unrelated to price, spread, or future returns.

Because every field is generated independently of future price movement,
**no feature derived from a single row or window of this data should have
a genuine causal relationship with the forward return**. This is the
premise the signal audit in `audit.py` tests directly rather than assumes.

## Window and label semantics (`data.py`)

`prepare_features` turns each raw row into engineered features: price
levels relative to mid (`(px - mid) / mid`), size columns (normalized or
raw depending on `size_normalization`), relative spread, top-of-book
volume imbalance, and one-step mid return.

`build_lob_windows` then builds **overlapping, stride-1** windows:

- Window `i` covers feature rows `[i, i + window_size)`.
- Its label is derived from the mid-price move between row
  `i + window_size - 1` (window's last row) and row
  `i + window_size - 1 + horizon` (the future row), thresholded at
  `threshold` (default `5e-4`) into `{-1, 0, +1}`.
- Consecutive windows share `window_size - 1` of their `window_size` rows.

## Where the leakage comes from

Two independent effects, both quantified in `leakage_audit.md`:

1. **Window overlap.** Windows `i` and `j` touch the same raw rows whenever
   `|i - j| < window_size + horizon` (`splits.window_span`). Splitting
   window *indices* uniformly at random (`torch.utils.data.random_split`,
   used by `train.py`/`train_tcn.py`) puts near-duplicate windows on both
   sides of the train/val boundary almost everywhere — `splits.overlap_fraction`
   measures this directly and is expected to be close to `1.0` under
   random splitting for realistic window sizes.
2. **Normalization boundary.** `prepare_features`'s default
   `size_normalization="global_max"` divides every size column by its max
   over the **entire** input DataFrame — computed before any split exists,
   so a validation-only row's magnitude shifts the normalized value seen by
   every training row. Fitting a train-only scaler afterward cannot undo
   this; the leak is already in the numbers.

## Split modes (`splits.py`)

| Mode | Behavior | `overlap_fraction` |
|---|---|---|
| `random_overlap_split` | Shuffle window indices uniformly, then cut (reproduces `train.py`/`train_tcn.py`). | ~1.0 for realistic window sizes |
| `chronological_split` | First `train_frac` windows (in time order) train, rest validate; no purge/embargo. | can still be >0 near the cut |
| `purged_embargoed_split` | Chronological cut, then drop training windows whose span reaches into validation (purge) and drop the first `embargo` (default `horizon`) validation windows after the cut. | exactly 0.0 by construction |

## Scaler boundary (`scaling.py`)

`TrainOnlyScaler` fits mean/std strictly on whatever array it's given —
it enforces nothing about *which* array that is. A genuinely leakage-free
pipeline requires **both**: `size_normalization="none"` in
`prepare_features`/`build_lob_windows` (so size columns start raw) *and*
fitting `TrainOnlyScaler` only on `split.train_idx`. `audit.py` does both;
`train.py`/`train_tcn.py` do neither (see [ADR
0001](./decisions/0001-evaluation-integrity-before-model-comparison.md) for
why the legacy path was left as-is rather than patched in place).

## Null baseline

`run_split_audit` includes a **shuffled-label null**: the `purged_embargoed`
split re-run with `y` globally permuted (independent of `X`) before
splitting. If the pipeline has no other hidden leak, validation accuracy
here should sit near the majority-class baseline. This is a sanity check on
the audit's own plumbing, not a claim about the synthetic features.

## What the results may — and may not — mean

- The audit's `signal_audit.verdict` answers a narrow question: does any
  engineered feature show a Bonferroni-corrected, permutation-tested
  correlation with the forward return on non-overlapping synthetic
  windows? A `"no_causal_signal_detected"` verdict is expected given how
  `simulate_lob` is constructed (see above) and says nothing about real
  LOB data.
- `split_audit.leakage_gap_random_minus_purged` quantifies how much the
  *legacy* split/normalization choices inflate apparent validation
  accuracy, independent of whether real signal exists.
- **None of this validates or invalidates DeepLOB or TCN as architectures.**
  The audit's classifier is a deterministic logistic regression chosen for
  reproducibility, not predictive power — it is a methodology probe, not a
  benchmark model.
- **None of this constitutes a trading, alpha, PnL, or Sharpe claim**, for
  the audit or for the legacy `reports/`/`backtest/`/`results/` artifacts
  they were computed to check.
- Everything here is specific to `simulator.py`'s synthetic random walk.
  None of it generalizes to real exchange order book data without
  independent validation on that data.

Full methodology, exact formulas, and how to regenerate
`reports/leakage_audit.json` yourself: [`leakage_audit.md`](./leakage_audit.md).
