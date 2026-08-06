# When Overlapping Windows Invent Predictability

A validation study of the leakage risks in this repository's synthetic
DeepLOB pipeline: sliding-window construction, train/validation splitting,
and feature normalization.

## Why this exists

The rest of this repository trains models on overlapping sliding windows
of a **synthetic, random-walk** order book and reports classification
accuracy, PnL, and Sharpe-like numbers from a simple backtest (see the
root `README.md` and `backtest/`, `reports/`). Those numbers were produced
with:

- a stride-1 overlapping window builder (`deep_lob.data.build_lob_windows`),
- `torch.utils.data.random_split` to form train/validation sets
  (`deep_lob.train`, `deep_lob.train_tcn`), and
- feature normalization fit on the *entire* dataframe, including rows that
  end up in validation (`deep_lob.data.prepare_features`, size columns).

Each of these is a well-documented way to make a model look like it has
learned something it has not (see `docs/references/`). This document adds
an independent, deterministic audit (`src/deep_lob/audit.py`) that
measures — rather than assumes — how much of the reported skill is an
artifact of window overlap and split methodology, and whether the
synthetic features carry any detectable causal relationship with future
price movement in the first place.

## What the audit checks

### 1. Signal audit — is there anything to find?

`run_signal_audit()` builds **non-overlapping full sample spans** (stride =
`window_size + horizon`, so neither the feature windows nor their future
label rows overlap) and tests each engineered feature's last-timestep value
for a linear relationship with the forward mid-price return, using a
permutation test (shuffle the forward return 1000 times, compare the
observed Pearson correlation to the shuffled distribution) rather than a
parametric p-value.

This matters because `deep_lob.simulator.simulate_lob` generates the mid
price as an i.i.d. multiplicative random walk
(`mids[t] = mids[t-1] * (1 + N(0, sigma))`) and order sizes as independent
uniform-integer noise, with **no mechanism connecting either to future
price movement**. Any statistically significant correlation found here on
non-overlapping data would indicate either a bug in feature construction
or a property of the random walk itself worth investigating — not a
tradable signal.

### 2. Split audit — how much does splitting method alone change the result?

`run_split_audit()` holds the data, labels, and model architecture fixed
(a small deterministic multinomial logistic regression — zero-initialized
weights, full-batch gradient descent, no RNG in the optimizer) and varies
**only** the train/validation split:

| Split | Description |
|---|---|
| `random_overlap` | Current default behaviour: window indices shuffled uniformly, then cut. Reproduces the leakage risk in `train.py`/`train_tcn.py`. |
| `chronological` | Train on the first `train_frac` windows in time order, validate on the rest. No purge/embargo. |
| `purged_embargoed` | Chronological cut, plus removal of training windows whose raw-row span reaches into the validation region, plus an embargo buffer after the cut (López de Prado, 2018). |

For each split the audit reports `overlap_fraction`: the share of
validation windows whose raw-row span intersects at least one training
window's span. `purged_embargoed` is constructed to make this exactly
`0.0`; `random_overlap` is expected to be close to `1.0` for realistic
window sizes.

### 3. Null baseline

The `purged_embargoed` split is re-run with labels **globally permuted**
(independent of the features) before splitting. If the pipeline is
implemented correctly, validation accuracy under this null should sit
close to the majority-class baseline — a sanity check that no other bug
(e.g., a label leaking into the feature matrix) is inflating results
independent of the window-overlap issue.

## How to run it

```bash
PYTHONPATH=src python -m deep_lob.audit \
  --out reports/leakage_audit.json \
  --n-rows 2000 --window-size 50 --horizon 10 --seed 0
```

This is fully deterministic given the same arguments: the simulator,
splits, and logistic regression all use fixed seeds / zero initialization,
so re-running produces byte-identical `per_feature` and `split_audit`
sections (see `tests/test_audit.py::test_run_full_audit_is_deterministic`).

The report is written to `reports/leakage_audit.json`. Regenerate it
locally to see current numbers — this document intentionally does not
hardcode a specific run's output, since JSON contents depend on
`--n-rows`/`--window-size`/`--horizon`/`--seed` and will drift as those
change.

### Running the tests

`pytest` is an optional dependency (`[project.optional-dependencies]` in `pyproject.toml`, available via `dev` or `test` extras), not a runtime dependency, so it is not installed by default. To run the audit's test suite reproducibly without installing anything globally, create a project-local virtual environment and install into that:

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/pytest tests/ -q
```

(`.venv/` is already git-ignored.) This installs only `numpy`, `pandas`,
`pytest`, and this package itself — `torch`/`pyyaml` are declared runtime
dependencies of the package but are not imported by anything under test
here (`deep_lob/__init__.py` is empty and `deep_lob.audit`/`deep_lob.data`/
`deep_lob.scaling`/`deep_lob.splits` only import `numpy`/`pandas`).

## How to interpret the report

- `signal_audit.verdict == "no_causal_signal_detected"` means no feature
  survived Bonferroni correction across all tested features at the chosen
  `alpha`. This is the expected outcome for this synthetic generator and
  is **not** evidence that LOB features are uninformative in general —
  only that this particular generator does not encode any.
- A large `split_audit.leakage_gap_random_minus_purged` (random-overlap
  validation accuracy minus purged-embargoed validation accuracy) is
  direct evidence that the current default splitting strategy overstates
  apparent skill, independent of whether any real signal exists.
- `split_audit.shuffled_label_null_baseline.val_accuracy` close to
  `split_audit.majority_class_baseline_accuracy` confirms the audit
  pipeline itself is not introducing a separate leak.

## Explicit scope limits

- **No trading, alpha, PnL, or Sharpe claims are made by this audit or this
  document.** The existing `backtest/` and `reports/` outputs elsewhere in
  this repository predate this audit and were produced with the leaky
  `random_split` path; they should be read as **unvalidated legacy
  artifacts**, not evidence of a working strategy.
- The audit's classifier (deterministic logistic regression on flattened
  windows) is a methodology probe, not a benchmark model. A negative
  signal-audit result does not prove no model could ever find structure;
  a positive one would need much more scrutiny before being trusted.
- All numbers are specific to the synthetic random-walk generator in
  `simulator.py`. They say nothing about real limit order book data.
- This is a validation-methodology study, not a claim that DeepLOB/TCN (or
  any architecture) is or isn't useful on real data.

## How the audit avoids the normalization leak on its own path

`deep_lob.data.prepare_features` normalizes order-book size columns by the
max over the **entire** input dataframe by default (`size_normalization=
"global_max"`, the legacy behaviour) — a normalization leak, since that max
is computed before any train/val split exists and therefore lets a
validation-only row's magnitude shift the normalized value seen by every
training row. Fitting `deep_lob.scaling.TrainOnlyScaler` *after* that
global-max normalization cannot undo it: the leak is already baked into the
numbers the scaler sees.

To make the audit itself leakage-free, `prepare_features` and
`build_lob_windows` accept `size_normalization="none"`, which leaves size
columns as raw, unnormalized values. `src/deep_lob/audit.py` always passes
`size_normalization="none"` (see `AUDIT_SIZE_NORMALIZATION`) for both the
signal audit and the split audit, so that `TrainOnlyScaler` — fit strictly
on `split.train_idx` inside `evaluate_split` — is the *only* source of
size-column statistics used anywhere in the audit. `tests/test_audit.py`
and `tests/test_scaling.py` assert this directly: held-out (validation-only)
size values are perturbed and the audit's fitted training mean/std are
checked to be byte-identical.

## Known pre-existing issue this audit does not fix in place

To preserve existing configs, results, and reproducibility of prior runs,
this audit does not modify the production training path
(`deep_lob.train`, `deep_lob.train_tcn`). Those still call
`prepare_features` / `build_lob_windows` with the default
`size_normalization="global_max"` and use `torch.utils.data.random_split`
on overlapping windows. `deep_lob.splits` provides `chronological_split`
and `purged_embargoed_split` as drop-in replacements (operating on
window-index arrays rather than dataset objects, so they compose with
`torch.utils.data.Subset`), and passing `size_normalization="none"` plus a
`TrainOnlyScaler` fit on the resulting train indices is the drop-in fix for
the normalization leak, for future pipeline changes.

See `docs/references/README.md` for the methodology sources behind these
recommendations.
