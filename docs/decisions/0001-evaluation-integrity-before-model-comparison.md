---
type: decision-record
status: accepted
scope: synthetic DeepLOB validation study
---

# 0001. Evaluation Integrity Before Model Comparison

## Status

Accepted.

## Context

The legacy pipeline (`train.py`, `train_tcn.py`, `evaluate.py`,
`backtest.py`, `run_experiment.py`) reported classification accuracy, PnL,
win rate, and Sharpe numbers comparing DeepLOB and TCN on synthetic order
book data (see `reports/`, `backtest/`, `results/`). Two properties of that
pipeline are documented sources of leakage for temporally overlapping
samples (see `docs/references/README.md`):

1. `data.build_lob_windows` produces stride-1 **overlapping** windows, so
   window `i` and window `j` share raw rows whenever
   `|i - j| < window_size + horizon`.
2. `train.py`/`train_tcn.py` split window *indices* with
   `torch.utils.data.random_split`, and `data.prepare_features`'s default
   `size_normalization="global_max"` normalizes size columns using
   statistics from the full dataframe — both computed before, and without
   regard to, any train/validation boundary.

Either effect alone is sufficient to make a model appear to have learned
something it has not, independent of whether DeepLOB or TCN is actually
"better." Comparing two architectures' accuracy or backtest numbers on top
of a leaky evaluation is not a valid comparison — it mostly measures how
much each architecture exploits the leak, not learned skill.

`src/deep_lob/audit.py` was built to measure this directly rather than
assume it: a signal audit (is there any real relationship between features
and future returns on non-overlapping windows?) and a split audit (how much
does validation accuracy change purely from splitting methodology, holding
data and model fixed?), both against a deterministic, leakage-free
baseline (`splits.purged_embargoed_split` + `scaling.TrainOnlyScaler` +
`size_normalization="none"`). See `docs/leakage_audit.md` for the full
write-up.

## Decision

1. The legacy training/evaluation path is **not** patched in place. It
   still uses `random_split` on overlapping windows and
   `size_normalization="global_max"`, and its outputs in `reports/`,
   `backtest/`, `results/` are kept, but labeled as unvalidated legacy
   artifacts (see root `README.md`) rather than removed or silently
   corrected.
2. A separate, deterministic audit path (`audit.py`, `splits.py`,
   `scaling.py`) is the source of truth for evaluation-methodology
   questions on this repository going forward.
3. **Any future model comparison** (new architecture vs. DeepLOB/TCN, or
   DeepLOB vs. TCN itself, re-run with intent to draw a conclusion) **must**
   use a temporal/purged evaluation protocol
   (`splits.purged_embargoed_split` or an equivalent purge + embargo
   scheme) and train-only feature transforms (`scaling.TrainOnlyScaler`
   fit strictly on `train_idx`, with `size_normalization="none"` upstream).
   Comparisons made with `random_split` and/or `global_max` normalization
   are not admissible as evidence of relative model skill.

## Consequences

- Existing `reports/`, `backtest/`, `results/` numbers remain reproducible
  and inspectable, but are explicitly disclaimed everywhere they're
  referenced (root `README.md`, `docs/leakage_audit.md`).
- New work that wants to make an architecture-comparison claim has a
  ready-made, tested toolset (`splits.py`, `scaling.py`) rather than having
  to invent purge/embargo logic ad hoc.
- The codebase carries two parallel paths (legacy vs. audit) rather than
  one, which is a real maintenance cost — accepted deliberately, per
  Alternatives Considered below.
- No claim of DeepLOB vs. TCN superiority, or of tradeable signal, should
  be read from this repository until a comparison is run through the
  audit-path protocol and its own leakage/null checks pass.

## Alternatives Considered

- **Patch `train.py`/`train_tcn.py` in place to use the purged split and
  train-only scaling.** Rejected for this change: it would silently
  invalidate the existing `reports/`/`backtest/`/`results/` artifacts'
  reproducibility (different train/val membership, different sample counts)
  without an independent record of what the leak's effect size actually
  was. Measuring the leak first, before removing it, is what makes the
  `leakage_gap_random_minus_purged` figure in the audit meaningful.
- **Delete the legacy path and its artifacts.** Rejected: they are useful
  as a concrete, reproducible example of the leakage pattern the audit
  measures, and deleting them would remove the ability to regenerate that
  comparison.
- **Ignore the leakage and continue reporting legacy accuracy/PnL/Sharpe as
  results.** Rejected: those numbers are known-confounded by overlap and
  normalization leakage and would misrepresent what has been validated.
- **Add a p-value or accuracy threshold and declare the signal audit
  "passed"/"failed" as a single verdict without the null baseline.**
  Rejected: without the shuffled-label null and majority-class baseline,
  a "significant" result could still be an artifact of the audit's own
  pipeline rather than the synthetic features, so both checks are kept.
