# When Overlapping Windows Invent Predictability

*A negative-control study of evaluation protocol in time-series machine learning.*

Draft, August 2026. All figures regenerate from `run_study.py` and
`run_ceiling.py`. Per-run evidence is committed under `results/`.

---

## Summary

A short-horizon limit-order-book classifier was trained and evaluated on
synthetic data containing **no relationship whatsoever** between features and
future price. Under the standard recipe — sliding windows at stride 1, a random
train/validation split, normalisation fitted on the full dataset — it reports
**66.0% accuracy** against a majority-class baseline of **39.8%**.

Under a purged and embargoed split, on the same data, with the same model and
the same seeds, it reports **32.9%** — *below* the baseline, which is the correct
answer when there is nothing to learn.

The 33-point difference is produced entirely by how the validation set was drawn.

We then derive, in closed form, an upper bound on how much accuracy a given
split geometry makes available, using only the window stride, the label horizon,
the threshold, and the train fraction. No model enters the derivation. An oracle
memoriser attains that bound to within 1.6 points at every horizon tested.

---

## 1. Why null data

The generator produces mid prices as an i.i.d. multiplicative random walk and
order sizes as independent uniform integers. No mechanism connects either to
future price movement. Predictive skill is not merely absent; it is **impossible
by construction**.

On real data, genuine skill and manufactured skill are indistinguishable — which
is what makes leakage durable. On a negative control, every point above baseline
is definitionally an artifact of the evaluation procedure. The quantity measured
here is not "how good is the model" but **how much predictability a pipeline can
invent from nothing**.

### What this adds, and what it doesn't

That leakage is widespread is settled. Kapoor and Narayanan surveyed 17 fields,
found 329 affected papers, and produced an eight-type taxonomy. Purging and
embargo as remedies for overlapping financial series are standard after López de
Prado. This study discovers neither.

What a negative control adds is the removal of the usual defence. The standard
objection to purged cross-validation is that it is too conservative — that
purging discards legitimate short-horizon signal along with the leak, so the
lower score reflects the remedy rather than an honest measurement. On a random
walk there is no short-horizon signal to discard. The honest protocol correctly
reports nothing; the leaky one reports 26 points above baseline. The gap cannot
be attributed to a remedy destroying something real, because nothing real was
there.

The second contribution is the ceiling in §5, which is a design-time quantity
rather than a measurement.

## 2. The pipeline under study

Deliberately unmodified, because it is the subject rather than the instrument.

**Stride-1 sliding windows.** Window *t* spans rows *t … t+99*; window *t+1* spans
*t+1 … t+100*. Adjacent examples share 99 of 100 input rows, and their labels —
the sign of the forward return over horizon *h*, thresholded at ±0.0005 — are
computed from price paths sharing *h−1* of *h* increments.

**A random split.** `random_split` shuffles windows uniformly before cutting, so
near-duplicate neighbours are distributed across the boundary.

**Normalisation fitted on the whole dataframe**, including validation rows.

**And, found during this work:** the repository's historical figures were produced
by scoring on the *entire* window set, including training windows. Those numbers
are training-set accuracy.

## 3. What was varied

Three splitting protocols × three horizons × twenty seeds = 180 runs.

| protocol | description | overlap |
|---|---|---|
| `random_split` | shuffle windows, then cut | 1.00 |
| `chronological` | train on earlier windows, validate on later | 0.11 |
| `purged_embargoed` | chronological, plus purge of training windows reaching into the validation span, plus an embargo buffer | 0.00 |

Architecture, data, optimiser and epochs are held fixed. Every run seeds model
initialisation, split draw and batch order, so the study is bit-reproducible even
though the pipeline it studies is not.

## 4. Results

Validation-only accuracy, mean over 20 seeds, 95% CI of the mean, and the
majority baseline on the same validation set.

| protocol | h | overlap | accuracy | 95% CI | baseline | **excess** |
|---|---|---|---|---|---|---|
| `random_split` | 5 | 1.00 | 0.5229 | [0.5130, 0.5328] | 0.5315 | −0.0086 |
| `random_split` | 10 | 1.00 | 0.5337 | [0.5179, 0.5495] | 0.3931 | **+0.1407** |
| `random_split` | 20 | 1.00 | 0.6603 | [0.6241, 0.6966] | 0.3980 | **+0.2624** |
| `chronological` | 5 | 0.11 | 0.5029 | [0.4836, 0.5222] | 0.5745 | −0.0716 |
| `chronological` | 10 | 0.11 | 0.3238 | [0.3117, 0.3358] | 0.4484 | −0.1246 |
| `chronological` | 20 | 0.12 | 0.3264 | [0.3071, 0.3457] | 0.4023 | −0.0759 |
| `purged_embargoed` | 5 | 0.00 | 0.4853 | [0.4736, 0.4971] | 0.5754 | −0.0901 |
| `purged_embargoed` | 10 | 0.00 | 0.3316 | [0.3197, 0.3436] | 0.4510 | −0.1193 |
| `purged_embargoed` | 20 | 0.00 | 0.3290 | [0.3093, 0.3487] | 0.4013 | −0.0723 |

Reported as **excess over baseline** because class balance shifts with horizon —
the baseline moves from 0.575 to 0.398 — so raw accuracies are not comparable
across rows.

**Two cells of nine show any skill, and both have `overlap = 1.00`.** Every honest
cell is below its own baseline: a model trained on noise underperforms a constant
predictor, which is correct.

**Overlap fraction is not the predictor.** `chronological` sits at 0.11 overlap and
behaves like `purged_embargoed` at 0.00, not like something 11% of the way to
`random_split`. A time cut leaves a thin seam and manufactures nothing. What
matters is whether near-duplicate *neighbours* land on opposite sides, which
shuffling guarantees and a chronological cut prevents.

**Leakage widens as well as inflates.** The leaky h20 cell has the largest spread
in the table — sd 0.0775 against 0.0421 for its purged counterpart, individual
seeds from 0.468 to 0.751.

## 5. The ceiling

The results above measure what one architecture achieves. A different question
has a closed-form answer: **given the geometry alone, how much accuracy is
available to be manufactured?**

For stride-1 windows the forward returns of adjacent windows are sums of *h*
i.i.d. increments sharing *h−1* of them, so

$$\rho = \mathrm{corr}(r_t, r_{t+1}) = \frac{h-1}{h}$$

which rises with the horizon. With the threshold expressed in units of the
*h*-step return's standard deviation, the probability that adjacent windows carry
the same label follows from the bivariate normal at correlation ρ. A memoriser
that copies its nearest training neighbour therefore scores, in expectation,

$$\text{ceiling} = P(\text{twin in train}) \cdot P(\text{labels agree}) + \big(1 - P(\text{twin})\big) \cdot \text{majority}$$

with *P*(twin) = 1 − (1−*f*)² under a random split at train fraction *f*.

Nothing about the model appears in this. It is computable before any training run.

**Verification.** An oracle handed the temporally nearest training window:

| h | ρ | P(agree) | ceiling | oracle | error |
|---|---|---|---|---|---|
| 5 | 0.800 | 0.6944 | 0.6884 | 0.6923 | +0.0039 |
| 10 | 0.900 | 0.7518 | 0.7378 | 0.7513 | +0.0135 |
| 20 | 0.950 | 0.8123 | 0.7940 | 0.8098 | +0.0158 |

Within 1.6 points at every horizon, from a formula with no fitted parameter.

**The bound binds only under a random split.** Under chronological and purged
splits the nearest training window is the *same* window for every validation
example, so the strategy degenerates to a constant predictor and scores at or
below the majority baseline — 0.19 to 0.40 in our runs. Copying your neighbour is
only useful when your neighbour has been scattered into the training set.

**Availability is not exploitation.** Measured accuracy can be read as the fraction
of the available leak a given model recovers:

| h | baseline | CNN | ceiling | share of available leak recovered |
|---|---|---|---|---|
| 5 | 0.5324 | 0.5229 | 0.6884 | none — the CNN stays at baseline |
| 10 | 0.3892 | 0.5337 | 0.7378 | 41% |
| 20 | 0.4039 | 0.6603 | 0.7940 | 66% |

At h5 there are 15.6 points of leak available and this architecture takes none of
them. So the ceiling describes the *exposure* a split geometry creates; whether a
particular model converts that exposure into a headline number is a separate,
model-dependent question. Both are worth knowing, and only the first can be
computed in advance.

**1-NN in raw feature space scores near chance** (0.36–0.43), below the majority
baseline. Euclidean distance over price-level features finds windows at similar
*price levels*, not temporal twins. The CNN's 0.52–0.66 is therefore not naive
distance matching — it partially recovers the leak without being able to identify
the twin outright.

## 6. The remedy, with a formula

The derivation generalises to any stride, and the generalisation is
prescriptive. With stride *s*, consecutive retained windows are *s* rows apart,
so their forward returns share *h − s* of *h* increments:

$$\rho(s) = \max\left(0, \frac{h-s}{h}\right)$$

At *s* = 1 this is near 1. At *s* ≥ *h* it is exactly 0: retained windows share
no increments, their labels are independent, and copying a neighbour cannot beat
the baseline. So the closed form predicts that **overlap leakage disappears once
the stride reaches the label horizon** — a design rule applicable before any
training run, and falsifiable.

It holds. Ceiling against oracle across the sweep:

| h | stride | ρ | ceiling | oracle | excess over majority | n windows |
|---|---|---|---|---|---|---|
| 20 | 1 | 0.950 | 0.7940 | 0.8117 | **+0.4111** | 4881 |
| 20 | 2 | 0.900 | 0.7243 | 0.7284 | +0.3261 | 2441 |
| 20 | 5 | 0.750 | 0.6059 | 0.6474 | +0.2462 | 977 |
| 20 | 10 | 0.500 | 0.4910 | 0.5255 | +0.1165 | 489 |
| 20 | 15 | 0.250 | 0.4074 | 0.3894 | −0.0094 | 326 |
| 20 | 20 | 0.000 | 0.3368 | 0.3367 | **−0.0837** | 245 |
| 10 | 1 | 0.900 | 0.7378 | 0.7530 | **+0.3647** | 4891 |
| 10 | 10 | 0.000 | 0.3428 | 0.3102 | **−0.1286** | 490 |
| 5 | 1 | 0.800 | 0.6884 | 0.6936 | **+0.1597** | 4896 |
| 5 | 5 | 0.000 | 0.4056 | 0.3990 | **−0.1541** | 980 |

Across all 24 (horizon, stride) combinations the closed form predicts the oracle
to within a few points, and at *s* = *h* the agreement is essentially exact —
0.3368 against 0.3367 at h20, 0.3428 against 0.3429 at h10 with *s* = 2*h*.

Two consequences worth separating.

**The rule.** Set stride ≥ horizon and overlap leakage is gone, not reduced. At
intermediate strides the formula quantifies the remaining exposure, so the
choice can be made numerically rather than by taste.

**The cost is real and should be stated.** Raising the stride divides the dataset
by *s*: 4881 windows become 245 at h20. That is the actual trade — leakage
against sample size — and it is the reason stride-1 windowing is so common. The
formula does not make the trade go away; it makes it visible.

Note also that beyond *s* = *h* the memoriser scores *below* the majority
baseline. With independent labels, copying a neighbour yields Σp², which is less
than max p. Once overlap is gone, imitating your neighbour is worse than
guessing the most common class — which is the correct behaviour and a useful
sanity check on the derivation.

## 7. Reproducibility of the pipeline itself

`train.py` seeds neither the split, the weight initialisation, nor the batch
order. Six unseeded repetitions of the same configuration — `random_split`, h=20,
identical data (generator seed 42) — produced validation accuracies spanning
**0.5036 to 0.7134** (mean 0.6395, sd 0.0830) and all-window accuracies spanning
**0.5206 to 0.7408** (mean 0.6712, sd 0.0864). The per-run metrics are committed
in `results/study-multiseed/summary.json` under `unseeded_arm`.

The pre-harness estimate of 0.674 to 0.749 recorded in earlier versions of this
section was untraceable to committed artifacts; the artifact above supersedes it.

Stated separately from the leakage finding because it is independent and travels
further: **any model comparison decided by less than roughly sixteen accuracy
points (two standard deviations of a single unseeded run) is indistinguishable
from running the same model twice.** That includes the DeepLOB-versus-TCN
comparison previously recorded in this repository.

Three lines of seeding make the pipeline bit-deterministic. They were never
written.

## 8. What this does not establish

- **No published result is shown to be wrong.** This shows what a procedure can
  manufacture on data with no signal. Whether a given real-world result suffers
  the same effect requires that result's data and code.
- **This is not a measurement of real markets.** The generator is a random walk
  with no microstructure — a feature for a negative control, a limitation for
  external validity.
- **The normalisation leak is present in every cell**, including the honest ones,
  because the pipeline was left unmodified. Its contribution is not separately
  identified and is *presumed* small next to 33 points. Presumed, not measured.
- **The ceiling assumes i.i.d. increments and a stride of 1.** Real price series
  have volatility clustering and fat tails; the correlation (h−1)/h is exact only
  under the generator used here.
- **Predicted class balance is accurate at h5 and h10** (within 0.3 sd across 20
  independent paths) and **under-predicts the majority share at h20 by about
  1.3 sd** (predicted 0.3547, mean over paths 0.3791). A single realised path
  carries directional imbalance that grows with horizon overlap. This affects the
  baseline term in the ceiling, not the verified P(agree) term.
- **One generator, one model family, 5000 rows.** Transfer is untested.

## 9. Reproducing

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[test]"
.venv/bin/python run_study.py       # 180 seeded runs, ~15 min on 4 workers
.venv/bin/python run_ceiling.py     # closed form + oracle + 1-NN, ~2 min
.venv/bin/python run_stride.py      # stride sweep and the remedy, ~2 min
```

Outputs land in `results/study/`, `results/ceiling/` and `results/stride/`, all committed.
Recorded environment: Python 3.12.13, torch 2.13.0, numpy 2.5.1, pandas 3.0.5.

`tests/test_null_generator.py` enforces the central premise: it runs a
permutation test over non-overlapping spans and fails if any feature shows a
significant relationship with the forward return after Bonferroni correction. It
has been observed to go red when a signal is deliberately introduced.

## 10. References

See [`docs/references/literature.md`](references/literature.md).
