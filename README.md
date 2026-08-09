# DeepLOB: A Limit-Order-Book Validation Study

An advanced, self-contained study of the data and evaluation path behind short-horizon limit-order-book modelling. It contains a synthetic order-book generator, DeepLOB and TCN implementations, and a reproducible audit of temporal leakage.

> **The result:** [When Overlapping Windows Invent Predictability](docs/when-overlapping-windows-invent-predictability.md) — on data with provably no signal, a random split of overlapping windows reports 66.0% accuracy against a 39.8% baseline; a purged split reports 32.9%. The closed-form ceiling on manufactured accuracy is derived and verified in §5.
>
> **Current status:** the historical accuracy, PnL, and Sharpe artifacts in this repository are legacy outputs from a random split of overlapping windows with full-data normalization. In addition, `evaluate_model()` scored the entire window set, including windows used for training, so those figures are training-set accuracy rather than held-out accuracy. They are not validated trading or alpha results. The active work is the validation study, [When Overlapping Windows Invent Predictability](docs/leakage_audit.md).

## Start Here

1. Read the [architecture](docs/architecture.md) for the code-path map.
2. Read [data and evaluation](docs/data-and-evaluation.md) for the synthetic-data assumptions and validation protocol.
3. Run the [leakage audit](docs/leakage_audit.md) before interpreting any model output.
4. Consult the [documentation map](docs/README.md) for the full record.

## Five-Minute Mental Model

```text
synthetic snapshots -> engineered features -> overlapping windows + labels
                                      |                 |
                                      |                 +-> DeepLOB / TCN (legacy path)
                                      |
                                      +-> temporal split + train-only scaling
                                                   -> deterministic audit (active path)
```

The simulator deliberately has no causal link from book sizes to future price movement. That makes it useful for studying how an evaluation pipeline can create apparent predictability even when the generator does not encode it.

## Repository Shape

```text
src/deep_lob/
  simulator.py     synthetic book snapshots and random-walk mid price
  data.py          feature construction and sliding-window labels
  models.py        DeepLOB and TCN definitions
  train*.py        historical training paths
  splits.py         random, chronological, and purged/embargoed protocols
  scaling.py        train-only feature scaling
  audit.py          deterministic leakage and signal audit
docs/
  architecture.md   module contracts and data flow
  data-and-evaluation.md  assumptions and result boundaries
  leakage_audit.md  active study protocol
  decisions/        durable evaluation decisions
```

## Reproduce the Audit

Create an isolated environment from the project metadata, then run:

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[test]"
PYTHONPATH=src .venv/bin/python -m pytest -q
PYTHONPATH=src .venv/bin/python -m deep_lob.audit --out reports/leakage_audit.json
```

The generated report is intentionally ignored by Git. It is evidence for the exact command and parameters used, not a permanent benchmark.

## What This Does Not Claim

- It does not establish a profitable strategy, alpha, or execution performance.
- It does not validate DeepLOB or TCN on real exchange data.
- It does not treat a model comparison as meaningful until it uses a documented temporal split and train-only transformations.

See [docs/README.md](docs/README.md) for architecture, data, decisions, references, and legacy context.
