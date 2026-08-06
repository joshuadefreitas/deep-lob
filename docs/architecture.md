---
type: architecture
status: active
scope: synthetic DeepLOB validation study
---

# Architecture

## Purpose

DeepLOB separates two things that must not be confused: an historical neural-model pipeline and the active evaluation-integrity audit that tests whether that pipeline can create apparent skill from a synthetic null generator.

## Data Flow

```text
simulate_lob / CSV
       |
       v
data.prepare_features
       |
       +--> build_lob_windows --> dataset --> train.py / train_tcn.py
       |                                  \-> models.py --> evaluate.py / backtest.py
       |
       +--> audit.py --> splits.py + scaling.py --> deterministic JSON report
```

## Module Contracts

| Module | Responsibility | Important boundary |
| --- | --- | --- |
| `simulator.py` | Generates random-walk mid prices and independent book sizes. | Synthetic null, not a market model. |
| `data.py` | Builds relative-price, size, spread, imbalance, and return features; creates windows and labels. | Legacy global size normalization is preserved; audit requests raw sizes. |
| `models.py` | DeepLOB CNN/Inception/LSTM and TCN architectures. | Architecture code is not evidence of predictive validity. |
| `train.py`, `train_tcn.py` | Historical model-training paths. | Their random window split is legacy and not the validated protocol. |
| `splits.py` | Random, chronological, and purged/embargoed index splits. | Window span includes features and the future label row. |
| `scaling.py` | `TrainOnlyScaler`, fitted only on training windows. | Held-out values must never influence fitted statistics. |
| `audit.py` | Signal and split audit using deterministic logistic regression. | Methodology probe, not a model benchmark. |

## Model Roles

DeepLOB represents local order-book structure through convolutional stages before temporal aggregation. The TCN uses dilated temporal convolutions. Both remain useful implementation subjects, but neither is credited with predictive power in this repository until evaluated under the documented temporal protocol.

## Extending Safely

Keep data generation, feature construction, split selection, scaling, model fitting, and result interpretation independently inspectable. Any new architecture must use the same train-only transformation and temporal-evaluation boundary before it is compared with an existing model.
