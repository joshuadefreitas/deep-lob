# DeepLOB

DeepLOB is an applied research project for learning short-horizon mid-price movement from limit order book data.

The repository keeps the full experiment visible: synthetic data generation, sliding-window construction, PyTorch models, evaluation, and a deliberately simple backtest. The goal is to study the relationship between data representation, model architecture, and signal behavior, not to claim a deployable trading strategy.

## Pipeline

```text
synthetic order book
        ↓
windowed tensors + labels
        ↓
DeepLOB CNN/Inception/LSTM or TCN
        ↓
classification metrics
        ↓
simple signal backtest
```

Each sample uses a 100-timestep window with 15 features. Labels represent a future mid-price move: rise, flat, or fall.

## Results from the committed experiment

| Model | Accuracy | Macro F1 | Sharpe | Max drawdown |
| --- | ---: | ---: | ---: | ---: |
| DeepLOB | 0.6263 | 0.6264 | 0.560 | 0.0095 |
| TCN | **0.6444** | **0.6518** | **0.667** | **0.0073** |

These figures are from synthetic data and the included backtest assumptions. They are useful for comparing this experiment's components, not evidence of live-market performance.

## Quickstart

Python 3.11+ is required. Install the project in an isolated environment:

```bash
git clone https://github.com/joshuadefreitas/deep-lob.git
cd deep-lob
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
pytest
```

Build the included synthetic dataset:

```bash
PYTHONPATH=src python -m deep_lob.simulator \
  --out data/raw/simulated_lob.csv \
  --n-rows 5000

PYTHONPATH=src python -m deep_lob.data \
  --csv data/raw/simulated_lob.csv \
  --out data/processed/lob_windows.npz \
  --window-size 100 \
  --horizon 10
```

The training and evaluation entry points are documented in `src/deep_lob/` and in the technical overview.

## Repository map

```text
src/deep_lob/       # simulator, tensorizer, models, training, evaluation, backtest
configs/            # experiment configurations
data/               # committed synthetic inputs and processed examples
reports/            # metrics and diagnostic plots
backtest/           # equity curves and summary statistics
docs/               # technical overview and paper
tests/              # import smoke tests
```

## Limitations

- The data is synthetic rather than an exchange feed.
- The backtest is intentionally simple and does not model realistic execution costs.
- The experiment is not a production trading system or investment recommendation.

Read the [technical overview](docs/deeplob_overview.md) for the modeling details.
