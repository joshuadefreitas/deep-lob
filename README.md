# DeepLOB — Deep Learning on Limit Order Books

This project explores **short-horizon price prediction** using **limit order book (LOB)** data and deep learning.

## 🎯 Objective

Forecast short-horizon mid-price movements or returns using deep models on LOB snapshots, inspired by research such as *DeepLOB*.

## 📁 Project Structure

- `data/raw/` — raw LOB/tick data (not versioned in git)
- `data/processed/` — cleaned & transformed tensors / parquet files
- `notebooks/` — exploratory work and research notebooks
- `src/deep_lob/` — production-grade Python code:
  - `data.py` — loading & transforming LOB data
  - `models.py` — PyTorch model architectures (CNN, CNN+LSTM, TCN)
  - `train.py` — training loop & evaluation
  - `config.py` — hyperparameters and paths
- `models/` — saved model checkpoints
- `experiments/` — logs, metrics, experiment configs

## 🛠 Tech Stack

- Python, PyTorch
- NumPy, pandas
- (Optionally) PyTorch Lightning / wandb for experiments

The goal is a **clean, research-grade implementation** that can be extended into live trading experiments later.
