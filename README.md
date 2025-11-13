# DeepLOB — Deep Learning on Limit Order Books

This project explores **short-horizon price prediction** using **limit order book (LOB)** data and deep learning.

## 🎯 Goal

Build and evaluate deep learning architectures (CNNs, LSTMs, Temporal CNNs) on LOB tensors to forecast:
- Mid-price direction
- Short-horizon returns
- Volatility bursts

## 🧱 Planned Components

- LOB tensor construction (multi-level bid/ask snapshots)
- PyTorch dataset & dataloader
- CNN + LSTM / Temporal CNN models
- GPU training & evaluation
- Signal backtesting over very short horizons

## 🛠 Tech Stack

- Python, PyTorch
- NumPy, pandas
- (Later) vectorbt / custom backtesting engine

This project is designed as a **flagship deep learning + microstructure** piece in my portfolio.
