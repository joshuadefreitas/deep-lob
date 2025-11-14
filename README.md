# 📘 DeepLOB – Learning Market Microstructure From Limit Order Books

Modern electronic markets are chaotic. Prices shift because thousands of tiny decisions—orders, cancellations, trades—flash in and out of the order book every second.  
This project builds a system that **learns to predict short‑horizon mid‑price movements** directly from this order book data.

It includes:  
- Synthetic LOB generator  
- Sliding‑window tensorization  
- PyTorch Dataset & DataLoaders  
- DeepLOB CNN → Inception → LSTM model  
- TCN (Temporal Convolutional Network)  
- Evaluation & backtesting engine  
- Architecture comparison  

---

## 1. Project Structure
```
deep-lob/
│
├── data/
│   ├── raw/            # Simulated or real LOB CSV files
│   └── processed/      # Sliding window tensors (X, y)
│
├── src/deep_lob/
│   ├── simulator.py    # Synthetic LOB generator
│   ├── data.py         # Window builder
│   ├── dataset.py      # PyTorch dataset
│   ├── models.py       # DeepLOB & TCN architectures
│   ├── train.py        # DeepLOB training script
│   ├── train_tcn.py    # TCN training script
│   ├── evaluate.py     # Metrics + confusion matrix
│   └── backtest.py     # Naive long/short backtester
│
├── reports/            # Metrics & plots
└── backtest/           # PnL, Sharpe, equity curves
```

---

## 2. How the Pipeline Works

### 2.1 Data  
Order book snapshots are generated (or loaded) and transformed into:
```
100 timesteps × 15 features → 1 training sample
```

Each sample receives a label:
- **+1** → mid‑price rises  
- **0** → stays flat  
- **–1** → falls  

### 2.2 Models  
**DeepLOB** learns spatial patterns across bid/ask levels and their temporal evolution.  
**TCN** uses dilated convolutions to extract long-range temporal patterns more efficiently.

---

## 3. Run the Pipeline

### 3.1 Build Synthetic Data
```bash
PYTHONPATH=src python -m deep_lob.simulator   --out data/raw/sim_lob.csv   --n-rows 5000

PYTHONPATH=src python -m deep_lob.data   --csv data/raw/sim_lob.csv   --out data/processed/lob_windows.npz   --window-size 100   --horizon 10
```

### 3.2 Train DeepLOB
```bash
PYTHONPATH=src python -m deep_lob.train
```

### 3.3 Train TCN
```bash
PYTHONPATH=src python -m deep_lob.train_tcn
```

### 3.4 Evaluate
```bash
PYTHONPATH=src python -m deep_lob.evaluate   --data data/processed/lob_windows.npz   --model models/deeplob_synthetic.pt
```

### 3.5 Backtest
```bash
PYTHONPATH=src python -m deep_lob.backtest   --data data/processed/lob_windows.npz   --raw data/raw/sim_lob.csv   --model models/deeplob_synthetic.pt   --window-size 100   --horizon 10
```

---

## 4. Model Comparison — DeepLOB vs TCN

### Classification Performance

| Model | Accuracy | Macro F1 |
|-------|----------|----------|
| DeepLOB | 0.6263 | 0.6264 |
| **TCN** | **0.6444** | **0.6518** |

### Backtest Performance

| Metric | DeepLOB | TCN |
|--------|---------|------|
| Total PnL | 1.7438 | **2.1585** |
| Avg return/trade | 0.000357 | **0.000441** |
| Win rate | 39.3% | **52.8%** |
| Sharpe | 0.560 | **0.667** |
| Max Drawdown | 0.0095 | **0.0073** |

**TCN clearly extracts stronger and more stable microstructure patterns.**

---

## 5. Why This Matters

This repository forms a foundation for real quant research on:
- market microstructure learning  
- short‑horizon price forecasting  
- execution & liquidity studies  
- pattern discovery in order flow  

With real exchange LOB feeds, this becomes a powerful applied ML framework for intraday modelling.

---

## 6. Next Steps
- Transformer‑style architectures  
- Regime conditioning  
- Multi‑asset LOB ingestion  
- Execution‑aware losses  
- Online / streaming models  

---

A full overview is available in `docs/deeplob_overview.md`.
