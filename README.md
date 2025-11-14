# DeepLOB — Deep Learning on Limit Order Book Data

This project implements a full research-grade DeepLOB pipeline:
- Synthetic LOB generator  
- Sliding-window tensor builder  
- PyTorch dataset + dataloader  
- DeepLOB CNN–Inception–LSTM model  
- Training loop with accuracy and loss tracking  
- Saved model weights  

## 📌 Folder Structure
```
deep-lob/
│
├── data/
│   ├── raw/            # Simulated or real LOB CSVs
│   └── processed/      # NPZ sliding windows (X, y)
│
├── src/deep_lob/
│   ├── simulator.py    # Synthetic LOB generator
│   ├── data.py         # Tensor builder
│   ├── dataset.py      # PyTorch dataset
│   ├── models.py       # DeepLOB model
│   └── train.py        # Training loop
│
└── models/             # Saved .pt weights
```

## 🚀 Training Output Example
```
Epoch 01 | train_loss=1.0252 | train_acc=0.434 | val_acc=0.439
Epoch 05 | train_loss=0.9234 | train_acc=0.597 | val_acc=0.439
Saved model weights to models/deeplob_synthetic.pt
```

## 🔧 How to Run
```bash
# 1) Build data
PYTHONPATH=src python -m deep_lob.simulator --out data/raw/lob.csv --n-rows 5000
PYTHONPATH=src python -m deep_lob.data --csv data/raw/lob.csv --out data/processed/lob_windows.npz --window-size 100 --horizon 10

# 2) Train model
PYTHONPATH=src python -m deep_lob.train
```

## 📘 DeepLOB Overview

See the detailed technical overview here:  
[../docs/deeplob_overview.md](../docs/deeplob_overview.md)

## 📊 Model Comparison — DeepLOB vs TCN (Synthetic LOB Data)

This project evaluates two architectures on identical synthetic limit order book data:

- **DeepLOB** (CNN + Inception + LSTM)
- **TCN** (Temporal Convolutional Network)

Both are trained using the same dataset, tensorization, and backtest engine.

---

### 1. Classification Performance

| Model | Val Accuracy | Macro F1 |
|-------|--------------|----------|
| **DeepLOB** | 0.6263 | 0.6264 |
| **TCN** | **0.6444** | **0.6518** |

**TCN improves generalization and temporal signal extraction.**

**Per-class metrics (TCN):**

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
| 0 (down) | 0.731 | 0.653 | 0.690 | 1581 |
| 1 (flat) | 0.563 | 0.576 | 0.569 | 1899 |
| 2 (up) | 0.669 | 0.726 | 0.696 | 1411 |

---

### 2. Backtest Performance (Long/Short Strategy)

| Metric | DeepLOB | TCN |
|--------|---------|------|
| **Total PnL** | 1.7438 | **2.1585** |
| **Avg return/trade** | 0.000357 | **0.000441** |
| **Win rate** | 39.3% | **52.8%** |
| **Sharpe** | 0.560 | **0.667** |
| **Max Drawdown** | 0.0095 | **0.0073** |

**TCN produces a materially better trading signal.**

---

### 3. Interpretation

Even though DeepLOB is a strong LOB architecture,  
**TCNs extract more actionable microstructure patterns** — especially short-term temporal dependencies.

This mirrors modern quant research, where TCNs often outperform recurrent models in noisy, high-frequency environments.

---

### 4. Files

- `reports/metrics_deeplob.json`  
- `reports/metrics_tcn.json`  
- `backtest/stats_deeplob.json`  
- `backtest/stats_tcn.json`  

You can recreate this comparison via:

```bash
PYTHONPATH=src python -m deep_lob.train
PYTHONPATH=src python -m deep_lob.train_tcn
PYTHONPATH=src python -m deep_lob.evaluate --model <model_path>
PYTHONPATH=src python -m deep_lob.backtest --model <model_path>
