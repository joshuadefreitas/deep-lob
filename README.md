<p align="center">
  <img src="../Banner.png" width="80%" alt="DeepLOB — Market Microstructure & Deep Learning"/>
</p>

# 📉 DeepLOB  
### Deep Learning on Limit Order Book Data

A research-grade implementation of a DeepLOB-style architecture for predicting short-horizon mid-price movements using limit order book (LOB) data.  
Combines market microstructure, tensorization, convolutional architectures, and temporal modeling into a modular ML system.

---

# 🚀 What’s Inside

### ✔️ Synthetic LOB Data Generator  
Produces realistic multi-level bid/ask snapshots.

### ✔️ Sliding-Window Tensorizer  
Converts raw snapshots into `(samples × window_size × features)` tensors.

### ✔️ PyTorch Dataset  
Clean dataset abstraction for training deep learning models.

### ✔️ DeepLOB Architecture  
CNN + Inception blocks + LSTM sequence modeling.

### ✔️ Full Training Pipeline  
Metrics, batching, validation split, and model checkpointing.

---

# 📁 Project Structure

```
deep-lob/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
├── src/deep_lob/
│   ├── simulator.py
│   ├── data.py
│   ├── dataset.py
│   ├── models.py
│   ├── train.py
│   └── config.py
├── models/
├── experiments/
└── tests/
```

---

# 🔧 How to Use

## 1. Generate synthetic LOB data
```bash
PYTHONPATH=src python -m deep_lob.simulator \
  --out data/raw/simulated_lob.csv \
  --n-rows 5000
```

## 2. Build sliding-window tensors
```bash
PYTHONPATH=src python -m deep_lob.data \
  --csv data/raw/simulated_lob.csv \
  --out data/processed/lob_windows.npz \
  --window-size 100 \
  --horizon 10
```

## 3. Train the model
```bash
PYTHONPATH=src python -m deep_lob.train
```

---

# 📊 Example Results (Synthetic)

- ~60% train accuracy  
- ~44% validation accuracy  

(Synthetic data contains noise — performance will improve with real LOB data.)

---

# 📘 Documentation

👉 **[Detailed Technical Overview](../docs/deeplob_overview.md)**

---

# 🧭 Next Steps

- Transformer / TCN models  
- Backtesting engine  
- Real LOB ingestion  
- Statistical microstructure features  
- Hyperparameter search (Optuna)

---

<p align="center">
  <span style="color:#6b7280;">
    Built for precision, research clarity, and long-term scalability.
  </span>
</p>
