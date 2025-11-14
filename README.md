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
