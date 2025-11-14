# DeepLOB – Deep Learning on Limit Order Book Data

A full implementation of a DeepLOB-style architecture, including:
- Synthetic LOB simulator
- Sliding window tensorizer
- Dataset utilities
- CNN + Inception + LSTM model
- Training pipeline
- Experiment-ready structure

## 📁 Project Structure
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

## 🚀 Key Features
- Synthetic LOB generator for prototyping  
- Sliding window conversion to 3D tensors (T × features)  
- Clean PyTorch Dataset class  
- DeepLOB architecture (CNN + Inception + LSTM)  
- Training loop with metrics and checkpointing  
- Fully modular and research-friendly layout  

## 🔧 How to Run

### 1. Generate synthetic data
```
PYTHONPATH=src python -m deep_lob.simulator   --out data/raw/simulated_lob.csv   --n-rows 5000
```

### 2. Build LOB tensors
```
PYTHONPATH=src python -m deep_lob.data   --csv data/raw/simulated_lob.csv   --out data/processed/lob_windows.npz   --window-size 100   --horizon 10
```

### 3. Train the model
```
PYTHONPATH=src python -m deep_lob.train
```

## 📊 Example Output
Training on synthetic data yields:
- ~60% train accuracy
- ~44% validation accuracy
(Reasonable for noise-heavy synthetic data.)

## 📘 Documentation
See `docs/deeplob_overview.md` for the full academic-style explanation of LOB microstructure, tensorization, and the DeepLOB architecture.

## 🧭 Next Steps
- Add TCN or Transformer models  
- Backtesting engine  
- Real LOB dataset ingestion layer  
