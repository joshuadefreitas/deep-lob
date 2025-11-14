# DeepLOB — Technical Overview

This document explains the DeepLOB system end‑to‑end: what the data represents, how it is transformed, how the models learn microstructure patterns, and how the evaluation/backtesting framework works.

## 1. What Is a Limit Order Book?

A limit order book (LOB) is the real‑time state of buyers and sellers in an electronic market.  
For each *price level*, we see:

- **Bid price** (buyers)
- **Bid size** (how many want to buy)
- **Ask price** (sellers)
- **Ask size** (how many want to sell)

The closest bid/ask levels form the *top of book*, further levels show deeper liquidity.

A single LOB snapshot is like a photograph of the market’s supply/demand.

## 2. What We Try to Predict

We do **short‑horizon mid‑price movement prediction**:

- +1 → price goes **up**
- 0 → price stays **flat**
- –1 → price goes **down**

This is the simplest possible framing for directional signal extraction.

## 3. Sliding‑Window Tensorization

Instead of giving a model a single LOB snapshot, we give it **a sequence of snapshots**.

A training sample looks like:

```
100 timesteps × 15 features
```

Why?

Because markets are temporal — the pattern in how the LOB *changes* is more informative than any single picture.

Example features:

- mid‑price
- bid prices/sizes (levels 1–3)
- ask prices/sizes (levels 1–3)
- micro price signals

These are extracted into a 3D tensor:  
**(window_length, features)**.

## 4. DeepLOB Architecture (CNN → Inception → LSTM)

DeepLOB was designed to capture *local spatial structure* across LOB levels and *temporal evolution* over time.

### 4.1 CNN Block
Extracts low‑level spatial features:  
e.g., “bid imbalance at levels 1–3”.

### 4.2 Inception Block
Multiple convolution filters in parallel produce:

- short‑range patterns
- medium‑range patterns
- longer‑range dependencies

This allows flexible pattern extraction.

### 4.3 LSTM Block
Learns temporal sequences of microstructure changes.

Together, this pipeline can detect subtle liquidity signals.

## 5. TCN Architecture

The Temporal Convolutional Network is simpler:

- uses 1D convolutions
- uses *dilated* filters to see far into the past
- captures long‑range patterns without recursion (faster to train)

In noisy environments like LOBs, TCNs are often more stable.

## 6. Training Procedure

We train both models using:

- Adam optimizer
- Cross‑entropy loss
- 80/20 train/validation split
- batched DataLoaders

Outputs tracked:

- train accuracy  
- validation accuracy  
- learning curves  
- confusion matrix

## 7. Evaluation

Evaluation computes:

- Accuracy  
- Macro F1  
- Per‑class precision/recall  
- Confusion matrix  

This allows detailed inspection of how well the model captures up/down/flat signals.

## 8. Backtesting

We translate predictions into a naïve trading strategy:

- +1 → long  
- 0 → flat  
- –1 → short  

Metrics:

- total PnL  
- average return per trade  
- win rate  
- Sharpe ratio  
- max drawdown  

This gives practical insight into whether the predicted signal is *tradeable*.

## 9. Interpretation of Results

On synthetic data:

- TCN outperforms DeepLOB in both classification and backtest
- TCN likely captures long‑range temporal dependencies better
- Synthetic data is highly noisy → useful for testing robustness

These results will evolve with real-market datasets, where structural patterns appear more clearly.

## 10. Future Extensions

Planned improvements:

- Transformer‑style encoders  
- Multi‑asset LOB ingestion  
- Execution‑aware labels  
- Regime‑aware models  
- Online (streaming) training  
- Real exchange LOB dataset support  

This document will expand as the project grows.
