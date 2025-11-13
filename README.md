<p align="center">
  <img src="./Banner.png" width="85%" alt="Joshua de Freitas — Data & AI Portfolio Banner"/>
</p>

<br>

<div align="center">

# Joshua de Freitas  
### Data Scientist · Analytics Engineer · AI Workflow Builder  

Crafting data & AI solutions across **time series**, **financial microstructure**, and **document intelligence**.

</div>

---

## 🧠 What I Work On

- **Statistical & ML modeling** – time series, classification, forecasting  
- **Analytics engineering** – SQL, ELT pipelines, semantic models, Power BI  
- **Applied AI systems** – LLM-powered workflows, APIs, automation around data  

I like projects that go **end-to-end**: from raw data → modeling → APIs → something real people can use.

---

## 🛠️ Tech Stack

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/R-276DC3?style=for-the-badge&logo=r&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/SQL-336791?style=for-the-badge&logo=postgresql&logoColor=white" />
  <img src="https://img.shields.io/badge/Time%20Series-00A8E8?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Data%20Pipelines-555555?style=for-the-badge" />
</p>

---

## 📂 Featured Projects

---

### 📈 1. Time Series Forecasting — Remittances to Mexico

**Goal:** Forecast remittance flows using classical time series techniques.  
**Keywords:** SARIMA, stationarity, seasonality, forecasting horizon, diagnostics  

**Highlights:**

- Exploratory analysis of remittance series (trend, seasonality, volatility)
- Model selection via ACF/PACF and information criteria
- SARIMA-based forecasting with residual diagnostics

🔗 **Project folder:** [`forecasting-remittances`](./forecasting-remittances)  
📄 `code/` – RMarkdown with full analysis & code  
📊 `report/` – PDF report of methods and findings  

---

### 🧾 2. AI Document Intelligence Platform (Invoices)

**Goal:** Build the backbone of an **AI-ready document platform** for finance teams — starting with invoices.

**What it does now:**

- Generates **synthetic invoice data** and loads it into a structured database
- Creates **clean analytical tables** (vendors, invoices, amounts, dates, status)
- Defines an initial **API spec** for querying invoices and building AI workflows on top

**Stack:**

- **Backend & Data:** Python, SQLite (swappable to Postgres), SQLAlchemy, pandas  
- **Structure:**
  - `data/sample_invoices.csv` – synthetic invoice dataset
  - `backend/pipeline/` – ETL pipeline to load & normalize invoices
  - `backend/api/` – FastAPI app skeleton (v1 spec in `docs/api_v1_spec.md`)  

🔗 **Project folder:** [`ai-document-platform`](./ai-document-platform)  

This project is designed as a foundation for:
- LLM-powered invoice Q&A  
- Automated approval flows  
- Spend analytics over invoices, vendors, and categories  

---

### 📘 3. DeepLOB — Deep Learning on Limit Order Book Data

**Goal:** Recreate a **DeepLOB-style** pipeline that predicts short-horizon price moves from limit order book (LOB) data.

**What it includes:**

- **LOB Simulator:**  
  Synthetic limit order book generator with mid-price, bid/ask levels and sizes  
  → `src/deep_lob/simulator.py`

- **Windowed Tensor Builder:**  
  Converts raw LOB snapshots into sliding windows  
  - Input: `data/raw/simulated_lob.csv`  
  - Output: `data/processed/lob_windows.npz`  
  → `src/deep_lob/data.py`

- **PyTorch Pipeline:**  
  - `DeepLOBDataset` → wraps precomputed windows & labels  
  - `DeepLOBModel` →  
    - Inception-style 1D convolutions across time  
    - LSTM on top of learned features  
    - 3-way classifier: up / flat / down  
  - `train.py` → full training loop with train/validation split, accuracy tracking, model checkpointing (`models/deeplob_synthetic.pt`)

**Quick structure:**

```bash
deep-lob/
├── data/
│   ├── raw/          # simulated LOB
│   └── processed/    # sliding-window tensors (X, y)
├── src/
│   └── deep_lob/
│       ├── simulator.py   # synthetic LOB generator
│       ├── data.py        # window builder
│       ├── dataset.py     # PyTorch Dataset
│       ├── models.py      # DeepLOB-style model
│       └── train.py       # training loop
└── models/
    └── deeplob_synthetic.pt
