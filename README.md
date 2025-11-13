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
- **Analytics engineering** – SQL, ELT pipelines, semantic models, dashboards  
- **Applied AI systems** – LLM-powered workflows, APIs, automation around data  

I like projects that go **end-to-end**: from raw data → modeling → APIs → something real people can use.

---

## 🛠️ Tech Stack

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/R-276DC3?style=for-the-badge&logo=r&logoColor=white" />
  <img src="https://img.shields.io/badge/SQL-336791?style=for-the-badge&logo=postgresql&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/Time%20Series-00A8E8?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Data%20Pipelines-555555?style=for-the-badge" />
</p>

---

## 📂 Featured Projects

---

### 📈 1. Time Series Forecasting — Remittances to Mexico

**Goal:** Forecast remittance flows using classical time series techniques.  
**Tools:** R, `forecast`, `stats`, `ggplot2`  
**Topics:** SARIMA, differencing, seasonality, diagnostics

- Exploratory analysis of trend, seasonality, and volatility  
- Model selection via ACF/PACF & information criteria  
- SARIMA-based forecasting with residual checks  

🔗 **Project folder:** [`forecasting-remittances`](./forecasting-remittances)  
📄 `code/` – RMarkdown with full analysis  
📊 `report/` – PDF write-up of methods and findings  

---

### 🧾 2. AI Document Intelligence Platform (Invoices)

**Goal:** Build the backbone of an **AI-ready document platform** for finance teams — starting with invoices.

**What it does now:**

- Generates **synthetic invoice data** and loads it into a structured database  
- Normalises data into **vendors / invoices / amounts / dates / status**  
- Defines an initial **API spec** for querying and AI workflows  

**Stack:**

- Python, SQLite (swappable to Postgres), SQLAlchemy, pandas  
- FastAPI skeleton & API v1 spec (`docs/api_v1_spec.md`)  

🔗 **Project folder:** [`ai-document-platform`](./ai-document-platform)  

Designed as a foundation for:

- LLM-powered invoice Q&A  
- Automated approval chains  
- Spend analytics over vendors, categories, time  

---

### 📘 3. DeepLOB — Deep Learning on Limit Order Book Data

**Goal:** Recreate a **DeepLOB-style** pipeline that predicts short-horizon mid-price moves from LOB data.

**Includes:**

- **LOB simulator** → synthetic bid/ask levels and sizes  
- **Window builder** → sliding windows into (T × features) tensors  
- **PyTorch pipeline**:
  - `DeepLOBDataset`  
  - `DeepLOBModel` (CNN + Inception + LSTM)  
  - `train.py` with train/validation split and checkpointing  

🔗 **Project folder:** [`deep-lob`](./deep-lob)  

Showcases:

- Market microstructure modeling  
- Deep learning architecture implementation  
- Full research-style training pipeline  

---

### 🧠 4. Neural Network for Authorship Classification *(coming soon)*

**Goal:** Predict which author wrote a text sample using neural networks.

**Planned stack:**

- Python, PyTorch  
- Tokenization & embeddings (e.g. Word2Vec / Transformer embeddings)  
- Text classification network  

**Planned highlights:**

- Dataset of multiple authors with distinct styles  
- Experiments comparing classical features vs deep embeddings  
- Interpretability of which phrases/styles matter most  

---

### 📉 5. Quant Factor Research Engine *(coming soon)*

**Goal:** Build a small **research platform** for testing equity/futures factors.

**Planned stack:**

- Python, pandas, NumPy  
- SQL / DuckDB for local analytics  
- Simple backtesting module  

**Ideas:**

- Cross-sectional value, momentum, and quality factors  
- Factor portfolio construction & performance plots  
- Risk/return stats and factor correlation analysis  

---

### 🤖 6. ML-Powered Trading Simulator *(coming soon)*

**Goal:** Combine a market simulator with ML-based decision policies.

**Planned components:**

- Simple **order book or price process** simulator  
- Policy model (e.g. RL agent or supervised signal)  
- Evaluation: PnL, drawdown, Sharpe, turnover  

Long-term idea: plug in DeepLOB signals and see how they behave in a toy execution setting.

---

## 🌱 Roadmap

- Add more **notebooks** explaining methodology  
- Add **evaluation dashboards** (metrics, confusion matrices, backtests)  
- Expand **docs/** for each project (technical + executive summaries)  

---

## 📫 Contact

If you’re interested in the technical details or collaboration:

- GitHub: **[@joshuadefreitas](https://github.com/joshuadefreitas)**  

Always open to conversations at the intersection of **data, finance, and AI systems**.
