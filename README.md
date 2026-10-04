# 🏪 RetailPulse: Smart Retail Data Science & Automated Restock Platform
> **An End-to-End Pure Data Science & Operations Research System for Hypermarket & Supermarket Intelligence (DMart / Walmart Style Operations)**

[![Python 3.14](https://img.shields.io/badge/python-3.14-blue.svg)](https://www.python.org/downloads/)
[![Flask 3.1.3](https://img.shields.io/badge/framework-Flask%203.1.3-green.svg)](https://flask.palletsprojects.com/)
[![SQLite WAL](https://img.shields.io/badge/database-SQLite%20(WAL%20Mode)-blue.svg)](https://www.sqlite.org/)
[![Tests](https://img.shields.io/badge/tests-9%2F9%20passed%20(100%25)-brightgreen.svg)](test_pipeline.py)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📌 1. Project Overview

In high-volume hypermarkets (e.g., DMart, Walmart, Reliance Smart), managing store inventory through static heuristics leads to two major financial pitfalls: **stock-outs of high-velocity staple goods** during peak evening rush hours (causing immediate lost revenue and customer attrition) and **over-ordering slow-moving deadstock** (locking up working capital and inflating annual holding costs by $18\text{--}22\%$).

**RetailPulse** is an end-to-end, **100% pure Data Science and Operations Research platform** that ingests daily Point-of-Sale (POS) cash register billing history and performs real-time data cleaning, stochastic inventory optimization, association rule mining, demand forecasting, customer loyalty clustering, and cashier staffing optimization.

```
+───────────────────────────────────────────────────────────────────────────────────────────────────+
|                                    RETAILPULSE PIPELINE FLOW                                      |
+───────────────────────────────────────────────────────────────────────────────────────────────────+
  Daily POS CSV / Stream  ──►  Real-Time Preprocessing  ──►  SQLite Relational Data Warehouse (WAL)
                               (IQR Outliers, Features)      (pos_transactions, inventory_levels)
                                                                    │
       ┌───────────────────────────────┬────────────────────────────┴───────────────────────────────┐
       ▼                               ▼                                                           ▼
 [ Operations Research ]      [ Market Basket Mining ]                                   [ Time-Series & ML ]
 • Safety Stock: Z·σ_d·√L     • Apriori Itemsets (mlxtend)                               • Holt-Winters Forecast
 • Reorder Point: d̄·L + SS    • Support, Confidence, Lift (4.76x)                        • 90% Prediction Bands
 • Wilson EOQ: √(2DS/H)       • Physical Shelf Planograms                                • RFM K-Means (k=4)
 • 9-Cell ABC-XYZ Matrix      • End-Cap Promo Bundles                                    • Poisson Cashier Model
       │                               │                                                           │
       └───────────────────────────────┼───────────────────────────────────────────────────────────┘
                                       ▼
                       [ Responsive Worker Terminal UI ]
                • 8 Operational Tabs  • Real-Time Live Ticker
                • 1-Click Automated Restock Purchase Order (CSV Export)
```

---

## 🔬 2. Core Mathematical & Statistical Foundations

| Module | Mathematical Formulation | Description & Operational Purpose |
| :--- | :--- | :--- |
| **Safety Stock ($SS$)** | $$SS = Z_{1-\alpha} \times \sigma_d \times \sqrt{L}$$ | Protects against daily demand variance ($\sigma_d$) with a $95\%$ service level ($Z_{0.95} = 1.645$) across supplier lead time ($L$). |
| **Reorder Point ($ROP$)** | $$ROP = (\bar{d} \times L) + SS$$ | The exact stock threshold that triggers an automated restocking order. |
| **Economic Order Quantity ($EOQ$)** | $$EOQ = \sqrt{\frac{2 \times D \times S}{H}}$$ | Wilson's classical formula minimizing annual ordering cost ($S$) and holding cost ($H$). |
| **ABC-XYZ Matrix** | Pareto $80/15/5\%$ Revenue $\times$ Coefficient of Variation ($CV = \frac{\sigma_d}{\bar{d}}$) | 9-cell segmentation distinguishing high-value predictable SKUs from volatile deadstock. |
| **Apriori Association Rules** | $$\text{Lift}(A \rightarrow B) = \frac{P(A \cap B)}{P(A) \times P(B)}$$ | Identifies high-affinity bundles (e.g., Bread $\rightarrow$ Butter at **$4.76\text{x}$ Lift**) for store shelf planograms. |
| **Holt-Winters Forecasting** | $$\hat{y}_{t+h\mid t} = \ell_t + h b_t + s_{t+h-m(k+1)}$$ | Triple Exponential Smoothing modeling Level ($\ell_t$), Trend ($b_t$), and 7-day weekly Seasonality ($s_t$) with $90\%$ confidence bands. |
| **RFM Customer Clustering** | $\log(x+1)$ transform $\rightarrow$ `StandardScaler` $\rightarrow$ `KMeans(k=4)` | Unsupervised customer loyalty segmentation into Champions, Regulars, At-Risk, and Value Hunters ($s = 0.311$). |
| **Queueing Staffing Model** | $$s_t = \max\left(1, \left\lceil \frac{\lambda_t \times T_{\text{service}}}{60 \times \rho_{\text{target}}} \right\rceil\right)$$ | Multi-server Poisson arrival model ($\lambda_t$) determining active cashier desks to prevent checkout bottleneck queues. |

---

## 📁 3. Repository Directory Structure

```text
retail-pulse-analytics/
├── data/                                         # Datasets & Relational Database
│   ├── sample_dmart_transactions.csv             # 11,035 POS billing line items (30 days)
│   ├── inventory_levels.csv                      # Warehouse stock, costs & supplier catalog
│   ├── product_catalog.csv                       # Master SKU metadata across 7 departments
│   └── retail_store.db                           # SQLite database in WAL concurrency mode
├── docs/                                         # Documentation & Presentation Assets
│   └── images/                                   # High-resolution (300 DPI) report figures (fig1 to fig12)
├── notebooks/
│   └── retail_datascience_complete_pipeline.ipynb # Interactive Jupyter Notebook Walkthrough
├── src/                                          # Modular Data Science & OR Engines
│   ├── __init__.py
│   ├── database.py                               # SQLite relational schema, WAL pooling & seeding
│   ├── data_processor.py                         # POS schema validation & feature derivation
│   ├── realtime_pipeline.py                      # Real-time ETL, IQR filtering & atomic stock decrements
│   ├── inventory_math.py                         # Safety Stock, ROP, Wilson EOQ, ABC-XYZ & PO Builder
│   ├── trending_engine.py                        # 7-Day Moving Averages & Z-Score Surge Detection
│   ├── market_basket.py                          # Apriori Association Mining & Shelf Planograms
│   ├── forecasting.py                            # Holt-Winters Triple Exponential Smoothing
│   ├── rfm_analytics.py                          # Customer RFM feature engineering & K-Means
│   └── queue_staffing.py                         # Poisson arrival modeling & cashier desk roster
├── static/                                       # Responsive UI Assets
│   ├── css/style.css                             # Custom modern CSS design system
│   └── js/app.js                                 # Asynchronous Chart.js client controller
├── templates/
│   └── index.html                                # 8-Tab Store Worker Dashboard single-page app
├── server.py                                     # Flask REST API backend server (Port 5001)
├── app.py                                        # Application entrypoint
├── test_pipeline.py                              # Automated test suite (9/9 unit tests passing)
├── generate_report_figures.py                    # Script to regenerate all 12 publication figures
├── sample_test_upload_pos.csv                    # 44-item sample POS bill for live upload testing
├── requirements.txt                              # Python library dependencies
├── .gitignore                                    # Standard Git ignore file
└── README.md                                     # Project master documentation
```

---

## ⚡ 4. Quickstart & Installation

### Prerequisites
* Python 3.9 or higher (Tested on **Python 3.14**)
* Modern web browser (Chrome, Edge, Firefox, Safari)

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/retail-pulse-analytics.git
cd retail-pulse-analytics
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Server
```bash
python server.py
```
Open your browser and navigate to:  
👉 **`http://localhost:5001`** (or `http://127.0.0.1:5001`)

---

## 🧪 5. Automated Testing & Verification

RetailPulse includes a comprehensive automated test suite verifying all 9 core subsystems (database concurrency, atomic stock decrements, inventory operations research, market basket mining, time-series forecasting, clustering, and REST endpoints).

Run the automated test suite:
```bash
python test_pipeline.py
```

### Test Output:
```text
----------------------------------------------------------------------
Ran 9 tests in 5.400s

OK
[SQL DB] Seeded 33 products into product_catalog table.
[SQL DB] Seeded 33 inventory records into inventory_levels table.
[SQL DB] Seeded 11035 billing transactions into pos_transactions table.
[PASS] SQLite DB test: Rows = 11035 Size = 2760.0 KB
[PASS] Realtime Preprocessing & Atomic Stock Deduction test passed: Stock 43 -> 41
[PASS] Trending Engine test: Analyzed 33 SKUs.
[PASS] Inventory Math test: Generated PO with 5 reorder line items.
[PASS] Market Basket test: Extracted 11 association rules. Top rule lift: 4.76x
[PASS] Forecasting test: 7-day projected demand = 3731 units.
[PASS] RFM Segmentation test: Segmented 450 customers. Silhouette score: 0.309
[PASS] Queue Staffing test: Computed 24-hour staffing schedule.
[PASS] Flask REST API tests: All 9 endpoints + Live Bill Simulation responded 200 OK.
```

---

## 🌟 6. Interactive Web Dashboard Features

1. **⚡ Store Worker Restock Hub:** Real-time stock runway tracking, critical stockout warnings (<24h), and **1-Click Restock Purchase Order (CSV)** download.
2. **🔥 Trending & Product Velocity:** Top 10 high-velocity charts, 7-day moving averages, and $Z$-score surge anomaly detection.
3. **📊 9-Cell ABC-XYZ Policy Simulator:** Interactive Cycle Service Level slider ($85\%\text{ to }99\%$) dynamically recalculating safety buffers.
4. **🛒 Market Basket & Planograms:** Apriori association cards displaying Support, Confidence, Lift, and physical aisle placement guidance.
5. **🔮 Holt-Winters Demand Forecasting:** 7-day projected demand curves with shaded $90\%$ prediction intervals.
6. **👥 Customer RFM Clusters:** 2D scatter visualization of spend vs. frequency with actionable customer retention cards.
7. **⏰ Peak Hours & Cashier Staffing:** Hourly customer footfall heatmap and recommended active checkout counter allocation.
8. **🗄️ SQLite DB Inspector & Real-Time Ticker:** Live database statistics (11,000+ rows, table schemas, live preprocessing transaction stream).

---

## 📄 7. License & Acknowledgements

* **License:** Distributed under the MIT License. See `LICENSE` for more information.
* **Academic Course:** Foundations of Data Science (PBL Component), Department of Artificial Intelligence & Machine Learning, Chennai Institute of Technology.
