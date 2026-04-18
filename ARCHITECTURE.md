# System Architecture

## 1. High-Level Overview
The AI Personal Finance Tracker is currently designed as a monolithic data application (Version 1.0). It processes user financial data, stores it locally, and runs advanced Machine Learning pipelines to generate predictive forecasting and anomaly detection. 

The architecture is built prioritizing mathematical accuracy and UI responsiveness, separating the heavy Machine Learning dataframes from the front-end rendering logic.

## 2. Technology Stack
* **Frontend / UI:** Streamlit
* **Data Processing:** Pandas, NumPy
* **Machine Learning:** * `Prophet` (Time-series forecasting)
  * `scikit-learn` / Isolation Forest (Anomaly detection)
* **Database:** SQLite (Local)
* **Visualization:** Plotly

## 3. Directory Structure
```text
personal-finance-tracker/
│
├── dashboard.py         # Main entry point for the Streamlit UI
├── requirements.txt     # Dependency management
├── README.md            # Project documentation and setup
├── ARCHITECTURE.md      # System design and data flow details
│
├── api/                 # (V2.0) FastAPI routing and endpoints
├── core/                # Core business logic and configurations
├── ml/                  # Machine learning models and training scripts
├── analysis/            # Data analysis and processing modules
│
├── data/                # Local database storage (Git-ignored)
│   └── expenses.db      # SQLite database file
│
└── scripts/             # Standalone database and utility scripts
    ├── seed_db.py       # Generates synthetic load-testing data
    ├── migrate_db.py    # Database schema migrations
    └── import_csv.py    # Legacy data ingestion