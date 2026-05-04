# ✈️ U.S. Domestic Flight Delay Predictor
### A Binary Classification Machine Learning Model

**Author:** Bobby Bagley
**Date:** April 2026

---

## Overview

This project builds a binary classification model that predicts whether a U.S. domestic flight will depart with a delay of **≥ 15 minutes** using only information available at the time of booking — no post-flight data, no real-time weather, no air traffic information.

The 15-minute threshold is the FAA/DOT standard for a reportable departure delay. The core constraint of the project is strict: if a feature is only knowable after the plane pushes back, it is excluded.

This notebook is the **machine learning companion** to a parallel data visualization project ([`domestic_flight_delay_analysis`](https://github.com/BobbyBagley/domestic-flight-delay-analysis)) that analyzes the same dataset through exploratory visualization — surfacing temporal, geographic, and operational delay patterns that directly informed the feature engineering decisions made here.

---

## Results

| Metric | Validation | Test | Drift |
|---|---|---|---|
| **ROC-AUC** | 0.7073 | 0.7049 | −0.0024 |
| **PR-AUC** | 0.3636 | 0.3608 | −0.0028 |
| **Log Loss** | 0.6209 | 0.6225 | +0.0016 |

All three metrics held within **0.003** of their validation scores — confirming generalization without overfitting. PR-AUC of **0.3608** is approximately **double the random classifier baseline** (~0.18), indicating meaningful discriminative power on the minority (delayed) class despite a 4.49:1 class imbalance.

---

## Dataset

| | |
|---|---|
| **Source** | [Flight Delay & Cancellation Dataset 2019–2023](https://www.kaggle.com/datasets/patrickzel/flight-delay-and-cancellation-dataset-2019-2023) |
| **Origin** | U.S. Department of Transportation — Bureau of Transportation Statistics (BTS) |
| **Format** | CSV (`flights_sample_3m.csv`) |
| **Raw Size** | ~3 million rows, 32 columns |
| **Working Sample** | 500,000 rows (stratified on `IS_DELAYED`) |
| **Time Span** | January 2019 – December 2023 |
| **Class Split** | ~82% On-Time / ~18% Delayed (4.49:1 ratio) |

The dataset is loaded directly from Kaggle using the `kagglehub` library. A free Kaggle account and API key are all that is required — no CSV file needs to be attached or downloaded manually.

---

## Methodology

### Data Leakage Prevention

**22 columns were dropped** before any modeling. Every post-flight field is excluded:

- **Operational:** `DEP_TIME`, `TAXI_OUT`, `WHEELS_OFF`, `WHEELS_ON`, `TAXI_IN`, `ARR_TIME`, `ARR_DELAY`, `ELAPSED_TIME`, `AIR_TIME`
- **Delay causes:** `DELAY_DUE_CARRIER`, `DELAY_DUE_WEATHER`, `DELAY_DUE_NAS`, `DELAY_DUE_SECURITY`, `DELAY_DUE_LATE_AIRCRAFT`
- **Redundant/ID:** `AIRLINE_DOT`, `DOT_CODE`, `FL_NUMBER`, `ORIGIN_CITY`, `DEST_CITY`
- **Filtered flags:** `CANCELLED`, `CANCELLATION_CODE`, `DIVERTED`

### Feature Engineering

All features are derivable from pre-departure information:

| Feature | Type | Source |
|---|---|---|
| `HOUR_OF_DAY` | Numeric | `CRS_DEP_TIME` |
| `CRS_ELAPSED_TIME` | Numeric | Raw column |
| `MONTH` | Numeric | `FL_DATE` |
| `DAY_OF_WEEK` | Numeric | `FL_DATE` |
| `YEAR` | Numeric | `FL_DATE` |
| `AIRLINE` | Categorical | Raw column |
| `ORIGIN` | Categorical | Raw column |
| `DEST` | Categorical | Raw column |
| `SEASON` | Categorical | Engineered from `MONTH` |
| `TIME_BUCKET` | Categorical | Engineered from `HOUR_OF_DAY` |

**Dropped due to multicollinearity:** `CRS_DEP_TIME` (r=1.00 with `HOUR_OF_DAY`), `DISTANCE` (r=0.98 with `CRS_ELAPSED_TIME`), `CRS_ARR_TIME` (r=0.70 with `CRS_DEP_TIME`), `IS_WEEKEND` (r=0.78 with `DAY_OF_WEEK`, r=0.01 with target).

### Data Split

70% Train / 15% Validation / 15% Test — stratified on `IS_DELAYED`:

| Partition | Rows |
|---|---|
| Train | 350,000 |
| Validation | 75,000 |
| Test | 75,000 |

The test set was touched **exactly once**, after final model selection on the validation set.

### Preprocessing Pipeline

A `scikit-learn` `Pipeline` + `ColumnTransformer` ensures all transformers are fit on training data only:

- **Numeric:** `SimpleImputer(median)` → `StandardScaler`
- **Categorical:** `SimpleImputer(most_frequent)` → `OneHotEncoder(handle_unknown='ignore')`

### Models Trained

All three models were tuned under **identical conditions** using `RandomizedSearchCV` (`n_iter=5`, `cv=3`, `scoring='roc_auc'`) before comparison — ensuring observed performance differences reflect model architecture, not differential tuning effort.

| Model | Imbalance Handling | Notes |
|---|---|---|
| Logistic Regression | `class_weight='balanced'` | Interpretable linear baseline |
| Random Forest | `class_weight='balanced'` | Parallel ensemble, CPU-only |
| XGBoost | `scale_pos_weight=4.49` | Sequential boosting, GPU-accelerated |

### Evaluation Metrics

Three complementary metrics were used — each measuring a distinct dimension of classifier quality:

| Metric | What it measures | Random baseline |
|---|---|---|
| **ROC-AUC** | Ranking delayed vs. on-time flights across all thresholds | 0.50 |
| **PR-AUC** | Precision-recall trade-off on the delayed (minority) class | ~0.18 |
| **Log Loss** | Quality of probability calibration — lower is better | ~0.54 |

---

## Key Findings

**XGBoost wins on discriminative performance** (ROC-AUC, PR-AUC), while **Random Forest produces better-calibrated probabilities** (Log Loss: 0.5195 vs. XGBoost's 0.6209). In a production system where the raw probability score is surfaced to users, this distinction matters.

**`TIME_BUCKET_Morning` is the single most important feature** — nearly 4× more important than the second-ranked feature. Morning departures are the daily reset point: a clean morning departure gives the rest of the day a chance; a delayed one begins the domino effect. This is consistent with the companion visualization study, which documented a near-linear climb in delay rate from 7.3% at 5AM to 28.3% at 9PM.

**Airline identity matters.** JetBlue (B6), Allegiant (G4), SkyWest (OO), Frontier (F9), and Endeavor Air (9E) all appear in the top 20 feature importances — consistent with the worst-performers identified independently in the visualization project.

---

## Compute Environment

| | Colab Free Tier | Local Machine (used) |
|---|---|---|
| CPU cores | 2 vCPUs | 16 logical processors |
| RAM | ~12GB | 32GB |
| GPU | T4 | NVIDIA RTX 3060 (16GB) |
| LR tuning time | 30+ min (killed) | ~21 minutes |
| XGBoost tuning | - | ~2 minutes (GPU) |

Logistic Regression training in the free Colab environment ran for over 30 minutes before the session was interrupted. Switching to a local machine with 16 processors and an RTX 3060 GPU reduced total tuning time across all three models to approximately 58 minutes.

---

## Getting Started

### Prerequisites

- Python 3.10+
- A free [Kaggle account](https://www.kaggle.com) and API key
- An NVIDIA GPU with CUDA support (optional — required only for XGBoost GPU acceleration)

### Installation

```bash
pip install -r requirements.txt
```

### Running the Notebook

1. Open `flight_delay_predictor_revised.ipynb` in VS Code or Jupyter
2. Run the environment setup cell
3. When prompted by `kagglehub.login()`, enter your Kaggle username and API key
4. Run all remaining cells in order

> **Note:** The dataset downloads automatically to a local cache directory via `kagglehub`. No manual file handling required.

> **GPU note:** If you do not have a CUDA-compatible GPU, change `device='cuda'` to `device='cpu'` in the XGBoost cell. Training will be slower but functionally identical.

---

## Repository Structure

```
├── flight_delay_predictor_revised.ipynb   # Main modeling notebook
├── requirements.txt                       # Python dependencies
├── ml_icml.tex                            # ICML-format project report (LaTeX)
├── references_ml.bib                      # BibTeX references
└── README.md                              # This file
```

---

## Libraries Used

| Library | Role |
|---|---|
| `pandas` | Data loading, cleaning, feature engineering |
| `numpy` | Numerical operations |
| `scikit-learn` | Pipeline, preprocessing, LR, RF, metrics, search |
| `xgboost` | Gradient boosting classifier |
| `matplotlib` / `seaborn` | EDA visualizations |
| `kagglehub` | Dataset acquisition from Kaggle |
| `torch` | CUDA availability check and VRAM clearing |

---

## Known Issues & Limitations

- **No weather covariates.** Real-time meteorological data is the strongest missing predictor. `DELAY_DUE_WEATHER` was excluded as post-flight leakage.
- **Random Forest CV nan.** At least one CV fold produced an undefined AUC, likely due to insufficient positive class examples in that fold. Validation AUC (computed on 75,000 rows) is unaffected and reliable.
- **COVID-19 structural break.** 2020 flight volumes fell by over 60% — including this period alongside 2021–2023 introduces non-stationarity.
- **Static threshold.** All metrics are reported at the default 0.5 threshold. A formal cost-sensitive threshold analysis was not conducted.
- **Standard KFold CV.** `RandomizedSearchCV` does not respect temporal ordering. `TimeSeriesSplit` would be more methodologically appropriate.

---

## Related Project

This notebook is the **machine learning companion** to a parallel data visualization project that performs exploratory analysis on the same dataset using matplotlib, seaborn, Altair, and Plotly.

See: [`domestic_flight_delay_analysis`](https://github.com/BobbyBagley/domestic-flight-delay-analysis)
