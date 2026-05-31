<div align="center">

# Tesla EV Deliveries & Production Analysis

### End-to-End Machine Learning Pipeline (2015-2025)

![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data_Analysis-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML_Models-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Boosting-189FDD?style=for-the-badge)
![Statsmodels](https://img.shields.io/badge/Statsmodels-Time_Series-4B8BBE?style=for-the-badge)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-11557C?style=for-the-badge)
![Seaborn](https://img.shields.io/badge/Seaborn-Statistical_Plots-76B947?style=for-the-badge)

<br>

[![Open in nbviewer](https://img.shields.io/badge/View_Notebook-nbviewer-orange?style=for-the-badge&logo=jupyter&logoColor=white)](https://nbviewer.org/github/sckandsharma/CEIP-Internship/blob/main/Week_2_Classicalmachine_learning/Sckand_Week2_assignment.ipynb)

<br>

> _"Can we predict how many Tesla vehicles will be delivered next month — and next year?"_

This project answers that question using **real-world Tesla data**, covering everything from raw data exploration to SARIMAX time series forecasting.

<br>

[View Notebook](#project-structure) · [Results](#key-results) · [Learnings](#what-i-learned)

</div>

---

## Project Overview

| Detail | Description |
|:------:|:------------|
| **Goal** | Predict monthly regional deliveries of Tesla EVs |
| **Dataset** | Tesla Deliveries & Production Data (2015-2025) from Kaggle |
| **Size** | 2,640 records x 12 features |
| **Approach** | Full ML lifecycle — EDA, Feature Engineering, Regression, Time Series |
| **Author** | Sckand Sharma |

---

## Pipeline at a Glance

```
Load Data
 |
 v
Explore — shape, dtypes, describe, head
 |
 v
EDA — KDE plots, count plots, box plots, heatmap, scatter plots
 |
 v
Feature Engineering — cyclical encoding, lag features, rolling mean, VIF
 |
 v
Preprocessing — ColumnTransformer, StandardScaler, OneHotEncoder, chronological split
 |
 v
Model Training — 9 regression models compared
 |
 v
Hyperparameter Tuning — GridSearchCV on Random Forest & XGBoost
 |
 v
Evaluation — Actual vs Predicted, Residuals Analysis
 |
 v
Time Series — ADF test, SARIMAX(1,1,1)(1,1,1,12), 2026 forecast
 |
 v
Conclusion & Insights
```

---

## Models Compared

| # | Model | Type |
|:-:|:------|:-----|
| 1 | Linear Regression | Linear |
| 2 | Ridge Regression | Linear (L2 regularization) |
| 3 | Lasso Regression | Linear (L1 regularization) |
| 4 | K-Neighbors Regressor | Instance-based |
| 5 | Decision Tree | Tree-based |
| 6 | **Random Forest** * | Ensemble (Bagging) |
| 7 | **XGBoost** * | Ensemble (Boosting) |
| 8 | AdaBoost | Ensemble (Boosting) |
| 9 | SVR | Kernel-based |

> \* **Top performers** — tuned further with GridSearchCV

---

## Key Results

| Metric | Best Model (Tuned Random Forest) |
|:------:|:--------------------------------:|
| **R2 Score** | ~0.95+ |
| **RMSE** | Low |
| **Residuals** | Centered around 0 (no bias) |

### Key Findings

- **Tree-based ensembles** (Random Forest, XGBoost) crushed linear models
- **Decision Tree** showed classic overfitting — high train R2, low test R2
- **SARIMAX** successfully forecasted 2026 deliveries with 95% confidence intervals
- **Production Units** was the strongest predictor of deliveries
- **Cheaper models** (Model 3, Model Y) have significantly higher delivery volumes

---

## What I Learned

### Data Handling & EDA

| Concept | What I Learned |
|:--------|:---------------|
| `df.shape`, `df.describe()`, `df.dtypes` | First things to check — understand your data before touching it |
| KDE Plots | Smooth version of histograms — shows distribution shape |
| Count Plots | Check if categorical features are balanced |
| Box Plots | Detect outliers using IQR method |
| Correlation Heatmap | Find which features are strongly related to the target |
| Skewness Check | Values below 1 — no severe skew — no PowerTransformer needed |

### Feature Engineering

| Technique | Why It Matters |
|:----------|:---------------|
| **Cyclical Encoding** (sin/cos) | Months are circular — December is close to January, not far from it |
| **Lag Features** | "Last month's deliveries" is the best predictor of this month's |
| **Rolling Mean** | Smooths out monthly noise — shows the underlying trend |
| **Interaction Features** | Price/Range ratio captures "value for money" — combinations can matter more than individual features |
| **VIF** | Checks multicollinearity — are any features redundant? |

### Preprocessing & Pipeline

| Concept | Key Insight |
|:--------|:------------|
| **Chronological Split** | Never use random split on time data — it causes data leakage |
| **Data Leakage** | Using future info to train = cheating. Model looks great in testing, fails in production |
| **`fit_transform` vs `transform`** | Fit on TRAIN only. Transform both. Never fit on test data |
| **ColumnTransformer** | Different columns need different treatment — numbers get scaled, text gets one-hot encoded |
| **StandardScaler** | Makes all features speak the same language (mean=0, std=1) |
| **OneHotEncoder** | Converts categories to binary columns — ML models only understand numbers |

### Modeling & Tuning

| Concept | Key Insight |
|:--------|:------------|
| **Overfitting** | High train score + low test score = model memorized, didn't learn |
| **Bias-Variance Tradeoff** | Too simple = underfitting. Too complex = overfitting. Ensembles find the sweet spot |
| **GridSearchCV** | Tries every combination of hyperparameters — brute force but effective |
| **Cross-Validation (cv=3)** | Splits training data 3 ways for more reliable scoring |
| **R2, MAE, RMSE** | Different angles on model accuracy — R2 for overall fit, MAE for avg error, RMSE penalizes big errors |

### Time Series

| Concept | Key Insight |
|:--------|:------------|
| **Stationarity** | Time series models need constant mean/variance over time |
| **ADF Test** | p-value > 0.05 — not stationary — differencing needed |
| **SARIMAX** | Handles trend (I), past values (AR), past errors (MA), and seasonality (S) — all in one model |
| **`order=(1,1,1)`** | p=1 (1 lag), d=1 (1 differencing), q=1 (1 MA term) |
| **`seasonal_order=(1,1,1,12)`** | Same structure but for yearly cycles (12-month period) |
| **Confidence Intervals** | Forecast widens over time — further predictions = more uncertainty |

---

## Project Structure

```
Tesla-EV-Analysis/
├── Sckand_Week2_assignment.ipynb    # Main notebook (full pipeline)
├── tesla_deliveries_dataset.csv     # Raw dataset
└── README.md                        # You are here
```

---

## How to Run

```bash
# 1. Clone this repo
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git

# 2. Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn xgboost statsmodels

# 3. Open the notebook
jupyter notebook Sckand_Week2_assignment.ipynb
```

---

## Tech Stack

<div align="center">

| Tool | Purpose |
|:----:|:--------|
| Python | Core language |
| Pandas | Data manipulation |
| NumPy | Numerical computing |
| Matplotlib | Base visualization |
| Seaborn | Statistical visualization |
| Scikit-Learn | ML models & preprocessing |
| XGBoost | Gradient boosting |
| Statsmodels | Time series (SARIMAX) |

</div>

---

<div align="center">

### If you found this useful, give it a star!

Made by **Sckand Sharma**

</div>
