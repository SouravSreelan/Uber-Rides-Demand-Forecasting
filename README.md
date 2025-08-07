
# 📈 Uber Rides Demand Forecasting

### Predict hourly ride demand using time-series and machine learning models to optimize supply-demand planning at scale.

---

## 🔍 Problem Statement

Uber operates in dynamic urban environments where rider demand fluctuates hourly. Efficient allocation of drivers, surge pricing, and wait time management all depend on **accurate demand forecasting**.

This project simulates and models hourly ride data to demonstrate how data scientists at Uber might forecast ride volume using **advanced forecasting methods**.

---

## 🎯 Objectives

- Predict hourly ride demand using multiple modeling approaches
- Compare model performance using standard forecasting metrics
- Quantify uncertainty and forecast intervals for real-world readiness
- Extract seasonality and trends to improve Uber’s marketplace efficiency

---

## 🧠 Models Used

| Model          | Strengths                              |
|----------------|-----------------------------------------|
| SARIMA         | Captures seasonality + trend            |
| Facebook Prophet | Robust to holidays + changepoints     |
| XGBoost        | Learns nonlinear patterns & feature lags |

---

## 🛠️ Key Features

- Simulated hourly data over a month with realistic patterns
- Feature engineering: hour, day of week, weekend indicator, lag features
- Multi-model forecasting:
  - `SARIMA` with `statsmodels`
  - `Prophet` from Meta
  - `XGBoost` with temporal regressors
- Metrics: MAE, RMSE
- Visualization: comparison of true vs predicted rides

---

---

## 🚀 Getting Started

1. Clone the repository  
   `git clone https://github.com/SouravSreelan/uber-rides-demand-forecasting.git`
2. Install dependencies  
   `pip install -r requirements.txt`
3. Run project  
   `python main.py`

---

## 📈 Use Case in Uber

Uber uses similar demand prediction pipelines to:
- Trigger surge pricing
- Dispatch drivers intelligently
- Monitor service availability and ETAs

---

