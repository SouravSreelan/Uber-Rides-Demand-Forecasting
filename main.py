
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
import xgboost as xgb
from datetime import timedelta

# Load Data (Example: NYC Taxi trip count)
url = 'https://data.cityofnewyork.us/api/views/buex-bi6w/rows.csv?accessType=DOWNLOAD'
df = pd.read_csv(url, parse_dates=['pickup_datetime'])

# For simulation purposes:
date_rng = pd.date_range(start='2023-01-01', end='2023-01-31 23:00', freq='H')
df = pd.DataFrame(date_rng, columns=['datetime'])
df['rides'] = (np.sin(np.linspace(0, 3 * np.pi, len(df))) + np.random.normal(0, 0.5, len(df))) * 100 + 500

# Feature Engineering
df['hour'] = df['datetime'].dt.hour
df['dayofweek'] = df['datetime'].dt.dayofweek
df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)

# SARIMA Model
sarima_model = SARIMAX(df['rides'], order=(1,1,1), seasonal_order=(1,1,1,24))
sarima_results = sarima_model.fit(disp=False)
df['sarima_forecast'] = sarima_results.predict(start=0, end=len(df)-1, dynamic=False)

# Prophet Model
prophet_df = df[['datetime', 'rides']].rename(columns={'datetime': 'ds', 'rides': 'y'})
prophet_model = Prophet(daily_seasonality=True)
prophet_model.fit(prophet_df)
future = prophet_model.make_future_dataframe(periods=48, freq='H')
forecast = prophet_model.predict(future)

# XGBoost Regressor
for lag in [1, 2, 24]:
    df[f'lag_{lag}'] = df['rides'].shift(lag)
df.dropna(inplace=True)
features = ['hour', 'dayofweek', 'is_weekend', 'lag_1', 'lag_2', 'lag_24']
X = df[features]
y = df['rides']
model = xgb.XGBRegressor()
model.fit(X, y)
df['xgb_forecast'] = model.predict(X)

plt.figure(figsize=(12,6))
plt.plot(df['datetime'], df['rides'], label='Actual', alpha=0.5)
plt.plot(df['datetime'], df['sarima_forecast'], label='SARIMA')
plt.plot(df['datetime'].iloc[-len(df['xgb_forecast']):], df['xgb_forecast'], label='XGBoost')
plt.legend()
plt.title("Uber Ride Demand Forecasting")
plt.xlabel("Datetime")
plt.ylabel("Predicted Rides")
plt.tight_layout()
plt.show()