import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model #type: ignore
import joblib
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Updated paths to look "up" one level then into the correct folders
model = load_model("models/energy_forecast_model.h5", compile=False)
scaler = joblib.load("models/energy_scaler.save")

# Load Latest Data
df = pd.read_csv("data/AEP_hourly.csv")
df["Datetime"] = pd.to_datetime(df["Datetime"])
df = df.sort_values("Datetime")

# Prepare Data for Prediction
def prepare_input(series, window=24):
    last_window = series[-window:]
    scaled = scaler.transform(last_window.reshape(-1,1))
    return scaled.reshape(1, window, 1)

# Predict Next Hour Consumption
last_series = df["AEP_MW"].values
X_input = prepare_input(last_series)
next_hour_scaled = model.predict(X_input)
next_hour = scaler.inverse_transform(next_hour_scaled)[0][0]

print(f"Predicted Energy Consumption for next hour: {next_hour:.2f} MW")

# Forecast Next 24 Hours
forecast = []
temp_series = last_series.copy()

for _ in range(24):
    X_input = prepare_input(temp_series)
    pred_scaled = model.predict(X_input)
    pred = scaler.inverse_transform(pred_scaled)[0][0]
    forecast.append(pred)
    temp_series = np.append(temp_series, pred)

# Monitoring Metrics
# Example: simple MAE on test data
from sklearn.metrics import mean_absolute_error
y_true = df["AEP_MW"].values[-24:]  # last 24 hours
y_pred = forecast[:24]  # first 24 predicted
mae = mean_absolute_error(y_true, y_pred)
print(f"Monitoring MAE for last 24 hours forecast: {mae:.2f} MW")
