import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential #type: ignore
from tensorflow.keras.layers import LSTM, Dense #type: ignore
import joblib

# Load Data
df = pd.read_csv("data/AEP_hourly.csv")

df["Datetime"] = pd.to_datetime(df["Datetime"])
df = df.sort_values("Datetime")

data = df["AEP_MW"].values.reshape(-1,1)

# Normalize
scaler = MinMaxScaler()
scaled = scaler.fit_transform(data)

# Create Time Series Windows
def create_sequences(data, window=24):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i+window])
        y.append(data[i+window])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled, 24)

# Train/Test Split
split = int(len(X)*0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Build LSTM
model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(24,1)))
model.add(LSTM(32))
model.add(Dense(1))

model.compile(optimizer="adam", loss="mse")

print("\nTraining Model...")
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Save into the models directory where the app logic lives
model.save("models/energy_forecast_model.h5")
joblib.dump(scaler, "models/energy_scaler.save")

print("\nModel and scaler saved successfully to /models folder")