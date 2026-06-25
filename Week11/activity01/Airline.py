from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.models import Sequential
import tensorflow as tf
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

CHARTS_DIR = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "charts2")
os.makedirs(CHARTS_DIR, exist_ok=True)

# Machine Learning Models

# Deep Learning Models

# Time Series Model

# Load dataset (replace with any time series data)
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
df = pd.read_csv(url, parse_dates=['Month'], index_col='Month')
print("\n Top five records: ", df.head())

# Plot data
plt.figure(figsize=(10, 5))
plt.plot(df, label="Passenger Count")
plt.legend()
plt.savefig(os.path.join(CHARTS_DIR, "01_passenger_count.png"),
            bbox_inches="tight", dpi=150)
plt.show()

# Convert data to supervised learning format
df['Passengers_Lag1'] = df['Passengers'].shift(1)
df.dropna(inplace=True)

# Train-Test Split
train_size = int(len(df) * 0.8)
train, test = df.iloc[:train_size], df.iloc[train_size:]

X_train, y_train = train[['Passengers_Lag1']], train['Passengers']
X_test, y_test = test[['Passengers_Lag1']], test['Passengers']

# Scale data for LSTM (built from the same globally-aligned Passengers_Lag1
# feature used by the other models, so lengths match y_test/X_test)
scaler = MinMaxScaler()
scaler.fit(train[['Passengers']])

X_train_lstm = scaler.transform(train[['Passengers_Lag1']].values)
y_train_lstm = scaler.transform(train[['Passengers']].values)
X_test_lstm = scaler.transform(test[['Passengers_Lag1']].values)
y_test_lstm = scaler.transform(test[['Passengers']].values)

# Reshape for LSTM input: (samples, timesteps, features)
X_train_lstm = X_train_lstm.reshape((X_train_lstm.shape[0], 1, 1))
X_test_lstm = X_test_lstm.reshape((X_test_lstm.shape[0], 1, 1))

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# XGBoost
xgb_model = xgb.XGBRegressor(objective="reg:squarederror")
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)

# LSTM Model
lstm_model = Sequential([
    LSTM(50, activation='relu', return_sequences=True, input_shape=(1, 1)),
    LSTM(50, activation='relu'),
    Dense(1)
])
lstm_model.compile(optimizer='adam', loss='mse')
lstm_model.fit(X_train_lstm, y_train_lstm, epochs=50, verbose=0, batch_size=1)
y_pred_lstm = lstm_model.predict(X_test_lstm)
y_pred_lstm = scaler.inverse_transform(y_pred_lstm)

# ANN Model
ann_model = Sequential([
    Dense(64, activation='relu', input_shape=(1,)),
    Dense(32, activation='relu'),
    Dense(1)
])
ann_model.compile(optimizer='adam', loss='mse')
ann_model.fit(X_train, y_train, epochs=50, verbose=0, batch_size=1)
y_pred_ann = ann_model.predict(X_test)

# ARIMA Model
arima_model = ARIMA(train['Passengers'], order=(5, 1, 0))
arima_model_fit = arima_model.fit()
y_pred_arima = arima_model_fit.forecast(steps=len(test))

# Evaluation Function


def evaluate_model(y_true, y_pred, model_name):
    print(f"Model: {model_name}")
    print(f"MAE: {mean_absolute_error(y_true, y_pred):.2f}")
    print(f"MSE: {mean_squared_error(y_true, y_pred):.2f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_true, y_pred)):.2f}")
    print(f"R2 Score: {r2_score(y_true, y_pred):.2f}")
    print("-" * 40)


# Evaluate All Models
evaluate_model(y_test, y_pred_lr, "Linear Regression")
evaluate_model(y_test, y_pred_xgb, "XGBoost")
evaluate_model(y_test, y_pred_lstm.flatten(), "LSTM")
evaluate_model(y_test, y_pred_ann.flatten(), "ANN")
evaluate_model(y_test, y_pred_arima, "ARIMA")

# Plot Predictions
plt.figure(figsize=(12, 6))
plt.plot(y_test.index, y_test, label="Actual", color="blue")
plt.plot(y_test.index, y_pred_lr, label="Linear Regression",
         linestyle="dashed", color="cyan")
plt.plot(y_test.index, y_pred_xgb, label="XGBoost",
         linestyle="dashdot", color="magenta")
plt.plot(y_test.index, y_pred_lstm.flatten(),
         label="LSTM", linestyle="dotted", color="green")
plt.plot(y_test.index, y_pred_ann.flatten(),
         label="ANN", linestyle="dashdot", color="black")
plt.plot(y_test.index, y_pred_arima, label="ARIMA",
         linestyle="dashed", color="red")
plt.legend()
plt.title("Model Predictions vs Actual")
plt.savefig(os.path.join(
    CHARTS_DIR, "02_model_predictions_vs_actual.png"), bbox_inches="tight", dpi=150)
plt.show()
