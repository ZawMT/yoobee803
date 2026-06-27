"""
Shared forecasting models for 3-point time series (2014, 2016, 2018 → predict 2020).

Each function signature:
    run_*(years, vals, ann_epochs, lstm_epochs) -> (pred_2018, pred_2020, mae)

Strategy:
  - Evaluate : train on [2014, 2016], predict 2018  (held-out test)
  - Forecast  : retrain on all 3 years, predict 2020
"""

import warnings
import os
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import tensorflow as tf

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.get_logger().setLevel("ERROR")
tf.random.set_seed(42)
np.random.seed(42)


def _norm_years(years):
    """Normalise years to [0, 1.5] scale: 2014=0, 2016=0.5, 2018=1.0, 2020=1.5."""
    return [(y - 2014) / 4.0 for y in years]


# ------------------------------------------------------------------
# 1. Linear Regression
# ------------------------------------------------------------------
def run_linear_regression(years, vals, **kwargs):
    X = np.array(_norm_years(years)).reshape(-1, 1)
    y = np.array(vals, dtype=float)

    model = LinearRegression().fit(X[:2], y[:2])
    pred_2018 = float(model.predict(X[2:3])[0])

    model_full = LinearRegression().fit(X, y)
    pred_2020 = float(model_full.predict(np.array([[1.5]]))[0])

    mae = abs(pred_2018 - vals[2])
    return round(pred_2018, 4), round(pred_2020, 4), round(mae, 4)


# ------------------------------------------------------------------
# 2. XGBoost
# ------------------------------------------------------------------
def run_xgboost(years, vals, **kwargs):
    X = np.array(_norm_years(years)).reshape(-1, 1)
    y = np.array(vals, dtype=float)

    params = dict(n_estimators=50, max_depth=2, learning_rate=0.3, verbosity=0)

    model = xgb.XGBRegressor(**params).fit(X[:2], y[:2])
    pred_2018 = float(model.predict(X[2:3])[0])

    model_full = xgb.XGBRegressor(**params).fit(X, y)
    pred_2020 = float(model_full.predict(np.array([[1.5]]))[0])

    mae = abs(pred_2018 - vals[2])
    return round(pred_2018, 4), round(pred_2020, 4), round(mae, 4)


# ------------------------------------------------------------------
# 3. ANN  (input: normalised year  →  output: estimate)
# ------------------------------------------------------------------
def run_ann(years, vals, ann_epochs=500, **kwargs):
    X = np.array(_norm_years(years)).reshape(-1, 1)
    scaler = MinMaxScaler()
    y_norm = scaler.fit_transform(np.array(vals).reshape(-1, 1)).flatten()

    def _build():
        m = Sequential([
            Dense(16, activation="relu", input_shape=(1,)),
            Dense(8, activation="relu"),
            Dense(1),
        ])
        m.compile(optimizer="adam", loss="mse")
        return m

    # Evaluate
    m1 = _build()
    m1.fit(X[:2], y_norm[:2], epochs=ann_epochs, verbose=0)
    pred_2018 = float(scaler.inverse_transform(
        m1.predict(X[2:3], verbose=0))[0][0])

    # Forecast
    m2 = _build()
    m2.fit(X, y_norm, epochs=ann_epochs, verbose=0)
    pred_2020 = float(scaler.inverse_transform(
        m2.predict(np.array([[1.5]]), verbose=0))[0][0])

    mae = abs(pred_2018 - vals[2])
    return round(pred_2018, 4), round(pred_2020, 4), round(mae, 4)


# ------------------------------------------------------------------
# 4. LSTM  (window = 1: uses previous value to predict next)
# ------------------------------------------------------------------
def run_lstm(years, vals, lstm_epochs=200, **kwargs):
    scaler = MinMaxScaler()
    v = scaler.fit_transform(np.array(vals).reshape(-1, 1)).flatten()

    def _build():
        m = Sequential([
            LSTM(8, input_shape=(1, 1)),
            Dense(1),
        ])
        m.compile(optimizer="adam", loss="mse")
        return m

    # Evaluate: train on (v[0]→v[1]), predict v[2] using v[1]
    m1 = _build()
    m1.fit(
        v[:1].reshape(1, 1, 1),
        v[1:2],
        epochs=lstm_epochs, verbose=0,
    )
    pred_2018_n = float(m1.predict(v[1:2].reshape(1, 1, 1), verbose=0)[0][0])
    pred_2018 = float(scaler.inverse_transform([[pred_2018_n]])[0][0])

    # Forecast: train on (v[0]→v[1], v[1]→v[2]), predict v[3] using v[2]
    m2 = _build()
    m2.fit(
        v[:2].reshape(2, 1, 1),
        v[1:3],
        epochs=lstm_epochs, verbose=0,
    )
    pred_2020_n = float(m2.predict(v[2:3].reshape(1, 1, 1), verbose=0)[0][0])
    pred_2020 = float(scaler.inverse_transform([[pred_2020_n]])[0][0])

    mae = abs(pred_2018 - vals[2])
    return round(pred_2018, 4), round(pred_2020, 4), round(mae, 4)


# ------------------------------------------------------------------
# 5. ARIMA  (AR(1) — simplest viable model for 2–3 data points)
# ------------------------------------------------------------------
def run_arima(years, vals, **kwargs):
    # Evaluate
    try:
        pred_2018 = float(ARIMA(vals[:2], order=(1, 0, 0)).fit().forecast(1)[0])
    except Exception:
        pred_2018 = vals[1]  # fallback: carry last value forward

    # Forecast
    try:
        pred_2020 = float(ARIMA(vals, order=(1, 0, 0)).fit().forecast(1)[0])
    except Exception:
        pred_2020 = vals[2]

    mae = abs(pred_2018 - vals[2])
    return round(pred_2018, 4), round(pred_2020, 4), round(mae, 4)


# Convenience registry used by prediction scripts
MODEL_REGISTRY = {
    "LR":    run_linear_regression,
    "XGB":   run_xgboost,
    "ANN":   run_ann,
    "LSTM":  run_lstm,
    "ARIMA": run_arima,
}
