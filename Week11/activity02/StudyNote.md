## Study Note — Why ANN and LSTM Are Slow on Short Time Series

### Context

In this pipeline, each wellbeing series has only 3 data points (2014, 2016, 2018).
ANN and LSTM models are trained independently for each of the ~5,000 series in the dataset.
This combination makes neural network models extremely slow compared to LR, XGBoost, and ARIMA.

---

### Reasons for Slowness

#### 1. TensorFlow Model Creation Overhead (Biggest Culprit)

Every series creates brand new Keras model objects from scratch. TensorFlow must compile
the model and trace the computation graph each time. This graph-tracing cost is nearly
fixed regardless of data size — meaning the same overhead is paid for 3 data points as
for 3,000. Across ~5,000 series × 2 models × 2 training runs, this results in approximately
**20,000 model creations**.

#### 2. Epoch Count × Series Count

With `predict_all` (reduced epochs):

| Model | Epochs per run | Runs | Series | Total epoch runs |
| --- | --- | --- | --- | --- |
| ANN  | 200 | 2 | ~5,000 | ~2,000,000 |
| LSTM | 100 | 2 | ~5,000 | ~1,000,000 |

Even fast epochs accumulate into hours at this scale.

### 3. Fundamental Mismatch Between Model and Data

Neural networks are designed for:
- Large datasets (thousands of samples)
- LSTM specifically requires long sequences to learn meaningful temporal patterns

A 3-point series provides only **2 training samples** per evaluation run. The model
spends hundreds of epochs trying to learn from 2 data points — most of that computation
produces no real learning benefit.

### 4. No Cross-Series Vectorisation

Each series is trained completely independently with no ability to batch training across
series. TensorFlow starts cold for every single series.

### 5. CPU-Bound Training

Without a GPU, every matrix operation in the neural network runs sequentially on CPU cores.

---

### The Core Lesson

ANN and LSTM are the **wrong tools** for a 3-point time series problem.

| Model | Designed for | How it solves the problem |
| --- | --- | --- |
| Linear Regression | Small, simple datasets | Closed-form solution — no iteration needed |
| ARIMA | Short time series | Statistical fitting — very fast |
| XGBoost | Tabular data, any size | Efficient tree-building algorithm |
| ANN / LSTM | Large datasets, long sequences | Iterative gradient descent — expensive by design |

LR and ARIMA run in microseconds per series because they have **closed-form solutions** —
no iterative gradient descent is required. Neural networks solve a much harder, more general
problem and carry the full computational cost of that generality even when the data is tiny.

---

### Practical Takeaway

For forecasting with very short time series (2–5 data points), prefer:
- **Linear Regression** — fastest, interpretable, strong baseline
- **ARIMA** — designed for time series, handles trends well
- **XGBoost** — good middle ground, handles non-linearity

Reserve ANN and LSTM for series with at least dozens of observations where their
capacity to learn complex patterns can actually be utilised.
