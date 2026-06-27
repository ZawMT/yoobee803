# Forecast Report — All Demographics

## Approach

| Item | Detail |
| --- | --- |
| Target | All wellbeing measures × all demographic combinations |
| Training years | 2014, 2016 |
| Evaluation year | 2018 (held-out test) |
| Forecast year | 2020 |
| Models | ANN |
| Metric | Mean Absolute Error (MAE) |
| ANN epochs | 5 |

> **Note:** ANN/LSTM epochs are reduced compared to the Total Population report to keep runtime manageable.

---

## 1. Series Summary

| Item | Count |
| --- | --- |
| Total series processed | 100 |
| Skipped (incomplete years) | 3 |
| Models compared | 1 |

---

## 2. Model Comparison — Mean MAE Across All Series (lower is better)

| Model | Mean MAE |
| --- | --- |
| ANN | 1.1986 |  **Best**

**Best performing model: ANN** (mean MAE = 1.1986)

---

## 3. 2020 Forecast Summary (across all series)

| Model | Mean Forecast | Min | Max |
| --- | --- | --- | --- |
| ANN | 18.3772 | -0.3125 | 43.1432 |

---

## 4. Mean MAE by Demographic Group

| Demographic | ANN MAE |
| --- | --- |
| Age group (years) | 0.6579 |
| Ethnicity | 2.9112 |
| Family type | 0.6117 |
| Highest qualification | 1.4987 |
| Household income | 0.6963 |
| Housing tenure | 0.4462 |
| Labour force status | 2.0566 |
| Life stage (years) | 0.4631 |
| Migrant status | 1.1306 |
| Personal income | 0.5893 |
| Region | 1.7993 |
| Sex | 0.5808 |
| Total population | 0.0101 |

---

## 5. Best Predicted Series (lowest ANN MAE)

| Measure | Demographic | Dem. Category | Actual 2018 | ANN MAE |
| --- | --- | --- | --- | --- |
| Ability to express identity | Ethnicity | European | 1.4 | 0.0011 |
| Ability to express identity | Highest qualification | Level 7 and above | 1.7 | 0.0051 |
| Ability to express identity | Region | Wellington | 28.0 | 0.0099 |
| Ability to express identity | Total population | Total population | 33.5 | 0.0101 |
| Ability to express identity | Housing tenure | Owner-occupied | 1.5 | 0.0143 |

---

## 6. Hardest to Predict Series (highest ANN MAE)

| Measure | Demographic | Dem. Category | Actual 2018 | ANN MAE |
| --- | --- | --- | --- | --- |
| Ability to express identity | Ethnicity | Pacific peoples | 52.5 | 21.6272 |
| Ability to express identity | Region | Taranaki | 36.1 | 10.251 |
| Ability to express identity | Highest qualification | Level 5-6 diploma | 33.4 | 6.8861 |
| Ability to express identity | Region | Otago | 39.6 | 6.6309 |
| Ability to express identity | Labour force status | Not in the labour force | 36.3 | 5.8356 |
