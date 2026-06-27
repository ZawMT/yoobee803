# Forecast Report — All Demographics

## Approach

| Item | Detail |
| --- | --- |
| Target | All wellbeing measures × all demographic combinations |
| Training years | 2014, 2016 |
| Evaluation year | 2018 (held-out test) |
| Forecast year | 2020 |
| Models | LSTM |
| Metric | Mean Absolute Error (MAE) |
| LSTM epochs | 5 |
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
| LSTM | 0.9895 |  **Best**

**Best performing model: LSTM** (mean MAE = 0.9895)

---

## 3. 2020 Forecast Summary (across all series)

| Model | Mean Forecast | Min | Max |
| --- | --- | --- | --- |
| LSTM | 18.3250 | 0.7176 | 40.2363 |

---

## 4. Mean MAE by Demographic Group

| Demographic | LSTM MAE |
| --- | --- |
| Age group (years) | 0.3384 |
| Ethnicity | 2.6994 |
| Family type | 0.2384 |
| Highest qualification | 1.0883 |
| Household income | 0.6669 |
| Housing tenure | 0.3344 |
| Labour force status | 1.0338 |
| Life stage (years) | 0.4428 |
| Migrant status | 1.1153 |
| Personal income | 0.3051 |
| Region | 1.8499 |
| Sex | 0.1779 |
| Total population | 0.2261 |

---

## 5. Best Predicted Series (lowest LSTM MAE)

| Measure | Demographic | Dem. Category | Actual 2018 | LSTM MAE |
| --- | --- | --- | --- | --- |
| Ability to express identity | Ethnicity | European | 1.4 | 0.0002 |
| Ability to express identity | Highest qualification | Level 7 and above | 1.7 | 0.0006 |
| Ability to express identity | Age group (years) | 55-64 years | 2.1 | 0.0011 |
| Ability to express identity | Housing tenure | Owner-occupied | 1.5 | 0.0013 |
| Ability to express identity | Life stage (years) | 45-64 years | 2.0 | 0.0027 |

---

## 6. Hardest to Predict Series (highest LSTM MAE)

| Measure | Demographic | Dem. Category | Actual 2018 | LSTM MAE |
| --- | --- | --- | --- | --- |
| Ability to express identity | Ethnicity | Pacific peoples | 52.5 | 20.5252 |
| Ability to express identity | Region | Taranaki | 36.1 | 10.3109 |
| Ability to express identity | Region | Otago | 39.6 | 7.7119 |
| Ability to express identity | Migrant status | Recent migrant | 37.5 | 5.412 |
| Ability to express identity | Highest qualification | Level 5-6 diploma | 33.4 | 5.0846 |
