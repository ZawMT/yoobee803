# Forecast Report — All Demographics

## Approach

| Item | Detail |
| --- | --- |
| Target | All wellbeing measures × all demographic combinations |
| Training years | 2014, 2016 |
| Evaluation year | 2018 (held-out test) |
| Forecast year | 2020 |
| Models | LR, XGB, ARIMA |
| Metric | Mean Absolute Error (MAE) |

> **Note:** ARIMA uses AR(1) — the simplest viable model for 2–3 observations.

---

## 1. Series Summary

| Item | Count |
| --- | --- |
| Total series processed | 5170 |
| Skipped (incomplete years) | 5 |
| Models compared | 3 |

---

## 2. Model Comparison — Mean MAE Across All Series (lower is better)

| Model | Mean MAE |
| --- | --- |
| LR | 3.6201 |
| XGB | 2.4088 |  **Best**
| ARIMA | 2.4716 |

**Best performing model: XGB** (mean MAE = 2.4088)

---

## 3. 2020 Forecast Summary (across all series)

| Model | Mean Forecast | Min | Max |
| --- | --- | --- | --- |
| LR | 25.4465 | 0.4000 | 93.2000 |
| XGB | 25.3643 | 1.0011 | 93.1989 |
| ARIMA | 24.8598 | 0.9091 | 91.8119 |

---

## 4. Mean MAE by Demographic Group

| Demographic | LR MAE | XGB MAE | ARIMA MAE |
| --- | --- | --- | --- |
| Age group (years) | 3.4191 | 2.3406 | 2.2872 |
| Ethnicity | 3.826 | 2.616 | 2.9656 |
| Family type | 3.112 | 2.1686 | 2.0995 |
| Highest qualification | 3.2083 | 2.1972 | 2.1526 |
| Household income | 2.8979 | 2.1337 | 2.4221 |
| Housing tenure | 2.4672 | 1.8385 | 1.8505 |
| Labour force status | 3.9939 | 2.5683 | 2.2196 |
| Life stage (years) | 3.0911 | 2.183 | 2.1234 |
| Migrant status | 3.5778 | 2.4519 | 2.5958 |
| Personal income | 2.658 | 2.0802 | 2.2899 |
| Region | 5.1377 | 2.9988 | 3.1307 |
| Sex | 2.426 | 1.866 | 1.8333 |
| Total population | 2.075 | 1.7031 | 1.7312 |

---

## 5. Best Predicted Series (lowest XGB MAE)

| Measure | Demographic | Dem. Category | Actual 2018 | XGB MAE |
| --- | --- | --- | --- | --- |
| Ability to express identity | Sex | Female | 1.9 | 0.0 |
| Life worthwhile | Age group (years) | 35-44 years | 8.1 | 0.0 |
| Life worthwhile | Highest qualification | Level 7 and above | 8.2 | 0.0 |
| Life worthwhile | Highest qualification | No qualification | 8.0 | 0.0 |
| Life worthwhile | Household income | $150,001 or more | 8.3 | 0.0 |

---

## 6. Hardest to Predict Series (highest XGB MAE)

| Measure | Demographic | Dem. Category | Actual 2018 | XGB MAE |
| --- | --- | --- | --- | --- |
| Feeling of safety using/waiting for public transport at night | Region | Northland | 60.3 | 27.701 |
| Spent less on hobbies or other special interests than liked to keep costs down | Region | Bay of Plenty | 67.2 | 22.3987 |
| Satisfaction with job in the last 4 weeks | Age group (years) | 75 years and over | 52.7 | 21.6991 |
| Trust held for parliament | Labour force status | Unemployed | 21.3 | 20.301 |
| Put up with feeling cold | Ethnicity | Pacific peoples | 59.1 | 18.4983 |
