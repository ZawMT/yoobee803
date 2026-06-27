# Data Preprocessing Report

## 3. Null / Missing Values Check

| Column | Null Count |
| --- | --- |
| Year | 0 |
| Wellbeing measure | 0 |
| Wellbeing measure category | 0 |
| Demographic | 0 |
| Demographic category | 0 |
| Type | 0 |
| Estimate(percent) | 9 |
| Absolute sampling error (percentage points) | 9 |
| Flag | 14382 |

> **Note:** Flag column has 14382 nulls — these represent rows with no quality concern (reliable data).

---

## 4. Duplicate Rows Check

| Check | Result |
| --- | --- |
| Duplicate rows found | 0 |

---

## 5. Data Anomaly Check

### Flag Distribution (Data Quality Indicators)

| Flag | Meaning | Count |
| --- | --- | --- |
| `` | No flag (reliable) | 14382 |
| `*` | * suppressed / use with care | 914 |
| `**` | ** use with caution | 213 |
| `***` | *** very unreliable | 9 |
| `S` | S suppressed | 9 |

### Value Range Check

| Check | Result |
| --- | --- |
| Estimate(percent) out of range [0–100] | 0 rows |
| Negative sampling error values | 0 rows |

---

## 6. Cleaning & Transformation Steps

| Action | Detail |
| --- | --- |
| Dropped missing estimates | 9 rows removed |
| Stripped whitespace | All string columns |
| Normalised Flag column | NaN → empty string |
| Marked unreliable rows | 9 rows flagged (`***` or `S`) — kept for reference |

---

## Preprocessing Outcome Summary

| Metric | Value |
| --- | --- |
| Total rows loaded | 15527 |
| Rows after cleaning | 15518 |
| Rows dropped | 9 |
| Unreliable rows (kept) | 9 |
| Years in dataset | 2014, 2016, 2018 |
| Ready for modelling | YES |
