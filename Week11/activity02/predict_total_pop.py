"""
03_Predict_TotalPop.py
Forecast 2020 wellbeing estimates for the Total Population demographic only.
Trains on 2014 + 2016, evaluates on 2018, then forecasts 2020.
Writes results to reports/03_Predict_TotalPop.md.
"""

import os
import importlib.util
import pandas as pd
import numpy as np
from models import MODEL_REGISTRY

REPORTS_DIR = "reports"
MODEL_NAMES  = list(MODEL_REGISTRY.keys())
ANN_EPOCHS   = 5
LSTM_EPOCHS  = 5


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _import_module(filename):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    spec = importlib.util.spec_from_file_location(filename, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_report(results_df, mae_summary, best_model):
    os.makedirs(REPORTS_DIR, exist_ok=True)

    # --- per-series evaluation table ---
    eval_rows = []
    for _, r in results_df.iterrows():
        row = f"| {r['Measure']} | {r['Category']} |"
        row += f" {r['Actual 2018']} |"
        for m in MODEL_NAMES:
            row += f" {r[f'{m} Pred 2018']} ({r[f'{m} MAE']}) |"
        eval_rows.append(row)

    eval_header = (
        "| Measure | Category | Actual 2018 |"
        + "".join(f" {m} Pred (MAE) |" for m in MODEL_NAMES)
    )
    eval_sep = "| --- | --- | --- |" + " --- |" * len(MODEL_NAMES)

    # --- 2020 forecast table ---
    fc_rows = []
    for _, r in results_df.iterrows():
        row = f"| {r['Measure']} | {r['Category']} |"
        for m in MODEL_NAMES:
            row += f" {r[f'{m} 2020']} |"
        fc_rows.append(row)

    fc_header = "| Measure | Category |" + "".join(f" {m} 2020 |" for m in MODEL_NAMES)
    fc_sep    = "| --- | --- |" + " --- |" * len(MODEL_NAMES)

    # --- MAE summary table ---
    mae_rows = "\n".join(
        f"| {m} | {v:.4f} |{'  **Best**' if m == best_model else ''}"
        for m, v in mae_summary.items()
    )

    content = f"""# Forecast Report — Total Population

## Approach

| Item | Detail |
| --- | --- |
| Target demographic | Total population |
| Training years | 2014, 2016 |
| Evaluation year | 2018 (held-out test) |
| Forecast year | 2020 |
| Models | {", ".join(MODEL_NAMES)} |
| Metric | Mean Absolute Error (MAE) |

> **Note on LSTM / ANN:** With only 3 data points per series, neural networks
> have minimal training data. Results should be interpreted cautiously.
> ARIMA uses AR(1) — the simplest viable model for 2–3 observations.

---

## 1. Model Evaluation on 2018 (Predicted vs Actual)

{eval_header}
{eval_sep}
{chr(10).join(eval_rows)}

---

## 2. 2020 Forecast by Model

{fc_header}
{fc_sep}
{chr(10).join(fc_rows)}

---

## 3. Model Comparison — Mean MAE (lower is better)

| Model | Mean MAE |
| --- | --- |
{mae_rows}

**Best performing model: {best_model}** (mean MAE = {mae_summary[best_model]:.4f})

---

## 4. Series Count

| Item | Count |
| --- | --- |
| Total series processed | {len(results_df)} |
| Models compared | {len(MODEL_NAMES)} |
"""

    path = os.path.join(REPORTS_DIR, "03_Predict_TotalPop.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"\n  >> Report saved: {path}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def run():
    print("=" * 60)
    print("03: FORECASTING — TOTAL POPULATION")
    print("=" * 60)

    preprocessing = _import_module("preprocessing.py")
    df = preprocessing.load_and_preprocess()

    df_total = df[
        (df["Demographic"] == "Total population") &
        (df["Demographic category"] == "Total population")
    ].copy()

    group_cols = ["Wellbeing measure", "Wellbeing measure category", "Type"]
    results = []
    total_series = df_total.groupby(group_cols).ngroups

    for i, (keys, grp) in enumerate(df_total.groupby(group_cols), 1):
        measure, category, mtype = keys
        grp_sorted = grp.sort_values("Year")

        if set(grp_sorted["Year"].tolist()) != {2014, 2016, 2018}:
            print(f"  [{i}/{total_series}] SKIP (incomplete years): {measure} [{category}]")
            continue

        years = grp_sorted["Year"].tolist()
        vals  = grp_sorted["Estimate(percent)"].tolist()

        print(f"  [{i}/{total_series}] {measure} [{category}]")

        row = {
            "Measure":      measure,
            "Category":     category,
            "Type":         mtype,
            "Actual 2014":  vals[0],
            "Actual 2016":  vals[1],
            "Actual 2018":  vals[2],
        }

        for name, fn in MODEL_REGISTRY.items():
            p18, p20, mae = fn(years, vals, ann_epochs=ANN_EPOCHS, lstm_epochs=LSTM_EPOCHS)
            row[f"{name} Pred 2018"] = p18
            row[f"{name} MAE"]       = mae
            row[f"{name} 2020"]      = p20

        results.append(row)

    results_df   = pd.DataFrame(results)
    mae_summary  = {m: round(float(results_df[f"{m} MAE"].mean()), 4) for m in MODEL_NAMES}
    best_model   = min(mae_summary, key=mae_summary.get)

    print("\n" + "=" * 60)
    print("MEAN MAE PER MODEL  (lower = better)")
    print("=" * 60)
    for m, v in mae_summary.items():
        tag = "  <-- BEST" if m == best_model else ""
        print(f"  {m:8s}: {v:.4f}{tag}")

    _write_report(results_df, mae_summary, best_model)
    return results_df


if __name__ == "__main__":
    run()
