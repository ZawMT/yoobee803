"""
04_Predict_All.py
Forecast 2020 wellbeing estimates across ALL demographic combinations.
Trains on 2014 + 2016, evaluates on 2018, then forecasts 2020.
Writes results to reports/04_Predict_All.md.

Note: ANN and LSTM epochs are reduced vs 03_Predict_TotalPop.py to keep
runtime manageable across hundreds of series.
"""

import os
import importlib.util
from datetime import datetime
import pandas as pd
import numpy as np
from models import MODEL_REGISTRY

REPORTS_DIR  = "reports"
MODEL_NAMES  = list(MODEL_REGISTRY.keys())
ANN_EPOCHS    = 5
LSTM_EPOCHS   = 5
SERIES_LIMIT  = 100   # set to None to process all series


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _import_module(filename):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    spec = importlib.util.spec_from_file_location(filename, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_report(results_df, mae_summary, best_model, skipped, active_names):
    os.makedirs(REPORTS_DIR, exist_ok=True)

    # Top 5 and bottom 5 series by best-model MAE
    best_col = f"{best_model} MAE"
    top5  = results_df.nsmallest(5,  best_col)[
        ["Measure", "Demographic", "Dem. Category", "Actual 2018", best_col]
    ]
    bot5  = results_df.nlargest(5, best_col)[
        ["Measure", "Demographic", "Dem. Category", "Actual 2018", best_col]
    ]

    def _df_to_md(df):
        header = "| " + " | ".join(df.columns) + " |"
        sep    = "| " + " | ".join(["---"] * len(df.columns)) + " |"
        rows   = ["| " + " | ".join(str(v) for v in row) + " |"
                  for _, row in df.iterrows()]
        return "\n".join([header, sep] + rows)

    # MAE summary
    mae_rows = "\n".join(
        f"| {m} | {v:.4f} |{'  **Best**' if m == best_model else ''}"
        for m, v in mae_summary.items()
    )

    # Per-model mean 2020 forecast across all series
    fc_rows = "\n".join(
        f"| {m} | {results_df[f'{m} 2020'].mean():.4f} | "
        f"{results_df[f'{m} 2020'].min():.4f} | "
        f"{results_df[f'{m} 2020'].max():.4f} |"
        for m in active_names
    )

    # Demographic breakdown — mean MAE per demographic group
    demo_mae = (
        results_df.groupby("Demographic")[
            [f"{m} MAE" for m in active_names]
        ].mean().round(4)
    )
    demo_header = "| Demographic |" + "".join(f" {m} MAE |" for m in active_names)
    demo_sep    = "| --- |" + " --- |" * len(active_names)
    demo_rows   = "\n".join(
        "| " + demo + " |" + "".join(f" {row[f'{m} MAE']} |" for m in active_names)
        for demo, row in demo_mae.iterrows()
    )

    epoch_rows = ""
    if "ANN" in active_names:
        epoch_rows += f"| ANN epochs | {ANN_EPOCHS} |\n"
    if "LSTM" in active_names:
        epoch_rows += f"| LSTM epochs | {LSTM_EPOCHS} |"

    notes = []
    if "ANN" in active_names or "LSTM" in active_names:
        notes.append("ANN/LSTM epochs are reduced compared to the Total Population report to keep runtime manageable.")
    if "ARIMA" in active_names:
        notes.append("ARIMA uses AR(1) — the simplest viable model for 2–3 observations.")
    note_block = ("> **Note:** " + " ".join(notes)) if notes else ""

    content = f"""# Forecast Report — All Demographics

## Approach

| Item | Detail |
| --- | --- |
| Target | All wellbeing measures × all demographic combinations |
| Training years | 2014, 2016 |
| Evaluation year | 2018 (held-out test) |
| Forecast year | 2020 |
| Models | {", ".join(active_names)} |
| Metric | Mean Absolute Error (MAE) |
{epoch_rows}
{note_block}

---

## 1. Series Summary

| Item | Count |
| --- | --- |
| Total series processed | {len(results_df)} |
| Skipped (incomplete years) | {skipped} |
| Models compared | {len(active_names)} |

---

## 2. Model Comparison — Mean MAE Across All Series (lower is better)

| Model | Mean MAE |
| --- | --- |
{mae_rows}

**Best performing model: {best_model}** (mean MAE = {mae_summary[best_model]:.4f})

---

## 3. 2020 Forecast Summary (across all series)

| Model | Mean Forecast | Min | Max |
| --- | --- | --- | --- |
{fc_rows}

---

## 4. Mean MAE by Demographic Group

{demo_header}
{demo_sep}
{demo_rows}

---

## 5. Best Predicted Series (lowest {best_model} MAE)

{_df_to_md(top5)}

---

## 6. Hardest to Predict Series (highest {best_model} MAE)

{_df_to_md(bot5)}
"""

    path = os.path.join(REPORTS_DIR, "04_Predict_All.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"\n  >> Report saved: {path}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def run(models="all"):
    print("=" * 60)
    print("04: FORECASTING — ALL DEMOGRAPHICS")
    print("=" * 60)

    if models.strip().lower() == "all":
        active_names = MODEL_NAMES
    else:
        requested    = [m.strip().upper() for m in models.split(";") if m.strip()]
        unknown      = [m for m in requested if m not in MODEL_REGISTRY]
        active_names = [m for m in MODEL_NAMES if m in requested]  # preserve registry order
        if unknown:
            print(f"  Warning: unknown model(s) ignored: {unknown}")
        if not active_names:
            print("  No valid models selected — skipping predict_all.")
            return pd.DataFrame()

    active_registry = {k: MODEL_REGISTRY[k] for k in active_names}
    print(f"  Models: {', '.join(active_names)}\n")

    preprocessing = _import_module("preprocessing.py")
    df = preprocessing.load_and_preprocess()

    group_cols = [
        "Wellbeing measure",
        "Wellbeing measure category",
        "Demographic",
        "Demographic category",
        "Type",
    ]

    groups        = list(df.groupby(group_cols))
    total_series  = len(groups)
    results       = []
    skipped       = 0

    for i, (keys, grp) in enumerate(groups, 1):
        measure, category, demo, demo_cat, mtype = keys
        grp_sorted = grp.sort_values("Year")

        if set(grp_sorted["Year"].tolist()) != {2014, 2016, 2018}:
            skipped += 1
            continue

        years = grp_sorted["Year"].tolist()
        vals  = grp_sorted["Estimate(percent)"].tolist()

        if i % 50 == 0 or i == total_series:
            ts = datetime.now().strftime("%H:%M:%S")
            print(f"  [{ts}] Progress: {i}/{total_series} series ...")

        if SERIES_LIMIT and len(results) >= SERIES_LIMIT:
            print(f"  >> SERIES_LIMIT={SERIES_LIMIT} reached — stopping early.")
            break

        row = {
            "Measure":      measure,
            "Category":     category,
            "Demographic":  demo,
            "Dem. Category": demo_cat,
            "Type":         mtype,
            "Actual 2014":  vals[0],
            "Actual 2016":  vals[1],
            "Actual 2018":  vals[2],
        }

        for name, fn in active_registry.items():
            p18, p20, mae = fn(years, vals, ann_epochs=ANN_EPOCHS, lstm_epochs=LSTM_EPOCHS)
            row[f"{name} Pred 2018"] = p18
            row[f"{name} MAE"]       = mae
            row[f"{name} 2020"]      = p20

        results.append(row)

    results_df  = pd.DataFrame(results)
    mae_summary = {m: round(float(results_df[f"{m} MAE"].mean()), 4) for m in active_names}
    best_model  = min(mae_summary, key=mae_summary.get)

    print("\n" + "=" * 60)
    print(f"Processed : {len(results_df)} series  |  Skipped: {skipped}")
    print("MEAN MAE PER MODEL  (lower = better)")
    print("=" * 60)
    for m, v in mae_summary.items():
        tag = "  <-- BEST" if m == best_model else ""
        print(f"  {m:8s}: {v:.4f}{tag}")

    _write_report(results_df, mae_summary, best_model, skipped, active_names)
    return results_df


if __name__ == "__main__":
    run()
