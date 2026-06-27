import pandas as pd
import os

DATA_PATH = "Wellbeing_data_all.csv"
REPORTS_DIR = "reports"


def _ensure_reports_dir():
    os.makedirs(REPORTS_DIR, exist_ok=True)


def _write_report(filename, content):
    path = os.path.join(REPORTS_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  >> Report saved: {path}\n")


def _df_to_md_table(df):
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    separator = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = []
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join([header, separator] + rows)


def load_and_preprocess():
    _ensure_reports_dir()

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")

    # ==================================================================
    # STEP 1 & 2 — Data Preview
    # ==================================================================
    print("=" * 60)
    print("STEP 1: RAW DATA SAMPLE (first 5 rows)")
    print("=" * 60)
    print(df.head().to_string())

    print("\n" + "=" * 60)
    print("STEP 2: BASIC DATA INFO")
    print("=" * 60)
    print(f"Shape          : {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Columns        : {list(df.columns)}")
    print(f"\nData types:\n{df.dtypes.to_string()}")
    print(f"\nYears covered  : {sorted(df['Year'].unique())}")
    print(f"\nWellbeing measures ({df['Wellbeing measure'].nunique()} unique):")
    for m in sorted(df["Wellbeing measure"].unique()):
        print(f"  - {m}")
    print(f"\nDemographics ({df['Demographic'].nunique()} unique):")
    for d in sorted(df["Demographic"].unique()):
        print(f"  - {d}")
    print(f"\nTypes : {df['Type'].unique().tolist()}")
    print(f"Flags : {sorted(df['Flag'].dropna().unique().tolist())}")

    # Build 01_DataPreview.md
    measures_list = "\n".join(f"- {m}" for m in sorted(df["Wellbeing measure"].unique()))
    demographics_list = "\n".join(f"- {d}" for d in sorted(df["Demographic"].unique()))
    dtypes_table = "| Column | Type |\n| --- | --- |\n" + "\n".join(
        f"| {col} | {str(dtype)} |" for col, dtype in df.dtypes.items()
    )
    sample_table = _df_to_md_table(df.head())

    preview_md = f"""# Data Preview Report

## 1. Raw Data Sample (first 5 rows)

{sample_table}

---

## 2. Basic Data Info

| Property | Value |
| --- | --- |
| Rows | {df.shape[0]} |
| Columns | {df.shape[1]} |
| Years covered | {", ".join(str(y) for y in sorted(df["Year"].unique()))} |
| Wellbeing measures | {df["Wellbeing measure"].nunique()} unique |
| Demographics | {df["Demographic"].nunique()} unique |

### Column Data Types

{dtypes_table}

### Wellbeing Measures ({df["Wellbeing measure"].nunique()})

{measures_list}

### Demographics ({df["Demographic"].nunique()})

{demographics_list}

### Measurement Types

{chr(10).join(f"- {t}" for t in df["Type"].unique())}

### Flag Codes Present

{chr(10).join(f"- `{f}`" for f in sorted(df["Flag"].dropna().unique()))}
"""
    _write_report("01_DataPreview.md", preview_md)

    # ==================================================================
    # STEPS 3–6 — Preprocessing
    # ==================================================================

    # Step 3: Nulls
    print("=" * 60)
    print("STEP 3: NULL / MISSING VALUES CHECK")
    print("=" * 60)
    null_counts = df.isnull().sum()
    print(null_counts.to_string())

    # Step 4: Duplicates
    print("\n" + "=" * 60)
    print("STEP 4: DUPLICATE ROWS CHECK")
    print("=" * 60)
    dupe_count = df.duplicated().sum()
    print(f"Duplicate rows: {dupe_count}")

    # Step 5: Anomalies
    print("\n" + "=" * 60)
    print("STEP 5: DATA ANOMALY CHECK")
    print("=" * 60)

    flag_labels = {
        "": "No flag (reliable)",
        "*": "* suppressed / use with care",
        "**": "** use with caution",
        "***": "*** very unreliable",
        "S": "S suppressed",
    }
    flag_counts = df["Flag"].fillna("").value_counts()
    for flag, count in flag_counts.items():
        print(f"  {flag_labels.get(flag, flag)!r:45s}: {count}")

    df["Estimate(percent)"] = pd.to_numeric(df["Estimate(percent)"], errors="coerce")
    df["Absolute sampling error (percentage points)"] = pd.to_numeric(
        df["Absolute sampling error (percentage points)"], errors="coerce"
    )

    out_of_range = df[(df["Estimate(percent)"] < 0) | (df["Estimate(percent)"] > 100)]
    neg_error = df[df["Absolute sampling error (percentage points)"] < 0]
    print(f"\nEstimate out of range [0–100]: {len(out_of_range)} rows")
    print(f"Negative sampling error      : {len(neg_error)} rows")

    # Step 6: Clean
    print("\n" + "=" * 60)
    print("STEP 6: PREPROCESSING — CLEAN & TRANSFORM")
    print("=" * 60)

    rows_before = len(df)
    df = df[df["Estimate(percent)"].notna()].copy()
    dropped = rows_before - len(df)
    print(f"Dropped {dropped} rows with missing Estimate(percent)")

    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()

    df["Flag"] = df["Flag"].fillna("").str.strip()
    df["is_unreliable"] = df["Flag"].isin(["***", "S"])
    unreliable_count = int(df["is_unreliable"].sum())
    print(f"Rows marked unreliable (*** or S): {unreliable_count}")
    print(f"Final shape: {df.shape}")

    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)
    print(f"  Total rows loaded   : {rows_before}")
    print(f"  Rows after cleaning : {len(df)}")
    print(f"  Rows dropped        : {dropped}")
    print(f"  Unreliable rows     : {unreliable_count}")
    print(f"  Years in dataset    : {sorted(df['Year'].unique())}")
    print(f"  Ready for modelling : YES")

    # Build 02_DataPreprocess.md
    null_table = "| Column | Null Count |\n| --- | --- |\n" + "\n".join(
        f"| {col} | {count} |" for col, count in null_counts.items()
    )
    flag_table = "| Flag | Meaning | Count |\n| --- | --- | --- |\n" + "\n".join(
        f"| `{flag}` | {flag_labels.get(flag, flag)} | {count} |"
        for flag, count in flag_counts.items()
    )

    preprocess_md = f"""# Data Preprocessing Report

## 3. Null / Missing Values Check

{null_table}

> **Note:** Flag column has {int(null_counts["Flag"])} nulls — these represent rows with no quality concern (reliable data).

---

## 4. Duplicate Rows Check

| Check | Result |
| --- | --- |
| Duplicate rows found | {dupe_count} |

---

## 5. Data Anomaly Check

### Flag Distribution (Data Quality Indicators)

{flag_table}

### Value Range Check

| Check | Result |
| --- | --- |
| Estimate(percent) out of range [0–100] | {len(out_of_range)} rows |
| Negative sampling error values | {len(neg_error)} rows |

---

## 6. Cleaning & Transformation Steps

| Action | Detail |
| --- | --- |
| Dropped missing estimates | {dropped} rows removed |
| Stripped whitespace | All string columns |
| Normalised Flag column | NaN → empty string |
| Marked unreliable rows | {unreliable_count} rows flagged (`***` or `S`) — kept for reference |

---

## Preprocessing Outcome Summary

| Metric | Value |
| --- | --- |
| Total rows loaded | {rows_before} |
| Rows after cleaning | {len(df)} |
| Rows dropped | {dropped} |
| Unreliable rows (kept) | {unreliable_count} |
| Years in dataset | {", ".join(str(y) for y in sorted(df["Year"].unique()))} |
| Ready for modelling | YES |
"""
    _write_report("02_DataPreprocess.md", preprocess_md)

    return df


if __name__ == "__main__":
    df = load_and_preprocess()
