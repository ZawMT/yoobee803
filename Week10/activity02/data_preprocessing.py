import pandas as pd

# -- Reference for data validity range ──────────────────────────────────────────

# Value must be within this range (numeric)
valid_range_min_max = {
    "age": (0, 120),
    "trestbps": (50, 300),
    "chol": (100, 700),
    "thalach": (20, 220),
    "oldpeak": (0, 7)
}

# Value must be one of these (categorical)
valid_range_set = {
    "sex": (0, 1),
    "restecg": (0, 1, 2),
    "exang": (0, 1),
    "slope": (1, 2, 3),
    "ca": (0, 1, 2, 3, 4),
    "thal": (0, 1, 2, 3, 4, 5, 6, 7)
}


# ── Preprocessing Report ──────────────────────────────────────────────────────

def preprocess(df):
    print("\n" + "=" * 55)
    print("PREPROCESSING REPORT")
    print("=" * 55)

    # ── First few records
    print("\n--- First 5 Records ---")
    print(df.head().to_string())

    # ── Data types
    print("\n--- Column Data Types ---")
    print(df.dtypes.to_string())

    # ── Null check
    print("\n--- Null Values ---")
    nulls = df.isnull().sum()
    null_pct = (nulls / len(df) * 100).round(2)
    null_report = pd.DataFrame({"Missing": nulls, "Missing %": null_pct})
    cols_with_nulls = null_report[null_report["Missing"] > 0]
    if cols_with_nulls.empty:
        print("  No null values found in any column.")
    else:
        print(cols_with_nulls.to_string())
        null_ids = df.loc[df.isnull().any(axis=1), "ID"].tolist()
        print(f"  Row ID(s) with null values: {null_ids}")
        df = df.dropna()
        print(f"  → Dropped rows with nulls. Remaining rows: {len(df)}")

    # ── Duplicate check
    print("\n--- Duplicate Rows ---")
    data_cols = [c for c in df.columns if c != "ID"]
    dupe_mask = df.duplicated(subset=data_cols)
    dupes = dupe_mask.sum()
    if dupes == 0:
        print("  No duplicate rows found.")
    else:
        print(f"  {dupes} duplicate row(s) detected.")
        dupe_ids = df.loc[df.duplicated(
            subset=data_cols, keep=False), "ID"].tolist()
        print(f"  Row ID(s) involved in duplication: {dupe_ids}")
        df = df.drop_duplicates(subset=data_cols)
        print(f"  → Dropped duplicates. Remaining rows: {len(df)}")

    # ── Validation check against known valid ranges/sets
    print("\n--- Validation Check ---")
    anomaly_found = False
    for col, (lo, hi) in valid_range_min_max.items():
        if col not in df.columns:
            continue
        invalid = df[(df[col] < lo) | (df[col] > hi)]
        if len(invalid) > 0:
            anomaly_found = True
            invalid_ids = invalid["ID"].tolist()
            print(f"  {col:<12}: {len(invalid):>3} invalid value(s)  "
                  f"[valid range: {lo} – {hi}]  "
                  f"values: {sorted(invalid[col].unique().tolist())}  "
                  f"Row ID(s): {invalid_ids}")
        else:
            print(f"  {col:<12}: No invalid values detected.")

    for col, valid_set in valid_range_set.items():
        if col not in df.columns:
            continue
        invalid = df[~df[col].isin(valid_set)]
        if len(invalid) > 0:
            anomaly_found = True
            invalid_ids = invalid["ID"].tolist()
            print(f"  {col:<12}: {len(invalid):>3} invalid value(s)  "
                  f"[valid set: {sorted(valid_set)}]  "
                  f"values: {sorted(invalid[col].unique().tolist())}  "
                  f"Row ID(s): {invalid_ids}")
        else:
            print(f"  {col:<12}: No invalid values detected.")

    if not anomaly_found:
        print("  No anomalies detected in any column.")

    # ── Summary
    print("\n--- Summary ---")
    print(f"  Rows after cleaning : {df.shape[0]}")
    print(f"  Columns             : {df.shape[1]}")
    print(f"  Null values remain  : {df.isnull().sum().sum()}")

    return df
