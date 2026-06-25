import importlib

import pandas as pd

CSV_PATH = "airline-passengers.csv"

# Module name starts with a digit, so it can't be imported with a normal
# `import` statement.
descriptive = importlib.import_module("01_descriptive")


def load_data(path):
    df = pd.read_csv(path)
    print(f"[STATUS] Loaded '{path}' -> {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def check_nulls(df):
    null_counts = df.isnull().sum()
    total_nulls = null_counts.sum()
    print("\n[STATUS] Null value check:")
    print(null_counts)
    if total_nulls == 0:
        print("[STATUS] No null values found.")
    else:
        print(f"[STATUS] Found {total_nulls} null value(s).")
    return null_counts


def check_missing_values(df):
    # Missing months in the Month sequence (gaps in the time series)
    months = pd.to_datetime(df["Month"], format="%Y-%m", errors="coerce")
    expected = pd.date_range(start=months.min(), end=months.max(), freq="MS")
    missing_months = expected.difference(months)
    print("\n[STATUS] Missing value check (gaps in monthly sequence):")
    if len(missing_months) == 0:
        print("[STATUS] No missing months in the time series.")
    else:
        print(f"[STATUS] Missing {len(missing_months)} month(s):")
        print(list(missing_months.strftime("%Y-%m")))
    return missing_months


def check_anomalies(df):
    print("\n[STATUS] Data anomaly check (Passengers column):")

    # Non-numeric / negative / zero passenger counts
    non_numeric = df[pd.to_numeric(df["Passengers"], errors="coerce").isnull()]
    if not non_numeric.empty:
        print(f"[STATUS] Found {len(non_numeric)} non-numeric Passengers value(s):")
        print(non_numeric)
    else:
        print("[STATUS] All Passengers values are numeric.")

    numeric_passengers = pd.to_numeric(df["Passengers"], errors="coerce")
    negative_or_zero = df[numeric_passengers <= 0]
    if not negative_or_zero.empty:
        print(f"[STATUS] Found {len(negative_or_zero)} non-positive Passengers value(s):")
        print(negative_or_zero)
    else:
        print("[STATUS] No non-positive Passengers values found.")

    # Outlier detection using IQR
    q1 = numeric_passengers.quantile(0.25)
    q3 = numeric_passengers.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = df[(numeric_passengers < lower_bound) | (numeric_passengers > upper_bound)]
    if not outliers.empty:
        print(f"[STATUS] Found {len(outliers)} outlier(s) outside [{lower_bound:.1f}, {upper_bound:.1f}]:")
        print(outliers)
    else:
        print("[STATUS] No outliers detected (IQR method).")

    # Duplicate rows
    duplicates = df[df.duplicated()]
    if not duplicates.empty:
        print(f"[STATUS] Found {len(duplicates)} duplicate row(s).")
    else:
        print("[STATUS] No duplicate rows found.")


def preprocess(df):
    print("\n[STATUS] Starting preprocessing...")
    clean_df = df.copy()

    # Convert Month to datetime and Passengers to numeric
    clean_df["Month"] = pd.to_datetime(clean_df["Month"], format="%Y-%m", errors="coerce")
    clean_df["Passengers"] = pd.to_numeric(clean_df["Passengers"], errors="coerce")

    before = len(clean_df)

    # Drop rows with nulls created by failed conversions
    clean_df = clean_df.dropna(subset=["Month", "Passengers"])

    # Drop duplicate rows
    clean_df = clean_df.drop_duplicates()

    # Sort chronologically
    clean_df = clean_df.sort_values("Month").reset_index(drop=True)

    after = len(clean_df)
    print(f"[STATUS] Preprocessing complete: {before} rows -> {after} rows "
          f"({before - after} row(s) removed).")
    return clean_df


def main():
    df = load_data(CSV_PATH)

    print("\n[STATUS] First 5 records (raw):")
    print(df.head())

    check_nulls(df)
    check_missing_values(df)
    check_anomalies(df)

    clean_df = preprocess(df)

    print("\n[STATUS] First 5 records (preprocessed):")
    print(clean_df.head())

    print("\n[STATUS] Last 5 records (preprocessed):")
    print(clean_df.tail())

    descriptive.generate_all_charts(clean_df)

    print("\n[STATUS] Preprocessing pipeline finished successfully.")


if __name__ == "__main__":
    main()
