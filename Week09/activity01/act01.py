import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")


def load_data(path):
    return pd.read_excel(path)


def preprocess(df):
    print("=" * 50)
    print("PREPROCESSING")
    print("=" * 50)

    print(f"\nShape: {df.shape[0]} rows, {df.shape[1]} columns")

    # Null check
    nulls = df.isnull().sum()
    print(f"\nNull values:\n{nulls}")
    df.dropna(inplace=True)

    # Duplicate check (full row)
    dupes = df.duplicated().sum()
    print(f"\nDuplicate rows (all columns): {dupes}")
    df.drop_duplicates(inplace=True)

    # Duplicate check ignoring first and last column (User_ID and Churned)
    feature_cols = df.columns[1:-1].tolist()
    dupes_feature = df.duplicated(subset=feature_cols).sum()
    print(
        f"Duplicate rows (ignoring '{df.columns[0]}' and '{df.columns[-1]}'): {dupes_feature}")
    if dupes_feature > 0:
        print(
            f"  Dropping {dupes_feature} feature-duplicate row(s), keeping first occurrence.")
        df.drop_duplicates(subset=feature_cols, keep="first", inplace=True)

    # --- Data types ---
    print("\n--- Data Types (before correction) ---")
    print(df.dtypes)

    expected_types = {
        "User_ID":                  "int64",
        "Age":                      "int64",
        "Workouts_per_Week":        "int64",
        "Avg_Session_Duration_Min": "float64",
        "Steps_per_Day":            "int64",
        "Churned":                  "int64",
    }
    for col, dtype in expected_types.items():
        if col in df.columns and str(df[col].dtype) != dtype:
            df[col] = df[col].astype(dtype)
            print(f"  Corrected '{col}' → {dtype}")

    print("\n--- Data Types (after correction) ---")
    print(df.dtypes)

    # --- Inconsistencies ---
    print("\n--- Inconsistency Checks ---")

    # Strip whitespace and normalise casing on string columns
    str_cols = df.select_dtypes(include="object").columns.tolist()
    for col in str_cols:
        before = df[col].unique()
        df[col] = df[col].str.strip().str.title()
        after = df[col].unique()
        if list(before) != list(after):
            print(f"  '{col}': normalised casing/whitespace → {after}")

    # Out-of-range numeric values
    range_rules = {
        "Age":               (1, 120),
        "Workouts_per_Week": (0, 7),
        "Steps_per_Day":     (0, 100_000),
        "Avg_Session_Duration_Min": (0, 300),
    }
    for col, (lo, hi) in range_rules.items():
        if col not in df.columns:
            continue
        bad = df[(df[col] < lo) | (df[col] > hi)]
        if not bad.empty:
            print(
                f"  '{col}': {len(bad)} out-of-range values (expected {lo}–{hi}) — rows dropped")
            df = df[(df[col] >= lo) & (df[col] <= hi)]
        else:
            print(f"  '{col}': all values in range ({lo}–{hi}) ✓")

    # Churned must be 0 or 1
    invalid_churned = df[~df["Churned"].isin([0, 1])]
    if not invalid_churned.empty:
        print(
            f"  'Churned': {len(invalid_churned)} invalid values — rows dropped")
        df = df[df["Churned"].isin([0, 1])]
    else:
        print("  'Churned': only 0/1 values ✓")

    # Basic stats after cleaning
    print(f"\nDescriptive statistics:\n{df.describe()}")

    # Encode categorical columns
    le = LabelEncoder()
    for col in df.select_dtypes(include="object").columns:
        df[col] = le.fit_transform(df[col])
        print(
            f"\nEncoded '{col}': {dict(zip(le.classes_, le.transform(le.classes_)))}")

    print("\nPreprocessing complete.")
    return df


def show_sample(df, n=5):
    print("\n" + "=" * 50)
    print(f"FIRST {n} RECORDS")
    print("=" * 50)
    print(df.head(n).to_string(index=False))


def find_optimal_k(X_scaled, max_k=10):
    inertias = []
    k_range = range(2, max_k + 1)
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertias.append(km.inertia_)

    plt.figure(figsize=(8, 4))
    plt.plot(list(k_range), inertias, marker="o")
    plt.title("Elbow Method — Optimal K")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Inertia")
    plt.xticks(list(k_range))
    plt.tight_layout()
    plt.savefig("elbow_plot.png")
    plt.close()
    print("\nElbow plot saved → elbow_plot.png")

    # Pick k at the elbow — the point after the largest inertia drop
    drops = [inertias[i] - inertias[i + 1] for i in range(len(inertias) - 1)]
    best_k = list(k_range)[drops.index(max(drops)) + 1]
    return best_k


def cluster(df, n_clusters):
    print("\n" + "=" * 50)
    print("K-MEANS CLUSTERING")
    print("=" * 50)

    features = ["Age", "Workouts_per_Week",
                "Avg_Session_Duration_Min", "Steps_per_Day"]
    X = df[features].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"\nFeatures used: {features}")
    print(f"Optimal K selected: {n_clusters}")

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df["Cluster"] = km.fit_predict(X_scaled)

    print(
        f"\nCluster distribution:\n{df['Cluster'].value_counts().sort_index()}")

    summary = df.groupby("Cluster")[features].mean().round(2)
    print(f"\nCluster centroids (original scale):\n{summary}")

    # Scatter plot with distinct colors
    distinct_colors = ["#c11943", "#24a435", "#203fb0"]
    plt.figure(figsize=(9, 6))
    for c in sorted(df["Cluster"].unique()):
        subset = df[df["Cluster"] == c]
        color = distinct_colors[c]
        plt.scatter(subset["Steps_per_Day"], subset["Avg_Session_Duration_Min"],
                    label=f"Cluster {c}", color=color,
                    alpha=0.75, edgecolors="k", linewidths=0.4, s=60)
    plt.title("K-Means Clusters\n(Steps per Day vs Avg Session Duration)")
    plt.xlabel("Steps per Day")
    plt.ylabel("Avg Session Duration (min)")
    plt.legend(title="Cluster", framealpha=0.8)
    plt.tight_layout()
    plt.savefig("clusters_plot.png")
    plt.close()
    print("\nCluster scatter plot saved → clusters_plot.png")

    return df


def main():
    path = "Fitness_App_User_Data.xlsx"

    df = load_data(path)
    df = preprocess(df)
    show_sample(df)

    features = ["Age", "Workouts_per_Week",
                "Avg_Session_Duration_Min", "Steps_per_Day"]
    X_scaled = StandardScaler().fit_transform(df[features])

    optimal_k = find_optimal_k(X_scaled)
    df = cluster(df, n_clusters=optimal_k)

    print("\nDone.")


if __name__ == "__main__":
    main()
