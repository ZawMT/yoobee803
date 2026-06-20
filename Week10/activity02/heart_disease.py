import pandas as pd
from ucimlrepo import fetch_ucirepo

from data_preprocessing import preprocess
from descriptive_analysis import descriptive_analysis
from exploratory_analysis import exploratory_analysis
from predictive_analysis import predictive_analysis, predictive_analysis_multiclass
from clustering_analysis import clustering_analysis, clustering_analysis_multiclass


# ── Data Loading ──────────────────────────────────────────────────────────────

def load_data(fallback_csv="heart_disease_raw.csv"):
    print("=" * 55)
    print("LOADING DATASET — UCI Heart Disease (id=45)")
    print("=" * 55)
    try:
        dataset = fetch_ucirepo(id=45)
        X = dataset.data.features
        y = dataset.data.targets
        df = pd.concat([X, y], axis=1)
        print(f"\nFeatures  : {list(X.columns)}")
        print(f"Target    : {list(y.columns)}")
    except Exception as e:
        print(f"\n  Failed to fetch dataset ({e}). Falling back to '{fallback_csv}'.")
        df = pd.read_csv(fallback_csv)

    print(f"Shape     : {df.shape[0]} rows × {df.shape[1]} columns")
    return df


# ── ID Column + CSV Export ──────────────────────────────────────────────────

def add_id_column(df):
    if "ID" not in df.columns:
        df = df.copy()
        df.insert(0, "ID", range(1, len(df) + 1))
    return df


def export_to_csv(df, filename="heart_disease_raw.csv"):
    df.to_csv(filename, index=False)
    print(f"\nExported {len(df)} rows × {df.shape[1]} columns to '{filename}'")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    df = load_data()
    df = add_id_column(df)
    export_to_csv(df)
    descriptive_analysis(df)
    df = preprocess(df)
    exploratory_analysis(df)
    predictive_analysis(df)
    predictive_analysis_multiclass(df)
    clustering_analysis(df)
    clustering_analysis_multiclass(df)


if __name__ == "__main__":
    main()
