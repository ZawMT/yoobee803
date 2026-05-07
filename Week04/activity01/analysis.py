import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(filepath):
    df = pd.read_csv(filepath)
    df.index += 1
    return df


def check_missing(df):
    print("\n--- Missing Values (per column) ---")
    print(df.isnull().sum())
    if df.isnull().any().any():
        print("WARNING: Missing values detected!")
    else:
        print("No missing values found.")


def check_statistics(df):
    print("\n--- Basic Statistics ---")
    print(df.describe())


def check_duplicates(df):
    print("\n--- Duplicate Rows ---")
    print(df.duplicated())
    dup_count = df.duplicated().sum()
    print(f"Number of duplicate rows: {dup_count}")
    if dup_count > 0:
        print("WARNING: Duplicates detected!")
    else:
        print("No duplicates found.")


def check_outliers(df):
    print("\n--- Outliers (IQR method, per column) ---")
    numeric_df = df.select_dtypes(include="number")
    Q1 = numeric_df.quantile(0.25)
    Q3 = numeric_df.quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((numeric_df < (Q1 - 1.5 * IQR)) | (numeric_df > (Q3 + 1.5 * IQR)))
    print(outliers.sum())
    if outliers.any().any():
        print("WARNING: Outliers detected in one or more columns!")
    else:
        print("No outliers detected.")


def check_dtypes(df):
    print("\n--- Data Types ---")
    print(df.dtypes)


def plot_horizontal_bar(df):
    sorted_df = df.sort_values("Happiness_Score")
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(sorted_df["Country"], sorted_df["Happiness_Score"], color="steelblue")
    ax.set_xlabel("Happiness Score")
    ax.set_title("Happiness Score by Country")
    plt.tight_layout()
    plt.savefig("charts/happiness_bar.png")
    plt.close()
    print("Saved: charts/happiness_bar.png")


def plot_heatmap(df):
    numeric_df = df.select_dtypes(include="number")
    correlation = numeric_df.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig("charts/correlation_heatmap.png")
    plt.close()
    print("Saved: charts/correlation_heatmap.png")


def main():
    df = load_data("../data/world_happiness_dataset.csv")

    print("\nWorld Happiness Dataset:\n")
    print(df)

    check_missing(df)
    check_statistics(df)
    check_duplicates(df)
    check_outliers(df)
    check_dtypes(df)

    plot_horizontal_bar(df)
    plot_heatmap(df)


if __name__ == "__main__":
    main()
