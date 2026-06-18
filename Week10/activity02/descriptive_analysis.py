import os
import math
import matplotlib.pyplot as plt

from data_preprocessing import valid_range_min_max


# ── Descriptive Analysis ──────────────────────────────────────────────────────

def _grid_layout(n, ncols=3):
    ncols = min(ncols, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.flatten() if n > 1 else [axes]
    for ax in axes[n:]:
        ax.axis("off")
    return fig, axes


def descriptive_analysis(df, output_dir="charts/descriptive"):
    print("\n" + "=" * 55)
    print("DESCRIPTIVE ANALYSIS")
    print("=" * 55)

    os.makedirs(output_dir, exist_ok=True)

    print(f"\nShape : {df.shape[0]} rows × {df.shape[1]} columns")

    numeric_cols = list(valid_range_min_max.keys())
    numeric_cols = [c for c in numeric_cols if c in df.columns]
    categorical_cols = [c for c in df.columns
                         if c not in numeric_cols and c not in ("ID", "num")]
    categorical_cols = [c for c in categorical_cols if c in df.columns]

    print("\n--- Numeric Summary ---")
    print(df[numeric_cols].describe().to_string())

    print("\n--- Categorical Value Counts ---")
    for col in categorical_cols:
        print(f"\n{col}:")
        print(df[col].value_counts(dropna=False).sort_index().to_string())

    if "num" in df.columns:
        print("\n--- Target ('num') Distribution ---")
        print(df["num"].value_counts(dropna=False).sort_index().to_string())

    # ── Numeric distributions
    fig, axes = _grid_layout(len(numeric_cols))
    for ax, col in zip(axes, numeric_cols):
        ax.hist(df[col].dropna(), bins=20, color="steelblue", edgecolor="black")
        ax.set_title(col)
    fig.suptitle("Numeric Feature Distributions")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "numeric_distributions.png"), dpi=120)
    plt.close(fig)

    # ── Categorical distributions
    fig, axes = _grid_layout(len(categorical_cols))
    for ax, col in zip(axes, categorical_cols):
        counts = df[col].value_counts(dropna=False).sort_index()
        ax.bar(counts.index.astype(str), counts.values, color="steelblue", edgecolor="black")
        ax.set_title(col)
    fig.suptitle("Categorical Feature Distributions")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "categorical_distributions.png"), dpi=120)
    plt.close(fig)

    # ── Target distribution
    if "num" in df.columns:
        counts = df["num"].value_counts(dropna=False).sort_index()
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(counts.index.astype(str), counts.values, color="steelblue", edgecolor="black")
        ax.set_title("Target ('num') Distribution")
        ax.set_xlabel("num")
        ax.set_ylabel("count")
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "target_distribution.png"), dpi=120)
        plt.close(fig)

    print(f"\nCharts saved to '{output_dir}/'")
