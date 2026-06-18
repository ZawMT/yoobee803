import os
import matplotlib.pyplot as plt

from data_preprocessing import valid_range_min_max
from descriptive_analysis import _grid_layout


# ── Exploratory Analysis ───────────────────────────────────────────────────────

def _plot_corr_heatmap(df, cols, title, filename, output_dir):
    corr = df[cols].corr()
    print(f"\n--- {title} ---")
    print(corr.round(2).to_string())

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right")
    ax.set_yticks(range(len(cols)))
    ax.set_yticklabels(cols)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, filename), dpi=120)
    plt.close(fig)


def exploratory_analysis(df, output_dir="charts/exploratory"):
    print("\n" + "=" * 55)
    print("EXPLORATORY ANALYSIS")
    print("=" * 55)

    os.makedirs(output_dir, exist_ok=True)

    numeric_cols = [c for c in valid_range_min_max if c in df.columns]
    categorical_cols = [c for c in df.columns
                         if c not in numeric_cols and c not in ("ID", "num")]
    has_target = "num" in df.columns

    # ── Correlation heatmaps
    _plot_corr_heatmap(
        df, numeric_cols + (["num"] if has_target else []),
        "Correlation Heatmap — Continuous Features",
        "correlation_heatmap_continuous.png", output_dir)

    _plot_corr_heatmap(
        df, categorical_cols + (["num"] if has_target else []),
        "Correlation Heatmap — Categorical Features",
        "correlation_heatmap_categorical.png", output_dir)

    if has_target:
        target_levels = sorted(df["num"].dropna().unique())

        # ── Numeric features vs target
        fig, axes = _grid_layout(len(numeric_cols))
        for ax, col in zip(axes, numeric_cols):
            groups = [df.loc[df["num"] == lvl, col].dropna() for lvl in target_levels]
            ax.boxplot(groups, labels=target_levels)
            ax.set_title(col)
            ax.set_xlabel("num")
        fig.suptitle("Numeric Features vs Target")
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "numeric_vs_target.png"), dpi=120)
        plt.close(fig)

        # ── Categorical features vs target (disease rate)
        fig, axes = _grid_layout(len(categorical_cols))
        for ax, col in zip(axes, categorical_cols):
            rate = df.groupby(col)["num"].apply(lambda s: (s > 0).mean())
            ax.bar(rate.index.astype(str), rate.values, color="steelblue", edgecolor="black")
            ax.set_title(col)
            ax.set_ylabel("disease rate")
        fig.suptitle("Categorical Features vs Target (Disease Rate)")
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "categorical_vs_target.png"), dpi=120)
        plt.close(fig)

    print(f"\nCharts saved to '{output_dir}/'")
