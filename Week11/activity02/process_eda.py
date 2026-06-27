"""
process_eda.py
Exploratory Data Analysis on the NZ Wellbeing dataset.
Charts saved to reports/charts/eda/. Report saved to reports/EDA.md.
"""

import importlib.util
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

CHARTS_DIR  = os.path.join("reports", "charts", "eda")
REPORT_PATH = os.path.join("reports", "EDA.md")

COLORS = {
    "blue":  "#4C72B0",
    "green": "#55A868",
    "red":   "#C44E52",
    "amber": "#DD8452",
    "purple":"#8172B2",
}


def _import_module(filename):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    spec = importlib.util.spec_from_file_location(filename, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _save(fig, filename):
    path = os.path.join(CHARTS_DIR, filename)
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    print(f"  >> Chart: {path}")
    return path


# ------------------------------------------------------------------
# Charts
# ------------------------------------------------------------------

def _chart_distribution(df_clean):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(df_clean["Estimate(percent)"], bins=40,
            color=COLORS["blue"], edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Estimate (%)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Wellbeing Estimates — all measures, all demographics, all years")
    _save(fig, "01_distribution.png")


def _chart_mean_by_year(df_clean):
    means = df_clean.groupby("Year")["Estimate(percent)"].mean()
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(means.index.astype(str), means.values,
                  color=[COLORS["blue"], COLORS["amber"], COLORS["green"]], width=0.5)
    for bar, v in zip(bars, means.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{v:.1f}%", ha="center", fontsize=10)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Year")
    ax.set_ylabel("Mean Estimate (%)")
    ax.set_title("Mean Wellbeing Estimate by Year")
    _save(fig, "02_mean_by_year.png")


def _chart_trend_direction(pivot):
    changes   = pivot[2018] - pivot[2014]
    threshold = 1.0
    counts = {
        "Increasing (> +1pp)": int((changes >  threshold).sum()),
        "Stable (±1pp)":       int((changes.abs() <= threshold).sum()),
        "Declining (< -1pp)":  int((changes < -threshold).sum()),
    }
    fig, ax = plt.subplots(figsize=(7, 3))
    bars = ax.barh(
        list(counts.keys()), list(counts.values()),
        color=[COLORS["green"], COLORS["purple"], COLORS["red"]],
    )
    for bar, v in zip(bars, counts.values()):
        ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
                str(v), va="center", fontsize=10)
    ax.set_xlabel("Number of series")
    ax.set_title("Trend Direction 2014→2018  (Total Population, threshold ±1pp)")
    _save(fig, "03_trend_direction.png")
    return counts


def _chart_top_bottom_measures(df_total):
    measure_means = (
        df_total.groupby("Wellbeing measure")["Estimate(percent)"]
        .mean().sort_values()
    )
    n = min(10, len(measure_means) // 2)
    combined = pd.concat([measure_means.head(n), measure_means.tail(n)]).drop_duplicates()
    colors = [COLORS["red"]] * n + [COLORS["green"]] * n
    colors = colors[:len(combined)]

    fig, ax = plt.subplots(figsize=(10, max(5, len(combined) * 0.4)))
    combined.plot.barh(ax=ax, color=colors)
    ax.axvline(x=50, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Mean Estimate (%)")
    ax.set_title(f"Top & Bottom {n} Wellbeing Measures by Mean Estimate\n(Total Population, 2014–2018 | green=highest, red=lowest)")
    _save(fig, "04_top_bottom_measures.png")


def _chart_most_changed(pivot):
    change_df = (pivot[2018] - pivot[2014]).reset_index()
    change_df.columns = ["Measure", "Category", "Type", "Change"]
    change_df["AbsChange"] = change_df["Change"].abs()
    top = change_df.nlargest(15, "AbsChange")
    labels = top["Measure"] + "\n[" + top["Category"] + "]"
    colors = [COLORS["green"] if c > 0 else COLORS["red"] for c in top["Change"]]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(labels, top["Change"], color=colors)
    ax.axvline(x=0, color="black", linewidth=0.8)
    ax.set_xlabel("Change in Estimate (percentage points)")
    ax.set_title("Top 15 Most Changed Series 2014→2018\n(Total Population | green=increase, red=decline)")
    _save(fig, "05_most_changed.png")


def _chart_demographic_comparison(df_clean):
    demo_means = (
        df_clean[df_clean["Demographic"] != "Total population"]
        .groupby(["Demographic", "Demographic category"])["Estimate(percent)"]
        .mean()
        .reset_index()
        .sort_values("Estimate(percent)")
    )
    label = demo_means["Demographic category"] + " (" + demo_means["Demographic"] + ")"
    overall_mean = demo_means["Estimate(percent)"].mean()

    fig, ax = plt.subplots(figsize=(10, max(6, len(demo_means) * 0.28)))
    ax.barh(label, demo_means["Estimate(percent)"], color=COLORS["blue"])
    ax.axvline(x=overall_mean, color=COLORS["red"], linestyle="--", linewidth=0.9,
               label=f"Mean: {overall_mean:.1f}%")
    ax.set_xlabel("Mean Estimate (%)")
    ax.set_title("Mean Wellbeing Estimate by Demographic Category\n(all measures, all years)")
    ax.legend()
    _save(fig, "06_demographic_comparison.png")


def _chart_period_changes(pivot):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, (y1, y2) in zip(axes, [(2014, 2016), (2016, 2018)]):
        changes = pivot[y2] - pivot[y1]
        ax.hist(changes, bins=30, color=COLORS["blue"], edgecolor="white", linewidth=0.5)
        ax.axvline(x=0, color=COLORS["red"], linewidth=1)
        ax.set_xlabel("Change (pp)")
        ax.set_ylabel("Number of series")
        ax.set_title(f"{y1} → {y2}")
    fig.suptitle("Period-on-Period Change Distribution  (Total Population)", fontsize=12)
    fig.tight_layout()
    _save(fig, "07_period_changes.png")


# ------------------------------------------------------------------
# Report
# ------------------------------------------------------------------

def _write_report(df_clean, df_total, pivot, trend_counts):
    total_series = len(pivot)
    trend_table = "\n".join(
        f"| {k} | {v} | {100*v/total_series:.1f}% |"
        for k, v in trend_counts.items()
    )
    chart_list = "\n".join(
        f"![{name}](charts/eda/{name})"
        for name in [
            "01_distribution.png",
            "02_mean_by_year.png",
            "03_trend_direction.png",
            "04_top_bottom_measures.png",
            "05_most_changed.png",
            "06_demographic_comparison.png",
            "07_period_changes.png",
        ]
    )

    content = f"""# Exploratory Data Analysis

## Dataset Overview

| Metric | Value |
| --- | --- |
| Records after cleaning | {len(df_clean)} |
| Unique wellbeing measures | {df_clean["Wellbeing measure"].nunique()} |
| Unique demographics | {df_clean["Demographic"].nunique()} |
| Unique demographic categories | {df_clean["Demographic category"].nunique()} |
| Years | {", ".join(str(y) for y in sorted(df_clean["Year"].unique()))} |
| Total Population series | {total_series} |

---

## Trend Analysis — Total Population (2014→2018, threshold ±1pp)

| Direction | Series | % |
| --- | --- | --- |
{trend_table}

---

## Charts

{chart_list}
"""
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  >> Report: {REPORT_PATH}")


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

def run():
    os.makedirs(CHARTS_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)

    preprocessing = _import_module("preprocessing.py")
    df = preprocessing.load_and_preprocess()

    df_clean = df[~df["is_unreliable"]].copy()
    df_total = df_clean[
        (df_clean["Demographic"] == "Total population") &
        (df_clean["Demographic category"] == "Total population")
    ].copy()

    pivot = df_total.pivot_table(
        index=["Wellbeing measure", "Wellbeing measure category", "Type"],
        columns="Year",
        values="Estimate(percent)",
    )
    pivot = pivot[[2014, 2016, 2018]].dropna()

    print("\n" + "=" * 60)
    print("EDA: GENERATING CHARTS")
    print("=" * 60)

    _chart_distribution(df_clean)
    _chart_mean_by_year(df_clean)
    trend_counts = _chart_trend_direction(pivot)
    _chart_top_bottom_measures(df_total)
    _chart_most_changed(pivot)
    _chart_demographic_comparison(df_clean)
    _chart_period_changes(pivot)

    _write_report(df_clean, df_total, pivot, trend_counts)

    print("\n" + "=" * 60)
    print("EDA COMPLETE")
    print("=" * 60)
    return df_clean


if __name__ == "__main__":
    run()
