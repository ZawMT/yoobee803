import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

CHARTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "charts")


def _save(fig, filename):
    os.makedirs(CHARTS_DIR, exist_ok=True)
    path = os.path.join(CHARTS_DIR, filename)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[STATUS] Saved chart -> {path}")


def plot_time_series(df):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["Month"], df["Passengers"], color="tab:blue")
    ax.set_title("Monthly Airline Passengers Over Time")
    ax.set_xlabel("Month")
    ax.set_ylabel("Passengers")
    _save(fig, "01_time_series.png")


def plot_yearly_trend(df):
    yearly = df.groupby(df["Month"].dt.year)["Passengers"].sum()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(yearly.index.astype(str), yearly.values, color="tab:orange")
    ax.set_title("Total Passengers per Year")
    ax.set_xlabel("Year")
    ax.set_ylabel("Total Passengers")
    _save(fig, "02_yearly_trend.png")


def plot_monthly_seasonality(df):
    data = df.copy()
    data["MonthName"] = data["Month"].dt.strftime("%b")
    order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
             "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=data, x="MonthName", y="Passengers", order=order, ax=ax)
    ax.set_title("Passenger Distribution by Month (Seasonality)")
    ax.set_xlabel("Month")
    ax.set_ylabel("Passengers")
    _save(fig, "03_monthly_seasonality.png")


def plot_distribution(df):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df["Passengers"], kde=True, color="tab:green", ax=ax)
    ax.set_title("Distribution of Passenger Counts")
    ax.set_xlabel("Passengers")
    ax.set_ylabel("Frequency")
    _save(fig, "04_distribution.png")


def plot_rolling_average(df, window=12):
    data = df.set_index("Month")["Passengers"]
    rolling_mean = data.rolling(window=window).mean()
    rolling_std = data.rolling(window=window).std()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(data.index, data.values, label="Passengers", alpha=0.5)
    ax.plot(rolling_mean.index, rolling_mean.values, label=f"{window}-Month Rolling Mean", color="tab:red")
    ax.plot(rolling_std.index, rolling_std.values, label=f"{window}-Month Rolling Std", color="tab:purple")
    ax.set_title("Rolling Mean & Standard Deviation")
    ax.set_xlabel("Month")
    ax.set_ylabel("Passengers")
    ax.legend()
    _save(fig, "05_rolling_average.png")


def plot_month_year_heatmap(df):
    data = df.copy()
    data["Year"] = data["Month"].dt.year
    data["MonthNum"] = data["Month"].dt.month
    pivot = data.pivot(index="Year", columns="MonthNum", values="Passengers")

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(pivot, cmap="YlGnBu", annot=True, fmt=".0f", ax=ax)
    ax.set_title("Passengers Heatmap (Year vs Month)")
    ax.set_xlabel("Month")
    ax.set_ylabel("Year")
    _save(fig, "06_year_month_heatmap.png")


def generate_all_charts(df):
    """Run all descriptive visualizations for the given preprocessed DataFrame."""
    print("\n[STATUS] Generating descriptive visualisation charts...")
    plot_time_series(df)
    plot_yearly_trend(df)
    plot_monthly_seasonality(df)
    plot_distribution(df)
    plot_rolling_average(df)
    plot_month_year_heatmap(df)
    print(f"[STATUS] All charts saved under '{CHARTS_DIR}'.")


if __name__ == "__main__":
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "airline-passengers.csv")
    raw_df = pd.read_csv(csv_path)
    raw_df["Month"] = pd.to_datetime(raw_df["Month"], format="%Y-%m")
    generate_all_charts(raw_df)
