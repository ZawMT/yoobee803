"""
Week 9 - Activity 2: User Knowledge Modeling Dataset
Pipeline: Load → Preprocess → EDA → Clustering → Classification → Report
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import xgboost as xgb

CHARTS_DIR = "charts"
REPORT_PATH = "report.md"
os.makedirs(CHARTS_DIR, exist_ok=True)

sns.set_theme(style="whitegrid")
np.random.seed(42)


def save_chart(fig, filename):
    path = os.path.join(CHARTS_DIR, filename)
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    print(f"  >> Chart: {path}")
    return path


# ==============================================================
# STEP 1: LOAD DATASET
# ==============================================================
def step1_load():
    print("\n" + "=" * 60)
    print("STEP 1: LOAD DATASET")
    print("=" * 60)

    dataset = fetch_ucirepo(id=257)
    X = dataset.data.features
    y = dataset.data.targets.squeeze()

    df = pd.concat([X, y], axis=1)
    df.to_csv("data_downloaded.csv", index=False)
    print(f"  >> Data saved: data_downloaded.csv")

    print(f"Features shape : {X.shape}")
    print(f"Target shape   : {y.shape}")
    print(f"\nFeature columns: {list(X.columns)}")
    print(f"Target column  : {y.name}")
    print(f"Target classes : {sorted(y.unique())}")
    print(f"\nFirst 5 rows:")
    print(df.head().to_string())

    return X, y


# ==============================================================
# STEP 2: PREPROCESSING
# ==============================================================
def step2_preprocess(X, y):
    print("\n" + "=" * 60)
    print("STEP 2: PREPROCESSING")
    print("=" * 60)

    df = pd.concat([X, y], axis=1)

    # Normalise target labels — fix inconsistent casing/underscores
    target_col = y.name
    df[target_col] = (
        df[target_col]
        .str.strip()
        .str.replace("_", " ", regex=False)
        .str.title()
    )
    print(f"Normalised target labels: {sorted(df[target_col].unique())}")

    # Null check
    null_counts = df.isnull().sum()
    print(f"\nNull values:\n{null_counts.to_string()}")

    # Duplicate check
    dupes = df.duplicated().sum()
    print(f"\nDuplicate rows: {dupes}")

    # Drop duplicates if any
    if dupes > 0:
        df = df.drop_duplicates().reset_index(drop=True)
        print(f"  Dropped {dupes} duplicates. Remaining: {len(df)} rows")

    X_clean = df[X.columns].copy()
    y_clean = df[y.name].copy()

    # Class distribution
    print(f"\nClass distribution:")
    for cls, cnt in y_clean.value_counts().items():
        print(f"  {cls:12s}: {cnt:4d}  ({100*cnt/len(y_clean):.1f}%)")

    # Feature stats
    print(f"\nFeature statistics:")
    print(X_clean.describe().round(3).to_string())

    # Encode target
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_clean)
    print(f"\nLabel encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)
    X_scaled = pd.DataFrame(X_scaled, columns=X_clean.columns)

    print(f"\nPreprocessing complete.")
    print(f"  Rows : {len(X_clean)}")
    print(f"  Features : {list(X_clean.columns)}")
    print(f"  Target classes : {list(le.classes_)}")

    return X_clean, X_scaled, y_clean, y_encoded, le, scaler


# ==============================================================
# STEP 3: EDA
# ==============================================================
def step3_eda(X_clean, y_clean):
    print("\n" + "=" * 60)
    print("STEP 3: EDA")
    print("=" * 60)

    features = list(X_clean.columns)

    # --- Chart 1: Class balance ---
    fig, ax = plt.subplots(figsize=(6, 4))
    order = ["Very Low", "Low", "Middle", "High"]
    order = [c for c in order if c in y_clean.unique()]
    counts = y_clean.value_counts().reindex(order)
    ax.bar(counts.index, counts.values,
           color=["#C44E52", "#DD8452", "#4C72B0", "#55A868"])
    for i, v in enumerate(counts.values):
        ax.text(i, v + 1, str(v), ha="center", fontsize=10)
    ax.set_xlabel("Knowledge Level")
    ax.set_ylabel("Count")
    ax.set_title("Class Distribution — User Knowledge Level")
    save_chart(fig, "01_class_balance.png")

    # --- Chart 2: Feature distributions ---
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    axes = axes.flatten()
    for i, col in enumerate(features):
        axes[i].hist(X_clean[col], bins=25, color="#4C72B0",
                     edgecolor="white", linewidth=0.5)
        axes[i].set_title(col)
        axes[i].set_xlabel("Value")
        axes[i].set_ylabel("Count")
    for j in range(len(features), len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Feature Distributions", fontsize=13)
    fig.tight_layout()
    save_chart(fig, "02_feature_distributions.png")

    # --- Chart 3: Boxplots by class ---
    order = [c for c in ["Very Low", "Low", "Middle", "High"]
             if c in y_clean.unique()]
    df_plot = pd.concat([X_clean, y_clean], axis=1)
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    axes = axes.flatten()
    for i, col in enumerate(features):
        sns.boxplot(data=df_plot, x=y_clean.name, y=col,
                    order=order, ax=axes[i], palette="Set2")
        axes[i].set_title(col)
        axes[i].set_xlabel("Knowledge Level")
    for j in range(len(features), len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Feature Distribution by Knowledge Level", fontsize=13)
    fig.tight_layout()
    save_chart(fig, "03_boxplots_by_class.png")

    # --- Chart 4: Correlation heatmap ---
    fig, ax = plt.subplots(figsize=(7, 5))
    corr = X_clean.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, ax=ax, linewidths=0.5)
    ax.set_title("Feature Correlation Heatmap")
    save_chart(fig, "04_correlation_heatmap.png")

    print("  EDA charts saved.")


# ==============================================================
# STEP 4: CLUSTERING
# ==============================================================
def step4_clustering(X_scaled, y_clean):
    print("\n" + "=" * 60)
    print("STEP 4: CLUSTERING (K-Means)")
    print("=" * 60)

    # Elbow method
    inertias = []
    k_range = range(2, 11)
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertias.append(km.inertia_)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(list(k_range), inertias, marker="o", color="#4C72B0")
    ax.axvline(x=4, color="#C44E52", linestyle="--", linewidth=1,
               label="k=4 (matches class count)")
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("Inertia")
    ax.set_title("Elbow Method — Optimal k")
    ax.legend()
    save_chart(fig, "05_elbow.png")

    # Fit with k=4 (matches the 4 knowledge levels)
    k_optimal = 4
    km = KMeans(n_clusters=k_optimal, random_state=42, n_init=10)
    cluster_labels = km.fit_predict(X_scaled)

    print(f"  Fitted K-Means with k={k_optimal}")
    print(f"  Cluster sizes: {dict(zip(*np.unique(cluster_labels, return_counts=True)))}")

    # PCA 2D visualisation
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    explained = pca.explained_variance_ratio_

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Clusters
    cluster_colors = ["#C44E52", "#DD8452", "#4C72B0", "#55A868"]
    for cid in range(k_optimal):
        mask = cluster_labels == cid
        axes[0].scatter(X_pca[mask, 0], X_pca[mask, 1],
                        label=f"Cluster {cid}",
                        color=cluster_colors[cid], s=30, alpha=0.7)
    axes[0].set_title(f"K-Means Clusters (k={k_optimal})")
    axes[0].set_xlabel(f"PC1 ({explained[0]*100:.1f}%)")
    axes[0].set_ylabel(f"PC2 ({explained[1]*100:.1f}%)")
    axes[0].legend()

    # Actual labels
    order = ["Very Low", "Low", "Middle", "High"]
    order = [c for c in order if c in y_clean.unique()]
    colors_map = {"Very Low": "#C44E52", "Low": "#DD8452",
                  "Middle": "#4C72B0", "High": "#55A868"}
    for cls in order:
        mask = y_clean == cls
        axes[1].scatter(X_pca[mask, 0], X_pca[mask, 1],
                        label=cls, color=colors_map.get(cls, "grey"),
                        s=30, alpha=0.7)
    axes[1].set_title("Actual Knowledge Levels (PCA)")
    axes[1].set_xlabel(f"PC1 ({explained[0]*100:.1f}%)")
    axes[1].set_ylabel(f"PC2 ({explained[1]*100:.1f}%)")
    axes[1].legend()

    fig.suptitle("Clustering vs Actual Labels (PCA 2D projection)", fontsize=12)
    fig.tight_layout()
    save_chart(fig, "06_clustering_pca.png")

    return cluster_labels


# ==============================================================
# STEP 5: CLASSIFICATION
# ==============================================================
def step5_classification(X_scaled, y_encoded, le):
    print("\n" + "=" * 60)
    print("STEP 5: CLASSIFICATION")
    print("=" * 60)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    print(f"  Train: {len(X_train)}  |  Test: {len(X_test)}")

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "KNN":                 KNeighborsClassifier(n_neighbors=5),
        "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost":             xgb.XGBClassifier(n_estimators=100, random_state=42,
                                                  verbosity=0, eval_metric="mlogloss"),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results[name] = {"model": model, "acc": acc, "y_pred": y_pred}
        print(f"  {name:22s}: Accuracy = {acc:.4f}")

    # --- Chart: Model accuracy comparison ---
    fig, ax = plt.subplots(figsize=(8, 4))
    names = list(results.keys())
    accs  = [results[n]["acc"] for n in names]
    bars  = ax.barh(names, accs, color="#4C72B0")
    for bar, v in zip(bars, accs):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{v:.4f}", va="center", fontsize=10)
    ax.set_xlim(0, 1.1)
    ax.set_xlabel("Accuracy")
    ax.set_title("Model Accuracy Comparison")
    save_chart(fig, "07_model_accuracy.png")

    # Best model
    best_name = max(results, key=lambda n: results[n]["acc"])
    best      = results[best_name]
    print(f"\n  Best model: {best_name} (accuracy={best['acc']:.4f})")

    # --- Chart: Confusion matrix for best model ---
    cm  = confusion_matrix(y_test, best["y_pred"])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(cm, display_labels=le.classes_)
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion Matrix — {best_name}")
    save_chart(fig, "08_confusion_matrix.png")

    print(f"\n  Classification report ({best_name}):")
    print(classification_report(y_test, best["y_pred"],
                                target_names=le.classes_))

    # Feature importance (Random Forest)
    rf = results["Random Forest"]["model"]
    importances = pd.Series(rf.feature_importances_,
                            index=X_scaled.columns).sort_values()
    fig, ax = plt.subplots(figsize=(7, 4))
    importances.plot.barh(ax=ax, color="#4C72B0")
    ax.set_title("Feature Importance — Random Forest")
    ax.set_xlabel("Importance")
    save_chart(fig, "09_feature_importance.png")

    return results, best_name


# ==============================================================
# STEP 6: REPORT
# ==============================================================
def step6_report(X_clean, y_clean, results, best_name):
    print("\n" + "=" * 60)
    print("STEP 6: REPORT")
    print("=" * 60)

    class_dist = "\n".join(
        f"| {cls} | {cnt} | {100*cnt/len(y_clean):.1f}% |"
        for cls, cnt in y_clean.value_counts().items()
    )
    model_rows = "\n".join(
        f"| {name} | {res['acc']:.4f} |{'  **Best**' if name == best_name else ''}"
        for name, res in results.items()
    )
    charts = "\n".join(
        f"![{name}](charts/{name})"
        for name in [
            "01_class_balance.png",
            "02_feature_distributions.png",
            "03_boxplots_by_class.png",
            "04_correlation_heatmap.png",
            "05_elbow.png",
            "06_clustering_pca.png",
            "07_model_accuracy.png",
            "08_confusion_matrix.png",
            "09_feature_importance.png",
        ]
    )

    null_total = X_clean.isnull().sum().sum()
    dupe_count = len(X_clean) - len(X_clean.drop_duplicates())

    content = f"""# User Knowledge Modeling — Analysis Report

## Dataset Overview

| Property | Value |
| --- | --- |
| Source | UCI ML Repository #257 |
| Rows | {len(X_clean)} |
| Features | {len(X_clean.columns)} numerical |
| Target | UNS (User Knowledge Level) |
| Classes | Very Low, Low, Middle, High |

## Class Distribution (after label normalisation)

| Class | Count | % |
| --- | --- | --- |
{class_dist}

## Preprocessing

| Step | Result |
| --- | --- |
| Null values | {null_total} found |
| Duplicate rows | {dupe_count} found |
| Label normalisation | Unified `very_low` → `Very Low` |
| Feature scaling | StandardScaler applied |
| Target encoding | LabelEncoder applied |

## Clustering — K-Means

- Elbow method used to identify optimal k
- k=4 selected (matches number of knowledge level classes)
- PCA used for 2D visualisation of clusters vs actual labels

## Classification Results

| Model | Accuracy |
| --- | --- |
{model_rows}

**Best model: {best_name}**

## Charts

{charts}
"""
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  >> Report: {REPORT_PATH}")


# ==============================================================
# MAIN
# ==============================================================
if __name__ == "__main__":
    print("\n*** USER KNOWLEDGE MODELING PIPELINE ***")

    X, y                                    = step1_load()
    X_clean, X_scaled, y_clean, \
        y_encoded, le, scaler               = step2_preprocess(X, y)
    step3_eda(X_clean, y_clean)
    cluster_labels                          = step4_clustering(X_scaled, y_clean)
    results, best_name                      = step5_classification(X_scaled, y_encoded, le)
    step6_report(X_clean, y_clean, results, best_name)

    print("\n*** PIPELINE COMPLETE ***")
    print(f"  Charts : {CHARTS_DIR}/")
    print(f"  Report : {REPORT_PATH}")
