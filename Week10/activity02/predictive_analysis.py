import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, classification_report)


# ── Predictive Analysis (Binary: disease present vs. not) ─────────────────────

def _plot_confusion_matrix(cm, labels, title, filename, output_dir):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, filename), dpi=120)
    plt.close(fig)


def _print_metrics(name, y_test, y_pred):
    print(f"\n--- {name} ---")
    print(f"  Accuracy  : {accuracy_score(y_test, y_pred):.3f}")
    print(f"  Precision : {precision_score(y_test, y_pred):.3f}")
    print(f"  Recall    : {recall_score(y_test, y_pred):.3f}")
    print(f"  F1 Score  : {f1_score(y_test, y_pred):.3f}")


def _print_classification_report(name, y_test, y_pred):
    print(f"\n--- {name} ---")
    print(f"  Accuracy : {accuracy_score(y_test, y_pred):.3f}")
    print(classification_report(y_test, y_pred, zero_division=0))


def _plot_feature_importance(rf, columns, title, filename, output_dir):
    importances = pd.Series(rf.feature_importances_, index=columns).sort_values()
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(importances.index, importances.values, color="steelblue", edgecolor="black")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, filename), dpi=120)
    plt.close(fig)


def predictive_analysis(df, output_dir="charts/predictive"):
    print("\n" + "=" * 55)
    print("PREDICTIVE ANALYSIS — Binary Classification (num > 0)")
    print("=" * 55)

    os.makedirs(output_dir, exist_ok=True)

    y = (df["num"] > 0).astype(int)
    X = df.drop(columns=["ID", "num"], errors="ignore")

    print(f"\nClass balance (0 = no disease, 1 = disease): {y.value_counts().sort_index().to_dict()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(X_train_scaled, y_train)
    y_pred_lr = log_reg.predict(X_test_scaled)
    _print_metrics("Logistic Regression", y_test, y_pred_lr)
    _plot_confusion_matrix(
        confusion_matrix(y_test, y_pred_lr), ["No Disease", "Disease"],
        "Logistic Regression — Confusion Matrix",
        "logistic_regression_confusion_matrix.png", output_dir)

    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    _print_metrics("Random Forest", y_test, y_pred_rf)
    _plot_confusion_matrix(
        confusion_matrix(y_test, y_pred_rf), ["No Disease", "Disease"],
        "Random Forest — Confusion Matrix",
        "random_forest_confusion_matrix.png", output_dir)

    _plot_feature_importance(rf, X.columns, "Random Forest — Feature Importance",
                              "feature_importance.png", output_dir)

    print(f"\nCharts saved to '{output_dir}/'")


# ── Predictive Analysis (5-Class: severity 0–4) ────────────────────────────────

def predictive_analysis_multiclass(df, output_dir="charts/predictive/multiclass"):
    print("\n" + "=" * 55)
    print("PREDICTIVE ANALYSIS — 5-Class Severity (num 0–4)")
    print("=" * 55)

    os.makedirs(output_dir, exist_ok=True)

    y = df["num"]
    X = df.drop(columns=["ID", "num"], errors="ignore")
    labels = sorted(y.unique())

    print(f"\nClass balance (num 0–4): {y.value_counts().sort_index().to_dict()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    log_reg = LogisticRegression(max_iter=2000)
    log_reg.fit(X_train_scaled, y_train)
    y_pred_lr = log_reg.predict(X_test_scaled)
    _print_classification_report("Logistic Regression", y_test, y_pred_lr)
    _plot_confusion_matrix(
        confusion_matrix(y_test, y_pred_lr, labels=labels), labels,
        "Logistic Regression — Confusion Matrix (5-Class)",
        "logistic_regression_confusion_matrix.png", output_dir)

    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    _print_classification_report("Random Forest", y_test, y_pred_rf)
    _plot_confusion_matrix(
        confusion_matrix(y_test, y_pred_rf, labels=labels), labels,
        "Random Forest — Confusion Matrix (5-Class)",
        "random_forest_confusion_matrix.png", output_dir)

    _plot_feature_importance(rf, X.columns, "Random Forest — Feature Importance (5-Class)",
                              "feature_importance.png", output_dir)

    print(f"\nCharts saved to '{output_dir}/'")
