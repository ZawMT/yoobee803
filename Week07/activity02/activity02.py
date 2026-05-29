# =============================================================
# Week 7 - Activity 2: Fraud Detection using SVM
# Dataset: Credit Card Fraud Detection (Kaggle - mlg-ulb)
# =============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
    precision_recall_curve, auc
)


# =============================================================
# SECTION FUNCTIONS
# =============================================================

def load_data(filepath):
    print("=" * 60)
    print("1. LOADING DATASET")
    print("=" * 60)
    df = pd.read_csv(filepath)
    print(f"Shape: {df.shape}")
    print(f"\nFirst 5 rows:\n{df.head()}")
    return df


def explore_data(df):
    print("\n" + "=" * 60)
    print("2. INITIAL EXPLORATION")
    print("=" * 60)
    print(f"\nData types:\n{df.dtypes}")
    print(f"\nBasic statistics:\n{df.describe()}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    print(f"\nDuplicate rows: {df.duplicated().sum()}")
    print(f"\nClass distribution:\n{df['Class'].value_counts()}")
    print(f"\nFraud percentage: {df['Class'].mean() * 100:.4f}%")


def clean_data(df):
    print("\n" + "=" * 60)
    print("3. DATA CLEANING")
    print("=" * 60)

    before = len(df)
    df.drop_duplicates(inplace=True)
    print(f"Removed {before - len(df)} duplicate row(s). Rows remaining: {len(df)}")

    before = len(df)
    df.dropna(inplace=True)
    print(f"Removed {before - len(df)} row(s) with missing values. Rows remaining: {len(df)}")

    df.to_csv("outputs/cleaned_dataset.csv", index=False)
    print("Cleaned dataset saved to outputs/cleaned_dataset.csv")
    return df


def run_eda(df):
    print("\n" + "=" * 60)
    print("4. EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    # --- 4.1 Class Distribution ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    class_counts = df['Class'].value_counts()
    axes[0].bar(['Legitimate (0)', 'Fraud (1)'], class_counts.values,
                color=['steelblue', 'tomato'], edgecolor='black')
    axes[0].set_title('Class Distribution (Count)')
    axes[0].set_ylabel('Number of Transactions')
    for i, v in enumerate(class_counts.values):
        axes[0].text(i, v + 500, f'{v:,}', ha='center', fontweight='bold')
    axes[1].pie(class_counts.values, labels=['Legitimate (0)', 'Fraud (1)'],
                colors=['steelblue', 'tomato'], autopct='%1.3f%%', startangle=90)
    axes[1].set_title('Class Distribution (Proportion)')
    plt.tight_layout()
    plt.savefig("outputs/1_eda_class_distribution.png", dpi=150)
    plt.close()
    print("Saved: outputs/1_eda_class_distribution.png")

    # --- 4.2 Transaction Amount by Class ---
    plt.figure(figsize=(10, 5))
    df.boxplot(column='Amount', by='Class', patch_artist=True,
               boxprops=dict(facecolor='lightblue'),
               medianprops=dict(color='red', linewidth=2))
    plt.suptitle('')
    plt.title('Transaction Amount by Class')
    plt.xlabel('Class (0 = Legitimate, 1 = Fraud)')
    plt.ylabel('Amount ($)')
    plt.tight_layout()
    plt.savefig("outputs/2_eda_amount_by_class.png", dpi=150)
    plt.close()
    print("Saved: outputs/2_eda_amount_by_class.png")

    # --- 4.3 Transaction Time Distribution by Class ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.hist(df[df['Class'] == 0]['Time'], bins=50, color='steelblue', edgecolor='white')
    ax1.set_title('Legitimate Transactions')
    ax1.set_xlabel('Time (seconds from first transaction)')
    ax1.set_ylabel('Count')

    ax2.hist(df[df['Class'] == 1]['Time'], bins=50, color='tomato', edgecolor='white')
    ax2.set_title('Fraudulent Transactions')
    ax2.set_xlabel('Time (seconds from first transaction)')
    ax2.set_ylabel('Count')

    fig.suptitle('Transaction Time Distribution by Class', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("outputs/3_eda_time_distribution.png", dpi=150)
    plt.close()
    print("Saved: outputs/3_eda_time_distribution.png")

    # --- 4.4 Correlation Heatmap ---
    plt.figure(figsize=(18, 14))
    sns.heatmap(df.corr(), cmap='coolwarm', center=0, linewidths=0.3,
                annot=False, fmt='.1f')
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig("outputs/4_eda_correlation_heatmap.png", dpi=150)
    plt.close()
    print("Saved: outputs/4_eda_correlation_heatmap.png")

    # --- 4.5 Top features correlated with Class ---
    corr_with_class = df.corr()['Class'].drop('Class').abs().sort_values(ascending=False)
    print(f"\nTop 10 features correlated with Class:\n{corr_with_class.head(10)}")
    plt.figure(figsize=(10, 6))
    corr_with_class.head(15).plot(kind='bar', color='steelblue', edgecolor='black')
    plt.title('Top 15 Features by Absolute Correlation with Class')
    plt.ylabel('Absolute Correlation')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("outputs/5_eda_feature_correlation_with_class.png", dpi=150)
    plt.close()
    print("Saved: outputs/5_eda_feature_correlation_with_class.png")


def select_features(df):
    print("\n" + "=" * 60)
    print("5. FEATURE SELECTION")
    print("=" * 60)

    features = df.drop(columns=['Class'])
    target   = df['Class']

    # --- 5.1 Near-zero variance check ---
    var_threshold = VarianceThreshold(threshold=0.01)
    var_threshold.fit(features)
    low_variance_cols = features.columns[~var_threshold.get_support()].tolist()
    if low_variance_cols:
        print(f"\nDropping {len(low_variance_cols)} near-zero variance feature(s): {low_variance_cols}")
        features = features.drop(columns=low_variance_cols)
    else:
        print("\nNo near-zero variance features found — all features retained.")

    # --- 5.2 Correlation-based selection ---
    corr_with_class = df.corr()['Class'].drop('Class').abs()
    CORR_THRESHOLD  = 0.01
    low_corr_cols   = corr_with_class[corr_with_class < CORR_THRESHOLD].index.tolist()
    low_corr_cols   = [c for c in low_corr_cols if c in features.columns]
    if low_corr_cols:
        print(f"\nDropping {len(low_corr_cols)} low-correlation feature(s) "
              f"(|corr| < {CORR_THRESHOLD}): {low_corr_cols}")
        features = features.drop(columns=low_corr_cols)
    else:
        print(f"\nNo features below correlation threshold ({CORR_THRESHOLD}) — all features retained.")

    print(f"\nFeatures before selection: {df.shape[1] - 1}")
    print(f"Features after  selection: {features.shape[1]}")
    print(f"Features kept: {features.columns.tolist()}")

    selected_corr = corr_with_class[features.columns].sort_values(ascending=False)
    plt.figure(figsize=(12, 5))
    selected_corr.plot(kind='bar', color='steelblue', edgecolor='black')
    plt.title('Selected Features — Absolute Correlation with Class')
    plt.ylabel('Absolute Correlation')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("outputs/6_feature_selection_correlation.png", dpi=150)
    plt.close()
    print("Saved: outputs/6_feature_selection_correlation.png")

    return features, target


def preprocess(features, target):
    print("\n" + "=" * 60)
    print("6. FEATURE ENGINEERING & PREPROCESSING")
    print("=" * 60)

    X = features.copy()
    y = target.copy()

    cols_to_scale = [c for c in ['Time', 'Amount'] if c in X.columns]
    if cols_to_scale:
        scaler = StandardScaler()
        X[cols_to_scale] = scaler.fit_transform(X[cols_to_scale])
        print(f"Scaled {cols_to_scale} using StandardScaler.")

    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Target vector shape:  {y.shape}")
    return X, y


def split_data(X, y):
    print("\n" + "=" * 60)
    print("7. TRAIN / TEST SPLIT")
    print("=" * 60)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training set:   {X_train.shape[0]:,} samples")
    print(f"Testing set:    {X_test.shape[0]:,} samples")
    print(f"Train fraud %:  {y_train.mean() * 100:.4f}%")
    print(f"Test  fraud %:  {y_test.mean() * 100:.4f}%")
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    print("\n" + "=" * 60)
    print("8. SVM MODEL TRAINING (LinearSVC)")
    print("=" * 60)

    # LinearSVC is used instead of SVC(kernel='rbf') for scalability
    # class_weight='balanced' handles the severe class imbalance automatically
    model = LinearSVC(class_weight='balanced', max_iter=2000, random_state=42)
    print("Training SVM model ... (this may take a moment on 280K+ rows)")
    model.fit(X_train, y_train)
    print("Training complete.")
    return model


def evaluate_model(model, X_test, y_test, fraud_rate):
    print("\n" + "=" * 60)
    print("9. EVALUATION")
    print("=" * 60)

    y_pred = model.predict(X_test)

    cm              = confusion_matrix(y_test, y_pred)
    TN, FP, FN, TP  = cm.ravel()
    accuracy        = accuracy_score(y_test, y_pred)
    precision       = precision_score(y_test, y_pred, zero_division=0)
    recall          = recall_score(y_test, y_pred)
    f1              = f1_score(y_test, y_pred)
    specificity     = TN / (TN + FP)

    print(f"\nConfusion Matrix:\n{cm}")
    print(f"\n  True Negatives  (TN): {TN:,}")
    print(f"  False Positives (FP): {FP:,}")
    print(f"  False Negatives (FN): {FN:,}")
    print(f"  True Positives  (TP): {TP:,}")
    print(f"\n--- Evaluation Metrics ---")
    print(f"  Accuracy:             {accuracy:.4f}  ({accuracy*100:.2f}%)")
    print(f"  Precision:            {precision:.4f}")
    print(f"  Recall (Sensitivity): {recall:.4f}")
    print(f"  Specificity:          {specificity:.4f}")
    print(f"  F1-Score:             {f1:.4f}")
    print(f"\nFull Classification Report:\n")
    print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Fraud']))

    # --- Confusion Matrix Plot ---
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted Legit', 'Predicted Fraud'],
                yticklabels=['Actual Legit', 'Actual Fraud'])
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig("outputs/7_evaluation_confusion_matrix.png", dpi=150)
    plt.close()
    print("Saved: outputs/7_evaluation_confusion_matrix.png")

    # --- Precision-Recall Curve (AUPRC) ---
    decision_scores            = model.decision_function(X_test)
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, decision_scores)
    auprc                      = auc(recall_vals, precision_vals)

    plt.figure(figsize=(8, 6))
    plt.plot(recall_vals, precision_vals, color='darkorange', lw=2,
             label=f'Precision-Recall curve (AUPRC = {auprc:.4f})')
    plt.axhline(y=fraud_rate, color='navy', linestyle='--',
                label=f'Baseline (fraud rate = {fraud_rate:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig("outputs/8_evaluation_precision_recall_curve.png", dpi=150)
    plt.close()
    print(f"Saved: outputs/8_evaluation_precision_recall_curve.png")
    print(f"\nAUPRC: {auprc:.4f}")

    # --- Metrics Summary Bar Chart ---
    metrics = {
        'Accuracy':    accuracy,
        'Precision':   precision,
        'Recall':      recall,
        'Specificity': specificity,
        'F1-Score':    f1,
        'AUPRC':       auprc,
    }
    plt.figure(figsize=(10, 6))
    bars = plt.bar(metrics.keys(), metrics.values(),
                   color=['#1565C0', '#90CAF9', '#0D47A1', '#42A5F5', '#1E88E5', '#BBDEFB'],
                   edgecolor='black')
    for bar, val in zip(bars, metrics.values()):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{val:.4f}', ha='center', fontsize=11, fontweight='bold')
    plt.ylim(0, 1.15)
    plt.title('Model Evaluation Metrics Summary')
    plt.ylabel('Score')
    plt.tight_layout()
    plt.savefig("outputs/9_evaluation_metrics_summary.png", dpi=150)
    plt.close()
    print("Saved: outputs/9_evaluation_metrics_summary.png")

    return metrics


def print_summary(df, y, metrics):
    print("\n" + "=" * 60)
    print("10. SUMMARY")
    print("=" * 60)
    print(f"  Model:       LinearSVC (class_weight='balanced')")
    print(f"  Dataset:     {len(df):,} transactions | {int(y.sum())} fraud cases")
    print(f"  Train/Test:  80% / 20% (stratified)")
    print(f"  Accuracy:    {metrics['Accuracy']*100:.2f}%")
    print(f"  Precision:   {metrics['Precision']:.4f}")
    print(f"  Recall:      {metrics['Recall']:.4f}")
    print(f"  Specificity: {metrics['Specificity']:.4f}")
    print(f"  F1-Score:    {metrics['F1-Score']:.4f}")
    print(f"  AUPRC:       {metrics['AUPRC']:.4f}")
    print("\nAll outputs saved to: outputs/")
    print("=" * 60)


# =============================================================
# MAIN
# =============================================================

def main():
    os.makedirs("outputs", exist_ok=True)

    df                          = load_data("Fraud Detection-dataset.csv")
    explore_data(df)
    df                          = clean_data(df)
    run_eda(df)
    features, target            = select_features(df)
    X, y                        = preprocess(features, target)
    X_train, X_test, y_train, y_test = split_data(X, y)
    model                       = train_model(X_train, y_train)
    metrics                     = evaluate_model(model, X_test, y_test, df['Class'].mean())
    print_summary(df, y, metrics)


if __name__ == "__main__":
    main()
