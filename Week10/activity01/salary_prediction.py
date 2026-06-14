import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ── Data Loading ──────────────────────────────────────────────────────────────

def load_data(path):
    df = pd.read_csv(path, index_col=0)
    print("=" * 50)
    print("DATASET OVERVIEW")
    print("=" * 50)
    print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(df.head())
    return df


# ── Data Preprocessing ────────────────────────────────────────────────────────

def preprocess(df):
    print("\n" + "=" * 50)
    print("DATA PREPROCESSING")
    print("=" * 50)

    # Null check
    nulls = df.isnull().sum()
    print(f"\nNull values per column:\n{nulls}")
    if nulls.any():
        df = df.dropna()
        print(f"  → Dropped rows with nulls. Remaining rows: {len(df)}")
    else:
        print("  → No null values found.")

    # Duplicate check
    dupes = df.duplicated().sum()
    print(f"\nDuplicate rows: {dupes}")
    if dupes > 0:
        df = df.drop_duplicates()
        print(f"  → Dropped {dupes} duplicate(s). Remaining rows: {len(df)}")
    else:
        print("  → No duplicates found.")

    # Anomaly / outlier check using IQR
    print("\nAnomaly check (IQR method):")
    for col in df.columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower) | (df[col] > upper)]
        if len(outliers) > 0:
            print(f"  {col}: {len(outliers)} outlier(s) detected "
                  f"(range: {lower:.2f} – {upper:.2f})")
            print(f"    Values: {outliers[col].tolist()}")
        else:
            print(f"  {col}: No outliers detected.")

    print(f"\nFinal dataset shape after preprocessing: {df.shape}")
    return df


# ── Error Metrics ─────────────────────────────────────────────────────────────

def print_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    print(f"\n  MAE  (Mean Absolute Error):       ${mae:,.2f}")
    print(f"  MSE  (Mean Squared Error):        ${mse:,.2f}")
    print(f"  RMSE (Root Mean Squared Error):   ${rmse:,.2f}")
    print(f"  R²   (Coefficient of Determination): {r2:.4f}")
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}


# ── Models ────────────────────────────────────────────────────────────────────

def train_linear(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def train_polynomial(X_train, y_train, degree=3):
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(X_train, y_train)
    return model


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot_results(X_train, X_test, y_train, y_test,
                 lin_model, poly_model, x_col):
    x_line = np.linspace(X_train[x_col].min(),
                         X_test[x_col].max(), 200).reshape(-1, 1)
    x_df = pd.DataFrame(x_line, columns=[x_col])

    plt.figure(figsize=(9, 5))
    plt.scatter(X_train, y_train, color="steelblue", label="Train", alpha=0.8)
    plt.scatter(X_test,  y_test,  color="orange",    label="Test",  alpha=0.8)
    plt.plot(x_line, lin_model.predict(x_df),
             color="red",   linewidth=2, label="Linear Regression")
    plt.plot(x_line, poly_model.predict(x_df),
             color="green", linewidth=2, label="Polynomial Regression (deg=3)",
             linestyle="--")
    plt.xlabel("Years of Experience")
    plt.ylabel("Salary")
    plt.title("Salary Prediction — Linear vs Polynomial Regression")
    plt.legend()
    plt.tight_layout()
    plt.savefig("salary_prediction.png", dpi=120)
    plt.close()
    print("\nPlot saved to salary_prediction.png")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    df = load_data("salary-dataset.csv")
    df = preprocess(df)

    X = df[["YearsExperience"]]
    y = df["Salary"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Linear Regression
    print("\n" + "=" * 50)
    print("LINEAR REGRESSION")
    print("=" * 50)
    lin_model = train_linear(X_train, y_train)
    lin_pred = lin_model.predict(X_test)
    print(f"  Coefficient (slope): {lin_model.coef_[0]:.2f}")
    print(f"  Intercept:           {lin_model.intercept_:.2f}")
    lin_metrics = print_metrics(y_test, lin_pred)

    # Polynomial Regression (degree 3)
    print("\n" + "=" * 50)
    print("POLYNOMIAL REGRESSION (degree=3)")
    print("=" * 50)
    poly_model = train_polynomial(X_train, y_train, degree=3)
    poly_pred = poly_model.predict(X_test)
    poly_metrics = print_metrics(y_test, poly_pred)

    # Comparison summary
    print("\n" + "=" * 50)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 50)
    print(f"{'Metric':<8} {'Linear':>14} {'Polynomial':>14}")
    print("-" * 38)
    for key in ["MAE", "MSE", "RMSE", "R2"]:
        print(
            f"{key:<8} {lin_metrics[key]:>14,.2f} {poly_metrics[key]:>14,.2f}")

    # Sample predictions
    print("\n" + "=" * 50)
    print("SAMPLE PREDICTIONS")
    print("=" * 50)
    print(f"{'Years':<8} {'Linear':>14} {'Polynomial':>14}")
    print("-" * 38)
    for years in [2, 5, 10, 15]:
        row = pd.DataFrame([[years]], columns=["YearsExperience"])
        lp = lin_model.predict(row)[0]
        pp = poly_model.predict(row)[0]
        print(f"{years:<8} ${lp:>13,.0f} ${pp:>13,.0f}")

    plot_results(X_train, X_test, y_train, y_test,
                 lin_model, poly_model, "YearsExperience")

    # Interactive salary estimator
    print("\n" + "=" * 50)
    print("SALARY ESTIMATOR")
    print("=" * 50)
    while True:
        raw = input(
            "\nEnter years of experience (e.g. 1.4), or 'q' to quit: ").strip()
        if raw.lower() == "q":
            print("Goodbye!")
            break
        try:
            years = float(raw)
            if years < 0:
                print("  Please enter a non-negative number.")
                continue
            row = pd.DataFrame([[years]], columns=["YearsExperience"])
            lp = lin_model.predict(row)[0]
            pp = poly_model.predict(row)[0]
            print(f"  Linear Regression estimate:     ${lp:,.0f}")
            print(f"  Polynomial Regression estimate: ${pp:,.0f}")
        except ValueError:
            print("  Invalid input — please enter a number.")


if __name__ == "__main__":
    main()
