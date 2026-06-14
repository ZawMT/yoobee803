## Error Metrics — MAE, MSE, RMSE, and R²

Error metrics measure how well a regression model's predictions match the actual values. Lower error = better model (except R², where higher = better).

#### MAE — Mean Absolute Error

**Formula:** average of |actual − predicted|

- Takes the absolute difference between each actual and predicted value, then averages them
- Easy to interpret: it's in the **same unit as the target** (e.g. dollars)
- Treats all errors equally — a $1,000 error counts the same whether it's a small or large prediction

**Example:** MAE = $5,000 means on average the model is off by $5,000 per prediction.

#### MSE — Mean Squared Error

**Formula:** average of (actual − predicted)²

- Squares each error before averaging, so **large errors are penalised much more** than small ones
- The unit is squared (e.g. dollars²), which makes it hard to interpret directly
- Useful during model training because it is mathematically smooth and easy to optimise

**Example:** MSE = 25,000,000 means the average squared error is $25M — harder to read, but large outlier predictions will push this number up significantly.

#### RMSE — Root Mean Squared Error

**Formula:** √MSE

- Simply the square root of MSE, which brings it **back to the same unit as the target**
- Still penalises large errors more than MAE, but is now human-readable like MAE
- Most commonly used metric for regression because it balances interpretability and sensitivity to outliers

**Example:** RMSE = $5,000 means the typical prediction error is around $5,000, but the model is more sensitive to big misses than MAE.

> **MAE vs RMSE:** If RMSE is much larger than MAE, it means the model has a few large errors (outliers in prediction). If they are close, errors are fairly consistent.

#### R² — Coefficient of Determination

**Formula:** 1 − (SS_residual / SS_total)

- Measures **how much of the variance in the target is explained by the model**
- Ranges from 0 to 1 (can go negative for very bad models)
  - **R² = 1.0** → perfect predictions
  - **R² = 0.9** → model explains 90% of the variation in salary
  - **R² = 0.0** → model is no better than just predicting the mean every time
- Unlike MAE/MSE/RMSE, higher is better

**Example:** R² = 0.95 means 95% of the salary variation is explained by years of experience — a strong fit.

#### Quick Reference Summary

| Metric | Unit | Penalises outliers? | Higher is better? |
|--------|------|---------------------|-------------------|
| MAE    | Same as target | No (equal weight) | No — lower is better |
| MSE    | Squared | Yes (heavily) | No — lower is better |
| RMSE   | Same as target | Yes (moderately) | No — lower is better |
| R²     | None (0–1) | N/A | Yes |