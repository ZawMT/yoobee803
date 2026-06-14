## Polynomial Regression — What does `degree` (deg) mean?
In polynomial regression, **degree** controls the highest power of the input feature used to fit the curve.
With `YearsExperience` as `x`:

| Degree | Equation fitted |
|--------|----------------|
| 1 (Linear) | `salary = a·x + b` |
| 2 | `salary = a·x² + b·x + c` |
| 3 | `salary = a·x³ + b·x² + c·x + d` |

So **deg=3** means the model fits a **cubic curve** — it has one more bend than a quadratic (deg=2), which lets it capture more complex patterns in the data.

**Why deg=3 for salary data?**
The salary data is not perfectly linear — early-career salary gains are slower, then accelerate, then level off at senior levels. A cubic curve can capture that S-like shape better than a straight line.

**Trade-off:**
Higher degrees = more flexible curve, but too high (e.g. deg=10) causes **overfitting** — the curve wiggles to hit every training point but predicts poorly on new data. Degree 3 is a common starting point that balances fit vs. generalisability.

Degree is a hyperparameter — chosen before training, not something the model calculates from the data. The model learns the coefficients (a, b, c, d) automatically during training, but the degree is set upfront. Common ways to choose the right degree:    
- Try a few values (2, 3, 4) and compare R² / RMSE on the test set — pick the one that performs best   
- Cross-validation — more rigorous version of the above    
- Visual inspection — plot the curve and see if it fits the shape of the data   
In this case, deg=3 is a reasonable starting guess for salary data, but you could test deg=2 or deg=4 and compare the error metrics to see if another degree performs better.