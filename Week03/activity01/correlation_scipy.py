import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

df = pd.read_csv('age_networth.csv')

print(df.head())

r, p_value = stats.pearsonr(df['Age'], df['Net Worth'])
print(f"Pearson Correlation Coefficient: {r:.5f}")
print(f"P-value: {p_value:.5f}")

slope, intercept, r_value, p_value, std_err = stats.linregress(df['Age'], df['Net Worth'])

plt.figure(figsize=(8, 5))
plt.scatter(df['Age'], df['Net Worth'], color='steelblue', label='Data points')
plt.plot(df['Age'], slope * df['Age'] + intercept, color='red', label=f'Regression line (r = {r:.2f})')
plt.xlabel('Age')
plt.ylabel('Net Worth')
plt.title('Linear Relationship between Age and Net Worth (SciPy)')
plt.legend()
plt.tight_layout()
plt.show()
