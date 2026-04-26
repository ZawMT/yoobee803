import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('age_networth.csv')

correlation = df['Age'].corr(df['Net Worth'])
print(f"Pearson Correlation Coefficient: {correlation:.4f}")

m, b = np.polyfit(df['Age'], df['Net Worth'], 1)

plt.figure(figsize=(8, 5))
plt.scatter(df['Age'], df['Net Worth'], color='steelblue', label='Data points')
plt.plot(df['Age'], m * df['Age'] + b, color='red', label=f'Regression line (r = {correlation:.2f})')
plt.xlabel('Age')
plt.ylabel('Net Worth')
plt.title('Linear Relationship between Age and Net Worth')
plt.legend()
plt.tight_layout()
plt.show()
