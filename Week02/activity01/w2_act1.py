import glob
import os

import matplotlib.pyplot as plt
import pandas as pd

files = glob.glob('data/PRSA_Data_20130301-20170228/*.csv')
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

print("Task-1: Load and Inspect the Dataset")
print("====================================")
print("\nThe data about Air Quality in Beijing")
print(
    f"\nLoaded {len(files)} files which present data about {len(files)} locations.")
print("Here are the first 5 rows:\n")
print(df.head())
print(f"\nShape: {df.shape[0]} rows, {df.shape[1]} columns")
print("\nColumn names and data types:")
print(df.dtypes)

print("\n\nTask-2: Data Cleaning")
print("=====================")
print("\nNull value analysis:")
null_counts = df.isnull().sum()
null_pct = (null_counts / len(df) * 100).round(2)
null_summary = pd.DataFrame({
    'null_count': null_counts,
    'null_percentage': null_pct
})
null_summary = null_summary[null_summary['null_count'] > 0]

if null_summary.empty:
    print("No null values found.")
else:
    print(null_summary.to_string())
    print(f"\nTotal null values : {df.isnull().sum().sum()}")
    print(
        f"Rows with any null: {df.isnull().any(axis=1).sum()} out of {len(df)}")
    print(f"Columns with nulls: {len(null_summary)} out of {df.shape[1]}")

# Replace missing values
numeric_cols = df.select_dtypes(include='number').columns
categorical_cols = df.select_dtypes(include='object').columns

df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
df[categorical_cols] = df[categorical_cols].fillna(
    df[categorical_cols].mode().iloc[0])

print("\n--- Update afterData Cleaning ---")
print(f"Null values after filling: {df.isnull().sum().sum()}")

# Drop any rows that still have nulls
rows_before = len(df)
df.dropna(inplace=True)
rows_dropped = rows_before - len(df)

print(f"Rows dropped after filling: {rows_dropped}")
print(f"Remaining rows: {len(df)}")

print("\n\nTask-3: Basic Statistical Analysis")
print("===================================")

pollutant_cols = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO',
                  'O3', 'TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM']

for col in pollutant_cols:
    print(f"\n{col}:")
    print(f"  Mean               : {df[col].mean():.2f}")
    print(f"  Median             : {df[col].median():.2f}")
    print(f"  Minimum            : {df[col].min():.2f}")
    print(f"  Maximum            : {df[col].max():.2f}")
    print(f"  Standard Deviation : {df[col].std():.2f}")

os.makedirs('graphs', exist_ok=True)

print("\n\nTask-4: Data Filtering")
print("======================")

pollution_cols = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']

station_avg = df.groupby('station')[pollution_cols].mean().round(2)
station_avg = station_avg.sort_values('PM2.5', ascending=False)

print("\nAverage pollution levels per station (sorted by PM2.5):\n")
print(station_avg.to_string())

print("\n\nTask-5: Data Visualisation")
print("==========================")

# 1. Histogram of PM2.5
plt.figure(figsize=(10, 5))
plt.hist(df['PM2.5'], bins=50, color='steelblue', edgecolor='white')
plt.title('Distribution of PM2.5 Across All Stations (2013–2017)')
plt.xlabel('PM2.5 (µg/m³)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('graphs/histogram_pm25.png')
plt.close()
print("Saved: graphs/histogram_pm25.png")

# 2. Line plot of PM2.5 over time (yearly average)
yearly_avg = df.groupby('year')['PM2.5'].mean().round(2)
plt.figure(figsize=(10, 5))
plt.plot(yearly_avg.index, yearly_avg.values,
         marker='o', color='tomato', linewidth=2)
plt.title('Yearly Average PM2.5 in Beijing (2013–2017)')
plt.xlabel('Year')
plt.ylabel('Average PM2.5 (µg/m³)')
plt.xticks(yearly_avg.index)
plt.tight_layout()
plt.savefig('graphs/lineplot_pm25_yearly.png')
plt.close()
print("Saved: graphs/lineplot_pm25_yearly.png")

# 3. Boxplot of pollutants
plt.figure(figsize=(12, 6))
df[pollution_cols].boxplot()
plt.title('Distribution of Pollutants Across All Stations (2013–2017)')
plt.ylabel('Concentration')
plt.tight_layout()
plt.savefig('graphs/boxplot_pollutants.png')
plt.close()
print("Saved: graphs/boxplot_pollutants.png")

print("\nTask-6: Correlation Analysis")
print("=============================")

# Which variables are most correlated with PM2.5?
numeric_df = df[['PM2.5', 'PM10', 'SO2', 'NO2', 'CO',
                 'O3', 'TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM']]
correlations = numeric_df.corr()['PM2.5'].drop(
    'PM2.5').sort_values(ascending=False)

print("\nCorrelation of all variables with PM2.5:")
for col, val in correlations.items():
    print(f"  {col:<6}: {val:.4f}")

print(
    f"\nMost positively correlated : {correlations.idxmax()} ({correlations.max():.4f})")
print(
    f"Most negatively correlated : {correlations.idxmin()} ({correlations.min():.4f})")

# Does temperature affect pollution levels?
temp_corr = df['TEMP'].corr(df['PM2.5'])
print(f"\nTemperature vs PM2.5 correlation: {temp_corr:.4f}")
if temp_corr < -0.3:
    print("Interpretation: Higher temperatures tend to associate with lower PM2.5 levels.")
elif temp_corr > 0.3:
    print("Interpretation: Higher temperatures tend to associate with higher PM2.5 levels.")
else:
    print("Interpretation: Weak correlation between temperature and PM2.5.")

# Heatmap of correlations
plt.figure(figsize=(10, 8))
corr_matrix = numeric_df.corr()
plt.imshow(corr_matrix, cmap='coolwarm', aspect='auto')
plt.colorbar()
plt.xticks(range(len(corr_matrix.columns)),
           corr_matrix.columns, rotation=45, ha='right')
plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.savefig('graphs/correlation_heatmap.png')
plt.close()
print("\nSaved: graphs/correlation_heatmap.png")
