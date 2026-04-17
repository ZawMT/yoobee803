import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load the CSV file into a DataFrame
df = pd.read_csv("Housing.csv")

# Output folder for graphs
os.makedirs("graphs", exist_ok=True)

print("Shape (rows, columns):", df.shape)
print("\nDataset preview:\n", df.head())

# --- Aggregation by number of bedrooms ---
print("\nAverage price by number of bedrooms:")
print(df.groupby("bedrooms")["price"].mean().round(2))

# --- Aggregation by furnishing status ---
print("\nAverage price by furnishing status:")
print(df.groupby("furnishingstatus")["price"].mean().round(2))

# --- Aggregation by air conditioning ---
print("\nAverage price by air conditioning (yes/no):")
print(df.groupby("airconditioning")["price"].mean().round(2))

# --- Overall summary statistics ---
print("\nSummary statistics for numeric columns:")
print(df[["price", "area", "bedrooms", "bathrooms", "stories", "parking"]].describe().round(2))

# --- Graph 1: Price Distribution ---
plt.figure(figsize=(8, 5))
df["price"].hist(bins=30, color="steelblue", edgecolor="white")
plt.title("Distribution of House Prices")
plt.xlabel("Price")
plt.ylabel("Number of Houses")
plt.tight_layout()
plt.savefig("graphs/price_distribution.png")
plt.close()

# --- Graph 2: Price vs Area Scatter Plot ---
plt.figure(figsize=(8, 5))
plt.scatter(df["area"], df["price"], alpha=0.5, color="coral")
plt.title("Price vs Area")
plt.xlabel("Area (sq ft)")
plt.ylabel("Price")
plt.tight_layout()
plt.savefig("graphs/price_vs_area.png")
plt.close()

# --- Graph 3: Average Price by Bedrooms ---
plt.figure(figsize=(7, 5))
df.groupby("bedrooms")["price"].mean().round(0).plot(kind="bar", color="mediumseagreen", edgecolor="white")
plt.title("Average Price by Number of Bedrooms")
plt.xlabel("Bedrooms")
plt.ylabel("Average Price")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("graphs/avg_price_by_bedrooms.png")
plt.close()

# --- Graph 4: Average Price by Furnishing Status ---
plt.figure(figsize=(7, 5))
df.groupby("furnishingstatus")["price"].mean().round(0).plot(kind="bar", color="mediumpurple", edgecolor="white")
plt.title("Average Price by Furnishing Status")
plt.xlabel("Furnishing Status")
plt.ylabel("Average Price")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("graphs/avg_price_by_furnishing.png")
plt.close()

# --- Graph 5: Correlation Heatmap ---
plt.figure(figsize=(8, 6))
sns.heatmap(df[["price", "area", "bedrooms", "bathrooms", "stories", "parking"]].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("graphs/correlation_heatmap.png")
plt.close()

print("\nGraphs saved to the 'graphs/' folder.")
