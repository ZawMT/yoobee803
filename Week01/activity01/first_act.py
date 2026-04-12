import pandas as pd

# Load the Excel file into a DataFrame
df = pd.read_excel("Data_set_w1A1.xlsx")

# Inspect the structure
# print("Shape (rows, columns):", df.shape)
# print("\nColumn names:", df.columns.tolist())
# print("\nData types:\n", df.dtypes)

# View the data
print("\nFull dataset:\n", df)

# Access a single column
print("\nSales sum per category:\n", df[["category", "sales_sum"]])

# Access a single row by index
# print("\nFirst row:\n", df.iloc[0])
