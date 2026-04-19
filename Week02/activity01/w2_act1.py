import pandas as pd

df = pd.read_csv(
    'data/PRSA_Data_20130301-20170228/PRSA_Data_Aotizhongxin_20130301-20170228.csv')

print(f"\nThe data about Air Quality in Beijing")
print(f"\nThe following are the first 5 records for the location of Aoti Zhongxin:\n")
print(df.head())
print(f"\nShape: {df.shape[0]} rows, {df.shape[1]} columns")
print("\nColumn names and data types:")
print(df.dtypes)
