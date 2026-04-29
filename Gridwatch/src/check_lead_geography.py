import pandas as pd

df = pd.read_csv("../data/processed/lead_cleaned_ca.csv")

print("Columns:")
print(df.columns.tolist())

print("\nSample rows:")
print(df[["geography_id", "name", "state"]].head(20))

print("\nGeography ID length counts:")
print(df["geography_id"].astype(str).str.len().value_counts())

print("\nDoes 90011 exist in geography_id?")
print((df["geography_id"].astype(str) == "90011").any())

print("\nSample names:")
print(df["name"].head(20).tolist())