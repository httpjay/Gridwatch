import pandas as pd
from datetime import datetime

# Load raw census data
df = pd.read_csv("../data/raw/census_zcta_income_housing_2024_acs5.csv")

print("Original shape:", df.shape)

# Remove rows with missing values
df = df.dropna(subset=["median_household_income", "median_year_built"])

# Convert to numeric
df["median_household_income"] = pd.to_numeric(df["median_household_income"], errors="coerce")
df["median_year_built"] = pd.to_numeric(df["median_year_built"], errors="coerce")

# Create housing age feature
current_year = datetime.now().year
df["housing_age"] = current_year - df["median_year_built"]

# Keep only reasonable values
df = df[(df["median_household_income"] > 0) & (df["housing_age"] > 0)]

# Keep only needed columns
df = df[["zcta", "median_household_income", "housing_age"]]

print("Cleaned shape:", df.shape)
print(df.head())

# Save cleaned file
df.to_csv("../data/processed/census_cleaned.csv", index=False)

print("Saved cleaned census data.")