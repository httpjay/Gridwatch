import pandas as pd

df = pd.read_csv("../data/processed/lead_cleaned_ca.csv")

print("Shape:", df.shape)

# Keep only columns we need
base_df = df[[
    "name",
    "state",
    "energy_burden",
    "household_income",
    "avg_annual_energy_cost",
    "risk_label"
]]

# Rename for clarity
base_df = base_df.rename(columns={
    "name": "city"
})

print(base_df.head())

base_df.to_csv("../data/processed/base_city_dataset.csv", index=False)

print("Saved base city dataset")