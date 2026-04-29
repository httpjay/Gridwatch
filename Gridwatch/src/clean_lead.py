import pandas as pd

# Load LEAD data, skipping metadata rows
df = pd.read_csv("../data/raw/lead_energy_data.csv", skiprows=8)

print("Original shape:", df.shape)

# Keep only important columns
df = df[[
    "Geography ID",
    "Name",
    "State",
    "Energy Burden (% income)",
    "Avg. Annual Energy Cost ($)",
    "Household Income"
]]

# Rename columns
df = df.rename(columns={
    "Geography ID": "geography_id",
    "Name": "name",
    "State": "state",
    "Energy Burden (% income)": "energy_burden",
    "Avg. Annual Energy Cost ($)": "avg_annual_energy_cost",
    "Household Income": "household_income"
})

# Convert numeric columns
df["energy_burden"] = pd.to_numeric(df["energy_burden"], errors="coerce")
df["avg_annual_energy_cost"] = pd.to_numeric(df["avg_annual_energy_cost"], errors="coerce")
df["household_income"] = pd.to_numeric(df["household_income"], errors="coerce")

# Filter to California
df = df[df["state"] == "California"]

print("After California filter:", df.shape)
print(df.head())

# Create risk label
df["risk_label"] = (df["energy_burden"] >= 3).astype(int)

# Save cleaned California-only LEAD data
df.to_csv("../data/processed/lead_cleaned_ca.csv", index=False)

print("Saved cleaned DOE LEAD data to ../data/processed/lead_cleaned_ca.csv")