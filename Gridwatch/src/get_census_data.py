import pandas as pd

# Official Census ACS 5-year API endpoint
url = (
    "https://api.census.gov/data/2024/acs/acs5"
    "?get=NAME,B19013_001E,B25035_001E"
    "&for=zip%20code%20tabulation%20area:*"
)

df = pd.read_json(url)

# First row returned by Census is the header row
df.columns = df.iloc[0]
df = df.iloc[1:].reset_index(drop=True)

# Rename columns to cleaner names
df = df.rename(columns={
    "zip code tabulation area": "zcta",
    "B19013_001E": "median_household_income",
    "B25035_001E": "median_year_built"
})

# Keep only useful columns
df = df[["zcta", "median_household_income", "median_year_built"]]

# Convert numeric columns
df["median_household_income"] = pd.to_numeric(df["median_household_income"], errors="coerce")
df["median_year_built"] = pd.to_numeric(df["median_year_built"], errors="coerce")

# Optional: keep only California-like ZCTAs later if needed
print(df.head())
print(df.shape)

df.to_csv("../data/raw/census_zcta_income_housing_2024_acs5.csv", index=False)
print("Saved Census data successfully.")