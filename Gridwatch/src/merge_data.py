import pandas as pd

# Load both datasets
census_df = pd.read_csv("../data/processed/census_cleaned.csv", dtype={"zcta": str})
weather_df = pd.read_csv("../data/processed/weather_90011_summary.csv", dtype={"zcta": str})

# Keep only 90011 from census for this first prototype
census_90011 = census_df[census_df["zcta"] == "90011"]

print("Census row:")
print(census_90011)

print("\nWeather row:")
print(weather_df)

# Merge
merged_df = pd.merge(census_90011, weather_df, on="zcta", how="inner")

print("\nMerged dataset:")
print(merged_df)

# Save
merged_df.to_csv("../data/processed/master_dataset_v1.csv", index=False)

print("\nSaved merged dataset to ../data/processed/master_dataset_v1.csv")